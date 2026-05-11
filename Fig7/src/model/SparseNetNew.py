import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SparseNet(nn.Module):
    """
    Sparse coding model with ISTA inference and three constraint types:
      - 'non' : no constraint (baseline)
      - 's'   : Cauchy sparsity prior (penalise large coefficients)
      - 'cv'  : CV maximisation (encourage diverse per-neuron activations)

    Key fix vs original: the constraint penalty is included in BOTH the
    R-update (ISTA inner loop) AND the U-update (outer loop), so that
    the dictionary learns to support the desired coding property.
    """

    def __init__(self, K: int, M: int, R_lr: float = 0.1,
                 lmda: float = 5e-3, cons='cv', device=None,
                 cv_ratio: float = 0.02, alpha_s: float = 1.0,
                 recon_budget: float = 1.0):
        super(SparseNet, self).__init__()
        self.K = K          # number of dictionary atoms (neurons)
        self.M = M          # patch side length → patch dim = M²
        self.R_lr = R_lr    # ISTA learning rate
        self.lmda = lmda    # soft-thresholding parameter
        self.cons = cons    # constraint type: 'non', 's', 'cv'
        self.cv_ratio = cv_ratio  # target |CV_penalty| / |recon| ratio
        self.alpha_s = alpha_s
        self.recon_budget = recon_budget  # cv_constrained: allow recon ≤ budget × warmup_recon

        self.device = torch.device("cpu") if device is None else device
        self.U = nn.Linear(self.K, self.M ** 2, bias=False).to(self.device)
        self.R = None
        self.normalize_weights()

    # ── constraint helpers (operate on live tensors, keep gradients) ──

    def _population_cv(self, R):
        """
        Population sparsity: for each input patch, compute CV of |a|
        across K neurons, then average across the batch.
        CV_i = std(|R[i,:]|) / (mean(|R[i,:]|) + eps)   for patch i
        Returns scalar (mean CV across batch).
        """
        R_abs = torch.abs(R)                        # [batch, K]
        mu = R_abs.mean(dim=1)                       # [batch]
        sigma = R_abs.std(dim=1)                     # [batch]
        cv_per_patch = sigma / (mu + 1e-8)           # [batch]
        return cv_per_patch.mean()

    def _lifetime_cv(self, R):
        """
        Lifetime sparsity (supplementary): for each neuron k, compute CV
        of |a| across patches, then average across neurons.
        CV_k = std(|R[:,k]|) / (mean(|R[:,k]|) + eps)   for neuron k
        """
        R_abs = torch.abs(R)                        # [batch, K]
        mu = R_abs.mean(dim=0)                       # [K]
        sigma = R_abs.std(dim=0)                     # [K]
        cv_per_neuron = sigma / (mu + 1e-8)          # [K]
        return cv_per_neuron.mean()

    def _cauchy_sparsity(self, R):
        """
        Cauchy log-prior: sum log(1 + R²).
        Larger value → less sparse → should be ADDED to loss to penalise.
        """
        return torch.sum(torch.log(1 + R ** 2))

    def compute_constraint_loss(self, R, recon_detached=None):
        """
        Return the constraint penalty to be ADDED to reconstruction loss.
          - 'cv'                : adaptive α · (−CV).  α = cv_ratio × |recon|/|CV|
                                  so the CV term is always cv_ratio fraction of recon.
          - 'cv_constrained'    : same α·(−CV) — U is pushed toward high CV
                                  *unconditionally* (ISTA is the only place gated).
          - 'cv_constrained_v2' : α·(−CV) is applied to U **only when the last
                                  ISTA iter actually had CV active** (i.e. recon
                                  was within budget at convergence).  When recon
                                  drifted out of budget, U gets pure-recon
                                  gradient — letting the dictionary recover RE
                                  before being pulled back toward CV.
          - 's'                 : +alpha_s × cauchy
          - 'non'               : 0
        """
        if self.cons in ('cv', 'cv_constrained', 'cv_constrained_v2'):
            if self.cons == 'cv_constrained_v2' and not getattr(
                    self, 'cv_active_last', False):
                return torch.tensor(0.0, device=self.device)
            cv_val = self._population_cv(R)
            if recon_detached is not None and cv_val.detach().abs() > 1e-8:
                alpha = self.cv_ratio * recon_detached / cv_val.detach().abs()
            else:
                alpha = self.cv_ratio * 1000.0  # fallback for first step
            return -alpha * cv_val
        elif self.cons == 's':
            return self.alpha_s * self._cauchy_sparsity(R)
        else:
            return torch.tensor(0.0, device=self.device)

    # ── ISTA inference ────────────────────────────────────────────────

    def ista_(self, img_batch):
        """Find sparse coefficients R for a given batch, using ISTA.

        Strategy:
          - Phase 1 (warmup, iters 1..warmup): recon + L1 only → let R
            settle to a reasonable non-zero solution.
          - Phase 2 (iters warmup+1..max_iter): add constraint penalty
            (Cauchy or CV) on top of recon + L1.
          - CV is safe in phase 2 because R is already non-zero, avoiding
            the grad instability at R≈0.
          - Gradient clipping ONLY for CV in phase 2 (avoids slowing
            convergence for null / cauchy).
        """
        self.R = torch.zeros((img_batch.shape[0], self.K),
                             requires_grad=True, device=self.device)
        optim = torch.optim.SGD([self.R], lr=self.R_lr)
        converged = False
        max_iter = 100
        warmup = 20     # first N iters: recon + L1 only
        it = 0
        cv_active = False  # track whether CV penalty is active this iter
        recon_warmup = None  # reconstruction baseline at end of warmup (cv_constrained)

        while not converged and it < max_iter:
            it += 1
            old_R = self.R.clone().detach()
            optim.zero_grad()
            cv_active = False

            pred = self.U(self.R)
            recon = ((img_batch - pred) ** 2).sum()

            # Record reconstruction baseline at end of warmup phase
            # (only v1 uses it; v2 uses an absolute RE gate)
            if it == warmup and self.cons == 'cv_constrained':
                recon_warmup = recon.detach().clone()

            # Phase 2: add constraint after warmup
            if it > warmup and self.cons == 's':
                loss = recon + self.alpha_s * self._cauchy_sparsity(self.R)
            elif it > warmup and self.cons == 'cv':
                cv_val = self._population_cv(self.R)
                alpha = self.cv_ratio * recon.detach() / (cv_val.detach().abs() + 1e-8)
                loss = recon - alpha * cv_val
                cv_active = True
            elif it > warmup and self.cons == 'cv_constrained':
                cv_val = self._population_cv(self.R)
                alpha = self.cv_ratio * recon.detach() / (cv_val.detach().abs() + 1e-8)
                # v1 gate: relative-to-warmup baseline (self-tracking).
                # threshold = recon_warmup × budget.  Because recon usually
                # keeps decreasing past warmup, this gate is rarely closed in
                # practice — v1 is essentially equivalent to plain `cv` for
                # most runs.
                threshold = (recon_warmup * self.recon_budget
                             if recon_warmup is not None
                             else recon.detach() * 1.5)
                if recon.detach() <= threshold:
                    loss = recon - alpha * cv_val
                    cv_active = True
                else:
                    loss = recon
            elif it > warmup and self.cons == 'cv_constrained_v2':
                cv_val = self._population_cv(self.R)
                alpha = self.cv_ratio * recon.detach() / (cv_val.detach().abs() + 1e-8)
                # v2 gate: absolute RE target.  recon_budget is interpreted as
                # the target relative reconstruction error: CV is only pushed
                # when current RE ≤ recon_budget.  This forces the dictionary
                # to learn pure-recon until RE hits the target, only then does
                # CV pressure kick in.  Sweep recon_budget over (0.4, 0.5, 0.55,
                # 0.6) — values < 1.0 — to actually exercise the gate.
                re_now = (recon.detach().sqrt()
                          / (img_batch.norm() + 1e-12))
                if re_now <= self.recon_budget:
                    loss = recon - alpha * cv_val
                    cv_active = True
                else:
                    loss = recon
            else:
                loss = recon

            loss.backward(retain_graph=True)

            # gradient clipping ONLY when CV is active (prevents instability)
            if cv_active:
                with torch.no_grad():
                    grad_norm = self.R.grad.norm()
                    if grad_norm > 100.0:
                        self.R.grad.mul_(100.0 / grad_norm)

            optim.step()

            # proximal operator (soft thresholding)
            with torch.no_grad():
                self.R.data = F.relu(self.R.data - self.lmda) \
                            - F.relu(-self.R.data - self.lmda)

            with torch.no_grad():
                converged = (torch.norm(self.R - old_R)
                             / (torch.norm(old_R) + 1e-8)) < 0.01

        # whether CV penalty was active at the converged ISTA state — used by
        # cv_constrained_v2 to gate the U-update too
        self.cv_active_last = bool(cv_active)

        # store diagnostics (detached)
        with torch.no_grad():
            R_abs = torch.abs(self.R)
            self.E = ((img_batch - self.U(self.R)) ** 2).sum().item()
            self.m = R_abs.mean().item()
            self.sig = R_abs.std().item()
            self.cv_val = self._population_cv(self.R).item()
            self.cv_lifetime = self._lifetime_cv(self.R).item()
            self.S = self._cauchy_sparsity(self.R).item()

    # ── forward / helpers ─────────────────────────────────────────────

    def forward(self, img_batch):
        self.ista_(img_batch)
        pred = self.U(self.R)
        return pred

    def normalize_weights(self):
        with torch.no_grad():
            self.U.weight.data = F.normalize(self.U.weight.data, dim=0)

    def get_total_loss(self, img_batch, pred):
        """
        Combined loss for U update: reconstruction + constraint.
        Call AFTER forward() so that self.R is populated.
        """
        recon = ((img_batch - pred) ** 2).sum()
        penalty = self.compute_constraint_loss(self.R,
                                               recon_detached=recon.detach())
        return recon + penalty
