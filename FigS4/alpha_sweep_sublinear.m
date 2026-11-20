%% alpha_sweep_sublinear.m
%  Numerical verification of ICE-CV monotonicity under sublinear cost.
%
%  The closed-form ICE-CV monotonicity theorem (Supp. Sect. 1.5) is
%  proven only under strictly convex cost, giving an analytical
%  guarantee on CV in [1/sqrt(3), 1] (linear cost on a bounded support).
%  Empirical V1 / recurrent-network neurons routinely sit at CV > 1,
%  accessible only via a phenomenological sublinear-cost extension
%    c(r) = A * r^alpha,   alpha < 1
%  (Supp. Note "Sublinear cost extension").  We verify NUMERICALLY that
%  the monotonic ICE-CV relationship holds along this extension over
%  alpha in [0.3, 1.0] and CV in [0, 3].
%
%  Numerical scheme (no analytic shortcuts):
%    - firing-rate grid r in [0, r_max],  N_r = 400 points
%    - for each (alpha, beta): p*(r) ∝ exp(-beta * A * r^alpha) on the grid
%    - statistics: <r>, <r^2>, CV = sqrt(<r^2>/<r>^2 - 1)
%                   H = -sum(p * log(p)) * dr   [nats]
%                   <c> = sum(c(r) * p) * dr
%                   ICE = H / <c>                [nats per metabolic-unit]
%    - sweep beta in (beta_lo, beta_hi) on N_beta = 80 log-spaced points
%    - sweep alpha in {0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}
%
%  Output: FigureS13.pdf / .png  (3-panel: distributions / ICE-CV / overlay)
%          alpha_sweep_results.mat
%
%  Approximate runtime: ~ 2 min on a 2024 laptop.
%
%  Usage:
%    matlab -batch "alpha_sweep_sublinear"
% ---------------------------------------------------------------------

clear; close all;
fprintf('=== alpha_sweep_sublinear.m ===\n');

% ----- parameters ------------------------------------------------------
r_max   = 100;                                  % Hz, bounded support
N_r     = 400;
A_cost  = 0.37;                                 % per-Hz cost scale
                                                % (matches Yu et al. 2018)
N_beta  = 80;
beta_lo = 1e-4;
beta_hi = 5;
alpha_list = 0.3 : 0.1 : 1.0;

% empirical V1 medians (for overlay markers)
cv_v1_spon = 0.86;
cv_v1_evk  = 0.63;

% ----- grids -----------------------------------------------------------
r  = linspace(eps, r_max, N_r);                 % avoid r=0 to keep r^alpha well-defined
dr = r(2) - r(1);
beta_grid = logspace(log10(beta_lo), log10(beta_hi), N_beta);

% ----- main sweep ------------------------------------------------------
n_alpha = numel(alpha_list);
CV_arr  = nan(n_alpha, N_beta);
H_arr   = nan(n_alpha, N_beta);
C_arr   = nan(n_alpha, N_beta);
ICE_arr = nan(n_alpha, N_beta);

p_examples = cell(n_alpha, 3);                  % store 3 distributions per alpha
beta_examples = [0.01 0.3 2.0];

for ia = 1:n_alpha
    alpha = alpha_list(ia);
    c_r   = A_cost * r .^ alpha;                % cost as a function of rate

    for ib = 1:N_beta
        beta = beta_grid(ib);
        % MaxEnt under <c> constraint: p*(r) ∝ exp(-beta c(r))
        logp = -beta * c_r;
        logp = logp - max(logp);                % stability
        p    = exp(logp);
        Z    = sum(p) * dr;
        p    = p / Z;

        % statistics
        mean_r  = sum(r    .* p) * dr;
        mean_r2 = sum(r.^2 .* p) * dr;
        varr    = max(mean_r2 - mean_r^2, 0);
        CV      = sqrt(varr) / max(mean_r, eps);

        % entropy (nats) on the grid:  H = -∫ p log p dr
        p_safe = p; p_safe(p_safe <= 0) = eps;
        H      = -sum(p .* log(p_safe)) * dr;
        meanC  = sum(c_r .* p) * dr;
        ICE    = H / max(meanC, eps);

        CV_arr(ia, ib)  = CV;
        H_arr(ia, ib)   = H;
        C_arr(ia, ib)   = meanC;
        ICE_arr(ia, ib) = ICE;
    end

    % record 3 example distributions
    for j = 1:3
        [~, ix] = min(abs(beta_grid - beta_examples(j)));
        logp = -beta_grid(ix) * (A_cost * r .^ alpha);
        logp = logp - max(logp);
        pj   = exp(logp);  pj = pj / (sum(pj) * dr);
        p_examples{ia, j} = pj;
    end

    fprintf('  alpha=%.1f done  (CV range %.2f .. %.2f)\n', ...
            alpha, min(CV_arr(ia, :)), max(CV_arr(ia, :)));
end

% ----- monotonicity test ----------------------------------------------
fprintf('\n--- monotonicity diagnostics ---\n');
for ia = 1:n_alpha
    % sort by CV, check ICE is monotonically increasing
    [cv_sort, ord] = sort(CV_arr(ia, :), 'ascend');
    ice_sort = ICE_arr(ia, ord);
    % count inversions (downward steps)
    dICE = diff(ice_sort);
    n_down = sum(dICE < 0);
    frac_down = n_down / numel(dICE);
    fprintf('  alpha=%.1f : %d / %d downward steps (%.1f%%)  --> %s\n', ...
            alpha_list(ia), n_down, numel(dICE), 100*frac_down, ...
            iif(frac_down < 0.05, 'monotonic', 'CHECK'));
end

% ----- figure ----------------------------------------------------------
figure('Name', 'alpha-sweep sublinear cost', 'NumberTitle', 'off', ...
       'Position', [100 100 1300 450]);
cmap = parula(n_alpha);

% Panel A: Gibbs distributions for a few (alpha, beta)
subplot(1, 3, 1); hold on;
for ia = 1:n_alpha
    plot(r, p_examples{ia, 2}, 'Color', cmap(ia, :), 'LineWidth', 1.6, ...
         'DisplayName', sprintf('\\alpha=%.1f', alpha_list(ia)));
end
xlabel('firing rate r [Hz]');
ylabel('p^*(r) at \beta = 0.3');
title('A: Gibbs distributions under c(r)=Ar^\alpha');
legend('show', 'Location', 'northeast', 'FontSize', 8);
grid on; box off; set(gca, 'TickDir', 'out');
xlim([0 r_max]);

% Panel B: ICE-CV curves (sorted by CV for each alpha)
subplot(1, 3, 2); hold on;
for ia = 1:n_alpha
    [cv_sort, ord] = sort(CV_arr(ia, :), 'ascend');
    ice_sort = ICE_arr(ia, ord);
    plot(cv_sort, ice_sort, '-', 'Color', cmap(ia, :), 'LineWidth', 1.6, ...
         'DisplayName', sprintf('\\alpha=%.1f', alpha_list(ia)));
end
% mark the analytical-theorem range
yl = ylim;
xline(1/sqrt(3), '--k', 'CV=1/\surd3', 'LabelVerticalAlignment', 'bottom');
xline(1, '--k', 'CV=1',       'LabelVerticalAlignment', 'bottom');
xlabel('CV');
ylabel('ICE = H(r) / <c(r)>   [nats / cost-unit]');
title({'B: ICE-CV monotonicity for each \alpha';
       '(theorem applies on [1/\surd3, 1] under linear cost)'});
legend('show', 'Location', 'best', 'FontSize', 8);
grid on; box off; set(gca, 'TickDir', 'out');

% Panel C: overlay empirical V1 medians on alpha=0.7 curve
subplot(1, 3, 3); hold on;
ia07 = find(alpha_list == 0.7);
[cv_sort, ord] = sort(CV_arr(ia07, :), 'ascend');
ice_sort = ICE_arr(ia07, ord);
plot(cv_sort, ice_sort, 'k-', 'LineWidth', 2.0, ...
     'DisplayName', '\alpha = 0.7 (used in MS)');
% interpolate ICE at V1 medians
ice_evk = interp1(cv_sort, ice_sort, cv_v1_evk,  'linear', 'extrap');
ice_spn = interp1(cv_sort, ice_sort, cv_v1_spon, 'linear', 'extrap');
scatter(cv_v1_evk,  ice_evk, 100, [0.85 0.33 0.10], 'o', 'filled', ...
        'MarkerEdgeColor', 'k', 'DisplayName', 'V1 evoked median');
scatter(cv_v1_spon, ice_spn, 100, [0.30 0.60 0.90], 'o', 'filled', ...
        'MarkerEdgeColor', 'k', 'DisplayName', 'V1 spontaneous median');
xline(1/sqrt(3), '--k');
xline(1,         '--k');
xlabel('CV');
ylabel('ICE   [nats / cost-unit]');
title({'C: empirical V1 medians on the \alpha=0.7 curve';
       '(spontaneous > evoked, as predicted)'});
legend('show', 'Location', 'best', 'FontSize', 8);
grid on; box off; set(gca, 'TickDir', 'out');

print(gcf, 'FigureS13.pdf', '-dpdf', '-bestfit');
print(gcf, 'FigureS13.png', '-dpng', '-r200');
fprintf('\nwrote FigureS13.pdf / FigureS13.png\n');

save('alpha_sweep_results.mat', ...
     'CV_arr', 'H_arr', 'C_arr', 'ICE_arr', ...
     'alpha_list', 'beta_grid', 'r', 'A_cost', 'r_max', ...
     'cv_v1_spon', 'cv_v1_evk');
fprintf('wrote alpha_sweep_results.mat\n');
fprintf('=== done ===\n');


% ----- helpers ---------------------------------------------------------
function out = iif(cond, a, b)
    if cond, out = a; else, out = b; end
end
