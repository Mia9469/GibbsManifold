%% redo_fig5_plots.m
%  Apply the Cost-unit fix in-place and redraw Fig 2 (MI vs Cost +
%  theoretical Gibbs frontier with auto-anchored R_0).  Assumes the
%  Figure5_new.m workspace is still loaded (variables PcostE, PcostI,
%  MIE_ST, MIE_temporal, MIE_spatial, HrE_ST, ..., TimeT, dt, T, Trials,
%  orange).  No simulation re-run needed.

if ~exist('PcostE', 'var') || ~exist('MIE_ST', 'var') || ~exist('TimeT', 'var')
    error(['redo_fig5_plots: required variables not found in workspace. ', ...
           'Run Figure5_new.m once first (or re-run only the simulation loop).']);
end

%% --- Cost-unit fix (idempotent guard) ---
% Old:  PcostE = 2.4 + 0.37 * mean(spike_count)
% New:  PcostE = 2.4 + 0.37 * mean(spike_count)/T_obs
%       <=>   new = 2.4 + (old - 2.4)/T_obs
% Guard: if PcostE is already in the realistic single-digit-W range
%        (max <= 30 W say), skip the correction so we don't shrink twice.
T_obs = TimeT * dt;
if max(PcostE(:)) > 30
    fprintf('Applying cost-unit fix:  PcostE = 2.4 + (PcostE - 2.4) / T_obs  (T_obs = %.3g s)\n', T_obs);
    PcostE = 2.4 + (PcostE - 2.4) / T_obs;
    PcostI = 2.4 + (PcostI - 2.4) / T_obs;
else
    fprintf('PcostE max already small (%.2f W); skipping cost-unit correction (already applied).\n', max(PcostE(:)));
end

fprintf('PcostE range after fix:  min=%.2f, median=%.2f, max=%.2f W\n', ...
    min(PcostE(:)), median(PcostE(:)), max(PcostE(:)));

%% --- Pick R_0 ---
% Three options, in order of preference:
%   (i)   manual override  (set R_0_OVERRIDE to a number; leave [] to skip)
%   (ii)  5%-above scan    (smallest R_0 such that <=5% of data points
%                           sit above the frontier -- principled choice
%                           for an upper-envelope anchor; matches the
%                           single-neuron diagnostic in Fig4/check_R0_singleneuron.m)
%   (iii) min-FR anchor    (fallback: makes the lowest-cost data point
%                           sit exactly ON the frontier; can give very
%                           small R_0 if that point is noisy)
R_0_OVERRIDE = 1.0;     % set to [] to disable; otherwise this is used as R_0_rn

C_0 = 2.4;
C_1 = 0.37;
ln2 = log(2);

PcostE_mean = mean(PcostE, 1, 'omitnan');
MIE_ST_mean = mean(MIE_ST, 1, 'omitnan');
valid_anchor = (PcostE_mean > C_0 + 1e-3) & (MIE_ST_mean > 0);
valid_idx = find(valid_anchor);

% Pre-compute candidate anchors for diagnostics regardless of override
[~, ia_min] = min(PcostE_mean(valid_anchor));
a_idx_min = valid_idx(ia_min);
C_anchor  = PcostE_mean(a_idx_min);
I_anchor  = MIE_ST_mean(a_idx_min);
R0_min    = I_anchor * ln2 / (1 + log((C_anchor - C_0) / C_1));

% 5%-above scan over the valid (cost, MI) pairs
frontier_I = @(C, R0) (R0/ln2) .* (1 + log((C - C_0)/C_1));
costs_v    = PcostE_mean(valid_anchor)';
MIs_v      = MIE_ST_mean(valid_anchor)';
frac_above = @(R0) mean(MIs_v - frontier_I(costs_v, R0) > 0) * 100;
R0_scan    = linspace(0.1, max(R0_min*5, 5), 500);
fa         = arrayfun(frac_above, R0_scan);
mask5      = fa <= 5;
if any(mask5)
    R0_5pct = min(R0_scan(mask5));
else
    R0_5pct = R0_min;
end

% Choose
if ~isempty(R_0_OVERRIDE)
    R_0_rn   = R_0_OVERRIDE;
    chose    = sprintf('manual override R_0 = %.3f', R_0_rn);
else
    R_0_rn   = R0_5pct;
    chose    = sprintf('5%%-above scan R_0 = %.3f', R_0_rn);
end

fprintf('Frontier R_0 candidates:\n');
fprintf('  min-FR anchor:        R_0 = %.3f  (anchor C=%.2f W, MI=%.2f bits/s)\n', ...
    R0_min, C_anchor, I_anchor);
fprintf('  5%%-above scan:        R_0 = %.3f\n', R0_5pct);
fprintf('  -> using:             %s\n', chose);

%% --- Frontier curves ---
x_ref  = C_0 + (0.001:0.1:max(15, max(PcostE_mean)));
r_vec  = (x_ref - C_0) / C_1;
y1_ref = (R_0_rn / ln2) * (1 + log(r_vec));        % I_frontier  [bits/s]
y2_ref = (R_0_rn / ln2) ./ (x_ref - C_0);          % beta = dI/dC [bits/J]
ICE    = y1_ref ./ x_ref;                          % ICE_frontier [bits/J]

%% --- Re-draw Figure 2 ---
if ~exist('orange', 'var'), orange = [1 0.5 0.2]; end
figure('Color','w','Name','Fig 5  MI vs Cost (cost fixed, R_0 anchored)');

subplot(2,2,1); hold on;
plot(x_ref, y1_ref, 'k-', 'LineWidth', 1.4);
for ii = 1:size(MIE_ST, 2)
    scatter(PcostE(:,ii), MIE_ST(:,ii), 36, orange, 's', 'filled');
end
plot(C_anchor, I_anchor, 'kx', 'MarkerSize', 14, 'LineWidth', 2);
xlabel('Cost E [W]'); ylabel('MI [bits/s]');
title(sprintf('I-C (E pop, S+T)   R_0=%.2f', R_0_rn));
xlim([2 max(x_ref)]); box off;

subplot(2,2,3); hold on;
plot(x_ref, y2_ref, 'k--', 'LineWidth', 1.0);
plot(x_ref, ICE,    'k-',  'LineWidth', 1.4);
for ii = 1:size(MIE_ST, 2)
    scatter(PcostE(:,ii), MIE_ST(:,ii) ./ PcostE(:,ii), 36, orange, 's', 'filled');
end
xlabel('Cost E [W]'); ylabel('ICE [bits/J]');
title('ICE-C   (data + frontier ICE solid + \beta dashed)');
xlim([2 max(x_ref)]); box off;
legend({'\beta = dI/dC', 'ICE = I/C', 'data'}, 'Location', 'best', 'FontSize', 8);
