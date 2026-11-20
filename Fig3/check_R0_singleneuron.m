%% check_R0_singleneuron.m
%  Diagnostic script for picking the R_0 calibration of the single-neuron
%  panel in main-text Fig.~5A.
%
%  Run this after figure3_main.m has finished (it writes
%  MI_results_direct_corrected.mat).  The script:
%    1. loads the saved (cost, MI, FR) trajectory
%    2. prints firing-rate / MI / cost ranges
%    3. computes R_0 under three anchor choices:
%         min FR, median FR, max FR
%    4. plots I-C and ICE-C with the three candidate frontiers
%    5. reports calibration-dependent data-to-reference gaps descriptively
%
%  Usage:
%    >> figure3_main     % first, to populate MI_results_direct_corrected.mat
%    >> check_R0_singleneuron

clear; clc; close all;

%% ============================== load ==============================
MAT = 'MI_results_direct_corrected.mat';
if ~isfile(MAT)
    error('check_R0_singleneuron:missing_data', ...
        ['MI_results_direct_corrected.mat not found.  Run figure3_main.m first ', ...
         'so it saves the trajectory.']);
end
S = load(MAT);

% Required fields: MI_vec [1 x NWindows], cost1_grand [NWindows x 1],
% ra1_grand [NWindows x 1] (firing rate Hz).  Fall back to alternatives.
MI   = S.MI_vec(:);
cost = S.cost1_grand(:);
if isfield(S, 'ra1_grand')
    FR = S.ra1_grand(:);
else
    % Recover FR from cost via the C = 2.4 + 0.37*FR relation
    FR = (cost - 2.4) / 0.37;
end

% Drop any non-finite or non-physical points
ok = isfinite(MI) & isfinite(cost) & isfinite(FR) & cost > 2.4 & FR > 0;
MI   = MI(ok);
cost = cost(ok);
FR   = FR(ok);
fprintf('Loaded %d valid windows from %s\n', numel(MI), MAT);

%% ============================== diagnostic prints ==============================
fprintf('\n--- single-neuron model trajectory ---\n');
fprintf('  Firing rate (Hz):   min=%.3f  median=%.3f  max=%.3f\n', ...
    min(FR), median(FR), max(FR));
fprintf('  MI         (b/s):   min=%.3f  median=%.3f  max=%.3f\n', ...
    min(MI), median(MI), max(MI));
fprintf('  Cost proxy      :   min=%.3f  median=%.3f  max=%.3f\n', ...
    min(cost), median(cost), max(cost));

%% ============================== R_0 candidates ==============================
% Frontier:  I(C) = (R_0/ln2) * [1 + ln((C-C0)/C1)]
% Anchored at (C_a, I_a):  R_0 = I_a * ln2 / (1 + ln((C_a-C0)/C1))
C0 = 2.4;
C1 = 0.37;
ln2 = log(2);

R0_for_anchor = @(C_a, I_a) I_a * ln2 / (1 + log((C_a - C0)/C1));

% Three canonical anchors
[~, idx_min] = min(FR);
[~, idx_med] = min(abs(FR - median(FR)));   % nearest to median
[~, idx_max] = max(FR);

R0_min = R0_for_anchor(cost(idx_min), MI(idx_min));
R0_med = R0_for_anchor(cost(idx_med), MI(idx_med));
R0_max = R0_for_anchor(cost(idx_max), MI(idx_max));

fprintf('\n--- R_0 candidates ---\n');
fprintf('  min-FR  anchor:  FR=%.3f Hz, C=%.3f W, I=%.3f b/s  ->  R_0 = %.3f\n', ...
    FR(idx_min),  cost(idx_min), MI(idx_min), R0_min);
fprintf('  med-FR  anchor:  FR=%.3f Hz, C=%.3f W, I=%.3f b/s  ->  R_0 = %.3f\n', ...
    FR(idx_med),  cost(idx_med), MI(idx_med), R0_med);
fprintf('  max-FR  anchor:  FR=%.3f Hz, C=%.3f W, I=%.3f b/s  ->  R_0 = %.3f\n', ...
    FR(idx_max),  cost(idx_max), MI(idx_max), R0_max);

%% ============================== calibration-dependent gaps ==============================
frontier_I = @(C, R0) (R0/ln2) .* (1 + log((C - C0)/C1));
gap = @(R0) MI - frontier_I(cost, R0);

frac_above = @(R0) mean(gap(R0) > 0) * 100;

fprintf('\n--- fraction of data points above the calibrated reference ---\n');
fprintf('  R_0 = %.2f (min-FR  anchor):  %.1f%% above\n', R0_min, frac_above(R0_min));
fprintf('  R_0 = %.2f (med-FR  anchor):  %.1f%% above\n', R0_med, frac_above(R0_med));
fprintf('  R_0 = %.2f (max-FR  anchor):  %.1f%% above\n', R0_max, frac_above(R0_max));

%% ============================== plot ==============================
C_grid = linspace(C0 + C1/exp(1)*1.001, max(cost)*1.05, 400);

figure('Color','w','Position',[80 120 1100 460]);

% --- Panel A: I-C ---
subplot(1,2,1); hold on;
scatter(cost, MI, 24, [0.5 0.5 0.5], 'filled', 'MarkerFaceAlpha', 0.55);
plot(C_grid, frontier_I(C_grid, R0_min), '-',  'Color', [0.0 0.4 0.9], 'LineWidth', 1.8);
plot(C_grid, frontier_I(C_grid, R0_med), '-',  'Color', [0.0 0.7 0.4], 'LineWidth', 1.8);
plot(C_grid, frontier_I(C_grid, R0_max), '-',  'Color', [0.9 0.3 0.1], 'LineWidth', 1.8);
plot(cost(idx_min), MI(idx_min), 'o', 'MarkerSize', 12, ...
    'MarkerFaceColor', 'none', 'MarkerEdgeColor', [0.0 0.4 0.9], 'LineWidth', 2);
plot(cost(idx_med), MI(idx_med), 'o', 'MarkerSize', 12, ...
    'MarkerFaceColor', 'none', 'MarkerEdgeColor', [0.0 0.7 0.4], 'LineWidth', 2);
plot(cost(idx_max), MI(idx_max), 'o', 'MarkerSize', 12, ...
    'MarkerFaceColor', 'none', 'MarkerEdgeColor', [0.9 0.3 0.1], 'LineWidth', 2);
xlabel('Affine cost proxy', 'FontSize', 12);
ylabel('MI (bits/s)', 'FontSize', 12);
title('I-C plane (single-neuron)', 'FontSize', 13);
legend({'data', sprintf('R_0=%.2f (min)', R0_min), ...
                 sprintf('R_0=%.2f (med)', R0_med), ...
                 sprintf('R_0=%.2f (max)', R0_max)}, ...
       'Location', 'northwest', 'FontSize', 10);
ylim([0, max(MI)*1.2]);
box off;

% --- Panel B: information per proxy cost and reference slope ---
% Average yield and local slope are distinct. Because the affine proxy has
% a non-zero baseline, no origin-independent inequality between them is used.
subplot(1,2,2); hold on;
ICE_data = MI ./ cost;
beta_curve = @(C, R0) (R0 / ln2) ./ (C - C0);   % dI/dC at C, anchored R_0

scatter(cost, ICE_data, 24, [0.5 0.5 0.5], 'filled', 'MarkerFaceAlpha', 0.55);

plot(C_grid, frontier_I(C_grid, R0_min)./C_grid, '-',  'Color', [0.0 0.4 0.9], 'LineWidth', 1.8);
plot(C_grid, frontier_I(C_grid, R0_med)./C_grid, '-',  'Color', [0.0 0.7 0.4], 'LineWidth', 1.8);
plot(C_grid, frontier_I(C_grid, R0_max)./C_grid, '-',  'Color', [0.9 0.3 0.1], 'LineWidth', 1.8);

plot(C_grid, beta_curve(C_grid, R0_min), '--', 'Color', [0.0 0.4 0.9], 'LineWidth', 1.2);
plot(C_grid, beta_curve(C_grid, R0_med), '--', 'Color', [0.0 0.7 0.4], 'LineWidth', 1.2);
plot(C_grid, beta_curve(C_grid, R0_max), '--', 'Color', [0.9 0.3 0.1], 'LineWidth', 1.2);

xlabel('Affine cost proxy', 'FontSize', 12);
ylabel('information / calibrated cost proxy', 'FontSize', 12);
title('Information per proxy cost and reference slope', 'FontSize', 13);
legend({'data ICE', ...
        sprintf('ICE  R_0=%.2f (min)', R0_min), ...
        sprintf('ICE  R_0=%.2f (med)', R0_med), ...
        sprintf('ICE  R_0=%.2f (max)', R0_max), ...
        sprintf('\\beta  R_0=%.2f (min)', R0_min), ...
        sprintf('\\beta  R_0=%.2f (med)', R0_med), ...
        sprintf('\\beta  R_0=%.2f (max)', R0_max)}, ...
       'Location', 'northeast', 'FontSize', 8);
% Y-axis: cap upper bound at the data ICE max so beta tangent at small C
% (which diverges) doesn't squash the visible region.
y_top = max( max(ICE_data) * 1.3, ...
             max(frontier_I(C_grid, R0_min) ./ C_grid) * 1.1 );
ylim([0, y_top]);
box off;

sgtitle('Single-neuron R_0 anchor sensitivity', ...
    'FontSize', 13, 'FontWeight', 'bold');
