%% variance_decomp_CVP.m
%  Nested-ANOVA decomposition of per-neuron x per-trial firing-rate
%  variance into across-neuron, across-trial, and interaction
%  components, separately for spontaneous and evoked V1 trials.
%
%  Purpose (manuscript R2, "Two samplings of CV diverge in direction"):
%  the lifetime CV (CV_L, across trials per neuron) and the per-trial
%  population CV (CV_P, across neurons per trial) move in opposite
%  directions across the spontaneous-evoked contrast on this dataset.
%  The proposed mechanism is that the two samplings integrate over
%  different variance axes:
%    - CV_L is dominated by trial-to-trial drift (large in spontaneous)
%    - CV_P is dominated by across-neuron heterogeneity (large in
%      evoked, because stimulus tuning makes a few neurons drive the
%      population mean upward while most stay near baseline)
%  This script verifies that mechanism quantitatively on the v1v2-1
%  data.  Outputs a 2-panel supp figure (stacked variance bars +
%  paired across-neuron vs across-trial comparison) and a .mat with
%  the per-session decomposition.
%
%  Decomposition: for each session, build a firing-rate matrix R of
%  size [n_neurons x n_trials] (firing rate in Hz, computed by summing
%  refractory-corrected spike counts over the appropriate driven /
%  spontaneous window and dividing by the window duration).  Then:
%    grand_mean      = mean(R(:))
%    SS_total        = sum((R(:) - grand_mean).^2)
%    SS_neuron       = sum(n_trials * (rowMean - grand_mean).^2)
%    SS_trial        = sum(n_neurons * (colMean - grand_mean).^2)
%    SS_interaction  = SS_total - SS_neuron - SS_trial
%  reported as fractions of SS_total.  This is the standard two-way
%  ANOVA fixed-effects decomposition; the "interaction + residual"
%  term is treated as one bucket since with one observation per
%  (neuron, trial) cell they are not separately identifiable.
%
%  Output: FigureS14.pdf / .png  +  variance_decomp_CVP.mat
%  Approximate runtime: < 1 min.
%
%  Usage:
%    matlab -batch "variance_decomp_CVP"
% ---------------------------------------------------------------------

clear; close all;
fprintf('=== variance_decomp_CVP.m ===\n');

% ----- parameters ------------------------------------------------------
V1V2_FILES = {
    '105l001p16.mat'
    '106r001p26.mat'
    '106r002p70.mat'
    '107l002p67.mat'
    '107l003p143.mat'
};
DUR_DRV  = 1.28;                                % driven window (s)
DUR_SPN  = 1.50;                                % spontaneous window (s)
N_BINS_D = 1280;
N_BINS_S = 1500;

% ----- allocate per-session results ------------------------------------
nS = numel(V1V2_FILES);
F_evk = nan(nS, 3);                             % [neuron, trial, interaction] fractions
F_spn = nan(nS, 3);
CVL_evk_sess = nan(nS, 1);  CVP_evk_sess = nan(nS, 1);
CVL_spn_sess = nan(nS, 1);  CVP_spn_sess = nan(nS, 1);

for is = 1:nS
    fname = V1V2_FILES{is};
    fprintf('\n--- session %d/%d: %s ---\n', is, nS, fname);
    if ~isfile(fname)
        warning('  not found, skipping');
        continue;
    end
    S = load(fname);
    nd = S.neuralData;
    nt_total = max(nd.trialId);

    % collect driven and spontaneous rasters for V1
    rast_evk = {}; rast_spn = {};
    for i = 1:nt_total
        sv1 = nd.spikeRasters{i, 1};
        if size(sv1, 2) == N_BINS_D
            rast_evk{end + 1} = sv1;                                        %#ok<AGROW>
        elseif size(sv1, 2) == N_BINS_S
            rast_spn{end + 1} = sv1;                                        %#ok<AGROW>
        end
    end
    n_v1 = size(rast_evk{1}, 1);
    n_drv = numel(rast_evk);
    n_spn = numel(rast_spn);
    fprintf('  n_v1=%d, n_drv=%d, n_spn=%d\n', n_v1, n_drv, n_spn);

    % firing-rate matrices  R = [neurons x trials]
    R_evk = zeros(n_v1, n_drv);
    for tr = 1:n_drv
        R_evk(:, tr) = sum(rast_evk{tr} > 0, 2) ./ DUR_DRV;
    end
    R_spn = zeros(n_v1, n_spn);
    for tr = 1:n_spn
        R_spn(:, tr) = sum(rast_spn{tr} > 0, 2) ./ DUR_SPN;
    end

    F_evk(is, :) = decompose_two_way(R_evk);
    F_spn(is, :) = decompose_two_way(R_spn);

    % per-session CV_L / CV_P for context
    CVL_evk_sess(is) = median(std(R_evk, 0, 2) ./ max(mean(R_evk, 2), eps));
    CVP_evk_sess(is) = median(std(R_evk, 0, 1) ./ max(mean(R_evk, 1), eps));
    CVL_spn_sess(is) = median(std(R_spn, 0, 2) ./ max(mean(R_spn, 2), eps));
    CVP_spn_sess(is) = median(std(R_spn, 0, 1) ./ max(mean(R_spn, 1), eps));

    fprintf(['  evoked  : F_neuron=%.2f F_trial=%.2f F_int=%.2f  ' ...
             'CV_L=%.2f CV_P=%.2f\n'], F_evk(is,1), F_evk(is,2), F_evk(is,3), ...
             CVL_evk_sess(is), CVP_evk_sess(is));
    fprintf(['  spont   : F_neuron=%.2f F_trial=%.2f F_int=%.2f  ' ...
             'CV_L=%.2f CV_P=%.2f\n'], F_spn(is,1), F_spn(is,2), F_spn(is,3), ...
             CVL_spn_sess(is), CVP_spn_sess(is));

    clear S nd rast_evk rast_spn R_evk R_spn;
end

% ----- pooled summary ---------------------------------------------------
F_evk_mean = mean(F_evk, 1, 'omitnan');
F_evk_sem  = std (F_evk, 0, 1, 'omitnan') ./ sqrt(sum(isfinite(F_evk(:,1))));
F_spn_mean = mean(F_spn, 1, 'omitnan');
F_spn_sem  = std (F_spn, 0, 1, 'omitnan') ./ sqrt(sum(isfinite(F_spn(:,1))));

fprintf('\n--- pooled across 5 sessions ---\n');
fprintf('  evoked  : F_neuron = %.2f +/- %.2f   F_trial = %.2f +/- %.2f   F_int = %.2f\n', ...
        F_evk_mean(1), F_evk_sem(1), F_evk_mean(2), F_evk_sem(2), F_evk_mean(3));
fprintf('  spont   : F_neuron = %.2f +/- %.2f   F_trial = %.2f +/- %.2f   F_int = %.2f\n', ...
        F_spn_mean(1), F_spn_sem(1), F_spn_mean(2), F_spn_sem(2), F_spn_mean(3));

% paired tests (Wilcoxon signed rank, n=5 sessions)
[p_neuron, ~] = signrank(F_evk(:,1), F_spn(:,1));
[p_trial,  ~] = signrank(F_evk(:,2), F_spn(:,2));
fprintf('  Wilcoxon (paired, n=5): F_neuron evk vs spn p=%.3g | F_trial evk vs spn p=%.3g\n', ...
        p_neuron, p_trial);

% ----- figure -----------------------------------------------------------
col_evk = [0.85 0.33 0.10];
col_spn = [0.30 0.60 0.90];

figure('Name', 'CV_P variance decomposition', 'NumberTitle', 'off', ...
       'Position', [100 100 1100 450]);

% Panel A: stacked variance composition (pooled)
subplot(1, 2, 1); hold on;
yvals = [F_spn_mean; F_evk_mean];
hb = bar(yvals, 'stacked');
hb(1).FaceColor = [0.55 0.55 0.85];  hb(1).DisplayName = 'F_{neuron}  (across-neuron)';
hb(2).FaceColor = [0.85 0.55 0.55];  hb(2).DisplayName = 'F_{trial}   (across-trial)';
hb(3).FaceColor = [0.70 0.70 0.70];  hb(3).DisplayName = 'F_{int}     (interaction / residual)';
set(gca, 'XTick', 1:2, 'XTickLabel', {'spontaneous', 'evoked'});
ylabel('fraction of total variance');
title('A: variance decomposition (pooled, 5 sessions)');
legend('show', 'Location', 'eastoutside', 'FontSize', 9);
ylim([0 1]); grid on; box off; set(gca, 'TickDir', 'out');

% Panel B: paired across-neuron vs across-trial fractions
subplot(1, 2, 2); hold on;
xj = [1 2 3 4];
for is = 1:nS
    % across-neuron (evoked larger expected) and across-trial (spon larger expected)
    plot([1 2], [F_spn(is, 1), F_evk(is, 1)], '-o', 'Color', [0.55 0.55 0.85], ...
         'LineWidth', 1.2, 'MarkerFaceColor', [0.55 0.55 0.85], 'MarkerSize', 6);
    plot([3 4], [F_spn(is, 2), F_evk(is, 2)], '-o', 'Color', [0.85 0.55 0.55], ...
         'LineWidth', 1.2, 'MarkerFaceColor', [0.85 0.55 0.55], 'MarkerSize', 6);
end
set(gca, 'XTick', 1:4, 'XTickLabel', {'spn', 'evk', 'spn', 'evk'});
xtickangle(0);
ylabel('variance fraction');
text(1.5, 1.02, sprintf('F_{neuron}  p=%.2g', p_neuron), 'HorizontalAlignment', 'center');
text(3.5, 1.02, sprintf('F_{trial}   p=%.2g', p_trial), 'HorizontalAlignment', 'center');
title('B: paired across-session (Wilcoxon signed-rank)');
ylim([0 1.1]); xlim([0.5 4.5]); grid on; box off; set(gca, 'TickDir', 'out');

sgtitle('Variance decomposition explains opposite directions of CV_L and CV_P');

print(gcf, 'FigureS14.pdf', '-dpdf', '-bestfit');
print(gcf, 'FigureS14.png', '-dpng', '-r200');
fprintf('\nwrote FigureS14.pdf / FigureS14.png\n');

save('variance_decomp_CVP.mat', ...
     'F_evk', 'F_spn', 'F_evk_mean', 'F_evk_sem', 'F_spn_mean', 'F_spn_sem', ...
     'CVL_evk_sess', 'CVP_evk_sess', 'CVL_spn_sess', 'CVP_spn_sess', ...
     'p_neuron', 'p_trial', 'V1V2_FILES');
fprintf('wrote variance_decomp_CVP.mat\n');
fprintf('=== done ===\n');


% ----- local helper ----------------------------------------------------
function frac = decompose_two_way(R)
% Two-way fixed-effects decomposition.
%   R : [n_neurons x n_trials] firing-rate matrix
%   frac : [F_neuron, F_trial, F_interaction]  (each in [0,1], summing to 1)
[n_neurons, n_trials] = size(R);
grand = mean(R(:));
SS_total  = sum((R(:) - grand).^2);
rowMean   = mean(R, 2);                         % per-neuron mean over trials
colMean   = mean(R, 1);                         % per-trial mean over neurons
SS_neuron = n_trials  * sum((rowMean - grand).^2);
SS_trial  = n_neurons * sum((colMean - grand).^2);
SS_inter  = max(SS_total - SS_neuron - SS_trial, 0);
frac      = [SS_neuron, SS_trial, SS_inter] ./ max(SS_total, eps);
end
