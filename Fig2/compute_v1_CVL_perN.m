%% compute_v1_CVL_perN.m
%  Companion to replot_v1_info_cost.m.
%
%  Per-neuron lifetime CV (CV_L) on DRIVEN V1 trials of the v1v2-1
%  dataset (5 sessions pooled), aligned to the per-neuron grid used by
%  unified_v1_Hrho_ICE.m (so the resulting vectors can be matched
%  element-wise with hr / mi / C from that file).
%
%  CV_L definition matches the per-neuron CV_L used in figure2_new.m
%  Section 2b: for each neuron, compute its mean firing rate per trial
%  (spikes-per-trial / driven-window-duration), then CV_L = std / mean
%  across the n_driven trials.
%
%  Output: v1_CVL_perN.mat with vectors (one entry per V1 neuron, all
%  5 sessions concatenated in session order):
%    cvL_drv  - lifetime CV on driven trials                     [-]
%    FR_drv   - mean firing rate across driven trials            [Hz]
%    n_trials - per-neuron number of driven trials used          [int]
%    sess_id  - session index 1..5                               [int]
%
%  Approximate runtime: < 1 min on a 2024 laptop.
%
%  Usage:
%    matlab -batch "compute_v1_CVL_perN"
% ---------------------------------------------------------------------

clear; close all;
fprintf('=== compute_v1_CVL_perN.m ===\n');

% ----- parameters (matched to unified_v1_Hrho_ICE.m) ------------------
V1V2_FILES = {
    '105l001p16.mat'
    '106r001p26.mat'
    '106r002p70.mat'
    '107l002p67.mat'
    '107l003p143.mat'
};
DUR_DRIVEN = 1.28;                      % seconds
N_BINS_DRV = 1280;                      % 1.28 s @ 1 ms bins

cvL_drv = []; FR_drv = []; n_trials = []; sess_id = [];

for is = 1:numel(V1V2_FILES)
    t0 = tic;
    fname = V1V2_FILES{is};
    fprintf('\n--- session %d/%d: %s ---\n', is, numel(V1V2_FILES), fname);
    if ~isfile(fname)
        warning('  not found, skipping');
        continue;
    end
    S = load(fname);
    nd = S.neuralData;
    nt = max(nd.trialId);

    % collect driven trials (1280-bin rasters) for V1
    rast_v1_d = {};
    for i = 1:nt
        sv1 = nd.spikeRasters{i, 1};
        if size(sv1, 2) == N_BINS_DRV
            rast_v1_d{end + 1} = sv1;                                       %#ok<AGROW>
        end
    end
    n_drv = numel(rast_v1_d);
    if n_drv == 0
        warning('  no driven trials in %s, skipping', fname);
        continue;
    end
    n_v1 = size(rast_v1_d{1}, 1);
    fprintf('  n_v1=%d, n_drv=%d\n', n_v1, n_drv);

    % build per-neuron x per-trial firing-rate matrix (Hz)
    rate_mat = zeros(n_v1, n_drv);
    for tr = 1:n_drv
        rate_mat(:, tr) = sum(rast_v1_d{tr} > 0, 2) ./ DUR_DRIVEN;
    end

    cvL_n = std(rate_mat, 0, 2) ./ max(mean(rate_mat, 2), eps);
    FR_n  = mean(rate_mat, 2);

    cvL_drv  = [cvL_drv;  cvL_n];                                           %#ok<AGROW>
    FR_drv   = [FR_drv;   FR_n];                                            %#ok<AGROW>
    n_trials = [n_trials; n_drv * ones(n_v1, 1)];                           %#ok<AGROW>
    sess_id  = [sess_id;  is * ones(n_v1, 1)];                              %#ok<AGROW>

    fprintf('  session done in %.1f s.  CV_L median=%.2f, FR median=%.2f Hz\n', ...
            toc(t0), median(cvL_n, 'omitnan'), median(FR_n, 'omitnan'));

    clear S nd rast_v1_d rate_mat;
end

save('v1_CVL_perN.mat', 'cvL_drv', 'FR_drv', 'n_trials', 'sess_id', ...
                        'DUR_DRIVEN', 'N_BINS_DRV', 'V1V2_FILES');
fprintf('\n=== wrote v1_CVL_perN.mat (n=%d V1 neurons across %d sessions) ===\n', ...
        sum(isfinite(cvL_drv)), numel(unique(sess_id)));
