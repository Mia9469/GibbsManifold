%% compute_v1_spon_Hr.m
%  Companion to replot_v1_info_cost.m.
%
%  Per-neuron Direct-Method Hr on SPONTANEOUS V1 trials of the v1v2-1
%  dataset (5 sessions pooled).  Equivalent to the per-neuron block of
%  unified_v1_Hrho_ICE.m, but applied to the 1500-bin spontaneous
%  rasters instead of the 1280-bin driven rasters.  Since there is no
%  stimulus identity in the spontaneous condition, only H_total (= Hr)
%  is computed (no Hn / MI / rho).
%
%  Output: spon_Hr_perN.mat with vectors (one entry per V1 neuron, all
%  5 sessions concatenated):
%    hr_spon  — Direct-Method Hr   [bits/s]
%    FR_spon  — mean firing rate    [Hz]
%    C_spon   — calibrated affine cost proxy C(r) = C1 + C2 * FR
%    sess_id  — session index 1..5
%
%  Approximate runtime: ≈ 5 min on a 2024 laptop.
%
%  Usage:
%    matlab -batch "compute_v1_spon_Hr"
% ---------------------------------------------------------------------

clear; close all;
fprintf('=== compute_v1_spon_Hr.m ===\n');

% ----- parameters (matched to unified_v1_Hrho_ICE.m exactly) ----------
V1V2_FILES = {
    '105l001p16.mat'
    '106r001p26.mat'
    '106r002p70.mat'
    '107l002p67.mat'
    '107l003p143.mat'
};
N_BINS_SPON = 1500;                     % spontaneous trial length (1.5 s)
DUR_SPON    = 1.5;                      % seconds
T_list      = uint8([4 5 6 7 8 10 13 15 20 25 30 40]);
n_fit       = 8;
dt_bin      = 0.001;
refractory  = 1;
C1          = 2.4;                      % Yu et al. 2018 cost params
C2          = 0.37;

hr_spon = []; FR_spon = []; C_spon = []; sess_id = [];

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

    % collect spontaneous trials (1500-bin rasters) for V1
    rast_v1_s = {};
    for i = 1:nt
        sv1 = nd.spikeRasters{i, 1};
        if size(sv1, 2) == N_BINS_SPON
            rast_v1_s{end + 1} = sv1;                                       %#ok<AGROW>
        end
    end
    n_spon = numel(rast_v1_s);
    if n_spon == 0
        warning('  no spontaneous trials in %s, skipping', fname);
        continue;
    end
    n_v1 = size(rast_v1_s{1}, 1);
    fprintf('  n_v1=%d, n_spon=%d\n', n_v1, n_spon);

    % per-neuron Direct-Method Hr (no Hn since no stim)
    nT    = length(T_list);
    T_sec = double(T_list) * dt_bin;
    inv_T = 1 ./ T_sec;
    fit_idx = 1 : min(n_fit, nT);

    hr_n = nan(1, n_v1);
    FR_n = nan(1, n_v1);

    for neuron = 1:n_v1
        if mod(neuron, 25) == 0
            fprintf('    neuron %d/%d\n', neuron, n_v1);
        end

        spike_long = false(n_spon, N_BINS_SPON);
        for trial = 1:n_spon
            spike_long(trial, :) = rast_v1_s{trial}(neuron, :) > 0;
        end
        spike_long = apply_refractory_local(spike_long, refractory);

        FR_n(neuron) = sum(spike_long(:)) / (n_spon * N_BINS_SPON * dt_bin);

        H_total_vals = zeros(1, nT);
        for ti = 1:nT
            T = double(T_list(ti));
            H_total_vals(ti) = total_entropy_local(spike_long, T);
        end
        rate_total = H_total_vals ./ T_sec;

        [~, ft] = linear_fit_local(inv_T(fit_idx), rate_total(fit_idx));
        hr_n(neuron) = ft.intercept;                                       % bits/s
    end

    C_n = C1 + C2 * FR_n;
    hr_spon = [hr_spon, hr_n];                                              %#ok<AGROW>
    FR_spon = [FR_spon, FR_n];                                              %#ok<AGROW>
    C_spon  = [C_spon,  C_n ];                                              %#ok<AGROW>
    sess_id = [sess_id, is * ones(1, n_v1)];                                %#ok<AGROW>

    fprintf('  session done in %.1f min.  Hr median=%.2f bits/s, FR median=%.2f Hz\n', ...
            toc(t0)/60, median(hr_n, 'omitnan'), median(FR_n, 'omitnan'));

    clear S nd rast_v1_s;
end

save('spon_Hr_perN.mat', 'hr_spon', 'FR_spon', 'C_spon', 'sess_id', ...
                          'C1', 'C2', 'DUR_SPON', 'T_list', 'n_fit', ...
                          'dt_bin', 'refractory');
fprintf('\n=== wrote spon_Hr_perN.mat (n=%d V1 neurons across %d sessions) ===\n', ...
        sum(isfinite(hr_spon)), numel(unique(sess_id)));


% ====================================================================
% LOCAL HELPERS  (copied verbatim from unified_v1_Hrho_ICE.m)
% ====================================================================

function R = apply_refractory_local(R, refractory)
    if refractory <= 0, return; end
    [n_rows, n_cols] = size(R);
    for r = 1:n_rows
        last_spike = -inf;
        for c = 1:n_cols
            if R(r, c)
                if c - last_spike <= refractory
                    R(r, c) = false;
                else
                    last_spike = c;
                end
            end
        end
    end
end

function H = total_entropy_local(spike_matrix, T)
    [n_rows, n_cols] = size(spike_matrix);
    n_windows = n_cols - T + 1;
    if n_windows <= 0, H = 0; return; end
    words = zeros(n_rows * n_windows, T, 'like', spike_matrix);
    for j = 1:n_windows
        idx = (j - 1) * n_rows + (1:n_rows);
        words(idx, :) = spike_matrix(:, j:(j + T - 1));
    end
    counts = count_unique_words_local(words);
    p = counts / sum(counts);
    H = -sum(p .* log2(p));
end

function counts = count_unique_words_local(words)
    [~, ~, ic] = unique(words, 'rows');
    counts = accumarray(ic, 1);
end

function [slope, fit_struct] = linear_fit_local(x, y)
    p = polyfit(x(:), y(:), 1);
    slope = p(1);
    fit_struct.slope     = p(1);
    fit_struct.intercept = p(2);
end
