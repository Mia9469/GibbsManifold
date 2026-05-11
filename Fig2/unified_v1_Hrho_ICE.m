% UNIFIED_V1_HRHO_ICE  Per-neuron H, C, rho, ICE on V1 data from
%                      (i) v1v2-1 (5 sessions, 1.28 s driven window)
%                      (ii) pvc-11 (3 monkeys, 1.28 s driven window).
%
% Pipeline is identical to the per-neuron block in figure2_new.m:
%   - 1 ms bins, refractory=1, T_list=[4..40], n_fit=8
%   - no SNR/FR filter (all V1 units kept, matching figure2_new.m)
%   - driven window 0-1.28 s (1280 bins) for both datasets
%   - per-neuron H_total, P(s)-weighted H_noise, 1/T linear extrapolation
%   - rho    = H_noise_rate / H_total_rate
%   - C      = C1 + C2 * mean_FR (Yu et al. 2018)
%   - ICE    = H_total_rate / C   [bits/J]
%
% Outputs:
%   results.(dataset){k}.hr/hn/rho/mi/FR/C/ICE     (per-neuron vectors)
%   printed summary table
%   saved to unified_v1_Hrho_ICE.mat

clear; clc;

%% ================= parameters =================

V1V2_FILES = { ...
    '105l001p16.mat', ...
    '106r001p26.mat', ...
    '106r002p70.mat', ...
    '107l002p67.mat', ...
    '107l003p143.mat'};

PVC11_DIR   = fullfile(pwd, 'pvc11', 'data_and_scripts', 'spikes_gratings');
PVC11_FILES = { ...
    'data_monkey1_gratings.mat', ...
    'data_monkey2_gratings.mat', ...
    'data_monkey3_gratings.mat'};

OUTPUT_MAT = fullfile(pwd, 'unified_v1_Hrho_ICE.mat');

% Direct Method (match figure2_new.m exactly)
T_list     = [4 5 6 7 8 10 13 15 20 25 30 40];
n_fit      = 8;
dt_bin     = 0.001;
refractory = 1;

% Metabolic cost (Yu et al. 2018)
C1 = 2.4;     % W, baseline
C2 = 0.37;    % W/Hz, per-spike

% Driven window (both datasets): 0 - 1.28 s  => 1280 bins
DUR_DRIVEN = 1.28;
N_BINS     = round(DUR_DRIVEN / dt_bin);   % 1280

% pvc-11 trial subsampling (set Inf to use all 200)
MAX_TRIALS_PER_STIM = 60;

%% ================= allocate ====================

results.v1v2  = cell(1, numel(V1V2_FILES));
results.pvc11 = cell(1, numel(PVC11_FILES));

%% ================= v1v2-1 sessions =================

for is = 1:numel(V1V2_FILES)
    t0 = tic;
    fname = V1V2_FILES{is};
    fprintf('\n======================================\n');
    fprintf(' v1v2-1 session %d/%d: %s\n', is, numel(V1V2_FILES), fname);
    fprintf('======================================\n');

    S = load(fname);
    nd = S.neuralData;
    nt = max(nd.trialId);
    stim_labels = nd.stim;

    % Collect driven trials (1280 bin rasters) for V1 only
    rast_v1_d = {};
    stim_driven = [];
    for i = 1:nt
        sv1 = nd.spikeRasters{i, 1};
        if size(sv1, 2) == N_BINS
            rast_v1_d{end+1} = sv1;
            stim_driven(end+1) = stim_labels(i);
        end
    end
    n_driven = numel(rast_v1_d);
    if n_driven == 0
        warning('no driven trials in %s, skipping', fname);
        continue;
    end
    n_v1 = size(rast_v1_d{1}, 1);
    unique_stim = unique(stim_driven);
    n_stim = numel(unique_stim);
    fprintf('  n_v1=%d, n_driven=%d, n_stim=%d\n', n_v1, n_driven, n_stim);

    out = per_neuron_direct_method_v1v2(rast_v1_d, stim_driven, unique_stim, ...
        T_list, n_fit, dt_bin, refractory, C1, C2, DUR_DRIVEN);

    out.file    = fname;
    out.n_v1    = n_v1;
    out.dataset = 'v1v2-1';
    results.v1v2{is} = out;

    fprintf('  session done in %.1f min.  rho: mean=%.3f  median=%.3f  IQR=[%.3f, %.3f]\n', ...
        toc(t0)/60, mean(out.rho, 'omitnan'), median(out.rho, 'omitnan'), ...
        quantile(out.rho, 0.25), quantile(out.rho, 0.75));

    clear S nd rast_v1_d;
end

%% ================= pvc-11 monkeys =================

for im = 1:numel(PVC11_FILES)
    t0 = tic;
    fname = PVC11_FILES{im};
    fprintf('\n======================================\n');
    fprintf(' pvc-11 monkey %d/%d: %s\n', im, numel(PVC11_FILES), fname);
    fprintf('======================================\n');

    S = load(fullfile(PVC11_DIR, fname));
    EV = S.data.EVENTS;               % {num_units x num_stim x num_trials}
    [num_units, num_stim, num_trials_raw] = size(EV);
    if isfinite(MAX_TRIALS_PER_STIM)
        num_trials = min(MAX_TRIALS_PER_STIM, num_trials_raw);
    else
        num_trials = num_trials_raw;
    end
    fprintf('  num_units=%d, num_stim=%d, trials_used=%d/%d\n', ...
        num_units, num_stim, num_trials, num_trials_raw);

    % Build driven rasters: for each trial, a [num_units x N_BINS] raster,
    % consistent with rast_v1_d cell format used above.
    edges = 0 : dt_bin : DUR_DRIVEN;      % 1281 edges -> 1280 bins
    rast_v1_d = cell(1, num_stim * num_trials);
    stim_driven = zeros(1, num_stim * num_trials);
    row = 0;
    for s = 1:num_stim
        for t = 1:num_trials
            row = row + 1;
            stim_driven(row) = s;
            R = false(num_units, N_BINS);
            for n = 1:num_units
                ts = EV{n, s, t};
                if isempty(ts), continue; end
                h = histcounts(ts(:), edges);
                R(n, :) = h > 0;
            end
            rast_v1_d{row} = R;
        end
    end
    unique_stim = 1:num_stim;
    fprintf('  binarised. n_driven_trials=%d\n', row);

    out = per_neuron_direct_method_v1v2(rast_v1_d, stim_driven, unique_stim, ...
        T_list, n_fit, dt_bin, refractory, C1, C2, DUR_DRIVEN);

    out.file    = fname;
    out.n_v1    = num_units;
    out.dataset = 'pvc-11';
    results.pvc11{im} = out;

    fprintf('  monkey done in %.1f min.  rho: mean=%.3f  median=%.3f  IQR=[%.3f, %.3f]\n', ...
        toc(t0)/60, mean(out.rho, 'omitnan'), median(out.rho, 'omitnan'), ...
        quantile(out.rho, 0.25), quantile(out.rho, 0.75));

    clear S EV rast_v1_d;
end

%% ================= pool & print ==================

v1v2_pool  = pool_results(results.v1v2);
pvc11_pool = pool_results(results.pvc11);

fprintf('\n\n============================================================\n');
fprintf(' UNIFIED V1 per-neuron summary (no SNR/FR filter, 1.28 s window)\n');
fprintf('============================================================\n');
print_block('v1v2-1 pooled (5 sessions)', v1v2_pool);
print_block('pvc-11 pooled (3 monkeys)', pvc11_pool);

fprintf('\n Per-session / per-monkey breakdown:\n');
for is = 1:numel(results.v1v2)
    if ~isempty(results.v1v2{is})
        print_block(sprintf('  v1v2-1 %s', results.v1v2{is}.file), results.v1v2{is});
    end
end
for im = 1:numel(results.pvc11)
    if ~isempty(results.pvc11{im})
        print_block(sprintf('  pvc-11 %s', results.pvc11{im}.file), results.pvc11{im});
    end
end

%% ================= save =========================

save(OUTPUT_MAT, 'results', 'v1v2_pool', 'pvc11_pool', ...
     'T_list', 'n_fit', 'dt_bin', 'refractory', 'C1', 'C2', ...
     'DUR_DRIVEN', 'MAX_TRIALS_PER_STIM');
fprintf('\nSaved to %s\n', OUTPUT_MAT);


%% ================= core pipeline =================

function out = per_neuron_direct_method_v1v2(rast_cell, stim_driven, unique_stim, ...
    T_list, n_fit, dt_bin, refractory, C1, C2, dur_driven)
% For each V1 neuron: pool all driven trials -> H_total; per-stim trials
% with P(s) weighting -> H_noise; 1/T fit -> entropy rates.

    n_driven = numel(rast_cell);
    n_v1     = size(rast_cell{1}, 1);
    n_stim   = numel(unique_stim);
    N_BINS   = size(rast_cell{1}, 2);

    nT    = length(T_list);
    T_sec = T_list * dt_bin;
    inv_T = 1 ./ T_sec;
    fit_idx = 1 : min(n_fit, nT);

    hr  = nan(1, n_v1);
    hn  = nan(1, n_v1);
    mi  = nan(1, n_v1);
    rho = nan(1, n_v1);
    FR  = nan(1, n_v1);

    for neuron = 1:n_v1
        if mod(neuron, 20) == 0
            fprintf('    neuron %d/%d\n', neuron, n_v1);
        end

        spike_long_n = false(n_driven, N_BINS);
        for trial = 1:n_driven
            spike_long_n(trial, :) = rast_cell{trial}(neuron, :) > 0;
        end
        spike_long_n = apply_refractory_local(spike_long_n, refractory);

        FR(neuron) = sum(spike_long_n(:)) / (n_driven * N_BINS * dt_bin);

        H_total_vals = zeros(1, nT);
        H_noise_vals = zeros(1, nT);

        for ti = 1:nT
            T = T_list(ti);
            H_total_vals(ti) = total_entropy_local(spike_long_n, T);
            Hn = 0;
            for si = 1:n_stim
                s = unique_stim(si);
                mask = (stim_driven == s);
                if sum(mask) < 2, continue; end
                P_s = sum(mask) / length(stim_driven);
                Hn  = Hn + P_s * noise_entropy_local(spike_long_n(mask, :), T);
            end
            H_noise_vals(ti) = Hn;
        end

        rate_total = H_total_vals ./ T_sec;
        rate_noise = H_noise_vals ./ T_sec;

        [~, ft] = linear_fit_local(inv_T(fit_idx), rate_total(fit_idx));
        [~, fn] = linear_fit_local(inv_T(fit_idx), rate_noise(fit_idx));

        hr(neuron)  = ft.intercept;
        hn(neuron)  = fn.intercept;
        mi(neuron)  = max(0, ft.intercept - fn.intercept);
        if hr(neuron) > 0
            rho(neuron) = hn(neuron) / hr(neuron);
        end
    end

    C   = C1 + C2 * FR;
    ICE = hr ./ C;           % bits/J  (use Hr = coding capacity)

    out.hr  = hr;    out.hn  = hn;   out.mi  = mi;
    out.rho = rho;   out.FR  = FR;   out.C   = C;    out.ICE = ICE;
end


function p = pool_results(cell_array)
    p.hr  = [];   p.hn  = [];   p.mi  = [];
    p.rho = [];   p.FR  = [];   p.C   = [];   p.ICE = [];
    for k = 1:numel(cell_array)
        if isempty(cell_array{k}), continue; end
        p.hr  = [p.hr,  cell_array{k}.hr];
        p.hn  = [p.hn,  cell_array{k}.hn];
        p.mi  = [p.mi,  cell_array{k}.mi];
        p.rho = [p.rho, cell_array{k}.rho];
        p.FR  = [p.FR,  cell_array{k}.FR];
        p.C   = [p.C,   cell_array{k}.C];
        p.ICE = [p.ICE, cell_array{k}.ICE];
    end
end


function print_block(label, s)
    v = @(x) x(isfinite(x));
    q = @(x, p) quantile(v(x), p);
    fprintf('\n  %s (n=%d V1 neurons)\n', label, numel(v(s.rho)));
    fprintf('    H_r   : mean=%7.2f  median=%7.2f  IQR=[%7.2f, %7.2f]  bits/s\n', ...
        mean(v(s.hr)), median(v(s.hr)), q(s.hr,0.25), q(s.hr,0.75));
    fprintf('    H_n   : mean=%7.2f  median=%7.2f  IQR=[%7.2f, %7.2f]  bits/s\n', ...
        mean(v(s.hn)), median(v(s.hn)), q(s.hn,0.25), q(s.hn,0.75));
    fprintf('    MI    : mean=%7.2f  median=%7.2f  IQR=[%7.2f, %7.2f]  bits/s\n', ...
        mean(v(s.mi)), median(v(s.mi)), q(s.mi,0.25), q(s.mi,0.75));
    fprintf('    FR    : mean=%7.3f  median=%7.3f  IQR=[%7.3f, %7.3f]  Hz\n', ...
        mean(v(s.FR)), median(v(s.FR)), q(s.FR,0.25), q(s.FR,0.75));
    fprintf('    C     : mean=%7.3f  median=%7.3f  IQR=[%7.3f, %7.3f]  W\n', ...
        mean(v(s.C)),  median(v(s.C)),  q(s.C,0.25),  q(s.C,0.75));
    fprintf('    rho   : mean=%7.3f  median=%7.3f  IQR=[%7.3f, %7.3f]\n', ...
        mean(v(s.rho)), median(v(s.rho)), q(s.rho,0.25), q(s.rho,0.75));
    fprintf('    ICE   : mean=%7.3f  median=%7.3f  IQR=[%7.3f, %7.3f]  bits/J\n', ...
        mean(v(s.ICE)), median(v(s.ICE)), q(s.ICE,0.25), q(s.ICE,0.75));
end


%% ================= local helpers (copied from figure2_new.m) =================

function H = total_entropy_local(spike_matrix, T)
    [~, n_cols] = size(spike_matrix);
    remainder = mod(n_cols, T);
    trimmed_len = n_cols - remainder;
    all_data = reshape(spike_matrix(:, 1:trimmed_len)', 1, []);
    total_len = length(all_data);
    remainder2 = mod(total_len, T);
    words = reshape(all_data(1:total_len - remainder2), T, (total_len - remainder2)/T)';
    counts = count_unique_words_local(words);
    p = counts / sum(counts);
    H = -sum(p .* log2(p));
end

function H = noise_entropy_local(spike_matrix, T)
    [~, n_bins] = size(spike_matrix);
    n_windows = n_bins - T + 1;
    entropy_values = zeros(1, n_windows);
    for j = 1:n_windows
        words = spike_matrix(:, j:(j + T - 1));
        counts = count_unique_words_local(words);
        p = counts / sum(counts);
        entropy_values(j) = -sum(p .* log2(p));
    end
    valid = entropy_values > 0;
    if any(valid), H = mean(entropy_values(valid)); else, H = 0; end
end

function counts = count_unique_words_local(words)
    [~, T] = size(words);
    if T <= 52
        pow2_vec = pow2(0:T-1)';
        word_ids = double(words) * pow2_vec;
        [~, ~, ic] = unique(word_ids);
        counts = accumarray(ic, 1);
    else
        [~, ~, ic] = unique(words, 'rows');
        counts = accumarray(ic, 1);
    end
end

function data = apply_refractory_local(data, n)
    [n_rows, n_cols] = size(data);
    for i = 1:n_rows
        for j = 1:n_cols
            if data(i, j) == 1
                idx_end = min(j + n, n_cols);
                data(i, (j+1):idx_end) = 0;
            end
        end
    end
end

function [y_fit, fit] = linear_fit_local(x, y)
    p = polyfit(x, y, 1);
    y_fit = polyval(p, x);
    residuals = y - y_fit;
    SSE = sum(residuals .^ 2);
    SST = (length(y) - 1) * var(y);
    fit.slope     = p(1);
    fit.intercept = p(2);
    fit.R2        = 1 - SSE / SST;
    fit.RMSE      = sqrt(SSE / length(y));
end
