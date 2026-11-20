% SHUFFLE_NULL_V1V2  Per-neuron Direct-Method ρ, H_r, MI on v1v2-1 V1 data
%                    after per-trial-per-neuron random circular shift of the
%                    1.28 s driven raster, breaking stimulus-locked timing
%                    while preserving per-trial spike count.
%
% Method: identical to unified_v1_Hrho_ICE.m (no SNR/FR filter, 1280-bin
% window, T_list=[4..40], n_fit=8, dt=1ms, refractory=1), with one extra
% step:
%
%   For each driven trial t and each V1 neuron n:
%       tau(t,n) ~ Uniform{0, ..., N_BINS-1}      (independent across n,t)
%       raster(t, n, :) := circshift(raster(t, n, :), tau(t,n), 3)
%
% Expected behaviour under the deterministic-encoding null:
%   - mean_FR unchanged                      (circshift preserves spike count)
%   - H_total approximately unchanged        (shift preserves marginal P(w))
%   - H_noise -> H_total within each stim    (trials no longer aligned)
%   - rho -> 1, MI -> 0
%
% A measured rho < 1 (and MI > 0) on the unshuffled data, significantly
% below the shuffle null, demonstrates that the cross-trial structure
% used by the Direct Method is genuinely stimulus-locked rather than an
% artefact of the estimator.

clear; clc;

%% ================= parameters =================

V1V2_FILES = { ...
    '105l001p16.mat', ...
    '106r001p26.mat', ...
    '106r002p70.mat', ...
    '107l002p67.mat', ...
    '107l003p143.mat'};

OUTPUT_MAT = fullfile(pwd, 'shuffle_null_v1v2.mat');

% Direct Method (match unified_v1_Hrho_ICE.m / figure2_new.m exactly)
T_list     = [4 5 6 7 8 10 13 15 20 25 30 40];
n_fit      = 8;
dt_bin     = 0.001;
refractory = 1;

% Driven window: 0 - 1.28 s -> 1280 bins
DUR_DRIVEN = 1.28;
N_BINS     = round(DUR_DRIVEN / dt_bin);

% Reproducibility
rng(20260426, 'twister');

%% ================= run =================

results = cell(1, numel(V1V2_FILES));

for is = 1:numel(V1V2_FILES)
    t0 = tic;
    fname = V1V2_FILES{is};
    fprintf('\n======================================\n');
    fprintf(' SHUFFLE NULL v1v2-1 session %d/%d: %s\n', is, numel(V1V2_FILES), fname);
    fprintf('======================================\n');

    S = load(fname);
    nd = S.neuralData;
    nt = max(nd.trialId);
    stim_labels = nd.stim;

    rast_v1_d   = {};
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
    fprintf('  n_v1=%d, n_driven=%d, n_stim=%d\n', n_v1, n_driven, numel(unique_stim));

    % --- per-neuron-per-trial independent random circular shift ---
    rast_shuffled = cell(1, n_driven);
    taus = zeros(n_driven, n_v1);
    for trial = 1:n_driven
        R = rast_v1_d{trial};       % [n_v1 x N_BINS]
        Rs = false(size(R));
        for nn = 1:n_v1
            tau = randi([0, N_BINS-1]);
            taus(trial, nn) = tau;
            Rs(nn, :) = circshift(R(nn, :) > 0, tau, 2);
        end
        rast_shuffled{trial} = Rs;
    end
    fprintf('  applied independent circular shifts (mean tau=%.1f bins)\n', mean(taus(:)));

    % --- per-neuron Direct Method on shuffled rasters ---
    out = per_neuron_direct_method(rast_shuffled, stim_driven, unique_stim, ...
        T_list, n_fit, dt_bin, refractory, N_BINS);

    out.file    = fname;
    out.n_v1    = n_v1;
    results{is} = out;

    fprintf('  session done in %.1f min.\n', toc(t0)/60);
    fprintf('  rho_shuffled : mean=%.3f  median=%.3f  IQR=[%.3f, %.3f]\n', ...
        mean(out.rho, 'omitnan'), median(out.rho, 'omitnan'), ...
        quantile(out.rho, 0.25), quantile(out.rho, 0.75));
    fprintf('  MI_shuffled  : mean=%.3f  median=%.3f  IQR=[%.3f, %.3f]  bits/s\n', ...
        mean(out.mi, 'omitnan'), median(out.mi, 'omitnan'), ...
        quantile(out.mi, 0.25), quantile(out.mi, 0.75));

    clear S nd rast_v1_d rast_shuffled;
end

%% ================= pool & compare against measured ==================

pool.hr  = [];   pool.hn  = [];   pool.mi  = [];   pool.rho = [];   pool.FR = [];
for k = 1:numel(results)
    if isempty(results{k}), continue; end
    pool.hr  = [pool.hr,  results{k}.hr];
    pool.hn  = [pool.hn,  results{k}.hn];
    pool.mi  = [pool.mi,  results{k}.mi];
    pool.rho = [pool.rho, results{k}.rho];
    pool.FR  = [pool.FR,  results{k}.FR];
end

fprintf('\n\n===========================================================\n');
fprintf(' SHUFFLE NULL pooled across 5 sessions  (n=%d V1 neurons)\n', ...
    sum(isfinite(pool.rho)));
fprintf('===========================================================\n');
print_pool('shuffled', pool);

% Compare against measured numbers if unified file is present
unified_path = fullfile(pwd, 'unified_v1_Hrho_ICE.mat');
if isfile(unified_path)
    U = load(unified_path);
    fprintf('\n  (measured, from unified_v1_Hrho_ICE.mat for comparison)\n');
    print_pool('measured', U.v1v2_pool);

    % Per-neuron paired difference
    fprintf('\n  per-neuron paired differences (measured - shuffled):\n');
    if numel(U.v1v2_pool.rho) == numel(pool.rho)
        d_rho = U.v1v2_pool.rho - pool.rho;
        d_mi  = U.v1v2_pool.mi  - pool.mi;
        d_hr  = U.v1v2_pool.hr  - pool.hr;
        fprintf('    Delta_rho : mean=%+.3f  median=%+.3f\n', ...
            mean(d_rho, 'omitnan'), median(d_rho, 'omitnan'));
        fprintf('    Delta_MI  : mean=%+.2f  median=%+.2f  bits/s\n', ...
            mean(d_mi, 'omitnan'),  median(d_mi, 'omitnan'));
        fprintf('    Delta_Hr  : mean=%+.2f  median=%+.2f  bits/s\n', ...
            mean(d_hr, 'omitnan'),  median(d_hr, 'omitnan'));
    else
        fprintf('    (neuron-count mismatch, skipping paired test)\n');
    end
end

save(OUTPUT_MAT, 'results', 'pool', 'T_list', 'n_fit', 'dt_bin', ...
                 'refractory', 'DUR_DRIVEN');
fprintf('\nSaved to %s\n', OUTPUT_MAT);


%% ================= helpers =================

function out = per_neuron_direct_method(rast_cell, stim_driven, unique_stim, ...
    T_list, n_fit, dt_bin, refractory, N_BINS)
    n_driven = numel(rast_cell);
    n_v1     = size(rast_cell{1}, 1);
    n_stim   = numel(unique_stim);

    nT    = length(T_list);
    T_sec = T_list * dt_bin;
    inv_T = 1 ./ T_sec;
    fit_idx = 1 : min(n_fit, nT);

    hr  = nan(1, n_v1);   hn = nan(1, n_v1);
    mi  = nan(1, n_v1);   rho = nan(1, n_v1);   FR = nan(1, n_v1);

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

        hr(neuron) = ft.intercept;
        hn(neuron) = fn.intercept;
        mi(neuron) = max(0, ft.intercept - fn.intercept);
        if hr(neuron) > 0, rho(neuron) = hn(neuron) / hr(neuron); end
    end
    out.hr = hr; out.hn = hn; out.mi = mi; out.rho = rho; out.FR = FR;
end

function print_pool(label, p)
    v = @(x) x(isfinite(x));
    q = @(x, k) quantile(v(x), k);
    fprintf('  [%s]\n', label);
    fprintf('    H_r   : mean=%7.2f  median=%7.2f  IQR=[%7.2f, %7.2f]  bits/s\n', ...
        mean(v(p.hr)), median(v(p.hr)), q(p.hr,0.25), q(p.hr,0.75));
    fprintf('    H_n   : mean=%7.2f  median=%7.2f  IQR=[%7.2f, %7.2f]  bits/s\n', ...
        mean(v(p.hn)), median(v(p.hn)), q(p.hn,0.25), q(p.hn,0.75));
    fprintf('    MI    : mean=%7.2f  median=%7.2f  IQR=[%7.2f, %7.2f]  bits/s\n', ...
        mean(v(p.mi)), median(v(p.mi)), q(p.mi,0.25), q(p.mi,0.75));
    fprintf('    rho   : mean=%7.3f  median=%7.3f  IQR=[%7.3f, %7.3f]\n', ...
        mean(v(p.rho)), median(v(p.rho)), q(p.rho,0.25), q(p.rho,0.75));
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
    H = mean(entropy_values);
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
