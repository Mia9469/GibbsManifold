% PVC11_NOISE_ENTROPY  Direct-Method per-neuron rho = H(r|s)/H(r)
%                      on CRCNS pvc-11 (Kohn/Smith anesthetised macaque V1,
%                      drifting gratings, 3 monkeys, 12 directions, 200 trials).
%
% Robustness check for the claim in Supplementary Section
% sec:noise-entropy-check that rho ~= 0.85 on the v1v2-1 dataset is
% caused by sufentanil anaesthesia rather than session- or
% stimulus-specific artifacts.  pvc-11 uses the same lab + anaesthesia +
% array + stimulus family as v1v2-1, and has ~10x the trial count per
% condition, so if the framework-level conclusion survives here it is
% dataset-independent within the anaesthetised-macaque regime.
%
% Method: identical to figure2_new.m per-neuron Direct-Method pipeline
%   - bin spikes to 1 ms, take the 0.28-1.28 s window (1 s driven response)
%   - apply SNR >= 1.5 and mean-FR >= 1 Hz filters (Kohn canonical)
%   - for each neuron:  construct binary spike matrix across all 12 stim
%                       x n_trials_per_stim driven trials
%     * H_total(T) over pooled trials
%     * H_noise(T) = P(s)-weighted noise entropy per stimulus
%     * linear fit H/T vs 1/T over fit_idx -> intercept = rate in bits/s
%     * rho = H_noise_rate / H_total_rate
%
% Parameters at top of file can be tuned for speed / fidelity tradeoff.
%
% Outputs:
%   - rho_per_neuron        : [1 x N_total] ratio per accepted neuron
%   - rho_per_monkey{k}     : per-monkey vector
%   - printed summary stats (mean, median, IQR, fraction below thresholds)
%   - saved to pvc11_rho_check.mat

clear; clc;

%% ================= parameters =================

PVC11_DIR = fullfile(pwd, 'pvc11', 'data_and_scripts', 'spikes_gratings');
OUTPUT_MAT = fullfile(pwd, 'pvc11_rho_check.mat');

% Neuron filters (match Kohn canonical script_mean_firing_rates_gratings.m)
SNR_THRESHOLD = 1.5;
FR_THRESHOLD  = 1.0;     % Hz

% Direct-Method parameters (match figure2_new.m exactly)
T_list  = [4 5 6 7 8 10 13 15 20 25 30 40];   % word lengths in ms
n_fit   = 8;                                   % # points for 1/T linear fit
dt_bin  = 0.001;                               % 1 ms bin width
refrac  = 1;                                   % refractory period (bins)

% Trial subsampling for speed (set to Inf to use all 200 trials/stim)
MAX_TRIALS_PER_STIM = 60;

% Response window (Kohn canonical): 0.28 -> 1.28 s of the 1.28 s stim epoch
T_WIN_START = 0.28;
T_WIN_END   = 1.28;
N_BINS      = round((T_WIN_END - T_WIN_START) / dt_bin);  % 1000 bins

%% ================= load & filter each monkey =================

monkey_files = {'data_monkey1_gratings.mat', ...
                'data_monkey2_gratings.mat', ...
                'data_monkey3_gratings.mat'};

rho_per_monkey = cell(1, numel(monkey_files));
hr_per_monkey  = cell(1, numel(monkey_files));
hn_per_monkey  = cell(1, numel(monkey_files));
mi_per_monkey  = cell(1, numel(monkey_files));

total_elapsed = 0;

for imonkey = 2:numel(monkey_files)
    tic;
    fpath = fullfile(PVC11_DIR, monkey_files{imonkey});
    fprintf('\n======================================\n');
    fprintf(' Loading %s\n', monkey_files{imonkey});
    fprintf('======================================\n');
    load(fpath);  % returns `data`

    EV  = data.EVENTS;                % {num_units x num_stim x num_trials}
    SNR = data.SNR;                   % [num_units x 1]
    [num_units, num_stim, num_trials_raw] = size(EV);

    if isfinite(MAX_TRIALS_PER_STIM)
        num_trials = min(MAX_TRIALS_PER_STIM, num_trials_raw);
    else
        num_trials = num_trials_raw;
    end
    fprintf('  num_units=%d, num_stim=%d, trials_used=%d (of %d)\n', ...
            num_units, num_stim, num_trials, num_trials_raw);

    % Pre-build binary spike matrices [n_stim x n_trials x n_bins] per neuron.
    % We store as cell{neuron} -> [num_stim*num_trials x N_BINS] pooled, plus
    % a stim_label vector of length num_stim*num_trials for P(s) weighting.
    edges   = T_WIN_START : dt_bin : T_WIN_END;    % 1001 edges -> 1000 bins
    spike_mats = cell(1, num_units);
    for n = 1:num_units
        spike_mats{n} = false(num_stim * num_trials, N_BINS);
    end
    stim_labels = zeros(1, num_stim * num_trials);

    row = 0;
    for s = 1:num_stim
        for t = 1:num_trials
            row = row + 1;
            stim_labels(row) = s;
            for n = 1:num_units
                ts = EV{n, s, t};
                if isempty(ts), continue; end
                ts = ts(:);
                h  = histcounts(ts, edges);  % length N_BINS
                spike_mats{n}(row, :) = h > 0;
            end
        end
    end
    fprintf('  binarised. rows=%d (stim*trials)\n', row);

    % Apply SNR and mean-FR filters
    mean_FR = zeros(1, num_units);
    for n = 1:num_units
        mean_FR(n) = sum(spike_mats{n}(:)) / (row * N_BINS * dt_bin);
    end
    keep = (SNR(:)' >= SNR_THRESHOLD) & (mean_FR >= FR_THRESHOLD);
    kept_idx = find(keep);
    fprintf('  kept %d/%d neurons (SNR>=%.1f & FR>=%.1f Hz)\n', ...
            length(kept_idx), num_units, SNR_THRESHOLD, FR_THRESHOLD);

    % Direct-Method per-neuron loop
    n_kept = length(kept_idx);
    hr_vec  = zeros(1, n_kept);
    hn_vec  = zeros(1, n_kept);
    mi_vec  = zeros(1, n_kept);
    rho_vec = zeros(1, n_kept);

    nT    = length(T_list);
    T_sec = T_list * dt_bin;
    inv_T = 1 ./ T_sec;
    fit_range = 1 : min(n_fit, nT);

    for ii = 1:n_kept
        n = kept_idx(ii);
        mat = spike_mats{n};                             % [rows x N_BINS]
        mat = apply_refractory_local(mat, refrac);

        H_total = zeros(1, nT);
        H_noise = zeros(1, nT);

        for ti = 1:nT
            T = T_list(ti);
            H_total(ti) = total_entropy_local(mat, T);
            % weighted noise entropy across stimuli
            Hn = 0;
            for s = 1:num_stim
                mask = (stim_labels == s);
                Ps   = sum(mask) / length(stim_labels);
                if sum(mask) < 2, continue; end
                Hn = Hn + Ps * noise_entropy_local(mat(mask, :), T);
            end
            H_noise(ti) = Hn;
        end

        rate_total = H_total ./ T_sec;
        rate_noise = H_noise ./ T_sec;
        [~, fit_t] = linear_fit_local(inv_T(fit_range), rate_total(fit_range));
        [~, fit_n] = linear_fit_local(inv_T(fit_range), rate_noise(fit_range));

        hr_vec(ii)  = fit_t.intercept;
        hn_vec(ii)  = fit_n.intercept;
        mi_vec(ii)  = max(0, fit_t.intercept - fit_n.intercept);
        if hr_vec(ii) > 0
            rho_vec(ii) = hn_vec(ii) / hr_vec(ii);
        else
            rho_vec(ii) = NaN;
        end

        if mod(ii, 10) == 0
            fprintf('    neuron %d/%d (orig #%d): rho=%.3f  Hr=%.2f  Hn=%.2f\n', ...
                    ii, n_kept, n, rho_vec(ii), hr_vec(ii), hn_vec(ii));
        end
    end

    rho_per_monkey{imonkey} = rho_vec;
    hr_per_monkey{imonkey}  = hr_vec;
    hn_per_monkey{imonkey}  = hn_vec;
    mi_per_monkey{imonkey}  = mi_vec;

    elapsed = toc;
    total_elapsed = total_elapsed + elapsed;
    fprintf('  monkey %d done in %.1f min\n', imonkey, elapsed/60);

    % Per-monkey summary
    valid = isfinite(rho_vec) & (rho_vec >= 0) & (rho_vec <= 1);
    v = rho_vec(valid);
    fprintf('  rho (monkey%d, n=%d):  mean=%.3f  median=%.3f  IQR=[%.3f, %.3f]\n', ...
            imonkey, length(v), mean(v), median(v), ...
            quantile(v, 0.25), quantile(v, 0.75));

    clear data EV SNR spike_mats mat stim_labels;
end

%% ================= pool across 3 monkeys =================

rho_all = [rho_per_monkey{:}];
hr_all  = [hr_per_monkey{:}];
hn_all  = [hn_per_monkey{:}];
mi_all  = [mi_per_monkey{:}];

valid_all = isfinite(rho_all) & (rho_all >= 0) & (rho_all <= 1);
r = rho_all(valid_all);

fprintf('\n\n==============================================\n');
fprintf(' PVC-11 POOLED rho = H(r|s)/H(r)  (n=%d neurons)\n', length(r));
fprintf('==============================================\n');
fprintf('  mean    = %.3f\n', mean(r));
fprintf('  median  = %.3f\n', median(r));
fprintf('  IQR     = [%.3f, %.3f]  (width %.3f)\n', ...
        quantile(r, 0.25), quantile(r, 0.75), ...
        quantile(r, 0.75) - quantile(r, 0.25));
fprintf('  frac rho<0.3: %.3f;  rho<0.5: %.3f;  rho<0.7: %.3f\n', ...
        mean(r < 0.3), mean(r < 0.5), mean(r < 0.7));
fprintf('\n  Reference (v1v2-1 anaesthetised V1, n=159):\n');
fprintf('    mean=0.852  median=0.871  IQR=[0.831, 0.895]  (from figure2_new.m)\n');
fprintf('\n  Elapsed: %.1f min total\n', total_elapsed/60);

%% ================= save =================
save(OUTPUT_MAT, 'rho_per_monkey', 'hr_per_monkey', 'hn_per_monkey', ...
                 'mi_per_monkey', 'rho_all', 'hr_all', 'hn_all', 'mi_all', ...
                 'T_list', 'n_fit', 'dt_bin', 'refrac', ...
                 'SNR_THRESHOLD', 'FR_THRESHOLD', 'MAX_TRIALS_PER_STIM');
fprintf('Saved to %s\n', OUTPUT_MAT);


%% ========== local functions (copied verbatim from figure2_new.m) ==========

function H = total_entropy_local(spike_matrix, T)
    [n_rows, n_cols] = size(spike_matrix);
    remainder = mod(n_cols, T);
    trimmed_len = n_cols - remainder;
    all_data = reshape(spike_matrix(:, 1:trimmed_len)', 1, []);
    total_len = length(all_data);
    remainder2 = mod(total_len, T);
    n_words = (total_len - remainder2) / T;
    words = reshape(all_data(1:total_len - remainder2), T, n_words)';
    counts = count_unique_words_local(words);
    p = counts / sum(counts);
    H = -sum(p .* log2(p));
end

function H = noise_entropy_local(spike_matrix, T)
    [n_trials, n_bins] = size(spike_matrix);
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
    [n_words, T] = size(words);
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
