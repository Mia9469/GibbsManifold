%% redo_MI_from_saved.m
%  Recompute MI / Hr / Hn / ICE from the previously-saved Binary_cell
%  in MI_results_direct.mat, with:
%    1. cross-stim formulation (matches figure2_new.m / V1 pipeline)
%    2. 1/T extrapolation diagnostic (low / mid / high cost windows)
%    3. drop the first WIN_DROP_INIT initialisation windows
%
%  Skips the slow simulation loop entirely.  Run after figure4_MI_new.m
%  has produced MI_results_direct.mat at least once.

clear; clc; close all;

%% ===================== load saved simulation =====================
S = load('MI_results_direct.mat');
Binary_cell = S.Binary_cell;
NStim       = S.NStim;
NReps       = S.NReps;
NWindows    = S.NWindows;
ra1_store   = S.ra1_store;
cv1_store   = S.cv1_store;
ss1_store   = S.ss1_store;
ent1_store  = S.ent1_store;
cost1_store = S.cost1_store;
EI_store    = S.EI_store;

% Recover bin / window dimensions from the data itself
nBins_win = size(Binary_cell{1, 1, 1}, 2);
fprintf('Loaded MI_results_direct.mat:  NStim=%d, NReps=%d, NWindows=%d, nBins_win=%d\n', ...
    NStim, NReps, NWindows, nBins_win);

%% ===================== Direct-Method parameters =====================
dt_bin = 10;                                    % ms (must match figure4_MI_new.m)
dt_sec = dt_bin / 1000;
T_list_mi = [4 5 6 7 8 10 11 12 13 15 17 20 22 25 28 30 40 50 60 70 80 90 100];
T_list_mi = T_list_mi(T_list_mi <= nBins_win);

% Linear-fit range for the 1/T extrapolation.
%   With 10 ms bins, the LIF autocorrelation timescale is several tens of
%   ms, so word lengths T <= ~15 bins (= 150 ms) sit in the finite-sample
%   bias-saturation plateau and produce a biased intercept.  The diagnostic
%   plot shows linear behaviour starting around T >= 20 bins.  We therefore
%   fit on T = 20-100 bins (indices 12:end of T_list_mi above).
T_FIT_MIN_BINS = 20;
T_FIT_MAX_BINS = 100;
fit_mask = (T_list_mi >= T_FIT_MIN_BINS) & (T_list_mi <= T_FIT_MAX_BINS);
fit_idx_global = find(fit_mask);
fprintf('1/T fit window: T = %d-%d bins  (%d points)\n', ...
    T_FIT_MIN_BINS, T_FIT_MAX_BINS, numel(fit_idx_global));

WIN_DROP_INIT = 1;     % drop first w windows (initialisation artefact)

%% ===================== cross-stim MI loop =====================
Hr_vec = zeros(1, NWindows);
Hn_vec = zeros(1, NWindows);
MI_vec = zeros(1, NWindows);

rate_total_store = nan(NWindows, length(T_list_mi));
rate_noise_store = nan(NWindows, length(T_list_mi));
fit_total_R2     = nan(1, NWindows);
fit_noise_R2     = nan(1, NWindows);

for w = 1:NWindows
    fprintf('  Window %d / %d\n', w, NWindows);

    spike_long_all = zeros(NStim * NReps, nBins_win);
    row = 0;
    for s = 1:NStim
        for r = 1:NReps
            row = row + 1;
            spike_long_all(row, :) = Binary_cell{s, r, w};
        end
    end
    if sum(spike_long_all(:)) == 0, continue; end

    nT = length(T_list_mi);
    H_total_T = zeros(1, nT);
    H_noise_T = zeros(1, nT);

    for ti = 1:nT
        H_total_T(ti) = total_entropy_pool(spike_long_all, T_list_mi(ti));
    end
    for s = 1:NStim
        spike_rep_s = zeros(NReps, nBins_win);
        for r = 1:NReps
            spike_rep_s(r, :) = Binary_cell{s, r, w};
        end
        if sum(spike_rep_s(:)) == 0, continue; end
        for ti = 1:nT
            H_noise_T(ti) = H_noise_T(ti) + ...
                (1/NStim) * noise_entropy_avg(spike_rep_s, T_list_mi(ti));
        end
    end

    T_sec = T_list_mi * dt_sec;
    inv_T = 1 ./ T_sec;
    rate_total = H_total_T ./ T_sec;
    rate_noise = H_noise_T ./ T_sec;
    fit_idx = fit_idx_global;
    p_t = polyfit(inv_T(fit_idx), rate_total(fit_idx), 1);
    p_n = polyfit(inv_T(fit_idx), rate_noise(fit_idx), 1);

    Hr_vec(w) = p_t(2);
    Hn_vec(w) = p_n(2);
    MI_vec(w) = max(0, p_t(2) - p_n(2));

    rate_total_store(w, :) = rate_total;
    rate_noise_store(w, :) = rate_noise;
    yhat_t = polyval(p_t, inv_T(fit_idx));
    yhat_n = polyval(p_n, inv_T(fit_idx));
    sst_t = sum((rate_total(fit_idx) - mean(rate_total(fit_idx))).^2);
    sst_n = sum((rate_noise(fit_idx) - mean(rate_noise(fit_idx))).^2);
    fit_total_R2(w) = 1 - sum((rate_total(fit_idx) - yhat_t).^2) / max(sst_t, eps);
    fit_noise_R2(w) = 1 - sum((rate_noise(fit_idx) - yhat_n).^2) / max(sst_n, eps);
end

%% ===================== grand stats =====================
cv1_grand   = squeeze(mean(mean(cv1_store, 1), 2));
ss1_grand   = squeeze(mean(mean(ss1_store, 1), 2));
ent1_grand  = squeeze(mean(mean(ent1_store, 1), 2));
ra1_grand   = squeeze(mean(mean(ra1_store, 1), 2));
cost1_grand = squeeze(mean(mean(cost1_store, 1), 2));
EI_grand    = squeeze(mean(mean(EI_store, 1), 2));

%% ===================== 1/T extrapolation diagnostic =====================
valid_w = find(MI_vec > 0);
if numel(valid_w) >= 3
    [~, ord] = sort(cost1_grand(valid_w));
    pick = valid_w(ord([1, ceil(end/2), end]));   % low / mid / high cost
    figure('Color','w','Position',[60 60 1100 360]);
    T_sec = T_list_mi * dt_sec;
    inv_T = 1 ./ T_sec;
    fit_idx = fit_idx_global;
    for ip = 1:numel(pick)
        w = pick(ip);
        subplot(1, 3, ip); hold on;
        % Highlight the fit range with a shaded band
        x_band = [inv_T(fit_idx(1)), inv_T(fit_idx(end))];
        yl = ylim; if isequal(yl, [0 1]), yl = [0, max(rate_total_store(w,:))*1.1]; end
        patch([x_band fliplr(x_band)], [yl(1) yl(1) yl(2) yl(2)], ...
              [1 0.95 0.85], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
        plot(inv_T, rate_total_store(w, :), 'ko-', 'MarkerFaceColor', 'k', ...
            'MarkerSize', 5, 'LineWidth', 1.0);
        plot(inv_T, rate_noise_store(w, :), 'bs-', 'MarkerFaceColor', 'b', ...
            'MarkerSize', 5, 'LineWidth', 1.0);
        p_t = polyfit(inv_T(fit_idx), rate_total_store(w, fit_idx), 1);
        p_n = polyfit(inv_T(fit_idx), rate_noise_store(w, fit_idx), 1);
        x_ext = linspace(0, max(inv_T)*1.05, 100);
        plot(x_ext, polyval(p_t, x_ext), 'k--', 'LineWidth', 1.2);
        plot(x_ext, polyval(p_n, x_ext), 'b--', 'LineWidth', 1.2);
        plot(0, p_t(2), 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 8);
        plot(0, p_n(2), 'bs', 'MarkerFaceColor', 'b', 'MarkerSize', 8);
        xlim([-max(inv_T)*0.05, max(inv_T)*1.05]);
        xlabel('1/T  (Hz)'); ylabel('rate = H/T  (bits/s)');
        title(sprintf('w=%d, C=%.2fW, MI=%.2f, R^2_{tot}=%.3f', ...
            w, cost1_grand(w), MI_vec(w), fit_total_R2(w)));
    end
    sgtitle('1/T extrapolation diagnostic  (low / mid / high cost windows)');
    fprintf('\n--- 1/T extrapolation diagnostic ---\n');
    fprintf('  fit R^2 (total):  min=%.3f  median=%.3f  max=%.3f\n', ...
        min(fit_total_R2(valid_w)), median(fit_total_R2(valid_w)), ...
        max(fit_total_R2(valid_w)));
    fprintf('  fit R^2 (noise):  min=%.3f  median=%.3f  max=%.3f\n', ...
        min(fit_noise_R2(valid_w)), median(fit_noise_R2(valid_w)), ...
        max(fit_noise_R2(valid_w)));
    fprintf('  R^2 < 0.95 in %d/%d windows\n', ...
        sum(fit_total_R2(valid_w) < 0.95), numel(valid_w));
end

%% ===================== drop initialisation windows =====================
init_drop = 1 : min(WIN_DROP_INIT, NWindows);
fprintf('\n--- Dropping initialisation windows %s ---\n', mat2str(init_drop));
Hr_vec(init_drop) = NaN;
Hn_vec(init_drop) = NaN;
MI_vec(init_drop) = NaN;
cv1_grand(init_drop)   = NaN;
ss1_grand(init_drop)   = NaN;
ent1_grand(init_drop)  = NaN;
ra1_grand(init_drop)   = NaN;
cost1_grand(init_drop) = NaN;
EI_grand(init_drop)    = NaN;

%% ===================== ICE =====================
ICE_vec = MI_vec(:)' ./ cost1_grand(:)';

%% ===================== Cost vs MI / ICE plots =====================
figure('Color','w','Position',[60 480 900 360]);
subplot(1, 2, 1);
plot(cost1_grand, MI_vec, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 5);
xlabel('Cost [W]'); ylabel('MI [bits/s]'); title('Cost vs MI (after fixes)');
grid on;

subplot(1, 2, 2);
plot(cost1_grand, ICE_vec, 'bs', 'MarkerFaceColor', 'b', 'MarkerSize', 5);
xlabel('Cost [W]'); ylabel('ICE [bits/J]'); title('Cost vs ICE (after fixes)');
grid on;

%% ===================== summary print =====================
fprintf('\n--- Trajectory summary (init dropped) ---\n');
fprintf('  Cost (W):   min=%.2f  median=%.2f  max=%.2f\n', ...
    min(cost1_grand), median(cost1_grand, 'omitnan'), max(cost1_grand));
fprintf('  MI (b/s):   min=%.2f  median=%.2f  max=%.2f\n', ...
    min(MI_vec), median(MI_vec, 'omitnan'), max(MI_vec));
fprintf('  ICE (b/J):  min=%.2f  median=%.2f  max=%.2f\n', ...
    min(ICE_vec), median(ICE_vec, 'omitnan'), max(ICE_vec));

%% ===================== save updated MAT =====================
save('MI_results_direct.mat', '-struct', 'S');   % first restore everything
save('MI_results_direct.mat', 'MI_vec', 'Hr_vec', 'Hn_vec', 'ICE_vec', ...
     'cost1_grand', 'cv1_grand', 'ra1_grand', 'EI_grand', ...
     'rate_total_store', 'rate_noise_store', 'fit_total_R2', 'fit_noise_R2', ...
     '-append');
fprintf('\nSaved updated arrays to MI_results_direct.mat (other fields preserved).\n');


%% ============================ local helpers ============================
function H = total_entropy_pool(spike_matrix, T)
    [~, n_cols] = size(spike_matrix);
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

function H = noise_entropy_avg(spike_matrix, T)
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
