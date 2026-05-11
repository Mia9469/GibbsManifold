%% ====================================================================
%% figure4_MI — 替换 line 618 到文件末尾
%% 包含: jackknife error bar + 带 error bar 的画图
%% ====================================================================

%% Compute Mean Statistics Across All Stimuli and Reps
cv1_grand   = squeeze(mean(mean(cv1_store, 1), 2));   % (NWindows x 1)
ss1_grand   = squeeze(mean(mean(ss1_store, 1), 2));
ent1_grand  = squeeze(mean(mean(ent1_store, 1), 2));
ra1_grand   = squeeze(mean(mean(ra1_store, 1), 2));
cost1_grand = squeeze(mean(mean(cost1_store, 1), 2));

% SEM across reps (average over stim first, then SEM over reps)
% _store is [NStim x NReps x NWindows]
cv1_sem   = squeeze(std(mean(cv1_store, 1), 0, 2))   / sqrt(NReps);  % mean over stim → [1 x NReps x NW] → std over reps
ss1_sem   = squeeze(std(mean(ss1_store, 1), 0, 2))   / sqrt(NReps);
ent1_sem  = squeeze(std(mean(ent1_store, 1), 0, 2))  / sqrt(NReps);
ra1_sem   = squeeze(std(mean(ra1_store, 1), 0, 2))   / sqrt(NReps);
cost1_sem = squeeze(std(mean(cost1_store, 1), 0, 2)) / sqrt(NReps);

%% Compute ICE (Information Rate / Power)
T_obs = winsize * duration / 1000;   % observation duration [s]
MI_rate_vec = MI_vec / T_obs;        % information rate [bits/s]
ICE_vec = MI_rate_vec ./ cost1_grand';  % [bits/J]

%% Jackknife MI — leave-one-rep-out
MI_jack  = zeros(NReps, NWindows);
Hr_jack  = zeros(NReps, NWindows);
Hrs_jack = zeros(NReps, NWindows);

for w = 1:NWindows
    for k = 1:NReps   % leave out rep k
        % H(r|s) with rep k removed
        H_noise_k = zeros(1, NStim);
        for s = 1:NStim
            pooled = [];
            for r = 1:NReps
                if r == k, continue; end
                if ~isempty(R_cell{s, r, w})
                    pooled = [pooled R_cell{s, r, w}];
                end
            end
            if ~isempty(pooled)
                binEdges = 0:0.1:(max(pooled)+0.1);
                if length(binEdges) < 2, binEdges = [0 0.1]; end
                [n, ~] = histcounts(pooled, binEdges);
                h_prob = n / sum(n);
                H_noise_k(s) = -sum(h_prob(h_prob > 0) .* log2(h_prob(h_prob > 0)));
            end
        end
        Hrs_jack(k,w) = mean(H_noise_k);

        % H(r) with rep k removed
        pooled_all = [];
        for s = 1:NStim
            for r = 1:NReps
                if r == k, continue; end
                if ~isempty(R_cell{s, r, w})
                    pooled_all = [pooled_all R_cell{s, r, w}];
                end
            end
        end
        if ~isempty(pooled_all)
            binEdges = 0:0.1:(max(pooled_all)+0.1);
            if length(binEdges) < 2, binEdges = [0 0.1]; end
            [n, ~] = histcounts(pooled_all, binEdges);
            h_prob = n / sum(n);
            Hr_jack(k,w) = -sum(h_prob(h_prob > 0) .* log2(h_prob(h_prob > 0)));
        end

        MI_jack(k,w) = Hr_jack(k,w) - Hrs_jack(k,w);
    end
end

% Jackknife SEM: SE = sqrt((N-1)/N * sum((x_jack - mean)^2))
N_jk = NReps;
MI_sem  = sqrt((N_jk-1)/N_jk * sum((MI_jack  - mean(MI_jack,1)).^2, 1));
Hr_sem  = sqrt((N_jk-1)/N_jk * sum((Hr_jack  - mean(Hr_jack,1)).^2, 1));
Hrs_sem = sqrt((N_jk-1)/N_jk * sum((Hrs_jack - mean(Hrs_jack,1)).^2, 1));

% ICE jackknife
MI_rate_jack = MI_jack / T_obs;
ICE_jack = MI_rate_jack ./ cost1_grand';
ICE_sem = sqrt((N_jk-1)/N_jk * sum((ICE_jack - mean(ICE_jack,1)).^2, 1));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%          MI Summary Plots            %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

timewindow = (1:NWindows) * winsize * duration / 1000;   % Time axis (s)

%% Plot MI, H(r), H(r|s) vs Time — with error bars
figure;
subplot(2, 2, 1)
errorbar(timewindow, Hr_vec, Hr_sem, 'ko-', 'LineWidth', 1.5, 'CapSize', 4); hold on;
errorbar(timewindow, Hrs_vec, Hrs_sem, 'bs-', 'LineWidth', 1.5, 'CapSize', 4);
errorbar(timewindow, MI_vec, MI_sem, 'r^-', 'LineWidth', 1.5, 'CapSize', 4);
legend('H(r)', 'H(r|s)', 'MI');
xlabel('Time [s]'); ylabel('Entropy / MI [bits]');
title('Mutual Information over Learning');

subplot(2, 2, 2)
errorbar(timewindow, MI_vec, MI_sem, 'r^-', 'LineWidth', 1.5, 'CapSize', 4);
xlabel('Time [s]'); ylabel('MI [bits]');
title('MI = H(r) - H(r|s)');

subplot(2, 2, 3)
errorbar(timewindow, ICE_vec, ICE_sem, 'm*-', 'LineWidth', 1.5, 'CapSize', 4);
xlabel('Time [s]'); ylabel('ICE [bits/J]');
title('Information-Cost Efficiency (ICE)');

subplot(2, 2, 4)
errorbar(timewindow, cv1_grand, cv1_sem, 'b*-', 'LineWidth', 1.2, 'CapSize', 3); hold on;
errorbar(timewindow, ss1_grand, ss1_sem, 'ko-', 'LineWidth', 1.2, 'CapSize', 3);
legend('CV', 'Sparseness');
xlabel('Time [s]'); ylabel('Value');
title('CV & Sparseness over Learning');

%% Plot CV vs MI / H(r) / H(r|s) — with error bars on both axes
figure;
subplot(1, 3, 1)
errorbar(cv1_grand, Hr_vec, Hr_sem, Hr_sem, cv1_sem, cv1_sem, 'ko', 'CapSize', 3); hold on;
xlabel('CV'); ylabel('H(r) [bits]'); title('CV vs H(r)');

subplot(1, 3, 2)
errorbar(cv1_grand, Hrs_vec, Hrs_sem, Hrs_sem, cv1_sem, cv1_sem, 'bs', 'CapSize', 3); hold on;
xlabel('CV'); ylabel('H(r|s) [bits]'); title('CV vs H(r|s)');

subplot(1, 3, 3)
errorbar(cv1_grand, MI_vec, MI_sem, MI_sem, cv1_sem, cv1_sem, 'r^', 'CapSize', 3); hold on;
xlabel('CV'); ylabel('MI [bits]'); title('CV vs MI');

%% Save MI Results (including jackknife)
save('MI_results.mat', ...
    'Hr_vec', 'Hrs_vec', 'MI_vec', 'ICE_vec', 'timewindow', ...
    'MI_sem', 'Hr_sem', 'Hrs_sem', 'ICE_sem', ...
    'MI_jack', 'Hr_jack', 'Hrs_jack', 'ICE_jack', ...
    'cv1_grand', 'ss1_grand', 'ent1_grand', 'ra1_grand', 'cost1_grand', ...
    'cv1_sem', 'ss1_sem', 'ent1_sem', 'ra1_sem', 'cost1_sem', ...
    'R_cell', 'NStim', 'NReps', 'NWindows', ...
    'ss1_store', 'cv1_store', 'ent1_store', 'ra1_store', ...
    'cost1_store', 'cvISI_store', 'mISI_store', 'EI_store');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%     Final Diagnostic Plots           %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Input Signal Visualization (last stim, last rep, last run)
figure;
for i = 1:8
    subplot(8, 1, i)
    plot(input_record(i, :));
    ylim([0 0.05])
end

%% Spike-Triggered Average (STA) — last rep of last stim
if length(v_rec) >= 100000
    [stae, rate_sta] = sta_mia([-ex_rec(1:100000); in_rec(1:100000)], ...
        v_rec(1:100000), dt, 100, 20);

    [stae, rate_sta] = sta_mia([-ex_rec(end-100000:end); in_rec(end-100000:end)], ...
        v_rec(end-100000:end), dt, 100, 20);
end

%% Smoothed E/I Balance — last rep of last stim
if ~isempty(ex_rec)
    windowSize = 60000;
    b = (1 / windowSize) * ones(1, windowSize);
    a = 1;
    ex_smooth = filter(b, a, -ex_rec);
    in_smooth = filter(b, a,  in_rec);

    figure;
    subplot(1, 2, 1);
    plot(ex_smooth(1:20000:end)); hold on;
    plot(in_smooth(1:20000:end))

    subplot(1, 2, 2);
    plot(ex_smooth(1:20000:end) ./ in_smooth(1:20000:end));
    ylim([1 3])
end

fprintf('\n=== Done. MI results saved to MI_results.mat ===\n');
