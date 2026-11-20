%% figure2_stats_only.m
%  Lightweight version of figure2_new.m: recompute ONLY the per-trial
%  population-level CV, sparseness, and mean firing rate for V1 and V2,
%  then save them.  Skips ALL Direct-Method MI computation and ALL
%  plotting.
%
%  Use this when:
%    - You need the per-trial cv_v1_s/cv_v1/ss_v1_s/ss_v1/mrate_*
%      vectors (e.g. to feed fig2_stats), but the slow MI re-run is
%      not worth it.
%    - The previously-saved MI_analysis_figure2_new.mat does NOT contain
%      mrate / per-trial CV / per-trial S (it doesn't).
%
%  Output: figure2_stats_only.mat (a few hundred kB), with:
%    cv_v1_s, cv_v1, ss_v1_s, ss_v1, mrate_v1_s, mrate_v1   (V1)
%    cv_v2_s, cv_v2, ss_v2_s, ss_v2, mrate_v2_s, mrate_v2   (V2)
%    cost1_ps, cost1_p, cost2_ps, cost2_p                    (calibrated cost proxy)
%    n_v1, n_v2, dur_spon, dur_driven
%
%  Runtime: ~10-30 s for 6400 trials.
%
%  Then run fig2_stats:
%    >> figure2_stats_only
%    >> load('figure2_stats_only.mat')
%    >> fig2_stats

clear; clc;
fprintf('[figure2_stats_only] starting...\n');
t0 = tic;

%% Load
load('105l001p16.mat')
nt          = max(neuralData.trialId);
stim_labels = neuralData.stim;
fprintf('[figure2_stats_only] loaded 105l001p16.mat (nt=%d trials)\n', nt);

%% Calibrated affine cost-proxy coefficients (adapted from Yu et al. 2018)
C1 = 2.4;        % proxy baseline
C2 = 0.37;       % proxy slope

n_v1 = 159;
n_v2 = 25;
dur_spon   = 1.5;       % s (1500 bins at 1 ms)
dur_driven = 1.28;      % s (1280 bins at 1 ms)

%% Allocate
cv_v1_s   = [];  cv_v2_s   = [];
cv_v1     = [];  cv_v2     = [];
ss_v1_s   = [];  ss_v2_s   = [];
ss_v1     = [];  ss_v2     = [];
mrate_v1_s = []; mrate_v2_s = [];
mrate_v1   = []; mrate_v2   = [];
cost1_ps  = [];  cost2_ps  = [];
cost1_p   = [];  cost2_p   = [];

n_spon = 0; n_driven = 0;

%% Single pass over trials
for i = 1:nt
    sv1 = neuralData.spikeRasters{i, 1};
    sv2 = neuralData.spikeRasters{i, 2};

    if size(sv1, 2) == 1500
        % --- spontaneous ---
        n_spon = n_spon + 1;
        rates_v1 = full(sum(sv1, 2)) / dur_spon;     % [n_v1 x 1] Hz
        rates_v2 = full(sum(sv2, 2)) / dur_spon;     % [n_v2 x 1] Hz

        cv_v1_s   = [cv_v1_s,   std(rates_v1) / mean(rates_v1)];
        cv_v2_s   = [cv_v2_s,   std(rates_v2) / mean(rates_v2)];
        ss_v1_s   = [ss_v1_s,   (1 - mean(rates_v1)^2 / mean(rates_v1.^2)) / (1 - 1/n_v1)];
        ss_v2_s   = [ss_v2_s,   (1 - mean(rates_v2)^2 / mean(rates_v2.^2)) / (1 - 1/n_v2)];
        mrate_v1_s = [mrate_v1_s, rates_v1];
        mrate_v2_s = [mrate_v2_s, rates_v2];
        cost1_ps  = [cost1_ps,  C1 + C2 * mean(rates_v1)];
        cost2_ps  = [cost2_ps,  C1 + C2 * mean(rates_v2)];
    else
        % --- driven (1280 bins) ---
        n_driven = n_driven + 1;
        rates_v1 = full(sum(sv1, 2)) / dur_driven;
        rates_v2 = full(sum(sv2, 2)) / dur_driven;

        cv_v1     = [cv_v1,     std(rates_v1) / mean(rates_v1)];
        cv_v2     = [cv_v2,     std(rates_v2) / mean(rates_v2)];
        ss_v1     = [ss_v1,     (1 - mean(rates_v1)^2 / mean(rates_v1.^2)) / (1 - 1/n_v1)];
        ss_v2     = [ss_v2,     (1 - mean(rates_v2)^2 / mean(rates_v2.^2)) / (1 - 1/n_v2)];
        mrate_v1  = [mrate_v1,  rates_v1];
        mrate_v2  = [mrate_v2,  rates_v2];
        cost1_p   = [cost1_p,   C1 + C2 * mean(rates_v1)];
        cost2_p   = [cost2_p,   C1 + C2 * mean(rates_v2)];
    end
end

fprintf('[figure2_stats_only] done in %.1f s\n', toc(t0));
fprintf('  spontaneous trials: %d   driven trials: %d\n', n_spon, n_driven);
fprintf('  V1 cv_pop  spon: median=%.3f  evk: median=%.3f\n', median(cv_v1_s), median(cv_v1));
fprintf('  V1 ss_pop  spon: median=%.3f  evk: median=%.3f\n', median(ss_v1_s), median(ss_v1));

save('figure2_stats_only.mat', ...
     'cv_v1_s','cv_v1','ss_v1_s','ss_v1','mrate_v1_s','mrate_v1', ...
     'cv_v2_s','cv_v2','ss_v2_s','ss_v2','mrate_v2_s','mrate_v2', ...
     'cost1_ps','cost1_p','cost2_ps','cost2_p', ...
     'n_v1','n_v2','dur_spon','dur_driven','n_spon','n_driven');
fprintf('[figure2_stats_only] saved figure2_stats_only.mat\n');
