%% fig2_stats.m
%  Compute the reviewer-requested statistics for Fig 2 (V1 spontaneous vs
%  evoked) from the figure2_new.m workspace:
%    - sample sizes (n_spon, n_evk trials)
%    - Mann-Whitney U test (unpaired, unequal trial counts)
%    - Cliff's delta effect size
%    - 95% bootstrap CI on the (spon - evk) median difference
%
%  Run AFTER figure2_new.m has populated the workspace (or after
%  load('MI_analysis_figure2_new.mat')).  Saves a small CSV summary that
%  can be pasted into supp.tex Table tab:fig2-stats.

if ~exist('cv_v1_s', 'var') || ~exist('cv_v1', 'var') || ...
   ~exist('ss_v1_s', 'var') || ~exist('ss_v1', 'var') || ...
   ~exist('mrate_v1_s', 'var') || ~exist('mrate_v1', 'var')
    if isfile('figure2_stats_only.mat')
        load('figure2_stats_only.mat', ...
             'cv_v1_s','cv_v1','ss_v1_s','ss_v1','mrate_v1_s','mrate_v1');
        fprintf('[fig2_stats] loaded from figure2_stats_only.mat\n');
    else
        try
            load('MI_analysis_figure2_new.mat', ...
                 'cv_v1_s','cv_v1','ss_v1_s','ss_v1','mrate_v1_s','mrate_v1');
        catch ME
            error(['fig2_stats: required variables not in workspace and ', ...
                   'neither figure2_stats_only.mat nor MI_analysis_figure2_new.mat ', ...
                   'contain them.  Run figure2_stats_only.m first.\nDetails: %s'], ...
                  ME.message);
        end
    end
end

% Per-trial summaries (V1 only; reviewer comment is about Fig 2 V1 panel)
% mrate_v1_s and mrate_v1 are stacked across trials -> reduce to per-trial
% mean rate so each "trial" is one row of the comparison.
n_v1 = 159;
mr_spon = mean(reshape(mrate_v1_s, n_v1, []), 1);   % [1 x n_spon_trials]
mr_evk  = mean(reshape(mrate_v1,   n_v1, []), 1);    % [1 x n_evk_trials]

% These are per-trial population-level CV/S vectors saved by figure2_new.m.
% Force them to full (non-sparse) double so fprintf/quantile/ranksum work
% even if upstream spikeRasters were stored as sparse logicals.
to_full = @(x) full(double(x(:)'));
quantities = struct( ...
    'CV_pop',       struct('spon', to_full(cv_v1_s), 'evk', to_full(cv_v1)), ...
    'S_pop',        struct('spon', to_full(ss_v1_s), 'evk', to_full(ss_v1)), ...
    'mean_rate_Hz', struct('spon', to_full(mr_spon), 'evk', to_full(mr_evk)));

names = fieldnames(quantities);

%% ============================== compute ==============================
B = 10000;
rng(20260427, 'twister');

fprintf('=== Fig 2 V1 statistics (spontaneous vs evoked) ===\n');
fprintf('  N = %d V1 neurons per trial.  Per-trial sample sizes:\n', n_v1);
fprintf('    n_spontaneous trials: %d\n', numel(quantities.CV_pop.spon));
fprintf('    n_evoked       trials: %d\n', numel(quantities.CV_pop.evk));

rows = {};
for i = 1:numel(names)
    qn = names{i};
    s = quantities.(qn).spon;
    e = quantities.(qn).evk;
    s = s(isfinite(s));
    e = e(isfinite(e));

    [p, ~, st] = ranksum(s, e);                        % Mann-Whitney U
    U = st.ranksum;
    cliff_d = cliffs_delta(s, e);                      % effect size
    diff_med   = median(s) - median(e);
    boot_diff  = zeros(1, B);
    for b = 1:B
        bs = s(randi(numel(s), 1, numel(s)));
        be = e(randi(numel(e), 1, numel(e)));
        boot_diff(b) = median(bs) - median(be);
    end
    ci = quantile(boot_diff, [0.025, 0.975]);

    fprintf('\n[%s]\n', qn);
    fprintf('  spontaneous: n=%-3d   median=%.4f   IQR=[%.4f, %.4f]\n', ...
        numel(s), median(s), quantile(s, 0.25), quantile(s, 0.75));
    fprintf('  evoked     : n=%-3d   median=%.4f   IQR=[%.4f, %.4f]\n', ...
        numel(e), median(e), quantile(e, 0.25), quantile(e, 0.75));
    fprintf('  Mann-Whitney U=%.0f   p=%.3g\n', U, p);
    fprintf('  Cliff''s delta = %+.3f\n', cliff_d);
    fprintf('  bootstrap 95%% CI on (spon - evk) median diff: [%.4f, %.4f]   (point=%+.4f)\n', ...
        ci(1), ci(2), diff_med);

    rows(end+1, :) = {qn, numel(s), numel(e), median(s), median(e), ...
                      diff_med, ci(1), ci(2), U, p, cliff_d}; %#ok<AGROW>
end

%% ============================== save CSV ==============================
T = cell2table(rows, 'VariableNames', ...
    {'quantity','n_spon','n_evk','median_spon','median_evk', ...
     'diff_median','CI95_lo','CI95_hi','U','p','cliff_delta'});
writetable(T, 'fig2_stats.csv');
fprintf('\nSaved table to fig2_stats.csv\n');


function d = cliffs_delta(x, y)
    % Cliff's delta in [-1, 1].  Positive when x stochastically > y.
    nx = numel(x); ny = numel(y);
    [~, ~, R] = unique([x(:); y(:)]);
    Rx = R(1:nx);
    Ry = R(nx+1:end);
    d = (sum(Rx) / nx - sum(Ry) / ny) * 2 / (nx + ny);
    % Equivalent definition that avoids sign convention drift:
    diffs = sign(x(:) - y(:)');
    d = sum(diffs(:)) / (nx * ny);
end
