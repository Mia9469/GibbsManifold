%% check_trial_structure.m
%  Verify the trial composition of 105l001p16.mat that figure2_new.m
%  consumes.  Reports:
%    - nt = max(neuralData.trialId) (script's loop count)
%    - actual size of neuralData.spikeRasters (rows x area columns)
%    - distribution of column-counts (1500 = spontaneous, 1280 = driven)
%    - range of trialId values (any duplicates / sub-trials)
%    - spontaneous and driven trial counts (should match the 3473 vs 3473
%      report from fig2_stats)
%
%  Run from Fig2/.

clear; clc;

if ~isfile('105l001p16.mat')
    error('check_trial_structure:missing', '105l001p16.mat not found in current dir.');
end

S = load('105l001p16.mat');
fprintf('--- top-level fields in 105l001p16.mat ---\n');
disp(fieldnames(S));

if ~isfield(S, 'neuralData')
    error('check_trial_structure:missing_field', 'neuralData not present.');
end
nd = S.neuralData;
fprintf('\n--- fields of neuralData ---\n');
disp(fieldnames(nd));

% trialId / spikeRasters shape
nt_loop  = max(nd.trialId);                       % what figure2_new.m uses
sz_rast  = size(nd.spikeRasters);
fprintf('\nnt = max(neuralData.trialId) = %d\n', nt_loop);
fprintf('size(neuralData.spikeRasters) = [%d x %d]\n', sz_rast(1), sz_rast(2));
fprintf('numel(neuralData.trialId)     = %d\n', numel(nd.trialId));

if isfield(nd, 'stim')
    fprintf('numel(neuralData.stim)        = %d  (unique stim values: %d)\n', ...
        numel(nd.stim), numel(unique(nd.stim)));
end

% Column-length distribution -> spontaneous (1500) vs driven (1280)
col_lens = zeros(1, sz_rast(1));
for i = 1:sz_rast(1)
    sv1 = nd.spikeRasters{i, 1};
    if isempty(sv1)
        col_lens(i) = -1;
    else
        col_lens(i) = size(sv1, 2);
    end
end
[uvals, ~, ic] = unique(col_lens);
ucounts = accumarray(ic, 1);
fprintf('\n--- distribution of spikeRasters{i,1} column counts ---\n');
for k = 1:numel(uvals)
    if uvals(k) == -1
        tag = '(empty)';
    elseif uvals(k) == 1500
        tag = '(spontaneous, 1.5 s)';
    elseif uvals(k) == 1280
        tag = '(driven, 1.28 s)';
    else
        tag = '(unknown)';
    end
    fprintf('  cols=%-5d  count=%-5d  %s\n', uvals(k), ucounts(k), tag);
end

n_spon  = sum(col_lens == 1500);
n_driv  = sum(col_lens == 1280);
n_other = sum(col_lens > 0 & col_lens ~= 1500 & col_lens ~= 1280);
fprintf('\n  spontaneous trials (1500 bins): %d\n', n_spon);
fprintf('  driven      trials (1280 bins): %d\n', n_driv);
fprintf('  other / unknown              : %d\n', n_other);

% Uniqueness of trialId
ut = unique(nd.trialId);
fprintf('\n  unique trialId values         : %d\n', numel(ut));
fprintf('  duplicates                    : %d\n', numel(nd.trialId) - numel(ut));

% Diagnostic: peek at one each
spon_idx = find(col_lens == 1500, 1, 'first');
driv_idx = find(col_lens == 1280, 1, 'first');
if ~isempty(spon_idx)
    fprintf('\n  example spontaneous trial row=%d:\n', spon_idx);
    fprintf('    spikeRasters{%d,1} size = [%d x %d]\n', spon_idx, ...
        size(nd.spikeRasters{spon_idx,1},1), size(nd.spikeRasters{spon_idx,1},2));
    fprintf('    spikeRasters{%d,2} size = [%d x %d]\n', spon_idx, ...
        size(nd.spikeRasters{spon_idx,2},1), size(nd.spikeRasters{spon_idx,2},2));
    if isfield(nd,'stim')
        fprintf('    stim(%d) = %g\n', spon_idx, nd.stim(spon_idx));
    end
end
if ~isempty(driv_idx)
    fprintf('\n  example driven      trial row=%d:\n', driv_idx);
    fprintf('    spikeRasters{%d,1} size = [%d x %d]\n', driv_idx, ...
        size(nd.spikeRasters{driv_idx,1},1), size(nd.spikeRasters{driv_idx,1},2));
    if isfield(nd,'stim')
        fprintf('    stim(%d) = %g\n', driv_idx, nd.stim(driv_idx));
    end
end

% Per-stimulus repetition count (if stim is defined)
if isfield(nd,'stim')
    [u, ~, ic2] = unique(nd.stim);
    cts = accumarray(ic2, 1);
    fprintf('\n  per-stimulus trial counts (first 10):\n');
    for k = 1:min(10, numel(u))
        fprintf('    stim=%-6g  trials=%d\n', u(k), cts(k));
    end
    if numel(u) > 10
        fprintf('    ... (%d total stim values)\n', numel(u));
    end
end
