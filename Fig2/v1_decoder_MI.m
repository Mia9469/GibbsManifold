%% v1_decoder_MI.m
%  V1 stimulus-identity decoding analysis: gives a lower bound on
%  I(stimulus; population_response) by training a held-out classifier
%  and converting balanced accuracy to a confusion-matrix MI estimate.
%
%  Why this matters: the manuscript currently estimates per-neuron MI
%  with the Direct Method, which gives an upper bound (it counts every
%  spike pattern that varies across stim presentations as informative,
%  including silence-induced co-non-firing).  A held-out decoder gives
%  the COMPLEMENTARY lower bound: any stimulus information that is
%  actually accessible by a downstream readout is captured here.
%
%  Decoder protocol:
%    - Use only DRIVEN trials (1280-bin rasters from spikeRasters)
%    - Population response = per-neuron spike count over the full 1280-bin
%      window (159-dim feature per trial)
%    - Stimulus identity = neuralData.stim
%    - Stratified 5-fold CV; one LDA + one regularised LDA + one linear
%      kernel SVM for robustness
%    - Report: per-class accuracy, balanced accuracy, confusion matrix,
%      I(stim; predicted_class) (Cover-Thomas plug-in MI from confusion
%      matrix counts; gives a *lower bound* on I(stim; response))
%    - Compare against chance baseline by 100x label permutation
%
%  Output: console summary + v1_decoder_MI.mat (results struct).

clear; clc;

%% ============================== load ==============================
if ~isfile('105l001p16.mat')
    error('v1_decoder_MI:missing', '105l001p16.mat not found.');
end
S = load('105l001p16.mat');
nd = S.neuralData;
nt = max(nd.trialId);

% Driven trials only
n_v1 = 159;
X = [];   % [n_trials x n_v1]
y = [];   % [n_trials x 1] stim labels
for i = 1:nt
    sv = nd.spikeRasters{i, 1};
    if isempty(sv) || size(sv,2) ~= 1280, continue; end
    X(end+1, :) = sum(sv, 2)';   %#ok<AGROW>  spike counts per neuron
    y(end+1, 1) = nd.stim(i);    %#ok<AGROW>
end
n_trials = size(X, 1);
fprintf('Driven trials kept: %d, neurons: %d\n', n_trials, n_v1);

% Drop trials with non-finite stim
ok = isfinite(y);
X = X(ok, :);  y = y(ok);
n_trials = numel(y);

stim_vals = unique(y);
n_stim = numel(stim_vals);
fprintf('Unique stimulus values: %d (range %g .. %g)\n', n_stim, min(stim_vals), max(stim_vals));
ct = accumarray(arrayfun(@(s) find(stim_vals==s,1), y), 1);
fprintf('Trials per stim:  min=%d, median=%d, max=%d\n', min(ct), median(ct), max(ct));

% Discard stim classes with too few trials (need >=5 for 5-fold CV)
keep_stim = stim_vals(ct >= 5);
keep_idx  = ismember(y, keep_stim);
X = X(keep_idx, :);
y = y(keep_idx);
n_trials = numel(y);
n_stim = numel(unique(y));
fprintf('After dropping <5-trial stims:  %d trials, %d stim classes\n\n', n_trials, n_stim);

%% ============================== 5-fold CV decoder ==============================
K = 5;
rng(20260427, 'twister');
cv = cvpartition(y, 'KFold', K, 'Stratify', true);

methods = {'LDA','RegLDA','LinSVM'};
all_pred = struct();
for m = 1:numel(methods)
    all_pred.(methods{m}) = nan(size(y));
end

for k = 1:K
    tr = training(cv, k);
    te = test(cv, k);
    Xtr = X(tr,:);  ytr = y(tr);
    Xte = X(te,:);

    % LDA
    try
        mdl = fitcdiscr(Xtr, ytr, 'DiscrimType', 'linear');
        all_pred.LDA(te) = predict(mdl, Xte);
    catch
        % singular-covariance fallback handled by RegLDA below
    end
    % Regularised LDA (handles singular covariance)
    mdl = fitcdiscr(Xtr, ytr, 'DiscrimType', 'pseudoLinear');
    all_pred.RegLDA(te) = predict(mdl, Xte);
    % Linear-kernel SVM (one-vs-one)
    t = templateSVM('KernelFunction','linear','Standardize',true);
    mdl = fitcecoc(Xtr, ytr, 'Coding','onevsone', 'Learners', t);
    all_pred.LinSVM(te) = predict(mdl, Xte);
end

%% ============================== metrics ==============================
results = struct();
for m = 1:numel(methods)
    name = methods{m};
    yhat = all_pred.(name);
    valid = ~isnan(yhat);
    [acc, bal_acc, ConfMat, MI_conf] = decoder_metrics(y(valid), yhat(valid));
    results.(name).acc      = acc;
    results.(name).bal_acc  = bal_acc;
    results.(name).ConfMat  = ConfMat;
    results.(name).MI_bits  = MI_conf;   % LOWER bound on I(stim; response)
end

%% ============================== permutation-shuffled baseline ==============================
fprintf('Computing 100x label-shuffle null (RegLDA only)...\n');
B = 100;
MI_null = zeros(B,1);
for b = 1:B
    yp = y(randperm(numel(y)));
    cv_null = cvpartition(yp, 'KFold', K, 'Stratify', true);
    yhat_null = nan(size(yp));
    for k = 1:K
        tr = training(cv_null, k);
        te = test(cv_null, k);
        mdl = fitcdiscr(X(tr,:), yp(tr), 'DiscrimType', 'pseudoLinear');
        yhat_null(te) = predict(mdl, X(te,:));
    end
    [~,~,~,MI_null(b)] = decoder_metrics(yp, yhat_null);
end
results.null_MI_bits_p95   = quantile(MI_null, 0.95);
results.null_MI_bits_mean  = mean(MI_null);
results.null_MI_bits_std   = std(MI_null);

%% ============================== mean firing rate / cost ==============================
% Per-trial mean rate (Hz) -> per-trial cost C = 2.4 + 0.37 * mean(rate)
T_obs = 1.28;   % seconds
mean_rate_Hz = mean(X / (T_obs * 1), 2);   % mean across neurons of (counts / 1.28s)
% (X holds counts over 1280 ms = 1.28 s window)
C_per_trial  = 2.4 + 0.37 * mean_rate_Hz;
results.mean_rate_Hz_med = median(mean_rate_Hz);
results.cost_W_med       = median(C_per_trial);

%% ============================== ICE_decoder ==============================
% ICE = MI_decoder / median_cost  [bits / J]
% (median_cost = average per-trial cost; MI_decoder is per-trial info)
results.ICE_decoder_bits_per_J = struct();
for m = 1:numel(methods)
    name = methods{m};
    % MI_decoder is per-trial bits; per-trial cost = C_W * T_obs Joules.
    % ICE = MI / (C * T) = bits per Joule.
    results.ICE_decoder_bits_per_J.(name) = ...
        results.(name).MI_bits / (results.cost_W_med * T_obs);
end

%% ============================== print ==============================
fprintf('\n========== V1 stimulus-identity decoder summary ==========\n');
fprintf('  N=%d V1 neurons, n_trials=%d, n_stim=%d  (chance accuracy = %.3f)\n', ...
    n_v1, n_trials, n_stim, 1/n_stim);
fprintf('  median trial mean rate: %.2f Hz   median cost: %.2f W\n\n', ...
    results.mean_rate_Hz_med, results.cost_W_med);

for m = 1:numel(methods)
    name = methods{m};
    fprintf('[%s]\n', name);
    fprintf('  accuracy        : %.4f\n', results.(name).acc);
    fprintf('  balanced acc.   : %.4f   (chance %.3f)\n', results.(name).bal_acc, 1/n_stim);
    fprintf('  decoder MI      : %.4f bits  (lower bound on I(stim; response))\n', ...
        results.(name).MI_bits);
    fprintf('  information/cost proxy (decoder): %.4f\n', ...
        results.ICE_decoder_bits_per_J.(name));
end
fprintf('\n[null label-shuffle (RegLDA, B=%d)]\n', B);
fprintf('  MI mean=%.4f, std=%.4f, 95th-pct=%.4f bits\n', ...
    results.null_MI_bits_mean, results.null_MI_bits_std, results.null_MI_bits_p95);

% Compare against Direct-Method per-neuron MI from earlier analysis
fprintf('\n--- Comparison against Direct-Method per-neuron MI (V1 figure2_new.m) ---\n');
fprintf('  Direct Method:        per-neuron MI mean ~16.7 bits/s ~ %.1f bits/trial (1.28 s)\n', 16.7*T_obs);
fprintf('  Decoder lower bound:  %.2f bits/trial (RegLDA)\n', results.RegLDA.MI_bits);
fprintf('  Decoder ratio to DM: %.2f\n', results.RegLDA.MI_bits / (16.7*T_obs));
fprintf('  (large gap = much of the Direct-Method MI is not linearly decodable\n');
fprintf('   from population spike counts alone)\n');

save('v1_decoder_MI.mat', 'results', 'X', 'y', 'all_pred', 'MI_null', 'mean_rate_Hz', 'C_per_trial');
fprintf('\nSaved results to v1_decoder_MI.mat\n');


%% ============================== helpers ==============================
function [acc, bal_acc, ConfMat, MI_bits] = decoder_metrics(y, yhat)
    classes = unique([y; yhat]);
    K = numel(classes);
    ConfMat = zeros(K, K);
    for i = 1:K
        for j = 1:K
            ConfMat(i,j) = sum(y == classes(i) & yhat == classes(j));
        end
    end
    acc = sum(diag(ConfMat)) / sum(ConfMat(:));
    cls_n = sum(ConfMat, 2);
    diag_n = diag(ConfMat);
    nonempty = cls_n > 0;
    bal_acc = mean(diag_n(nonempty) ./ cls_n(nonempty));
    % I(true; pred) plug-in MI from joint counts
    Pjt = ConfMat / sum(ConfMat(:));
    Pi  = sum(Pjt, 2);
    Pj  = sum(Pjt, 1);
    M = 0;
    for i = 1:K
        for j = 1:K
            if Pjt(i,j) > 0 && Pi(i) > 0 && Pj(j) > 0
                M = M + Pjt(i,j) * log2(Pjt(i,j) / (Pi(i) * Pj(j)));
            end
        end
    end
    MI_bits = M;
end
