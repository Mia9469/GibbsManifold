% figure2_new.m
% Computes and plots population- and lifetime-level statistics
% (CV, sparseness, entropy, and information-cost efficiency) for
% neural spike data from two cortical areas (V1 and V2),
% separately for spontaneous and visually driven conditions.
%
% MUTUAL INFORMATION via DIRECT METHOD (Strong et al. 1998):
%   - Temporal MI computed from binary spike words (not ISI histograms)
%   - 1/T extrapolation to reduce finite-word-length dependence
%   - Spatial-only MI still uses histogram method on spike counts
%   - Per-neuron, population, and spatial-only MI levels
%
% Data source: CRCNS V1/V2 Dataset (https://crcns.org/data-sets/vc/v1v2-1)
% Data file: 105l001p16.mat and similar session files
% Requires: neuralData.spikeRasters{trial, area}, neuralData.stim, neuralData.trialId

clear; close all;

% Add path to toolbox directory (for reference; we use local copies of core functions)
addpath('/Users/mia/Library/Mobile Documents/com~apple~CloudDocs/compute_entropy');

%% Load data
load('105l001p16.mat')
nt = max(neuralData.trialId);  % total number of trials
stim_labels = neuralData.stim;  % stimulus identify for each trial
unique_stim = unique(stim_labels);  % unique stimulus identities
n_stim = length(unique_stim);  % number of distinct stimuli

%% Initialise accumulators

% Population-level CV (coefficient of variation)
cv_v1_s = [];  cv_v2_s = [];   % spontaneous
cv_v1   = [];  cv_v2   = [];   % driven

% Population-level sparseness
ss_v1_s = [];  ss_v2_s = [];   % spontaneous
ss_v1   = [];  ss_v2   = [];   % driven

% Population mean firing rates
mrate_v1   = [];  mrate_v2   = [];   % driven
mrate_v1_s = [];  mrate_v2_s = [];   % spontaneous

% Lifetime CV and sparseness
ss1_l   = [];  cv1_l   = [];   % V1, driven
ss2_l   = [];  cv2_l   = [];   % V2, driven
ss1_l_s = [];  cv1_l_s = [];   % V1, spontaneous
ss2_l_s = [];  cv2_l_s = [];   % V2, spontaneous

% Entropy and metabolic cost (population-level)
entr1_ps = [];  entr2_ps = [];   % entropy, spontaneous
cost1_ps = [];  cost2_ps = [];   % cost,    spontaneous
entr1_p  = [];  entr2_p  = [];   % entropy, driven
cost1_p  = [];  cost2_p  = [];   % cost,    driven

% Entropy (lifetime-level)
entr1_ls = [];  entr2_ls = [];   % spontaneous
entr1_l  = [];  entr2_l  = [];   % driven

% Raster cell arrays
rast_v1_s = {};  rast_v1_d = {};
rast_v2_s = {};  rast_v2_d = {};

% Metabolic cost parameters (from Yu et al. 2018, human gray matter)
% Calibrated affine cost proxy C(r) = C1 + C2*r.
C1 = 2.4;    % baseline proxy term
C2 = 0.37;   % rate-dependent proxy coefficient

% Trial durations [s]
dur_driven = 1.28;   % driven trial duration [s]
dur_spon   = 1.5;    % spontaneous trial duration [s]

% Storage for ISI-based firing rates (used for spatial-only MI)
rate_v1_driven_pooled = cell(1, nt);
rate_v1_driven_byNeuron = cell(159, nt);
rate_v1_spon_pooled = cell(1, nt);
rate_v1_spon_byNeuron = cell(159, nt);
rate_v2_driven_pooled = cell(1, nt);
rate_v2_driven_byNeuron = cell(25, nt);
rate_v2_spon_pooled = cell(1, nt);
rate_v2_spon_byNeuron = cell(25, nt);
stim_driven = [];
trial_count_driven = 0;
trial_count_spon = 0;

%% -----------------------------------------------------------------------
%% SECTION 1: Population-level CV, sparseness, and entropy
%% -----------------------------------------------------------------------
% Spontaneous trials have 1500 time bins; driven trials have 1280 bins.

for i = 1:nt

    spikes_v1 = neuralData.spikeRasters{i, 1};
    spikes_v2 = neuralData.spikeRasters{i, 2};

    if size(spikes_v1, 2) == 1500
        % ----- Spontaneous condition (1.5 s window) -----
        dur = 1.5;

        % Population CV
        cv_v1_s = [cv_v1_s,  std(sum(spikes_v1, 2) / dur) ./ mean(sum(spikes_v1, 2) / dur)];
        cv_v2_s = [cv_v2_s,  std(sum(spikes_v2, 2) / dur) ./ mean(sum(spikes_v2, 2) / dur)];

        % Population sparseness (Vinje-Gallant bias correction: divide by 1-1/N)
        % S_TR = 1 - <r>^2/<r^2> has maximum (1-1/N) for a population of
        % N neurons (one neuron firing alone), so dividing by (1-1/N)
        % normalises the maximum to 1.
        ss_v1_s = [ss_v1_s, (1 - mean(sum(spikes_v1,2)/dur)^2 ./ mean((sum(spikes_v1,2)/dur).^2)) / (1 - 1/159)];
        ss_v2_s = [ss_v2_s, (1 - mean(sum(spikes_v2,2)/dur)^2 ./ mean((sum(spikes_v2,2)/dur).^2)) / (1 - 1/25)];

        % Mean firing rates across neurons
        mrate_v1_s = [mrate_v1_s, sum(spikes_v1, 2) / dur];
        mrate_v2_s = [mrate_v2_s, sum(spikes_v2, 2) / dur];

        % Entropy of the population firing-rate distribution (V1)
        [n, ~] = histcounts(sum(spikes_v1, 2) / dur, 100);
        h = n / sum(n);
        entr = 0;
        for j = h
            if j > 0
                entr = entr - j * log(j);
            end
        end
        entr1_ps = [entr1_ps, entr];
        cost1_ps = [cost1_ps, C1 + C2 * mean(sum(spikes_v1, 2) / dur)];

        % Entropy of the population firing-rate distribution (V2)
        [n, ~] = histcounts(sum(spikes_v2, 2) / dur, 100);
        h = n / sum(n);
        entr = 0;
        for j = h
            if j > 0
                entr = entr - j * log(j);
            end
        end
        entr2_ps = [entr2_ps, entr];
        cost2_ps = [cost2_ps, C1 + C2 * mean(sum(spikes_v2, 2) / dur)];

        % Store rasters
        rast_v1_s{end+1} = spikes_v1;
        rast_v2_s{end+1} = spikes_v2;

        % Store ISI-based rates for spontaneous condition
        trial_count_spon = trial_count_spon + 1;
        rate_v1_spon_pooled_trial = [];
        rate_v2_spon_pooled_trial = [];

        for neuron = 1:size(spikes_v1, 1)
            spike_times = find(spikes_v1(neuron, :)) * dur / size(spikes_v1, 2);
            if length(spike_times) >= 2
                isi = diff(spike_times);
                rate_isi = 1000 ./ isi;
                rate_v1_spon_byNeuron{neuron, trial_count_spon} = rate_isi;
                rate_v1_spon_pooled_trial = [rate_v1_spon_pooled_trial, rate_isi];
            end
        end

        for neuron = 1:size(spikes_v2, 1)
            spike_times = find(spikes_v2(neuron, :)) * dur / size(spikes_v2, 2);
            if length(spike_times) >= 2
                isi = diff(spike_times);
                rate_isi = 1000 ./ isi;
                rate_v2_spon_byNeuron{neuron, trial_count_spon} = rate_isi;
                rate_v2_spon_pooled_trial = [rate_v2_spon_pooled_trial, rate_isi];
            end
        end

        rate_v1_spon_pooled{trial_count_spon} = rate_v1_spon_pooled_trial;
        rate_v2_spon_pooled{trial_count_spon} = rate_v2_spon_pooled_trial;

    else
        % ----- Driven condition (1.28 s window) -----
        dur = 1.28;

        % Population CV
        cv_v1 = [cv_v1, std(sum(spikes_v1, 2) / dur) ./ mean(sum(spikes_v1, 2) / dur)];
        cv_v2 = [cv_v2, std(sum(spikes_v2, 2) / dur) ./ mean(sum(spikes_v2, 2) / dur)];

        % Population sparseness (Vinje-Gallant bias correction: divide by 1-1/N)
        ss_v1 = [ss_v1, (1 - mean(sum(spikes_v1,2)/dur)^2 ./ mean((sum(spikes_v1,2)/dur).^2)) / (1 - 1/159)];
        ss_v2 = [ss_v2, (1 - mean(sum(spikes_v2,2)/dur)^2 ./ mean((sum(spikes_v2,2)/dur).^2)) / (1 - 1/25)];

        % Mean firing rates across neurons
        mrate_v1 = [mrate_v1, sum(spikes_v1, 2) / dur];
        mrate_v2 = [mrate_v2, sum(spikes_v2, 2) / dur];

        % Entropy of the population firing-rate distribution (V1)
        [n, ~] = histcounts(sum(spikes_v1, 2) / dur, 100);
        h = n / sum(n);
        entr = 0;
        for j = h
            if j > 0
                entr = entr - j * log(j);
            end
        end
        entr1_p = [entr1_p, entr];
        cost1_p = [cost1_p, C1 + C2 * mean(sum(spikes_v1, 2) / dur)];

        % Entropy of the population firing-rate distribution (V2)
        [n, ~] = histcounts(sum(spikes_v2, 2) / dur, 100);
        h = n / sum(n);
        entr = 0;
        for j = h
            if j > 0
                entr = entr - j * log(j);
            end
        end
        entr2_p = [entr2_p, entr];
        cost2_p = [cost2_p, C1 + C2 * mean(sum(spikes_v2, 2) / dur)];

        % Store rasters
        rast_v1_d{end+1} = spikes_v1;
        rast_v2_d{end+1} = spikes_v2;

        % Compute ISI-based firing rates for spatial-only MI
        trial_count_driven = trial_count_driven + 1;
        rate_v1_pooled = [];
        rate_v2_pooled = [];

        % V1: Store per-neuron rates
        for neuron = 1:size(spikes_v1, 1)
            spike_times = find(spikes_v1(neuron, :)) * dur / size(spikes_v1, 2);
            if length(spike_times) >= 2
                isi = diff(spike_times);
                rate_isi = 1000 ./ isi;
                rate_v1_driven_byNeuron{neuron, trial_count_driven} = rate_isi;
                rate_v1_pooled = [rate_v1_pooled, rate_isi];
            end
        end

        % V2: Store per-neuron rates
        for neuron = 1:size(spikes_v2, 1)
            spike_times = find(spikes_v2(neuron, :)) * dur / size(spikes_v2, 2);
            if length(spike_times) >= 2
                isi = diff(spike_times);
                rate_isi = 1000 ./ isi;
                rate_v2_driven_byNeuron{neuron, trial_count_driven} = rate_isi;
                rate_v2_pooled = [rate_v2_pooled, rate_isi];
            end
        end

        % Store pooled rates and stimulus label
        rate_v1_driven_pooled{trial_count_driven} = rate_v1_pooled;
        rate_v2_driven_pooled{trial_count_driven} = rate_v2_pooled;
        stim_driven = [stim_driven, stim_labels(i)];
    end
end

%% -----------------------------------------------------------------------
%% SECTION 2: MI computation using Direct Method (Strong et al. 1998)
%% -----------------------------------------------------------------------
% Direct method parameters
T_list = [4 5 6 7 8 10 13 15 20 25 30 40];  % word lengths in bins (ms)
n_fit = 8;                                     % number of points for linear fit
dt_bin = 0.001;                                % 1 ms bin width
refractory = 1;                                % refractory period (bins)

n_v1 = 159;
n_v2 = 25;
n_driven = trial_count_driven;
n_spon = trial_count_spon;

unique_stim_driven = unique(stim_driven);

%% ===== SPATIAL-ONLY MI (histogram method on spike counts) =====
% This is the correct approach for spatial-only since it measures
% the distribution of per-neuron spike count rates.

fprintf('\n=== Computing Spatial-Only MI (histogram method) ===\n');

% --- V1 Spatial-Only MI ---
spatial_rates_v1_byTrial = cell(1, length(rast_v1_d));
all_spatial_rates_v1_driven = [];
for trial = 1:length(rast_v1_d)
    spikes = rast_v1_d{trial};
    rates = sum(spikes, 2)' / dur_driven;  % 1x159 firing rates
    spatial_rates_v1_byTrial{trial} = rates;
    all_spatial_rates_v1_driven = [all_spatial_rates_v1_driven, rates];
end

if ~isempty(all_spatial_rates_v1_driven)
    binEdges_spatial = 0:1:(max(all_spatial_rates_v1_driven)+1);
    [n_r_spatial, ~] = histcounts(all_spatial_rates_v1_driven, binEdges_spatial);
    p_r_spatial = n_r_spatial / sum(n_r_spatial);
    p_r_spatial = p_r_spatial(p_r_spatial > 0);
    Hr_spatial_v1 = -sum(p_r_spatial .* log2(p_r_spatial));

    Hrs_spatial_v1 = 0;
    for s = unique_stim_driven
        stim_mask = (stim_driven == s);
        P_s = sum(stim_mask) / length(stim_driven);
        rates_this_stim = [];
        for trial = find(stim_mask)
            rates_this_stim = [rates_this_stim, spatial_rates_v1_byTrial{trial}];
        end
        if ~isempty(rates_this_stim)
            [n_s_spatial, ~] = histcounts(rates_this_stim, binEdges_spatial);
            p_s_spatial = n_s_spatial / sum(n_s_spatial);
            p_s_spatial = p_s_spatial(p_s_spatial > 0);
            if ~isempty(p_s_spatial)
                H_rs_spatial = -sum(p_s_spatial .* log2(p_s_spatial));
                Hrs_spatial_v1 = Hrs_spatial_v1 + P_s * H_rs_spatial;
            end
        end
    end
    MI_spatial_v1 = Hr_spatial_v1 - Hrs_spatial_v1;
    fprintf('  V1 Spatial-Only: H(r)=%.4f bits, H(r|s)=%.4f bits, MI=%.4f bits\n', ...
        Hr_spatial_v1, Hrs_spatial_v1, MI_spatial_v1);
else
    Hr_spatial_v1 = 0; Hrs_spatial_v1 = 0; MI_spatial_v1 = 0;
end

% --- V2 Spatial-Only MI ---
spatial_rates_v2_byTrial = cell(1, length(rast_v2_d));
all_spatial_rates_v2_driven = [];
for trial = 1:length(rast_v2_d)
    spikes = rast_v2_d{trial};
    rates = sum(spikes, 2)' / dur_driven;
    spatial_rates_v2_byTrial{trial} = rates;
    all_spatial_rates_v2_driven = [all_spatial_rates_v2_driven, rates];
end

if ~isempty(all_spatial_rates_v2_driven)
    binEdges_spatial = 0:1:(max(all_spatial_rates_v2_driven)+1);
    [n_r_spatial, ~] = histcounts(all_spatial_rates_v2_driven, binEdges_spatial);
    p_r_spatial = n_r_spatial / sum(n_r_spatial);
    p_r_spatial = p_r_spatial(p_r_spatial > 0);
    Hr_spatial_v2 = -sum(p_r_spatial .* log2(p_r_spatial));

    Hrs_spatial_v2 = 0;
    for s = unique_stim_driven
        stim_mask = (stim_driven == s);
        P_s = sum(stim_mask) / length(stim_driven);
        rates_this_stim = [];
        for trial = find(stim_mask)
            rates_this_stim = [rates_this_stim, spatial_rates_v2_byTrial{trial}];
        end
        if ~isempty(rates_this_stim)
            [n_s_spatial, ~] = histcounts(rates_this_stim, binEdges_spatial);
            p_s_spatial = n_s_spatial / sum(n_s_spatial);
            p_s_spatial = p_s_spatial(p_s_spatial > 0);
            if ~isempty(p_s_spatial)
                H_rs_spatial = -sum(p_s_spatial .* log2(p_s_spatial));
                Hrs_spatial_v2 = Hrs_spatial_v2 + P_s * H_rs_spatial;
            end
        end
    end
    MI_spatial_v2 = Hr_spatial_v2 - Hrs_spatial_v2;
    fprintf('  V2 Spatial-Only: H(r)=%.4f bits, H(r|s)=%.4f bits, MI=%.4f bits\n', ...
        Hr_spatial_v2, Hrs_spatial_v2, MI_spatial_v2);
else
    Hr_spatial_v2 = 0; Hrs_spatial_v2 = 0; MI_spatial_v2 = 0;
end

% --- Spatial-Only Spontaneous H(r) ---
all_spatial_rates_v1_spon = [];
for trial = 1:length(rast_v1_s)
    spikes = rast_v1_s{trial};
    rates = sum(spikes, 2)' / dur_spon;
    all_spatial_rates_v1_spon = [all_spatial_rates_v1_spon, rates];
end
if ~isempty(all_spatial_rates_v1_spon)
    binEdges_spatial = 0:1:(max(all_spatial_rates_v1_spon)+1);
    [n_r, ~] = histcounts(all_spatial_rates_v1_spon, binEdges_spatial);
    p_r = n_r / sum(n_r); p_r = p_r(p_r > 0);
    Hr_spatial_v1_spon = -sum(p_r .* log2(p_r));
else
    Hr_spatial_v1_spon = 0;
end

all_spatial_rates_v2_spon = [];
for trial = 1:length(rast_v2_s)
    spikes = rast_v2_s{trial};
    rates = sum(spikes, 2)' / dur_spon;
    all_spatial_rates_v2_spon = [all_spatial_rates_v2_spon, rates];
end
if ~isempty(all_spatial_rates_v2_spon)
    binEdges_spatial = 0:1:(max(all_spatial_rates_v2_spon)+1);
    [n_r, ~] = histcounts(all_spatial_rates_v2_spon, binEdges_spatial);
    p_r = n_r / sum(n_r); p_r = p_r(p_r > 0);
    Hr_spatial_v2_spon = -sum(p_r .* log2(p_r));
else
    Hr_spatial_v2_spon = 0;
end

%% ===== PER-NEURON TEMPORAL MI (Direct Method) =====
% For each neuron: construct spike_long (all driven trials), compute total entropy
% For noise: per stimulus, stack trials of that stimulus, compute noise entropy
% Extrapolate H/T vs 1/T to get entropy rates, MI = Hr - Hn

fprintf('\n=== Computing Per-Neuron Temporal MI (Direct Method) ===\n');

mi_v1_perNeuron = zeros(1, n_v1);
hr_v1_perNeuron_total = zeros(1, n_v1);
hn_v1_perNeuron = zeros(1, n_v1);

% Per-stimulus Hr: [n_v1 x n_stim_driven] for fair spon vs driven comparison
n_stim_driven = length(unique_stim_driven);
hr_v1_perNeuron_perStim = zeros(n_v1, n_stim_driven);
hn_v1_perNeuron_perStim = zeros(n_v1, n_stim_driven);
mi_v1_perNeuron_perStim = zeros(n_v1, n_stim_driven);

for neuron = 1:n_v1
    fprintf('  V1 neuron %d/%d\r', neuron, n_v1);

    % Construct spike_long: [n_driven_trials x 1280]
    spike_long_n = zeros(n_driven, 1280);
    for trial = 1:n_driven
        spike_long_n(trial, :) = rast_v1_d{trial}(neuron, :);
    end

    % Apply refractory period
    spike_long_n = apply_refractory_local(spike_long_n, refractory);

    % Compute H_total for each T
    nT = length(T_list);
    H_total_vals = zeros(1, nT);
    H_noise_vals = zeros(1, nT);

    % Per-stimulus H_total for each T: [n_stim_driven x nT]
    H_total_perStim = zeros(n_stim_driven, nT);

    for ti = 1:nT
        T = T_list(ti);
        H_total_vals(ti) = total_entropy_local(spike_long_n, T);

        % Noise entropy: weighted average across stimuli
        H_noise_weighted = 0;
        for si = 1:n_stim_driven
            s = unique_stim_driven(si);
            stim_mask = (stim_driven == s);
            P_s = sum(stim_mask) / length(stim_driven);

            % spike_repeat for this neuron and stimulus
            trial_idx = find(stim_mask);
            if length(trial_idx) < 2
                continue;
            end
            spike_repeat_ns = zeros(length(trial_idx), 1280);
            for k = 1:length(trial_idx)
                spike_repeat_ns(k, :) = rast_v1_d{trial_idx(k)}(neuron, :);
            end
            spike_repeat_ns = apply_refractory_local(spike_repeat_ns, refractory);

            H_noise_s = noise_entropy_local(spike_repeat_ns, T);
            H_noise_weighted = H_noise_weighted + P_s * H_noise_s;

            % Per-stimulus total entropy (only trials for this stim)
            H_total_perStim(si, ti) = total_entropy_local(spike_repeat_ns, T);
        end
        H_noise_vals(ti) = H_noise_weighted;
    end

    % Extrapolate: H/T vs 1/T, intercept = entropy rate (bits/s)
    T_sec = T_list * dt_bin;
    inv_T = 1 ./ T_sec;
    rate_total = H_total_vals ./ T_sec;
    rate_noise = H_noise_vals ./ T_sec;

    n_fit_use = min(n_fit, nT);
    fit_idx = 1:n_fit_use;

    [~, fit_total] = linear_fit_local(inv_T(fit_idx), rate_total(fit_idx));
    [~, fit_noise] = linear_fit_local(inv_T(fit_idx), rate_noise(fit_idx));

    Hr_n = fit_total.intercept;
    Hn_n = fit_noise.intercept;
    MI_n = max(0, Hr_n - Hn_n);

    mi_v1_perNeuron(neuron) = MI_n;
    hr_v1_perNeuron_total(neuron) = Hr_n;
    hn_v1_perNeuron(neuron) = Hn_n;

    % Per-stimulus extrapolation
    for si = 1:n_stim_driven
        rate_total_s = H_total_perStim(si, :) ./ T_sec;
        rate_noise_s = H_noise_vals ./ T_sec;  % noise is already per-stim weighted
        % Re-compute noise for this specific stimulus
        H_noise_this_stim = zeros(1, nT);
        s = unique_stim_driven(si);
        trial_idx = find(stim_driven == s);
        if length(trial_idx) >= 2
            for ti = 1:nT
                spike_repeat_ns = zeros(length(trial_idx), 1280);
                for k = 1:length(trial_idx)
                    spike_repeat_ns(k, :) = rast_v1_d{trial_idx(k)}(neuron, :);
                end
                spike_repeat_ns = apply_refractory_local(spike_repeat_ns, refractory);
                H_noise_this_stim(ti) = noise_entropy_local(spike_repeat_ns, T_list(ti));
            end
            rate_noise_this = H_noise_this_stim ./ T_sec;
            [~, fit_t] = linear_fit_local(inv_T(fit_idx), rate_total_s(fit_idx));
            [~, fit_n] = linear_fit_local(inv_T(fit_idx), rate_noise_this(fit_idx));
            hr_v1_perNeuron_perStim(neuron, si) = fit_t.intercept;
            hn_v1_perNeuron_perStim(neuron, si) = fit_n.intercept;
            mi_v1_perNeuron_perStim(neuron, si) = max(0, fit_t.intercept - fit_n.intercept);
        end
    end
end
fprintf('\n  V1 per-neuron MI done. Mean=%.2f bits/s, Max=%.2f bits/s\n', ...
    mean(mi_v1_perNeuron), max(mi_v1_perNeuron));

% --- V2 Per-Neuron Temporal MI ---
mi_v2_perNeuron = zeros(1, n_v2);
hr_v2_perNeuron_total = zeros(1, n_v2);
hn_v2_perNeuron = zeros(1, n_v2);

% Per-stimulus Hr for V2
hr_v2_perNeuron_perStim = zeros(n_v2, n_stim_driven);
hn_v2_perNeuron_perStim = zeros(n_v2, n_stim_driven);
mi_v2_perNeuron_perStim = zeros(n_v2, n_stim_driven);

for neuron = 1:n_v2
    fprintf('  V2 neuron %d/%d\r', neuron, n_v2);

    spike_long_n = zeros(n_driven, 1280);
    for trial = 1:n_driven
        spike_long_n(trial, :) = rast_v2_d{trial}(neuron, :);
    end
    spike_long_n = apply_refractory_local(spike_long_n, refractory);

    nT = length(T_list);
    H_total_vals = zeros(1, nT);
    H_noise_vals = zeros(1, nT);
    H_total_perStim = zeros(n_stim_driven, nT);

    for ti = 1:nT
        T = T_list(ti);
        H_total_vals(ti) = total_entropy_local(spike_long_n, T);

        H_noise_weighted = 0;
        for si = 1:n_stim_driven
            s = unique_stim_driven(si);
            stim_mask = (stim_driven == s);
            P_s = sum(stim_mask) / length(stim_driven);

            trial_idx = find(stim_mask);
            if length(trial_idx) < 2
                continue;
            end
            spike_repeat_ns = zeros(length(trial_idx), 1280);
            for k = 1:length(trial_idx)
                spike_repeat_ns(k, :) = rast_v2_d{trial_idx(k)}(neuron, :);
            end
            spike_repeat_ns = apply_refractory_local(spike_repeat_ns, refractory);

            H_noise_s = noise_entropy_local(spike_repeat_ns, T);
            H_noise_weighted = H_noise_weighted + P_s * H_noise_s;

            H_total_perStim(si, ti) = total_entropy_local(spike_repeat_ns, T);
        end
        H_noise_vals(ti) = H_noise_weighted;
    end

    T_sec = T_list * dt_bin;
    inv_T = 1 ./ T_sec;
    rate_total = H_total_vals ./ T_sec;
    rate_noise = H_noise_vals ./ T_sec;

    n_fit_use = min(n_fit, nT);
    fit_idx = 1:n_fit_use;

    [~, fit_total] = linear_fit_local(inv_T(fit_idx), rate_total(fit_idx));
    [~, fit_noise] = linear_fit_local(inv_T(fit_idx), rate_noise(fit_idx));

    Hr_n = fit_total.intercept;
    Hn_n = fit_noise.intercept;
    MI_n = max(0, Hr_n - Hn_n);

    mi_v2_perNeuron(neuron) = MI_n;
    hr_v2_perNeuron_total(neuron) = Hr_n;
    hn_v2_perNeuron(neuron) = Hn_n;

    % Per-stimulus extrapolation
    for si = 1:n_stim_driven
        rate_total_s = H_total_perStim(si, :) ./ T_sec;
        s = unique_stim_driven(si);
        trial_idx = find(stim_driven == s);
        if length(trial_idx) >= 2
            H_noise_this_stim = zeros(1, nT);
            for ti = 1:nT
                spike_repeat_ns = zeros(length(trial_idx), 1280);
                for k = 1:length(trial_idx)
                    spike_repeat_ns(k, :) = rast_v2_d{trial_idx(k)}(neuron, :);
                end
                spike_repeat_ns = apply_refractory_local(spike_repeat_ns, refractory);
                H_noise_this_stim(ti) = noise_entropy_local(spike_repeat_ns, T_list(ti));
            end
            rate_noise_this = H_noise_this_stim ./ T_sec;
            [~, fit_t] = linear_fit_local(inv_T(fit_idx), rate_total_s(fit_idx));
            [~, fit_n] = linear_fit_local(inv_T(fit_idx), rate_noise_this(fit_idx));
            hr_v2_perNeuron_perStim(neuron, si) = fit_t.intercept;
            hn_v2_perNeuron_perStim(neuron, si) = fit_n.intercept;
            mi_v2_perNeuron_perStim(neuron, si) = max(0, fit_t.intercept - fit_n.intercept);
        end
    end
end
fprintf('\n  V2 per-neuron MI done. Mean=%.2f bits/s, Max=%.2f bits/s\n', ...
    mean(mi_v2_perNeuron), max(mi_v2_perNeuron));

%% ===== NOISE-ENTROPY FRACTION rho = H(r|s)/H(r) (per-neuron) =====
% Sanity check for the deterministic-encoding approximation H(r|s) ~ 0
% used in the main-text Gibbs framework. Reported in Supplementary
% Section "Empirical validation of the deterministic-encoding approximation".
valid_v1 = isfinite(hn_v1_perNeuron) & isfinite(hr_v1_perNeuron_total) ...
         & hr_v1_perNeuron_total > 0;
rho_v1 = hn_v1_perNeuron(valid_v1) ./ hr_v1_perNeuron_total(valid_v1);
fprintf('\n=== Noise-entropy fraction rho = H(r|s)/H(r) (V1, n=%d) ===\n', ...
        sum(valid_v1));
fprintf('  mean=%.3f, median=%.3f, IQR=[%.3f, %.3f]\n', ...
        mean(rho_v1), median(rho_v1), quantile(rho_v1,0.25), quantile(rho_v1,0.75));
fprintf('  fraction with rho<0.3: %.2f;  with rho<0.5: %.2f\n', ...
        mean(rho_v1<0.3), mean(rho_v1<0.5));
valid_v2 = isfinite(hn_v2_perNeuron) & isfinite(hr_v2_perNeuron_total) ...
         & hr_v2_perNeuron_total > 0;
if any(valid_v2)
    rho_v2 = hn_v2_perNeuron(valid_v2) ./ hr_v2_perNeuron_total(valid_v2);
    fprintf('=== Noise-entropy fraction rho (V2, n=%d) ===\n', sum(valid_v2));
    fprintf('  mean=%.3f, median=%.3f, IQR=[%.3f, %.3f]\n', ...
            mean(rho_v2), median(rho_v2), quantile(rho_v2,0.25), quantile(rho_v2,0.75));
end

%% ===== POPULATION TEMPORAL MI (Direct Method) =====
% Pool all neurons: each row = one neuron from one trial
% spike_long_pop = [n_driven * n_neurons, 1280]

fprintf('\n=== Computing Population Temporal MI (Direct Method) ===\n');

% --- V1 Population ---
fprintf('  V1 population...\n');
spike_long_pop_v1 = zeros(n_driven * n_v1, 1280);
row = 0;
for trial = 1:n_driven
    for neuron = 1:n_v1
        row = row + 1;
        spike_long_pop_v1(row, :) = rast_v1_d{trial}(neuron, :);
    end
end
spike_long_pop_v1 = apply_refractory_local(spike_long_pop_v1, refractory);

nT = length(T_list);
H_total_pop_v1 = zeros(1, nT);
H_noise_pop_v1 = zeros(1, nT);

for ti = 1:nT
    T = T_list(ti);
    fprintf('    T=%d bins\r', T);
    H_total_pop_v1(ti) = total_entropy_local(spike_long_pop_v1, T);

    % Noise: per stimulus, stack all neurons from all trials of that stimulus
    H_noise_weighted = 0;
    for si = 1:length(unique_stim_driven)
        s = unique_stim_driven(si);
        stim_mask = (stim_driven == s);
        P_s = sum(stim_mask) / length(stim_driven);
        trial_idx = find(stim_mask);

        if length(trial_idx) < 2
            continue;
        end

        spike_repeat_pop = zeros(length(trial_idx) * n_v1, 1280);
        row = 0;
        for k = 1:length(trial_idx)
            for neuron = 1:n_v1
                row = row + 1;
                spike_repeat_pop(row, :) = rast_v1_d{trial_idx(k)}(neuron, :);
            end
        end
        spike_repeat_pop = apply_refractory_local(spike_repeat_pop, refractory);

        H_noise_s = noise_entropy_local(spike_repeat_pop, T);
        H_noise_weighted = H_noise_weighted + P_s * H_noise_s;
    end
    H_noise_pop_v1(ti) = H_noise_weighted;
end

T_sec = T_list * dt_bin;
inv_T = 1 ./ T_sec;
rate_total_v1 = H_total_pop_v1 ./ T_sec;
rate_noise_v1 = H_noise_pop_v1 ./ T_sec;

n_fit_use = min(n_fit, nT);
fit_idx = 1:n_fit_use;
[~, fit_total_v1] = linear_fit_local(inv_T(fit_idx), rate_total_v1(fit_idx));
[~, fit_noise_v1] = linear_fit_local(inv_T(fit_idx), rate_noise_v1(fit_idx));

Hr_v1 = fit_total_v1.intercept;
Hn_v1 = fit_noise_v1.intercept;
MI_v1 = max(0, Hr_v1 - Hn_v1);

fprintf('\n  V1 Population: Hr=%.2f bits/s, Hn=%.2f bits/s, MI=%.2f bits/s\n', Hr_v1, Hn_v1, MI_v1);

% --- V2 Population ---
fprintf('  V2 population...\n');
spike_long_pop_v2 = zeros(n_driven * n_v2, 1280);
row = 0;
for trial = 1:n_driven
    for neuron = 1:n_v2
        row = row + 1;
        spike_long_pop_v2(row, :) = rast_v2_d{trial}(neuron, :);
    end
end
spike_long_pop_v2 = apply_refractory_local(spike_long_pop_v2, refractory);

H_total_pop_v2 = zeros(1, nT);
H_noise_pop_v2 = zeros(1, nT);

for ti = 1:nT
    T = T_list(ti);
    fprintf('    T=%d bins\r', T);
    H_total_pop_v2(ti) = total_entropy_local(spike_long_pop_v2, T);

    H_noise_weighted = 0;
    for si = 1:length(unique_stim_driven)
        s = unique_stim_driven(si);
        stim_mask = (stim_driven == s);
        P_s = sum(stim_mask) / length(stim_driven);
        trial_idx = find(stim_mask);

        if length(trial_idx) < 2
            continue;
        end

        spike_repeat_pop = zeros(length(trial_idx) * n_v2, 1280);
        row = 0;
        for k = 1:length(trial_idx)
            for neuron = 1:n_v2
                row = row + 1;
                spike_repeat_pop(row, :) = rast_v2_d{trial_idx(k)}(neuron, :);
            end
        end
        spike_repeat_pop = apply_refractory_local(spike_repeat_pop, refractory);

        H_noise_s = noise_entropy_local(spike_repeat_pop, T);
        H_noise_weighted = H_noise_weighted + P_s * H_noise_s;
    end
    H_noise_pop_v2(ti) = H_noise_weighted;
end

rate_total_v2 = H_total_pop_v2 ./ T_sec;
rate_noise_v2 = H_noise_pop_v2 ./ T_sec;

[~, fit_total_v2] = linear_fit_local(inv_T(fit_idx), rate_total_v2(fit_idx));
[~, fit_noise_v2] = linear_fit_local(inv_T(fit_idx), rate_noise_v2(fit_idx));

Hr_v2 = fit_total_v2.intercept;
Hn_v2 = fit_noise_v2.intercept;
MI_v2 = max(0, Hr_v2 - Hn_v2);

fprintf('\n  V2 Population: Hr=%.2f bits/s, Hn=%.2f bits/s, MI=%.2f bits/s\n', Hr_v2, Hn_v2, MI_v2);

%% ===== PER-STIMULUS POPULATION MI (Direct Method) =====
% For each stimulus direction, stack all neurons × trials-of-that-stim
% → per-stimulus Hr and Hn → average across stimuli for fair comparison

fprintf('\n=== Computing Per-Stimulus Population MI (Direct Method) ===\n');

Hr_v1_perStim_pop = zeros(1, n_stim_driven);
Hn_v1_perStim_pop = zeros(1, n_stim_driven);
MI_v1_perStim_pop = zeros(1, n_stim_driven);

for si = 1:n_stim_driven
    s = unique_stim_driven(si);
    trial_idx = find(stim_driven == s);
    if length(trial_idx) < 2, continue; end
    fprintf('  V1 stim %d/%d (n_trials=%d)\n', si, n_stim_driven, length(trial_idx));

    spike_stim_pop = zeros(length(trial_idx) * n_v1, 1280);
    row = 0;
    for k = 1:length(trial_idx)
        for neuron = 1:n_v1
            row = row + 1;
            spike_stim_pop(row, :) = rast_v1_d{trial_idx(k)}(neuron, :);
        end
    end
    spike_stim_pop = apply_refractory_local(spike_stim_pop, refractory);

    H_total_s = zeros(1, nT);
    H_noise_s = zeros(1, nT);
    for ti = 1:nT
        T = T_list(ti);
        H_total_s(ti) = total_entropy_local(spike_stim_pop, T);
        H_noise_s(ti) = noise_entropy_local(spike_stim_pop, T);
    end

    rate_total_s = H_total_s ./ T_sec;
    rate_noise_s = H_noise_s ./ T_sec;
    [~, ft] = linear_fit_local(inv_T(fit_idx), rate_total_s(fit_idx));
    [~, fn] = linear_fit_local(inv_T(fit_idx), rate_noise_s(fit_idx));
    Hr_v1_perStim_pop(si) = ft.intercept;
    Hn_v1_perStim_pop(si) = fn.intercept;
    MI_v1_perStim_pop(si) = max(0, ft.intercept - fn.intercept);
end

Hr_v2_perStim_pop = zeros(1, n_stim_driven);
Hn_v2_perStim_pop = zeros(1, n_stim_driven);
MI_v2_perStim_pop = zeros(1, n_stim_driven);

for si = 1:n_stim_driven
    s = unique_stim_driven(si);
    trial_idx = find(stim_driven == s);
    if length(trial_idx) < 2, continue; end
    fprintf('  V2 stim %d/%d (n_trials=%d)\n', si, n_stim_driven, length(trial_idx));

    spike_stim_pop = zeros(length(trial_idx) * n_v2, 1280);
    row = 0;
    for k = 1:length(trial_idx)
        for neuron = 1:n_v2
            row = row + 1;
            spike_stim_pop(row, :) = rast_v2_d{trial_idx(k)}(neuron, :);
        end
    end
    spike_stim_pop = apply_refractory_local(spike_stim_pop, refractory);

    H_total_s = zeros(1, nT);
    H_noise_s = zeros(1, nT);
    for ti = 1:nT
        T = T_list(ti);
        H_total_s(ti) = total_entropy_local(spike_stim_pop, T);
        H_noise_s(ti) = noise_entropy_local(spike_stim_pop, T);
    end

    rate_total_s = H_total_s ./ T_sec;
    rate_noise_s = H_noise_s ./ T_sec;
    [~, ft] = linear_fit_local(inv_T(fit_idx), rate_total_s(fit_idx));
    [~, fn] = linear_fit_local(inv_T(fit_idx), rate_noise_s(fit_idx));
    Hr_v2_perStim_pop(si) = ft.intercept;
    Hn_v2_perStim_pop(si) = fn.intercept;
    MI_v2_perStim_pop(si) = max(0, ft.intercept - fn.intercept);
end

fprintf('  V1 Pop per-stim: Hr=%.2f±%.2f, Hn=%.2f±%.2f, MI=%.2f±%.2f bits/s\n', ...
    mean(Hr_v1_perStim_pop), std(Hr_v1_perStim_pop), ...
    mean(Hn_v1_perStim_pop), std(Hn_v1_perStim_pop), ...
    mean(MI_v1_perStim_pop), std(MI_v1_perStim_pop));
fprintf('  V2 Pop per-stim: Hr=%.2f±%.2f, Hn=%.2f±%.2f, MI=%.2f±%.2f bits/s\n', ...
    mean(Hr_v2_perStim_pop), std(Hr_v2_perStim_pop), ...
    mean(Hn_v2_perStim_pop), std(Hn_v2_perStim_pop), ...
    mean(MI_v2_perStim_pop), std(MI_v2_perStim_pop));

%% ===== CHECKPOINT: Save driven results =====
save('figure2_driven_checkpoint.mat', '-v7.3');
fprintf('Checkpoint saved: figure2_driven_checkpoint.mat\n');
% To resume from here after a crash, run:
%   load('figure2_driven_checkpoint.mat');
% then execute from this point onward.

%% ===== SPONTANEOUS: Total Entropy Rate (Direct Method) =====
% No stimulus in spontaneous, so only compute Hr (no MI).

fprintf('\n=== Computing Spontaneous Entropy Rate (Direct Method) ===\n');

% --- V1 Spontaneous ---
% Per-neuron spontaneous Hr
hr_v1_perNeuron_spon = zeros(1, n_v1);
for neuron = 1:n_v1
    spike_long_spon = zeros(n_spon, 1500);
    for trial = 1:n_spon
        spike_long_spon(trial, :) = rast_v1_s{trial}(neuron, :);
    end
    spike_long_spon = apply_refractory_local(spike_long_spon, refractory);

    H_total_spon = zeros(1, nT);
    T_list_spon = T_list(T_list <= 1500);
    nT_spon = length(T_list_spon);
    for ti = 1:nT_spon
        H_total_spon(ti) = total_entropy_local(spike_long_spon, T_list_spon(ti));
    end

    T_sec_spon = T_list_spon * dt_bin;
    inv_T_spon = 1 ./ T_sec_spon;
    rate_total_spon = H_total_spon(1:nT_spon) ./ T_sec_spon;

    n_fit_spon = min(n_fit, nT_spon);
    [~, fit_spon] = linear_fit_local(inv_T_spon(1:n_fit_spon), rate_total_spon(1:n_fit_spon));
    hr_v1_perNeuron_spon(neuron) = fit_spon.intercept;
end

% Population spontaneous Hr
spike_long_spon_pop_v1 = zeros(n_spon * n_v1, 1500);
row = 0;
for trial = 1:n_spon
    for neuron = 1:n_v1
        row = row + 1;
        spike_long_spon_pop_v1(row, :) = rast_v1_s{trial}(neuron, :);
    end
end
spike_long_spon_pop_v1 = apply_refractory_local(spike_long_spon_pop_v1, refractory);

T_list_spon = T_list(T_list <= 1500);
nT_spon = length(T_list_spon);
H_total_spon_pop_v1 = zeros(1, nT_spon);
for ti = 1:nT_spon
    H_total_spon_pop_v1(ti) = total_entropy_local(spike_long_spon_pop_v1, T_list_spon(ti));
end
T_sec_spon = T_list_spon * dt_bin;
inv_T_spon = 1 ./ T_sec_spon;
rate_total_spon_v1 = H_total_spon_pop_v1 ./ T_sec_spon;
n_fit_spon = min(n_fit, nT_spon);
[~, fit_spon_v1] = linear_fit_local(inv_T_spon(1:n_fit_spon), rate_total_spon_v1(1:n_fit_spon));
Hr_v1_spon_direct = fit_spon_v1.intercept;

fprintf('  V1 Spontaneous: Hr_pop=%.2f bits/s, Hr_perNeuron_mean=%.2f bits/s\n', ...
    Hr_v1_spon_direct, mean(hr_v1_perNeuron_spon));

% --- V2 Spontaneous ---
hr_v2_perNeuron_spon = zeros(1, n_v2);
for neuron = 1:n_v2
    spike_long_spon = zeros(n_spon, 1500);
    for trial = 1:n_spon
        spike_long_spon(trial, :) = rast_v2_s{trial}(neuron, :);
    end
    spike_long_spon = apply_refractory_local(spike_long_spon, refractory);

    T_list_spon = T_list(T_list <= 1500);
    nT_spon = length(T_list_spon);
    H_total_spon = zeros(1, nT_spon);
    for ti = 1:nT_spon
        H_total_spon(ti) = total_entropy_local(spike_long_spon, T_list_spon(ti));
    end

    T_sec_spon = T_list_spon * dt_bin;
    inv_T_spon = 1 ./ T_sec_spon;
    rate_total_spon = H_total_spon ./ T_sec_spon;

    n_fit_spon = min(n_fit, nT_spon);
    [~, fit_spon] = linear_fit_local(inv_T_spon(1:n_fit_spon), rate_total_spon(1:n_fit_spon));
    hr_v2_perNeuron_spon(neuron) = fit_spon.intercept;
end

spike_long_spon_pop_v2 = zeros(n_spon * n_v2, 1500);
row = 0;
for trial = 1:n_spon
    for neuron = 1:n_v2
        row = row + 1;
        spike_long_spon_pop_v2(row, :) = rast_v2_s{trial}(neuron, :);
    end
end
spike_long_spon_pop_v2 = apply_refractory_local(spike_long_spon_pop_v2, refractory);

T_list_spon = T_list(T_list <= 1500);
nT_spon = length(T_list_spon);
H_total_spon_pop_v2 = zeros(1, nT_spon);
for ti = 1:nT_spon
    H_total_spon_pop_v2(ti) = total_entropy_local(spike_long_spon_pop_v2, T_list_spon(ti));
end
T_sec_spon = T_list_spon * dt_bin;
inv_T_spon = 1 ./ T_sec_spon;
rate_total_spon_v2 = H_total_spon_pop_v2 ./ T_sec_spon;
n_fit_spon = min(n_fit, nT_spon);
[~, fit_spon_v2] = linear_fit_local(inv_T_spon(1:n_fit_spon), rate_total_spon_v2(1:n_fit_spon));
Hr_v2_spon_direct = fit_spon_v2.intercept;

fprintf('  V2 Spontaneous: Hr_pop=%.2f bits/s, Hr_perNeuron_mean=%.2f bits/s\n', ...
    Hr_v2_spon_direct, mean(hr_v2_perNeuron_spon));

%% -----------------------------------------------------------------------
%% SECTION 2b: Lifetime-level CV, sparseness, and entropy
%% -----------------------------------------------------------------------

% V1 driven (159 neurons)
for i = 1:size(mrate_v1, 1)
    ss1_l = [ss1_l, 1 - mean(mrate_v1(i,:))^2 ./ mean(mrate_v1(i,:).^2)];
    cv1_l = [cv1_l, std(mrate_v1(i,:)) ./ mean(mrate_v1(i,:))];
    [n, ~] = histcounts(mrate_v1(i,:), 100);
    h = n / sum(n);
    entr = 0;
    for j = h
        if j > 0
            entr = entr - j * log(j);
        end
    end
    entr1_l = [entr1_l, entr];
end

% V2 driven (25 neurons)
for i = 1:size(mrate_v2, 1)
    ss2_l = [ss2_l, 1 - mean(mrate_v2(i,:))^2 ./ mean(mrate_v2(i,:).^2)];
    cv2_l = [cv2_l, std(mrate_v2(i,:)) ./ mean(mrate_v2(i,:))];
    [n, ~] = histcounts(mrate_v2(i,:), 100);
    h = n / sum(n);
    entr = 0;
    for j = h
        if j > 0
            entr = entr - j * log(j);
        end
    end
    entr2_l = [entr2_l, entr];
end

% V1 spontaneous (159 neurons)
for i = 1:size(mrate_v1_s, 1)
    ss1_l_s = [ss1_l_s, 1 - mean(mrate_v1_s(i,:))^2 ./ mean(mrate_v1_s(i,:).^2)];
    cv1_l_s = [cv1_l_s, std(mrate_v1_s(i,:)) ./ mean(mrate_v1_s(i,:))];
    [n, ~] = histcounts(mrate_v1_s(i,:), 100);
    h = n / sum(n);
    entr = 0;
    for j = h
        if j > 0
            entr = entr - j * log(j);
        end
    end
    entr1_ls = [entr1_ls, entr];
end

% V2 spontaneous (25 neurons)
for i = 1:size(mrate_v2_s, 1)
    ss2_l_s = [ss2_l_s, 1 - mean(mrate_v2_s(i,:))^2 ./ mean(mrate_v2_s(i,:).^2)];
    cv2_l_s = [cv2_l_s, std(mrate_v2_s(i,:)) ./ mean(mrate_v2_s(i,:))];
    [n, ~] = histcounts(mrate_v2_s(i,:), 100);
    h = n / sum(n);
    entr = 0;
    for j = h
        if j > 0
            entr = entr - j * log(j);
        end
    end
    entr2_ls = [entr2_ls, entr];
end

%% -----------------------------------------------------------------------
%% SECTION 3: Summary and ICE Computation
%% -----------------------------------------------------------------------

fprintf('\n=== MI Summary (Direct Method) ===\n');
fprintf('  Spatial-Only MI (histogram, bits):\n');
fprintf('    V1: MI=%.4f, H(r)=%.4f, H(r|s)=%.4f\n', MI_spatial_v1, Hr_spatial_v1, Hrs_spatial_v1);
fprintf('    V2: MI=%.4f, H(r)=%.4f, H(r|s)=%.4f\n', MI_spatial_v2, Hr_spatial_v2, Hrs_spatial_v2);
fprintf('  Per-Neuron Temporal MI (direct, bits/s):\n');
fprintf('    V1: mean=%.2f, max=%.2f\n', mean(mi_v1_perNeuron), max(mi_v1_perNeuron));
fprintf('    V2: mean=%.2f, max=%.2f\n', mean(mi_v2_perNeuron), max(mi_v2_perNeuron));
fprintf('  Population Temporal MI (direct, bits/s):\n');
fprintf('    V1: MI=%.2f, Hr=%.2f, Hn=%.2f\n', MI_v1, Hr_v1, Hn_v1);
fprintf('    V2: MI=%.2f, Hr=%.2f, Hn=%.2f\n', MI_v2, Hr_v2, Hn_v2);
fprintf('  Spontaneous Hr (direct, bits/s):\n');
fprintf('    V1: Hr_pop=%.2f, Hr_perNeuron_mean=%.2f\n', Hr_v1_spon_direct, mean(hr_v1_perNeuron_spon));
fprintf('    V2: Hr_pop=%.2f, Hr_perNeuron_mean=%.2f\n', Hr_v2_spon_direct, mean(hr_v2_perNeuron_spon));

% ICE computation
% Direct method outputs are in bits/s; normalization is by the calibrated cost proxy.
% Spatial-only MI is in bits (not bits/s), so divide by dur_driven for ICE

mean_cost_v1_driven = mean(cost1_p);
mean_cost_v2_driven = mean(cost2_p);
mean_cost_v1_spon = mean(cost1_ps);
mean_cost_v2_spon = mean(cost2_ps);

% Spatial-only ICE (bits -> bits/s by dividing by dur_driven, then / cost)
ICE_spatial_v1 = MI_spatial_v1 / (dur_driven * mean_cost_v1_driven);
ICE_spatial_v2 = MI_spatial_v2 / (dur_driven * mean_cost_v2_driven);

% Per-neuron temporal information per calibrated cost proxy
ICE_perNeuron_v1 = mean(mi_v1_perNeuron) / mean_cost_v1_driven;
ICE_perNeuron_v2 = mean(mi_v2_perNeuron) / mean_cost_v2_driven;

% Population temporal information per calibrated cost proxy
ICE_pop_v1 = MI_v1 / mean_cost_v1_driven;
ICE_pop_v2 = MI_v2 / mean_cost_v2_driven;

fprintf('\n  information / calibrated cost proxy:\n');
fprintf('    V1: Spatial=%.4f, PerNeuron=%.4f, Population=%.4f\n', ICE_spatial_v1, ICE_perNeuron_v1, ICE_pop_v1);
fprintf('    V2: Spatial=%.4f, PerNeuron=%.4f, Population=%.4f\n', ICE_spatial_v2, ICE_perNeuron_v2, ICE_pop_v2);

%% -----------------------------------------------------------------------
%% SECTION 4: Plotting
%% -----------------------------------------------------------------------

%% Figure 1: Three-Level MI Summary (bar chart)
figure('Name', 'Three-Level MI Summary', 'NumberTitle', 'off', 'Position', [100 100 1400 600]);

level_names = {'Spatial-Only\n(bits)', 'Per-Neuron\n(bits/s)', 'Population\n(bits/s)'};

% V1
subplot(1, 2, 1)
mi_levels_v1 = [MI_spatial_v1, mean(mi_v1_perNeuron), MI_v1];
bars_v1 = bar(mi_levels_v1, 'FaceColor', [0.2 0.6 0.9], 'EdgeColor', 'k', 'LineWidth', 1.5);
set(gca, 'XTickLabel', level_names, 'FontSize', 11);
ylabel('MI', 'FontSize', 11);
title('V1: MI across Spatial-Temporal Levels', 'FontSize', 12);
grid on
ylim_max_v1 = max(mi_levels_v1(~isnan(mi_levels_v1) & ~isinf(mi_levels_v1)));
if isempty(ylim_max_v1) || ylim_max_v1 <= 0, ylim_max_v1 = 1; end
ylim([0, ylim_max_v1*1.3]);
for i = 1:length(mi_levels_v1)
    if ~isnan(mi_levels_v1(i)) && ~isinf(mi_levels_v1(i))
        text(i, mi_levels_v1(i) + ylim_max_v1*0.03, sprintf('%.3f', mi_levels_v1(i)), ...
            'HorizontalAlignment', 'center', 'FontSize', 10);
    end
end

% V2
subplot(1, 2, 2)
mi_levels_v2 = [MI_spatial_v2, mean(mi_v2_perNeuron), MI_v2];
bars_v2 = bar(mi_levels_v2, 'FaceColor', [0.9 0.4 0.2], 'EdgeColor', 'k', 'LineWidth', 1.5);
set(gca, 'XTickLabel', level_names, 'FontSize', 11);
ylabel('MI', 'FontSize', 11);
title('V2: MI across Spatial-Temporal Levels', 'FontSize', 12);
grid on
ylim_max_v2 = max(mi_levels_v2(~isnan(mi_levels_v2) & ~isinf(mi_levels_v2)));
if isempty(ylim_max_v2) || ylim_max_v2 <= 0, ylim_max_v2 = 1; end
ylim([0, ylim_max_v2*1.3]);
for i = 1:length(mi_levels_v2)
    if ~isnan(mi_levels_v2(i)) && ~isinf(mi_levels_v2(i))
        text(i, mi_levels_v2(i) + ylim_max_v2*0.03, sprintf('%.3f', mi_levels_v2(i)), ...
            'HorizontalAlignment', 'center', 'FontSize', 10);
    end
end

sgtitle('Spatial-Temporal Decomposition: MI at Each Level (Direct Method)', 'FontSize', 13);

%% Figure 2: Per-Neuron MI Distribution
figure('Name', 'Per-Neuron MI Distribution (Direct Method)', 'NumberTitle', 'off', 'Position', [100 100 1200 500]);

subplot(1, 2, 1)
histogram(mi_v1_perNeuron, 20, 'FaceColor', [0.2 0.6 0.9], 'EdgeColor', 'k');
xlabel('MI [bits/s]');
ylabel('# Neurons');
title(sprintf('V1: Per-Neuron MI Distribution (n=%d)', n_v1));
grid on
hold on
xline(mean(mi_v1_perNeuron), 'r--', 'LineWidth', 2, 'DisplayName', sprintf('Mean=%.2f', mean(mi_v1_perNeuron)));
legend;

subplot(1, 2, 2)
histogram(mi_v2_perNeuron, 15, 'FaceColor', [0.9 0.4 0.2], 'EdgeColor', 'k');
xlabel('MI [bits/s]');
ylabel('# Neurons');
title(sprintf('V2: Per-Neuron MI Distribution (n=%d)', n_v2));
grid on
hold on
xline(mean(mi_v2_perNeuron), 'r--', 'LineWidth', 2, 'DisplayName', sprintf('Mean=%.2f', mean(mi_v2_perNeuron)));
legend;

sgtitle('Per-Neuron Temporal MI (Direct Method, Strong et al. 1998)');

%% Figure 3: ICE Comparison
figure('Name', 'ICE Comparison', 'NumberTitle', 'off', 'Position', [100 100 900 500]);

ice_data = [ICE_spatial_v1, ICE_perNeuron_v1, ICE_pop_v1; ...
            ICE_spatial_v2, ICE_perNeuron_v2, ICE_pop_v2];
b = bar(ice_data);
b(1).FaceColor = [0.5 0.5 0.8]; b(1).DisplayName = 'Spatial-Only';
b(2).FaceColor = [0.8 0.4 0.2]; b(2).DisplayName = 'Per-Neuron (mean)';
b(3).FaceColor = [0.2 0.7 0.3]; b(3).DisplayName = 'Population';
set(gca, 'XTickLabel', {'V1', 'V2'});
ylabel('information / calibrated cost proxy');
title('Information-Cost Efficiency at Each Level');
legend('Location', 'best');
grid on

%% Figure 4: H(r) Decomposition (stacked bars: Hn + MI = Hr for driven)
figure('Name', 'Entropy Decomposition (Direct Method)', 'NumberTitle', 'off', 'Position', [100 100 1200 600]);

% V1
subplot(1, 2, 1)
hold on
% Per-neuron level (stacked)
data_v1 = [mean(hn_v1_perNeuron), mean(mi_v1_perNeuron); ...
           Hn_v1, MI_v1];
b = bar(data_v1, 'stacked');
b(1).FaceColor = [0.3 0.8 0.3]; b(1).DisplayName = 'Hn (noise)';
b(2).FaceColor = [0.8 0.3 0.3]; b(2).DisplayName = 'MI (info)';
set(gca, 'XTickLabel', {'Per-Neuron\n(mean)', 'Population'});
ylabel('Entropy Rate [bits/s]');
title('V1: Hr = Hn + MI (Direct Method)');
legend('Location', 'best');
grid on
% Add value labels
for idx = 1:2
    text(idx, data_v1(idx,1)/2, sprintf('%.1f', data_v1(idx,1)), ...
        'HorizontalAlignment', 'center', 'FontSize', 9, 'Color', 'w');
    text(idx, data_v1(idx,1) + data_v1(idx,2)/2, sprintf('%.1f', data_v1(idx,2)), ...
        'HorizontalAlignment', 'center', 'FontSize', 9, 'Color', 'w');
end

% V2
subplot(1, 2, 2)
hold on
data_v2 = [mean(hn_v2_perNeuron), mean(mi_v2_perNeuron); ...
           Hn_v2, MI_v2];
b = bar(data_v2, 'stacked');
b(1).FaceColor = [0.3 0.8 0.3]; b(1).DisplayName = 'Hn (noise)';
b(2).FaceColor = [0.8 0.3 0.3]; b(2).DisplayName = 'MI (info)';
set(gca, 'XTickLabel', {'Per-Neuron\n(mean)', 'Population'});
ylabel('Entropy Rate [bits/s]');
title('V2: Hr = Hn + MI (Direct Method)');
legend('Location', 'best');
grid on
for idx = 1:2
    text(idx, data_v2(idx,1)/2, sprintf('%.1f', data_v2(idx,1)), ...
        'HorizontalAlignment', 'center', 'FontSize', 9, 'Color', 'w');
    text(idx, data_v2(idx,1) + data_v2(idx,2)/2, sprintf('%.1f', data_v2(idx,2)), ...
        'HorizontalAlignment', 'center', 'FontSize', 9, 'Color', 'w');
end

sgtitle('Driven Entropy Decomposition: Noise + MI = Total (Direct Method)');

%% Figure 5: CV-Sparseness-Entropy scatter plots (same as original)
figure('Name', 'CV-Sparseness-Rate Scatter', 'NumberTitle', 'off');

subplot(1, 3, 1)
plot(mean(mrate_v1_s), cv_v1_s, 'b*'); hold on;
plot(mean(mrate_v2_s), cv_v2_s, 'b+'); hold on;
plot(mean(mrate_v1),   cv_v1,   'r*'); hold on;
plot(mean(mrate_v2),   cv_v2,   'r+'); hold on;
ylim([0.5 2]);  xlim([0 22])
xlabel('Mean Rate [Hz]'); ylabel('CV');
title('Population CV vs Rate');

subplot(1, 3, 2)
plot(mean(mrate_v1_s), ss_v1_s, 'b*'); hold on;
plot(mean(mrate_v2_s), ss_v2_s, 'b+'); hold on;
plot(mean(mrate_v1),   ss_v1,   'r*'); hold on;
plot(mean(mrate_v2),   ss_v2,   'r+'); hold on;
ylim([0.2 1]);  xlim([0 22])
xlabel('Mean Rate [Hz]'); ylabel('Sparseness');
title('Population Sparseness vs Rate');

subplot(1, 3, 3)
x = 0:0.01:3;
y = x.^2 ./ (x.^2 + 1);
plot(cv_v1_s, ss_v1_s, 'b*'); hold on;
plot(cv_v2_s, ss_v2_s, 'b+'); hold on;
plot(cv_v1,   ss_v1,   'r*'); hold on;
plot(cv_v2,   ss_v2,   'r+'); hold on;
plot(x, y);  xlim([0 3])
xlabel('CV'); ylabel('Sparseness');
title('CV vs Sparseness');

%% Figure 6: Information vs Cost scatter (population-level, V1)
%
% Three traces per panel for a complete comparison:
%   (1) Spontaneous H(r)    — blue  : coding readiness, no stimulus defined
%   (2) Evoked    H(r)      — orange faded : entropy of evoked response
%                                            (upper bound on attainable I/C
%                                            under deterministic encoding)
%   (3) Evoked    I(s; r)   — orange solid: deployed stimulus information
%                                           = H(r) - H(r|s) per stimulus,
%                                           pooled across stimuli for the
%                                           population summary point.
%
% The H–I gap visualises the noise-entropy fraction rho ~ 0.85 reported
% in the main text and in Supp Sect 3.2.
x = 0:0.1:30;
figure('Name', 'Information-Cost Tradeoff', 'NumberTitle', 'off');

col_spon = [0.30 0.60 0.90];   % blue   — spontaneous
col_evk  = [0.85 0.33 0.10];   % orange — evoked

% --- per-stimulus evoked summaries (one point per stim) ---------------------
% MI_v1_perStim_pop / Hr_v1_perStim_pop already exist (population-level Hr / MI
% under the deterministic-encoding approximation); each entry = one stimulus.
% Per-stimulus cost: average cost over trials of that stimulus.
% (`stim_driven` and `cost1_p` are appended in lockstep in the per-trial loop,
%  so they share length and indexing.)
cost_v1_perStim = nan(1, n_stim_driven);
for si = 1:n_stim_driven
    s_id = unique_stim_driven(si);
    mask = (stim_driven == s_id);
    if any(mask)
        cost_v1_perStim(si) = mean(cost1_p(mask));
    end
end

% ---------- Subplot 1: Entropy / Information vs Cost ----------
subplot(1, 2, 1)
% (1) Spontaneous H — blue
scatter(cost1_ps, entr1_ps, 30, col_spon, 'o', 'filled', 'MarkerFaceAlpha', 1.0); hold on;
% (2) Evoked H — orange faded
scatter(cost1_p,  entr1_p,  30, col_evk,  'o', 'filled', 'MarkerFaceAlpha', 0.3);
% (3) Evoked MI — orange solid (per-stimulus summary)
scatter(cost_v1_perStim, MI_v1_perStim_pop, 80, col_evk, 'o', 'filled', ...
        'MarkerFaceAlpha', 1.0, 'MarkerEdgeColor', 'k', 'LineWidth', 0.8);
y = 1 + log(x);
plot(x, y, 'k--', 'LineWidth', 1.5)
xlabel('calibrated cost proxy'); ylabel('Information [nats]');
title('V1: Information vs Cost (H, MI)');
legend('Spon H(r)', 'Evoked H(r)', 'Evoked MI(s;r)', 'Theory  1 + ln C', ...
       'Location', 'best');
xlim([0 30]); grid on

% ---------- Subplot 2: ICE = Information / Cost vs Cost ----------
subplot(1, 2, 2)
% (1) Spontaneous Hr/C — blue
scatter(cost1_ps, entr1_ps ./ (dur_spon   * cost1_ps), 30, col_spon, 'o', ...
        'filled', 'MarkerFaceAlpha', 1.0); hold on;
% (2) Evoked Hr/C — orange faded
scatter(cost1_p,  entr1_p  ./ (dur_driven * cost1_p),  30, col_evk,  'o', ...
        'filled', 'MarkerFaceAlpha', 0.3);
% (3) Evoked MI/C — orange solid (per-stimulus summary)
scatter(cost_v1_perStim, MI_v1_perStim_pop ./ (dur_driven * cost_v1_perStim), ...
        80, col_evk, 'o', 'filled', ...
        'MarkerFaceAlpha', 1.0, 'MarkerEdgeColor', 'k', 'LineWidth', 0.8);
y = 1 ./ x;
plot(x, y, 'k--', 'LineWidth', 1.5)
xlabel('calibrated cost proxy'); ylabel('information / calibrated cost proxy');
title('V1: ICE vs Cost (H/C, MI/C)');
legend('Spon H(r)/C', 'Evoked H(r)/C', 'Evoked MI(s;r)/C', 'Theory  1/C', ...
       'Location', 'best');
ylim([0 1]); grid on

%% Figure 7: 1/T Extrapolation Diagnostic (V1 Population)
figure('Name', '1/T Extrapolation (V1 Population)', 'NumberTitle', 'off', 'Position', [100 100 900 400]);

subplot(1, 2, 1)
plot(inv_T, rate_total_v1, 'bo-', 'MarkerFaceColor', 'b', 'DisplayName', 'H_{total}/T'); hold on;
plot(inv_T, rate_noise_v1, 'ro-', 'MarkerFaceColor', 'r', 'DisplayName', 'H_{noise}/T');
x_fit_line = [0, max(inv_T)];
plot(x_fit_line, fit_total_v1.slope * x_fit_line + fit_total_v1.intercept, 'b--', ...
    'DisplayName', sprintf('Hr = %.1f bits/s', Hr_v1));
plot(x_fit_line, fit_noise_v1.slope * x_fit_line + fit_noise_v1.intercept, 'r--', ...
    'DisplayName', sprintf('Hn = %.1f bits/s', Hn_v1));
plot(0, Hr_v1, 'bs', 'MarkerSize', 10, 'MarkerFaceColor', 'b', 'HandleVisibility', 'off');
plot(0, Hn_v1, 'rs', 'MarkerSize', 10, 'MarkerFaceColor', 'r', 'HandleVisibility', 'off');
xlabel('1/T [Hz]'); ylabel('Entropy Rate [bits/s]');
title(sprintf('V1 Population: MI = %.1f bits/s', MI_v1));
legend('Location', 'best'); grid on

subplot(1, 2, 2)
T_sec_plot = T_list * dt_bin;
inv_T_plot = 1 ./ T_sec_plot;
plot(inv_T_plot, H_total_pop_v2 ./ T_sec_plot, 'bo-', 'MarkerFaceColor', 'b', 'DisplayName', 'H_{total}/T'); hold on;
plot(inv_T_plot, H_noise_pop_v2 ./ T_sec_plot, 'ro-', 'MarkerFaceColor', 'r', 'DisplayName', 'H_{noise}/T');
x_fit_line = [0, max(inv_T_plot)];
plot(x_fit_line, fit_total_v2.slope * x_fit_line + fit_total_v2.intercept, 'b--', ...
    'DisplayName', sprintf('Hr = %.1f bits/s', Hr_v2));
plot(x_fit_line, fit_noise_v2.slope * x_fit_line + fit_noise_v2.intercept, 'r--', ...
    'DisplayName', sprintf('Hn = %.1f bits/s', Hn_v2));
plot(0, Hr_v2, 'bs', 'MarkerSize', 10, 'MarkerFaceColor', 'b', 'HandleVisibility', 'off');
plot(0, Hn_v2, 'rs', 'MarkerSize', 10, 'MarkerFaceColor', 'r', 'HandleVisibility', 'off');
xlabel('1/T [Hz]'); ylabel('Entropy Rate [bits/s]');
title(sprintf('V2 Population: MI = %.1f bits/s', MI_v2));
legend('Location', 'best'); grid on

sgtitle('1/T Extrapolation Diagnostic (Direct Method)');

%% Figure 8: Per-Neuron Hr comparison (Spon vs Driven)
figure('Name', 'Per-Neuron Entropy Analysis (Direct Method)', 'NumberTitle', 'off', 'Position', [100 100 1400 600]);

% V1: Hr spon vs driven
subplot(1, 3, 1)
scatter(hr_v1_perNeuron_spon, hr_v1_perNeuron_total, 50, 'o', ...
    'MarkerFaceColor', [0.2 0.6 0.9], 'MarkerEdgeColor', 'k', 'LineWidth', 1);
hold on
lim_max = max(max(hr_v1_perNeuron_spon), max(hr_v1_perNeuron_total));
plot([0 lim_max], [0 lim_max], 'r--', 'LineWidth', 1);
xlabel('Hr | Spontaneous [bits/s]');
ylabel('Hr | Driven [bits/s]');
title('V1: Total Entropy Rate (Spon vs Driven)');
axis square; grid on

% V2: Hr spon vs driven
subplot(1, 3, 2)
scatter(hr_v2_perNeuron_spon, hr_v2_perNeuron_total, 50, 'o', ...
    'MarkerFaceColor', [0.9 0.4 0.2], 'MarkerEdgeColor', 'k', 'LineWidth', 1);
hold on
lim_max = max(max(hr_v2_perNeuron_spon), max(hr_v2_perNeuron_total));
plot([0 lim_max], [0 lim_max], 'r--', 'LineWidth', 1);
xlabel('Hr | Spontaneous [bits/s]');
ylabel('Hr | Driven [bits/s]');
title('V2: Total Entropy Rate (Spon vs Driven)');
axis square; grid on

% Hr vs Hn (driven) — V1
subplot(1, 3, 3)
scatter(hr_v1_perNeuron_total, hn_v1_perNeuron, 50, 'o', ...
    'MarkerFaceColor', [0.8 0.4 0.2], 'MarkerEdgeColor', 'k', 'LineWidth', 1);
hold on
lim_max = max(max(hr_v1_perNeuron_total), max(hn_v1_perNeuron));
plot([0 lim_max], [0 lim_max], 'r--', 'LineWidth', 1);
xlabel('Hr (Total) [bits/s]');
ylabel('Hn (Noise) [bits/s]');
title('V1: Total vs Noise Entropy Rate');
axis square; grid on

sgtitle('Per-Neuron Entropy Analysis (Direct Method)');

%% Figure 9: Bar plots of mean CV and sparseness (same as original)
figure('Name', 'CV and Sparseness Summary', 'NumberTitle', 'off');

subplot(1, 2, 1)
model_series = [mean(cv_v1); mean(cv_v1_s)];
model_error  = [std(cv_v1).^2; std(cv_v1_s).^2];
b = bar(model_series, 'grouped');
hold on
[ngroups, nbars] = size(model_series);
x_bar = nan(nbars, ngroups);
for i = 1:nbars
    x_bar(i,:) = b(i).XEndPoints;
end
errorbar(x_bar', model_series, model_error, 'k', 'linestyle', 'none');
ylabel('CV')
xticklabels({'evoked', 'spontaneous'})
ylim([1 1.2])

subplot(1, 2, 2)
model_series = [mean(ss_v1); mean(ss_v1_s)];
model_error  = [std(ss_v1).^2; std(ss_v1_s).^2];
b = bar(model_series, 'grouped');
hold on
[ngroups, nbars] = size(model_series);
x_bar = nan(nbars, ngroups);
for i = 1:nbars
    x_bar(i,:) = b(i).XEndPoints;
end
errorbar(x_bar', model_series, model_error, 'k', 'linestyle', 'none');
ylabel('sparseness')
xticklabels({'evoked', 'spontaneous'})
ylim([0.5 0.54])

%% Statistical test
ranksum(cv_v1, cv_v1_s)

%% ========================================================================
%% Summary Figure: Spon Hr vs Per-Stimulus Driven Hr (= Hrs + MI)
%% ========================================================================
% Per-Neuron: for each neuron, average Hr/Hrs/MI across stimuli, then across neurons
%   → Driven Hr here is the single-stimulus Hr, NOT pooled across all stimuli
%   → This makes it directly comparable to Spon Hr (both = one-condition entropy)
%
% Population panels still use the pooled values (Hr_v1, Hn_v1, MI_v1)
%   because per-stimulus population MI requires too few rows for reliable
%   direct-method estimation. Consider them as upper-bound references.

figure;
set(gcf, 'Color', 'w', 'Position', [100 100 900 700]);
sgtitle('Spon H(r) vs Per-Stimulus Driven H(r) = H(r|s) + MI', ...
    'FontSize', 14, 'FontWeight', 'bold');

% Colors
col_hrs  = [0.55 0.55 0.55];   % grey for H(r|s)
col_mi   = [0.85 0.33 0.10];   % orange for MI
col_spon = [0.30 0.60 0.90];   % blue for spontaneous Hr

% ---- Panel 1: V1 Per-Neuron (per-stimulus driven) ----
subplot(2,2,1);
% Per-neuron mean across stimuli: [1 x n_v1]
hr_v1_driv_perN  = mean(hr_v1_perNeuron_perStim, 2)';   % mean across stim
hn_v1_driv_perN  = mean(hn_v1_perNeuron_perStim, 2)';
mi_v1_driv_perN  = mean(mi_v1_perNeuron_perStim, 2)';

spon_hr  = mean(hr_v1_perNeuron_spon);
driv_hrs = mean(hn_v1_driv_perN);
driv_mi  = mean(mi_v1_driv_perN);

data = [spon_hr 0; driv_hrs driv_mi];
b = bar(1:2, data, 0.5, 'stacked'); hold on;
b(1).FaceColor = 'flat';
b(1).CData = [col_spon; col_hrs];
b(2).FaceColor = col_mi;

n_v1 = length(hr_v1_perNeuron_spon);
spon_sem = std(hr_v1_perNeuron_spon) / sqrt(n_v1);
driv_hr_perN = hn_v1_driv_perN + mi_v1_driv_perN;  % per-neuron total
driv_sem = std(driv_hr_perN) / sqrt(n_v1);
errorbar(1, spon_hr, spon_sem, 'k', 'LineStyle', 'none', 'LineWidth', 1.5, 'CapSize', 8);
errorbar(2, driv_hrs+driv_mi, driv_sem, 'k', 'LineStyle', 'none', 'LineWidth', 1.5, 'CapSize', 8);

set(gca, 'XTick', [1 2], 'XTickLabel', {'Spon', 'Driven'});
ylabel('Entropy rate [bits/s]');
title(sprintf('V1 Per-Neuron (n=%d)', n_v1));
legend(b, {'H(r) / H(r|s)', 'MI'}, 'Location', 'best');
box off;

% ---- Panel 2: V1 Population (per-stimulus mean) ----
subplot(2,2,2);
driv_hrs_pop = mean(Hn_v1_perStim_pop);
driv_mi_pop  = mean(MI_v1_perStim_pop);

data = [Hr_v1_spon_direct 0; driv_hrs_pop driv_mi_pop];
b = bar(1:2, data, 0.5, 'stacked'); hold on;
b(1).FaceColor = 'flat';
b(1).CData = [col_spon; col_hrs];
b(2).FaceColor = col_mi;

% SEM across stimuli for driven Hr
driv_hr_pop_perStim = Hn_v1_perStim_pop + MI_v1_perStim_pop;
driv_sem_pop = std(driv_hr_pop_perStim) / sqrt(n_stim_driven);
errorbar(2, driv_hrs_pop+driv_mi_pop, driv_sem_pop, 'k', 'LineStyle', 'none', 'LineWidth', 1.5, 'CapSize', 8);

set(gca, 'XTick', [1 2], 'XTickLabel', {'Spon', 'Driven'});
ylabel('Entropy rate [bits/s]');
title(sprintf('V1 Population (n_{stim}=%d)', n_stim_driven));
legend(b, {'H(r) / H(r|s)', 'MI'}, 'Location', 'best');
box off;

% ---- Panel 3: V2 Per-Neuron (per-stimulus driven) ----
subplot(2,2,3);
hr_v2_driv_perN  = mean(hr_v2_perNeuron_perStim, 2)';
hn_v2_driv_perN  = mean(hn_v2_perNeuron_perStim, 2)';
mi_v2_driv_perN  = mean(mi_v2_perNeuron_perStim, 2)';

spon_hr  = mean(hr_v2_perNeuron_spon);
driv_hrs = mean(hn_v2_driv_perN);
driv_mi  = mean(mi_v2_driv_perN);

data = [spon_hr 0; driv_hrs driv_mi];
b = bar(1:2, data, 0.5, 'stacked'); hold on;
b(1).FaceColor = 'flat';
b(1).CData = [col_spon; col_hrs];
b(2).FaceColor = col_mi;

n_v2 = length(hr_v2_perNeuron_spon);
spon_sem = std(hr_v2_perNeuron_spon) / sqrt(n_v2);
driv_hr_perN = hn_v2_driv_perN + mi_v2_driv_perN;
driv_sem = std(driv_hr_perN) / sqrt(n_v2);
errorbar(1, spon_hr, spon_sem, 'k', 'LineStyle', 'none', 'LineWidth', 1.5, 'CapSize', 8);
errorbar(2, driv_hrs+driv_mi, driv_sem, 'k', 'LineStyle', 'none', 'LineWidth', 1.5, 'CapSize', 8);

set(gca, 'XTick', [1 2], 'XTickLabel', {'Spon', 'Driven'});
ylabel('Entropy rate [bits/s]');
title(sprintf('V2 Per-Neuron (n=%d)', n_v2));
legend(b, {'H(r) / H(r|s)', 'MI'}, 'Location', 'best');
box off;

% ---- Panel 4: V2 Population (per-stimulus mean) ----
subplot(2,2,4);
driv_hrs_pop = mean(Hn_v2_perStim_pop);
driv_mi_pop  = mean(MI_v2_perStim_pop);

data = [Hr_v2_spon_direct 0; driv_hrs_pop driv_mi_pop];
b = bar(1:2, data, 0.5, 'stacked'); hold on;
b(1).FaceColor = 'flat';
b(1).CData = [col_spon; col_hrs];
b(2).FaceColor = col_mi;

% SEM across stimuli for driven Hr
driv_hr_pop_perStim = Hn_v2_perStim_pop + MI_v2_perStim_pop;
driv_sem_pop = std(driv_hr_pop_perStim) / sqrt(n_stim_driven);
errorbar(2, driv_hrs_pop+driv_mi_pop, driv_sem_pop, 'k', 'LineStyle', 'none', 'LineWidth', 1.5, 'CapSize', 8);

set(gca, 'XTick', [1 2], 'XTickLabel', {'Spon', 'Driven'});
ylabel('Entropy rate [bits/s]');
title(sprintf('V2 Population (n_{stim}=%d)', n_stim_driven));
legend(b, {'H(r) / H(r|s)', 'MI'}, 'Location', 'best');
box off;

%% ========================================================================
%% Figure: ICE comparison & Information-Cost scatter
%% ========================================================================
% Panel A: Bar chart — Spon ICE vs Driven ICE (per-neuron, V1+V2)
% Panel B: Scatter — Spon H(r) vs Cost (blue), Driven H(r) vs Cost (blue 30%),
%          MI vs Cost (orange)
%
% Per-neuron cost: C(r_i) = C1 + C2 * mean_rate_i

% ── Per-neuron cost (driven & spon) ──
% mrate_v1: [159 x n_driven_trials], mrate_v1_s: [159 x n_spon_trials]
rate_perN_v1_driven = mean(mrate_v1, 2)';   % [1 x 159]
rate_perN_v1_spon   = mean(mrate_v1_s, 2)'; % [1 x 159]
rate_perN_v2_driven = mean(mrate_v2, 2)';   % [1 x 25]
rate_perN_v2_spon   = mean(mrate_v2_s, 2)'; % [1 x 25]

cost_perN_v1_driven = C1 + C2 * rate_perN_v1_driven;  % [1 x 159]
cost_perN_v1_spon   = C1 + C2 * rate_perN_v1_spon;
cost_perN_v2_driven = C1 + C2 * rate_perN_v2_driven;  % [1 x 25]
cost_perN_v2_spon   = C1 + C2 * rate_perN_v2_spon;

% ── Per-neuron driven quantities (mean across stim) ──
hr_v1_driv_perN  = mean(hr_v1_perNeuron_perStim, 2)';   % driven Hr [1x159]
mi_v1_driv_perN  = mean(mi_v1_perNeuron_perStim, 2)';   % driven MI [1x159]
hr_v2_driv_perN  = mean(hr_v2_perNeuron_perStim, 2)';
mi_v2_driv_perN  = mean(mi_v2_perNeuron_perStim, 2)';

% ── Per-neuron ICE ──
% Spon: Hr_spon / cost_spon (entropy-to-cost, upper bound)
ICE_perN_v1_spon   = hr_v1_perNeuron_spon ./ cost_perN_v1_spon;
ICE_perN_v2_spon   = hr_v2_perNeuron_spon ./ cost_perN_v2_spon;
% Driven: MI / cost_driven
ICE_perN_v1_driven = mi_v1_driv_perN ./ cost_perN_v1_driven;
ICE_perN_v2_driven = mi_v2_driv_perN ./ cost_perN_v2_driven;
% Driven: Hr / cost_driven (total entropy efficiency, for reference)
ICE_Hr_perN_v1_driven = hr_v1_driv_perN ./ cost_perN_v1_driven;
ICE_Hr_perN_v2_driven = hr_v2_driv_perN ./ cost_perN_v2_driven;

figure;
set(gcf, 'Color', 'w', 'Position', [100 100 1500 450]);

% ---- Panel A: ICE bar chart (Spon Hr/C vs Driven MI/C) ----
subplot(1,3,1); hold on;
bw = 0.30;

col_spon = [0.30 0.60 0.90];  % blue
col_mi   = [0.85 0.33 0.10];  % orange

for g = 1:2  % g=1: V1, g=2: V2
    if g == 1
        ice_spon = ICE_perN_v1_spon;
        ice_driv = ICE_perN_v1_driven;
    else
        ice_spon = ICE_perN_v2_spon;
        ice_driv = ICE_perN_v2_driven;
    end

    x_spon = g - 0.18;
    x_driv = g + 0.18;

    % Spon Hr/C (blue)
    bar(x_spon, mean(ice_spon), bw, 'FaceColor', col_spon, 'EdgeColor', 'none');
    errorbar(x_spon, mean(ice_spon), std(ice_spon)/sqrt(length(ice_spon)), ...
        'k', 'LineStyle', 'none', 'LineWidth', 1.5, 'CapSize', 6);

    % Driven MI/C (orange)
    bar(x_driv, mean(ice_driv), bw, 'FaceColor', col_mi, 'EdgeColor', 'none');
    errorbar(x_driv, mean(ice_driv), std(ice_driv)/sqrt(length(ice_driv)), ...
        'k', 'LineStyle', 'none', 'LineWidth', 1.5, 'CapSize', 6);
end

set(gca, 'XTick', [1 2], 'XTickLabel', {'V1', 'V2'});
ylabel('information / calibrated cost proxy');
title('A. Spon H(r)/C vs Driven MI/C');

hS = bar(nan, nan, 'FaceColor', col_spon, 'EdgeColor', 'none');
hD = bar(nan, nan, 'FaceColor', col_mi, 'EdgeColor', 'none');
legend([hS, hD], {'Spon H(r)/C', 'Driven MI/C'}, 'Location', 'best');
box off;

% ---- Panel B: Information vs Cost scatter (V1=o, V2=*) ----
subplot(1,3,2); hold on;

col_blue   = [0.30 0.60 0.90];
col_orange = [0.85 0.33 0.10];

% Spon H(r) vs Cost — solid blue (V1 circle, V2 star)
scatter(cost_perN_v1_spon, hr_v1_perNeuron_spon, 30, ...
    col_blue, 'o', 'filled', 'MarkerFaceAlpha', 1.0);
scatter(cost_perN_v2_spon, hr_v2_perNeuron_spon, 80, ...
    col_blue, 'p', 'filled', 'MarkerFaceAlpha', 1.0);

% Driven H(r) vs Cost — blue 30% (V1 circle, V2 star)
scatter(cost_perN_v1_driven, hr_v1_driv_perN, 30, ...
    col_blue, 'o', 'filled', 'MarkerFaceAlpha', 0.3);
scatter(cost_perN_v2_driven, hr_v2_driv_perN, 80, ...
    col_blue, 'p', 'filled', 'MarkerFaceAlpha', 0.3);

% MI vs Cost — orange (V1 circle, V2 star)
scatter(cost_perN_v1_driven, mi_v1_driv_perN, 30, ...
    col_orange, 'o', 'filled', 'MarkerFaceAlpha', 0.9);
scatter(cost_perN_v2_driven, mi_v2_driv_perN, 80, ...
    col_orange, 'p', 'filled', 'MarkerFaceAlpha', 0.9);

xlabel('calibrated cost proxy');
ylabel('Information [bits/s]');
title('B. Information vs Cost');

% Legend: 3 info types + 2 region markers
h1 = scatter(nan, nan, 40, col_blue, 'o', 'filled', 'MarkerFaceAlpha', 1.0);
h2 = scatter(nan, nan, 40, col_blue, 'o', 'filled', 'MarkerFaceAlpha', 0.3);
h3 = scatter(nan, nan, 40, col_orange, 'o', 'filled', 'MarkerFaceAlpha', 0.9);
h4 = scatter(nan, nan, 30, 'k', 'o', 'filled');
h5 = scatter(nan, nan, 80, 'k', 'p', 'filled');
legend([h1 h2 h3 h4 h5], ...
    {'Spon H(r)', 'Driven H(r)', 'Driven MI', 'V1', 'V2'}, ...
    'Location', 'best');
box off;

% ---- Panel C: ICE vs Cost scatter (V1=o, V2=*) ----
subplot(1,3,3); hold on;

% Spon ICE (Hr/C) vs Cost — solid blue
scatter(cost_perN_v1_spon, ICE_perN_v1_spon, 30, ...
    col_blue, 'o', 'filled', 'MarkerFaceAlpha', 1.0);
scatter(cost_perN_v2_spon, ICE_perN_v2_spon, 80, ...
    col_blue, 'p', 'filled', 'MarkerFaceAlpha', 1.0);

% Driven ICE (MI/C) vs Cost — orange
scatter(cost_perN_v1_driven, ICE_perN_v1_driven, 30, ...
    col_orange, 'o', 'filled', 'MarkerFaceAlpha', 0.9);
scatter(cost_perN_v2_driven, ICE_perN_v2_driven, 80, ...
    col_orange, 'p', 'filled', 'MarkerFaceAlpha', 0.9);

xlabel('calibrated cost proxy');
ylabel('information / calibrated cost proxy');
title('C. ICE vs Cost');

h1c = scatter(nan, nan, 40, col_blue, 'o', 'filled', 'MarkerFaceAlpha', 1.0);
h2c = scatter(nan, nan, 40, col_orange, 'o', 'filled', 'MarkerFaceAlpha', 0.9);
h3c = scatter(nan, nan, 30, 'k', 'o', 'filled');
h4c = scatter(nan, nan, 80, 'k', 'p', 'filled');
legend([h1c h2c h3c h4c], ...
    {'Spon H(r)/C', 'Driven MI/C', 'V1', 'V2'}, ...
    'Location', 'best');
box off;

%% Save results
save('MI_analysis_figure2_new.mat', ...
    'MI_v1', 'Hr_v1', 'Hn_v1', 'MI_v2', 'Hr_v2', 'Hn_v2', ...
    'MI_spatial_v1', 'Hr_spatial_v1', 'Hrs_spatial_v1', ...
    'MI_spatial_v2', 'Hr_spatial_v2', 'Hrs_spatial_v2', ...
    'mi_v1_perNeuron', 'mi_v2_perNeuron', ...
    'hr_v1_perNeuron_total', 'hn_v1_perNeuron', ...
    'hr_v2_perNeuron_total', 'hn_v2_perNeuron', ...
    'hr_v1_perNeuron_perStim', 'hn_v1_perNeuron_perStim', 'mi_v1_perNeuron_perStim', ...
    'hr_v2_perNeuron_perStim', 'hn_v2_perNeuron_perStim', 'mi_v2_perNeuron_perStim', ...
    'Hr_v1_perStim_pop', 'Hn_v1_perStim_pop', 'MI_v1_perStim_pop', ...
    'Hr_v2_perStim_pop', 'Hn_v2_perStim_pop', 'MI_v2_perStim_pop', ...
    'hr_v1_perNeuron_spon', 'hr_v2_perNeuron_spon', ...
    'Hr_v1_spon_direct', 'Hr_v2_spon_direct', ...
    'Hr_spatial_v1_spon', 'Hr_spatial_v2_spon', ...
    'cost1_p', 'cost2_p', 'cost1_ps', 'cost2_ps', ...
    'entr1_p', 'entr2_p', 'entr1_ps', 'entr2_ps', ...
    'dur_driven', 'dur_spon', ...
    'unique_stim_driven', ...
    'ICE_spatial_v1', 'ICE_spatial_v2', ...
    'ICE_perNeuron_v1', 'ICE_perNeuron_v2', ...
    'ICE_pop_v1', 'ICE_pop_v2', ...
    'stim_driven', 'T_list', 'dt_bin', ...
    'H_total_pop_v1', 'H_noise_pop_v1', ...
    'H_total_pop_v2', 'H_noise_pop_v2')

fprintf('\n=== Analysis Complete (Direct Method) ===\n');
fprintf('Results saved to: MI_analysis_figure2_new.mat\n');

%% ========================================================================
%% LOCAL FUNCTIONS (copied from compute_entropy_toolbox.m)
%% ========================================================================

function H = total_entropy_local(spike_matrix, T)
% TOTAL_ENTROPY_LOCAL  Compute total response entropy (bits)
%
% Concatenates all rows, cuts into binary words of length T,
% computes Shannon entropy.
%
% Input:
%   spike_matrix - [rows x time_bins], binary 0/1
%   T            - word length (bins)
% Output:
%   H - Shannon entropy (bits)

    [n_rows, n_cols] = size(spike_matrix);
    n_windows = n_cols - T + 1;
    if n_windows <= 0, H = 0; return; end

    % Use the same sliding word starts as the conditional-entropy estimate.
    words = zeros(n_rows * n_windows, T, 'like', spike_matrix);
    for j = 1:n_windows
        idx = (j - 1) * n_rows + (1:n_rows);
        words(idx, :) = spike_matrix(:, j:(j + T - 1));
    end

    % Count unique words
    counts = count_unique_words_local(words);

    % Shannon entropy: H = -sum(p * log2(p))
    p = counts / sum(counts);
    H = -sum(p .* log2(p));
end


function H = noise_entropy_local(spike_matrix, T)
% NOISE_ENTROPY_LOCAL  Compute noise entropy (bits)
%
% For repeated trials of the same stimulus, sliding window across time,
% count word distributions across trials at each position, average entropy.
%
% Input:
%   spike_matrix - [trials x time_bins], binary (same stimulus repeated)
%   T            - word length (bins)
% Output:
%   H - average noise entropy (bits)

    [n_trials, n_bins] = size(spike_matrix);
    n_windows = n_bins - T + 1;

    entropy_values = zeros(1, n_windows);

    for j = 1:n_windows
        % Extract words from all trials at this time position
        words = spike_matrix(:, j:(j + T - 1));

        % Count unique words
        counts = count_unique_words_local(words);

        % Shannon entropy
        p = counts / sum(counts);
        entropy_values(j) = -sum(p .* log2(p));
    end

    % Zero-entropy positions are valid observations and remain in the mean.
    H = mean(entropy_values);
end


function counts = count_unique_words_local(words)
% COUNT_UNIQUE_WORDS_LOCAL  Count occurrences of unique rows in a matrix
%
% Input:
%   words - [n_words x T] matrix, each row is a binary word (0/1)
% Output:
%   counts - occurrences of each unique word (column vector)
%
% Uses binary-to-integer hashing for O(n log n) performance.

    [n_words, T] = size(words);

    if T <= 52
        % Convert each binary word to a unique integer (double has 52-bit mantissa)
        pow2_vec = pow2(0:T-1)';           % [T x 1]
        word_ids = words * pow2_vec;        % [n_words x 1]
        [~, ~, ic] = unique(word_ids);
        counts = accumarray(ic, 1);
    else
        % For very long words, use sortrows on the matrix directly
        [~, ~, ic] = unique(words, 'rows');
        counts = accumarray(ic, 1);
    end
end


function data = apply_refractory_local(data, n)
% APPLY_REFRACTORY_LOCAL  Enforce refractory period
%
% After each spike (1), force the next n bins to 0.
%
% Input:
%   data - [rows x cols] binary matrix
%   n    - refractory period length (bins)
% Output:
%   data - processed matrix (same size)

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
% LINEAR_FIT_LOCAL  Simple linear regression
%
% Input:
%   x, y - data vectors
% Output:
%   y_fit - fitted values
%   fit   - struct with slope, intercept, R2, RMSE

    p = polyfit(x, y, 1);
    y_fit = polyval(p, x);

    residuals = y - y_fit;
    SSE = sum(residuals .^ 2);
    SST = (length(y) - 1) * var(y);

    fit.slope     = p(1);
    fit.intercept = p(2);
    if SST > 0
        fit.R2 = 1 - SSE / SST;
    else
        fit.R2 = NaN;
    end
    fit.RMSE = sqrt(SSE / length(y));
end
