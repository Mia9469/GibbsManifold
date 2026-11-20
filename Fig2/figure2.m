% figure2.m
% Computes and plots population- and lifetime-level statistics
% (CV, sparseness, entropy, and information-cost efficiency) for
% neural spike data from two cortical areas (V1 and V2),
% separately for spontaneous and visually driven conditions.
%
% ENHANCED WITH MUTUAL INFORMATION (MI):
%   - Separates trials by stimulus identity
%   - Computes H(r|s) : noise entropy (entropy per stimulus, averaged)
%   - Computes H(r)   : response entropy (entropy across all stimuli)
%   - Computes MI     : mutual information between response and stimulus
%   - Computes ICE    : information-cost efficiency (MI / metabolic cost)
%
% Data source: CRCNS V1/V2 Dataset (https://crcns.org/data-sets/vc/v1v2-1)
% Data file: 105l001p16.mat and similar session files
% Requires: neuralData.spikeRasters{trial, area}, neuralData.stim, neuralData.trialId

clear; close all;

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

% Trial durations [s] — needed for information rate = MI / dur
dur_driven = 1.28;   % driven trial duration [s]
dur_spon   = 1.5;    % spontaneous trial duration [s]

% ===== NEW: Storage for MI computation =====
% For each trial, store ISI-based firing rate and stimulus identity
% Structure: rate_v1_driven_byNeuron{neuron}{trial} = rates for that neuron in that trial
% Structure: rate_v1_driven_pooled{trial} = pooled rates for all neurons in that trial
% Separate storage for spontaneous and driven conditions
rate_v1_driven_pooled = cell(1, nt);      % V1 pooled rates for driven trials
rate_v1_driven_byNeuron = cell(159, nt);  % V1 per-neuron rates (159 V1 neurons)
rate_v1_spon_pooled = cell(1, nt);        % V1 pooled rates for spontaneous trials
rate_v1_spon_byNeuron = cell(159, nt);    % V1 per-neuron rates spontaneous
rate_v2_driven_pooled = cell(1, nt);      % V2 pooled rates for driven trials
rate_v2_driven_byNeuron = cell(25, nt);   % V2 per-neuron rates (25 V2 neurons)
rate_v2_spon_pooled = cell(1, nt);        % V2 pooled rates for spontaneous
rate_v2_spon_byNeuron = cell(25, nt);     % V2 per-neuron rates spontaneous
stim_driven = [];                         % stimulus labels for driven trials
trial_count_driven = 0;                   % count of driven trials processed
trial_count_spon = 0;                     % count of spontaneous trials

%% -----------------------------------------------------------------------
%% SECTION 1: Population-level CV, sparseness, and entropy
%% -----------------------------------------------------------------------
% Spontaneous trials have 1500 time bins; driven trials have 1280 bins.
% Window durations: 1.5 s (spontaneous) and 1.28 s (driven).

for i = 1:nt

    spikes_v1 = neuralData.spikeRasters{i, 1};
    spikes_v2 = neuralData.spikeRasters{i, 2};

    if size(spikes_v1, 2) == 1500
        % ----- Spontaneous condition (1.5 s window) -----
        dur = 1.5;

        % Population CV
        cv_v1_s = [cv_v1_s,  std(sum(spikes_v1, 2) / dur) ./ mean(sum(spikes_v1, 2) / dur)];
        cv_v2_s = [cv_v2_s,  std(sum(spikes_v2, 2) / dur) ./ mean(sum(spikes_v2, 2) / dur)];

        % Population sparseness (bias-corrected)
        ss_v1_s = [ss_v1_s, (1 - mean(sum(spikes_v1,2)/dur)^2 ./ mean((sum(spikes_v1,2)/dur).^2)) * (1 - 1/159)];
        ss_v2_s = [ss_v2_s, (1 - mean(sum(spikes_v2,2)/dur)^2 ./ mean((sum(spikes_v2,2)/dur).^2)) * (1 - 1/25)];

        % Mean firing rates across neurons
        mrate_v1_s = [mrate_v1_s, sum(spikes_v1, 2) / dur];
        mrate_v2_s = [mrate_v2_s, sum(spikes_v2, 2) / dur];

        % Entropy of the population firing-rate distribution (V1)
        [n, ~] = histcounts(sum(spikes_v1, 2) / dur, 100);
        h = n / sum(n);
        entr = 0;
        for j = h
            if j > 0
                entr = entr - j * log2(j);
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
                entr = entr - j * log2(j);
            end
        end
        entr2_ps = [entr2_ps, entr];
        cost2_ps = [cost2_ps, C1 + C2 * mean(sum(spikes_v2, 2) / dur)];

        % Store rasters
        rast_v1_s{end+1} = spikes_v1;
        rast_v2_s{end+1} = spikes_v2;
        
        % ===== NEW: Store ISI-based rates for spontaneous condition =====
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

        % Population sparseness (bias-corrected)
        ss_v1 = [ss_v1, (1 - mean(sum(spikes_v1,2)/dur)^2 ./ mean((sum(spikes_v1,2)/dur).^2)) * (1 - 1/159)];
        ss_v2 = [ss_v2, (1 - mean(sum(spikes_v2,2)/dur)^2 ./ mean((sum(spikes_v2,2)/dur).^2)) * (1 - 1/25)];

        % Mean firing rates across neurons
        mrate_v1 = [mrate_v1, sum(spikes_v1, 2) / dur];
        mrate_v2 = [mrate_v2, sum(spikes_v2, 2) / dur];

        % Entropy of the population firing-rate distribution (V1)
        [n, ~] = histcounts(sum(spikes_v1, 2) / dur, 100);
        h = n / sum(n);
        entr = 0;
        for j = h
            if j > 0
                entr = entr - j * log2(j);
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
                entr = entr - j * log2(j);
            end
        end
        entr2_p = [entr2_p, entr];
        cost2_p = [cost2_p, C1 + C2 * mean(sum(spikes_v2, 2) / dur)];

        % Store rasters
        rast_v1_d{end+1} = spikes_v1;
        rast_v2_d{end+1} = spikes_v2;

        % ===== NEW: Compute ISI-based firing rates for MI =====
        % For each neuron, compute instantaneous rate from ISIs
        % Store BOTH: per-neuron (for individual MI) and pooled (for population MI)
        
        trial_count_driven = trial_count_driven + 1;
        rate_v1_pooled = [];   % will accumulate all V1 rates
        rate_v2_pooled = [];   % will accumulate all V2 rates
        
        % V1: Store per-neuron rates in a cell array
        for neuron = 1:size(spikes_v1, 1)  % 159 neurons
            spike_times = find(spikes_v1(neuron, :)) * dur / size(spikes_v1, 2);
            if length(spike_times) >= 2
                isi = diff(spike_times);
                rate_isi = 1000 ./ isi;  % convert to Hz
                rate_v1_driven_byNeuron{neuron, trial_count_driven} = rate_isi;
                rate_v1_pooled = [rate_v1_pooled, rate_isi];  % accumulate
            end
        end
        
        % V2: Store per-neuron rates in a cell array
        for neuron = 1:size(spikes_v2, 1)  % 25 neurons
            spike_times = find(spikes_v2(neuron, :)) * dur / size(spikes_v2, 2);
            if length(spike_times) >= 2
                isi = diff(spike_times);
                rate_isi = 1000 ./ isi;  % convert to Hz
                rate_v2_driven_byNeuron{neuron, trial_count_driven} = rate_isi;
                rate_v2_pooled = [rate_v2_pooled, rate_isi];  % accumulate
            end
        end
        
        % Store pooled rates and stimulus label
        rate_v1_driven_pooled{trial_count_driven} = rate_v1_pooled;
        rate_v2_driven_pooled{trial_count_driven} = rate_v2_pooled;
        stim_driven = [stim_driven, stim_labels(i)];
    end
end

%% ===== NEW SECTION: Compute Mutual Information =====
%% -----------------------------------------------------------------------
%% Mutual Information (MI) for SPONTANEOUS responses
%% -----------------------------------------------------------------------
% Note: Spontaneous condition has no stimulus labels, so MI = 0.
% However, we compute entropy for comparison with driven condition.

fprintf('Computing statistics for spontaneous condition...\n');

% V1 spontaneous
all_rates_v1_spon = [];
for trial = 1:length(rate_v1_spon_pooled)
    if ~isempty(rate_v1_spon_pooled{trial})
        all_rates_v1_spon = [all_rates_v1_spon, rate_v1_spon_pooled{trial}];
    end
end

if ~isempty(all_rates_v1_spon)
    binEdges_r = 0:1:(max(all_rates_v1_spon)+1);
    if length(binEdges_r) < 2
        binEdges_r = [0 1];
    end
    [n_r, ~] = histcounts(all_rates_v1_spon, binEdges_r);
    p_r = n_r / sum(n_r);
    p_r = p_r(p_r > 0);
    Hr_v1_spon = -sum(p_r .* log2(p_r));
    MI_v1_spon = 0;  % No stimulus in spontaneous condition
    Hrs_v1_spon = Hr_v1_spon;  % H(r|s) = H(r) when no stimulus dependence
    fprintf('  V1 spontaneous: H(r) = %.4f bits (no stimulus, MI=0)\n', Hr_v1_spon);
else
    Hr_v1_spon = 0;
    Hrs_v1_spon = 0;
    MI_v1_spon = 0;
end

% V2 spontaneous
all_rates_v2_spon = [];
for trial = 1:length(rate_v2_spon_pooled)
    if ~isempty(rate_v2_spon_pooled{trial})
        all_rates_v2_spon = [all_rates_v2_spon, rate_v2_spon_pooled{trial}];
    end
end

if ~isempty(all_rates_v2_spon)
    binEdges_r = 0:1:(max(all_rates_v2_spon)+1);
    if length(binEdges_r) < 2
        binEdges_r = [0 1];
    end
    [n_r, ~] = histcounts(all_rates_v2_spon, binEdges_r);
    p_r = n_r / sum(n_r);
    p_r = p_r(p_r > 0);
    Hr_v2_spon = -sum(p_r .* log2(p_r));
    MI_v2_spon = 0;
    Hrs_v2_spon = Hr_v2_spon;
    fprintf('  V2 spontaneous: H(r) = %.4f bits (no stimulus, MI=0)\n\n', Hr_v2_spon);
else
    Hr_v2_spon = 0;
    Hrs_v2_spon = 0;
    MI_v2_spon = 0;
end

% ===== SPATIAL-ONLY ENTROPY FOR SPONTANEOUS =====
% Pool per-neuron spike-count firing rates across all spon trials.
% Each trial contributes N rates (one per neuron), capturing spatial diversity.

% V1 spontaneous: spatial-only H(r)
all_spatial_rates_v1_spon = [];
for trial = 1:length(rast_v1_s)
    spikes = rast_v1_s{trial};
    rates = sum(spikes, 2)' / 1.5;  % 1×159 firing rates
    all_spatial_rates_v1_spon = [all_spatial_rates_v1_spon, rates];
end

if ~isempty(all_spatial_rates_v1_spon)
    binEdges_spatial = 0:1:(max(all_spatial_rates_v1_spon)+1);
    [n_r, ~] = histcounts(all_spatial_rates_v1_spon, binEdges_spatial);
    p_r = n_r / sum(n_r);
    p_r = p_r(p_r > 0);
    Hr_spatial_v1_spon = -sum(p_r .* log2(p_r));
    fprintf('  V1 spatial-only spon: H(r) = %.4f bits\n', Hr_spatial_v1_spon);
else
    Hr_spatial_v1_spon = 0;
end

% V2 spontaneous: spatial-only H(r)
all_spatial_rates_v2_spon = [];
for trial = 1:length(rast_v2_s)
    spikes = rast_v2_s{trial};
    rates = sum(spikes, 2)' / 1.5;  % 1×25 firing rates
    all_spatial_rates_v2_spon = [all_spatial_rates_v2_spon, rates];
end

if ~isempty(all_spatial_rates_v2_spon)
    binEdges_spatial = 0:1:(max(all_spatial_rates_v2_spon)+1);
    [n_r, ~] = histcounts(all_spatial_rates_v2_spon, binEdges_spatial);
    p_r = n_r / sum(n_r);
    p_r = p_r(p_r > 0);
    Hr_spatial_v2_spon = -sum(p_r .* log2(p_r));
    fprintf('  V2 spatial-only spon: H(r) = %.4f bits\n', Hr_spatial_v2_spon);
else
    Hr_spatial_v2_spon = 0;
end

%% -----------------------------------------------------------------------
%% Mutual Information (MI) for DRIVEN responses (stimulus-dependent)
%% -----------------------------------------------------------------------

fprintf('Computing Mutual Information for V1...\n');

% ===== POPULATION-LEVEL MI =====
% Pool all rates across all stimuli
all_rates_v1 = [];
for trial = 1:length(rate_v1_driven_pooled)
    if ~isempty(rate_v1_driven_pooled{trial})
        all_rates_v1 = [all_rates_v1, rate_v1_driven_pooled{trial}];
    end
end

if ~isempty(all_rates_v1)
    % H(r) — Response entropy (unconditional)
    binEdges_r = 0:1:(max(all_rates_v1)+1);  % 1 Hz bins
    if length(binEdges_r) < 2
        binEdges_r = [0 1];
    end
    [n_r, ~] = histcounts(all_rates_v1, binEdges_r);
    p_r = n_r / sum(n_r);
    p_r = p_r(p_r > 0);
    Hr = -sum(p_r .* log2(p_r));
    
    % H(r|s) — Conditional entropy (per stimulus)
    % Correct definition: H(r|s) = sum over all stimuli: P(s) * H(r | s=s)
    Hrs = 0;
    for s = unique(stim_driven)
        stim_mask = (stim_driven == s);
        P_s = sum(stim_mask) / length(stim_driven);  % Probability of stimulus s
        
        rates_this_stim = [];
        for trial = find(stim_mask)
            if ~isempty(rate_v1_driven_pooled{trial})
                rates_this_stim = [rates_this_stim, rate_v1_driven_pooled{trial}];
            end
        end
        
        if ~isempty(rates_this_stim)
            [n_s, ~] = histcounts(rates_this_stim, binEdges_r);
            p_s = n_s / sum(n_s);
            p_s = p_s(p_s > 0);
            if ~isempty(p_s)
                H_rs = -sum(p_s .* log2(p_s));
                Hrs = Hrs + P_s * H_rs;  % Weighted average by stimulus probability
            end
        end
    end
    
    % MI = H(r) - H(r|s)
    MI_v1 = Hr - Hrs;
    
    fprintf('  Population-level MI:\n');
    fprintf('    H(r)   = %.4f bits\n', Hr);
    fprintf('    H(r|s) = %.4f bits\n', Hrs);
    fprintf('    MI     = %.4f bits\n\n', MI_v1);
    
    % ===== SPATIAL-ONLY MI (per-neuron spike-count rates) =====
    % Pool per-neuron firing rates across trials for spatial distribution.
    % Each trial contributes 159 rates (one per neuron).
    fprintf('  Computing Spatial-only MI (per-neuron rates)...\n');

    % Store per-trial spatial rate vectors for H(r|s)
    spatial_rates_v1_byTrial = cell(1, length(rast_v1_d));
    all_spatial_rates_v1_driven = [];
    for trial = 1:length(rast_v1_d)
        spikes = rast_v1_d{trial};
        rates = sum(spikes, 2)' / 1.28;  % 1×159 firing rates
        spatial_rates_v1_byTrial{trial} = rates;
        all_spatial_rates_v1_driven = [all_spatial_rates_v1_driven, rates];
    end

    if ~isempty(all_spatial_rates_v1_driven)
        % H(r) — pool all neuron rates across all driven trials
        binEdges_spatial = 0:1:(max(all_spatial_rates_v1_driven)+1);
        [n_r_spatial, ~] = histcounts(all_spatial_rates_v1_driven, binEdges_spatial);
        p_r_spatial = n_r_spatial / sum(n_r_spatial);
        p_r_spatial = p_r_spatial(p_r_spatial > 0);
        Hr_spatial_v1 = -sum(p_r_spatial .* log2(p_r_spatial));

        % H(r|s) — per stimulus, pool neuron rates across trials of that stimulus
        Hrs_spatial_v1 = 0;
        for s = unique(stim_driven)
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
        fprintf('  Spatial-only MI (V1):\n');
        fprintf('    H(r)   = %.4f bits\n', Hr_spatial_v1);
        fprintf('    H(r|s) = %.4f bits\n', Hrs_spatial_v1);
        fprintf('    MI     = %.4f bits\n\n', MI_spatial_v1);
    end
    
    % ===== PER-NEURON MI =====
    fprintf('  Computing per-neuron MI (159 V1 neurons)...\n');
    mi_v1_perNeuron = zeros(1, 159);
    
    for neuron = 1:159
        all_rates_neuron = [];
        for trial = 1:trial_count_driven
            if ~isempty(rate_v1_driven_byNeuron{neuron, trial})
                all_rates_neuron = [all_rates_neuron, rate_v1_driven_byNeuron{neuron, trial}];
            end
        end
        
        if ~isempty(all_rates_neuron) && length(all_rates_neuron) > 1
            binEdges_n = 0:1:(max(all_rates_neuron)+1);
            if length(binEdges_n) < 2
                binEdges_n = [0 1];
            end
            
            % H(r) for this neuron across all stimuli
            [n_r_neuron, ~] = histcounts(all_rates_neuron, binEdges_n);
            p_r_neuron = n_r_neuron / sum(n_r_neuron);
            p_r_neuron = p_r_neuron(p_r_neuron > 0);
            if ~isempty(p_r_neuron)
                Hr_neuron = -sum(p_r_neuron .* log2(p_r_neuron));
            else
                Hr_neuron = 0;
            end
            
            % H(r|s) for this neuron per stimulus
            % Correct definition: H(r|s) = sum over all stimuli: P(s) * H(r | s=s)
            Hrs_neuron = 0;
            for s = unique(stim_driven)
                stim_mask = (stim_driven == s);
                P_s = sum(stim_mask) / length(stim_driven);  % Probability of stimulus s
                
                rates_neuron_stim = [];
                for trial = find(stim_mask)
                    if ~isempty(rate_v1_driven_byNeuron{neuron, trial})
                        rates_neuron_stim = [rates_neuron_stim, rate_v1_driven_byNeuron{neuron, trial}];
                    end
                end
                
                if ~isempty(rates_neuron_stim)
                    [n_s_neuron, ~] = histcounts(rates_neuron_stim, binEdges_n);
                    p_s_neuron = n_s_neuron / sum(n_s_neuron);
                    p_s_neuron = p_s_neuron(p_s_neuron > 0);
                    if ~isempty(p_s_neuron)
                        H_rs_neuron = -sum(p_s_neuron .* log2(p_s_neuron));
                        Hrs_neuron = Hrs_neuron + P_s * H_rs_neuron;  % Weighted average by stimulus probability
                    end
                end
            end
            
            mi_v1_perNeuron(neuron) = max(0, Hr_neuron - Hrs_neuron);  % MI >= 0
        end
    end
    
    fprintf('    Mean per-neuron MI: %.4f bits\n', mean(mi_v1_perNeuron(mi_v1_perNeuron > 0)));
    fprintf('    Max per-neuron MI:  %.4f bits (neuron #%d)\n', max(mi_v1_perNeuron), find(mi_v1_perNeuron == max(mi_v1_perNeuron), 1));
    fprintf('    % selective neurons (MI > 0.01): %.1f%%\n\n', 100*sum(mi_v1_perNeuron > 0.01)/159);
    
    fprintf('  Information-Cost Efficiency (Population): %.4f\n\n', MI_v1 / (dur_driven * mean(cost1_p)));
else
    fprintf('  Warning: No driven trial data for MI computation\n');
    MI_v1 = 0;
    Hr = 0;
    Hrs = 0;
    mi_v1_perNeuron = zeros(1, 159);
end

% Repeat for V2
fprintf('Computing Mutual Information for V2...\n');
all_rates_v2 = [];
for trial = 1:length(rate_v2_driven_pooled)
    if ~isempty(rate_v2_driven_pooled{trial})
        all_rates_v2 = [all_rates_v2, rate_v2_driven_pooled{trial}];
    end
end

if ~isempty(all_rates_v2)
    binEdges_r = 0:1:(max(all_rates_v2)+1);
    if length(binEdges_r) < 2
        binEdges_r = [0 1];
    end
    [n_r, ~] = histcounts(all_rates_v2, binEdges_r);
    p_r = n_r / sum(n_r);
    p_r = p_r(p_r > 0);
    Hr_v2 = -sum(p_r .* log2(p_r));
    
    Hrs_v2 = 0;
    for s = unique(stim_driven)
        stim_mask = (stim_driven == s);
        P_s = sum(stim_mask) / length(stim_driven);  % Probability of stimulus s
        
        rates_this_stim = [];
        for trial = find(stim_mask)
            if ~isempty(rate_v2_driven_pooled{trial})
                rates_this_stim = [rates_this_stim, rate_v2_driven_pooled{trial}];
            end
        end
        if ~isempty(rates_this_stim)
            [n_s, ~] = histcounts(rates_this_stim, binEdges_r);
            p_s = n_s / sum(n_s);
            p_s = p_s(p_s > 0);
            if ~isempty(p_s)
                H_rs = -sum(p_s .* log2(p_s));
                Hrs_v2 = Hrs_v2 + P_s * H_rs;  % Weighted average by stimulus probability
            end
        end
    end
    
    MI_v2 = Hr_v2 - Hrs_v2;
    
    fprintf('  Population-level MI:\n');
    fprintf('    H(r)   = %.4f bits\n', Hr_v2);
    fprintf('    H(r|s) = %.4f bits\n', Hrs_v2);
    fprintf('    MI     = %.4f bits\n\n', MI_v2);
    
    % ===== SPATIAL-ONLY MI (per-neuron spike-count rates) =====
    % Pool per-neuron firing rates across trials for spatial distribution.
    % Each trial contributes 25 rates (one per neuron).
    fprintf('  Computing Spatial-only MI (per-neuron rates)...\n');

    spatial_rates_v2_byTrial = cell(1, length(rast_v2_d));
    all_spatial_rates_v2_driven = [];
    for trial = 1:length(rast_v2_d)
        spikes = rast_v2_d{trial};
        rates = sum(spikes, 2)' / 1.28;  % 1×25 firing rates
        spatial_rates_v2_byTrial{trial} = rates;
        all_spatial_rates_v2_driven = [all_spatial_rates_v2_driven, rates];
    end

    if ~isempty(all_spatial_rates_v2_driven)
        % H(r) — pool all neuron rates across all driven trials
        binEdges_spatial = 0:1:(max(all_spatial_rates_v2_driven)+1);
        [n_r_spatial, ~] = histcounts(all_spatial_rates_v2_driven, binEdges_spatial);
        p_r_spatial = n_r_spatial / sum(n_r_spatial);
        p_r_spatial = p_r_spatial(p_r_spatial > 0);
        Hr_spatial_v2 = -sum(p_r_spatial .* log2(p_r_spatial));

        % H(r|s) — per stimulus, pool neuron rates across trials of that stimulus
        Hrs_spatial_v2 = 0;
        for s = unique(stim_driven)
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
        fprintf('  Spatial-only MI (V2):\n');
        fprintf('    H(r)   = %.4f bits\n', Hr_spatial_v2);
        fprintf('    H(r|s) = %.4f bits\n', Hrs_spatial_v2);
        fprintf('    MI     = %.4f bits\n\n', MI_spatial_v2);
    end
    
    % ===== PER-NEURON MI =====
    fprintf('  Computing per-neuron MI (25 V2 neurons)...\n');
    mi_v2_perNeuron = zeros(1, 25);
    
    for neuron = 1:25
        all_rates_neuron = [];
        for trial = 1:trial_count_driven
            if ~isempty(rate_v2_driven_byNeuron{neuron, trial})
                all_rates_neuron = [all_rates_neuron, rate_v2_driven_byNeuron{neuron, trial}];
            end
        end
        
        if ~isempty(all_rates_neuron) && length(all_rates_neuron) > 1
            binEdges_n = 0:1:(max(all_rates_neuron)+1);
            if length(binEdges_n) < 2
                binEdges_n = [0 1];
            end
            
            [n_r_neuron, ~] = histcounts(all_rates_neuron, binEdges_n);
            p_r_neuron = n_r_neuron / sum(n_r_neuron);
            p_r_neuron = p_r_neuron(p_r_neuron > 0);
            if ~isempty(p_r_neuron)
                Hr_neuron = -sum(p_r_neuron .* log2(p_r_neuron));
            else
                Hr_neuron = 0;
            end
            
            Hrs_neuron = 0;
            for s = unique(stim_driven)
                stim_mask = (stim_driven == s);
                P_s = sum(stim_mask) / length(stim_driven);  % Probability of stimulus s
                
                rates_neuron_stim = [];
                for trial = find(stim_mask)
                    if ~isempty(rate_v2_driven_byNeuron{neuron, trial})
                        rates_neuron_stim = [rates_neuron_stim, rate_v2_driven_byNeuron{neuron, trial}];
                    end
                end
                
                if ~isempty(rates_neuron_stim)
                    [n_s_neuron, ~] = histcounts(rates_neuron_stim, binEdges_n);
                    p_s_neuron = n_s_neuron / sum(n_s_neuron);
                    p_s_neuron = p_s_neuron(p_s_neuron > 0);
                    if ~isempty(p_s_neuron)
                        H_rs_neuron = -sum(p_s_neuron .* log2(p_s_neuron));
                        Hrs_neuron = Hrs_neuron + P_s * H_rs_neuron;  % Weighted average by stimulus probability
                    end
                end
            end
            
            mi_v2_perNeuron(neuron) = max(0, Hr_neuron - Hrs_neuron);
        end
    end
    
    fprintf('    Mean per-neuron MI: %.4f bits\n', mean(mi_v2_perNeuron(mi_v2_perNeuron > 0)));
    fprintf('    Max per-neuron MI:  %.4f bits (neuron #%d)\n', max(mi_v2_perNeuron), find(mi_v2_perNeuron == max(mi_v2_perNeuron), 1));
    fprintf('    %% selective neurons (MI > 0.01): %.1f%%\n\n', 100*sum(mi_v2_perNeuron > 0.01)/25);
    
    fprintf('  Information-Cost Efficiency (Population): %.4f\n', MI_v2 / (dur_driven * mean(cost2_p)));
else
    fprintf('  Warning: No driven trial data for MI computation\n');
    MI_v2 = 0;
    Hr_v2 = 0;
    Hrs_v2 = 0;
    mi_v2_perNeuron = zeros(1, 25);
end

%% ===== NEW: Compute per-neuron H(r) for spontaneous and driven conditions =====
% For each neuron, compute H(r) separately in spon and driven conditions
% This allows comparison: H(r)_spon vs H(r)_driven (response entropy) and H(r|s)_driven (noise)

fprintf('Computing per-neuron entropy for spon/driven comparison...\n');

% V1 per-neuron entropy
Hr_v1_perNeuron_spon = zeros(1, 159);    % H(r) in spontaneous
Hr_v1_perNeuron_driven = zeros(1, 159);  % H(r) in driven (unconditional)
Hrs_v1_perNeuron = zeros(1, 159);        % H(r|s) in driven (conditional on stimulus)

for neuron = 1:159
    % Spontaneous
    all_rates_spon = [];
    for trial = 1:trial_count_spon
        if ~isempty(rate_v1_spon_byNeuron{neuron, trial})
            all_rates_spon = [all_rates_spon, rate_v1_spon_byNeuron{neuron, trial}];
        end
    end
    
    if ~isempty(all_rates_spon) && length(all_rates_spon) > 1
        binEdges_n = 0:1:(max(all_rates_spon)+1);
        if length(binEdges_n) < 2
            binEdges_n = [0 1];
        end
        [n_r, ~] = histcounts(all_rates_spon, binEdges_n);
        p_r = n_r / sum(n_r);
        p_r = p_r(p_r > 0);
        Hr_v1_perNeuron_spon(neuron) = -sum(p_r .* log2(p_r));
    end
    
    % Driven (unconditional overall)
    all_rates_driven = [];
    for trial = 1:trial_count_driven
        if ~isempty(rate_v1_driven_byNeuron{neuron, trial})
            all_rates_driven = [all_rates_driven, rate_v1_driven_byNeuron{neuron, trial}];
        end
    end
    
    if ~isempty(all_rates_driven) && length(all_rates_driven) > 1
        binEdges_n = 0:1:(max(all_rates_driven)+1);
        if length(binEdges_n) < 2
            binEdges_n = [0 1];
        end
        [n_r, ~] = histcounts(all_rates_driven, binEdges_n);
        p_r = n_r / sum(n_r);
        p_r = p_r(p_r > 0);
        Hr_v1_perNeuron_driven(neuron) = -sum(p_r .* log2(p_r));
    end
    
    % Driven conditional (already computed in MI loop above)
    % Need to recompute for this neuron's H(r|s)
    Hrs_neuron = 0;
    for s = unique(stim_driven)
        stim_mask = (stim_driven == s);
        P_s = sum(stim_mask) / length(stim_driven);
        
        rates_neuron_stim = [];
        for trial = find(stim_mask)
            if ~isempty(rate_v1_driven_byNeuron{neuron, trial})
                rates_neuron_stim = [rates_neuron_stim, rate_v1_driven_byNeuron{neuron, trial}];
            end
        end
        
        if ~isempty(rates_neuron_stim)
            [n_s, ~] = histcounts(rates_neuron_stim, binEdges_n);
            p_s = n_s / sum(n_s);
            p_s = p_s(p_s > 0);
            if ~isempty(p_s)
                H_rs = -sum(p_s .* log2(p_s));
                Hrs_neuron = Hrs_neuron + P_s * H_rs;
            end
        end
    end
    Hrs_v1_perNeuron(neuron) = Hrs_neuron;
end

% V2 per-neuron entropy
Hr_v2_perNeuron_spon = zeros(1, 25);    % H(r) in spontaneous
Hr_v2_perNeuron_driven = zeros(1, 25);  % H(r) in driven
Hrs_v2_perNeuron = zeros(1, 25);        % H(r|s) in driven

for neuron = 1:25
    % Spontaneous
    all_rates_spon = [];
    for trial = 1:trial_count_spon
        if ~isempty(rate_v2_spon_byNeuron{neuron, trial})
            all_rates_spon = [all_rates_spon, rate_v2_spon_byNeuron{neuron, trial}];
        end
    end
    
    if ~isempty(all_rates_spon) && length(all_rates_spon) > 1
        binEdges_n = 0:1:(max(all_rates_spon)+1);
        if length(binEdges_n) < 2
            binEdges_n = [0 1];
        end
        [n_r, ~] = histcounts(all_rates_spon, binEdges_n);
        p_r = n_r / sum(n_r);
        p_r = p_r(p_r > 0);
        Hr_v2_perNeuron_spon(neuron) = -sum(p_r .* log2(p_r));
    end
    
    % Driven (unconditional)
    all_rates_driven = [];
    for trial = 1:trial_count_driven
        if ~isempty(rate_v2_driven_byNeuron{neuron, trial})
            all_rates_driven = [all_rates_driven, rate_v2_driven_byNeuron{neuron, trial}];
        end
    end
    
    if ~isempty(all_rates_driven) && length(all_rates_driven) > 1
        binEdges_n = 0:1:(max(all_rates_driven)+1);
        if length(binEdges_n) < 2
            binEdges_n = [0 1];
        end
        [n_r, ~] = histcounts(all_rates_driven, binEdges_n);
        p_r = n_r / sum(n_r);
        p_r = p_r(p_r > 0);
        Hr_v2_perNeuron_driven(neuron) = -sum(p_r .* log2(p_r));
    end
    
    % Driven conditional
    Hrs_neuron = 0;
    for s = unique(stim_driven)
        stim_mask = (stim_driven == s);
        P_s = sum(stim_mask) / length(stim_driven);
        
        rates_neuron_stim = [];
        for trial = find(stim_mask)
            if ~isempty(rate_v2_driven_byNeuron{neuron, trial})
                rates_neuron_stim = [rates_neuron_stim, rate_v2_driven_byNeuron{neuron, trial}];
            end
        end
        
        if ~isempty(rates_neuron_stim)
            [n_s, ~] = histcounts(rates_neuron_stim, binEdges_n);
            p_s = n_s / sum(n_s);
            p_s = p_s(p_s > 0);
            if ~isempty(p_s)
                H_rs = -sum(p_s .* log2(p_s));
                Hrs_neuron = Hrs_neuron + P_s * H_rs;
            end
        end
    end
    Hrs_v2_perNeuron(neuron) = Hrs_neuron;
end

fprintf('  V1: mean H(r) spon=%.4f, driven=%.4f, H(r|s)=%.4f\n', ...
    mean(Hr_v1_perNeuron_spon), mean(Hr_v1_perNeuron_driven), mean(Hrs_v1_perNeuron));
fprintf('  V2: mean H(r) spon=%.4f, driven=%.4f, H(r|s)=%.4f\n\n', ...
    mean(Hr_v2_perNeuron_spon), mean(Hr_v2_perNeuron_driven), mean(Hrs_v2_perNeuron));

%% -----------------------------------------------------------------------
%% SECTION 2: Lifetime-level CV, sparseness, and entropy
%% -----------------------------------------------------------------------
% For each neuron, compute statistics across stimulus conditions (columns).
% Note: V1 has 159 neurons; V2 has 25 neurons. Use mrate size, not spikes size.

% V1 – driven (159 neurons)
for i = 1:size(mrate_v1, 1)
    ss1_l = [ss1_l, 1 - mean(mrate_v1(i,:))^2 ./ mean(mrate_v1(i,:).^2)];
    cv1_l = [cv1_l, std(mrate_v1(i,:)) ./ mean(mrate_v1(i,:))];
    [n, ~] = histcounts(mrate_v1(i,:), 100);
    h = n / sum(n);
    entr = 0;
    for j = h
        if j > 0
            entr = entr - j * log2(j);
        end
    end
    entr1_l = [entr1_l, entr];
end

% V2 – driven (25 neurons)
for i = 1:size(mrate_v2, 1)
    ss2_l = [ss2_l, 1 - mean(mrate_v2(i,:))^2 ./ mean(mrate_v2(i,:).^2)];
    cv2_l = [cv2_l, std(mrate_v2(i,:)) ./ mean(mrate_v2(i,:))];
    [n, ~] = histcounts(mrate_v2(i,:), 100);
    h = n / sum(n);
    entr = 0;
    for j = h
        if j > 0
            entr = entr - j * log2(j);
        end
    end
    entr2_l = [entr2_l, entr];
end

% V1 – spontaneous (159 neurons)
for i = 1:size(mrate_v1_s, 1)
    ss1_l_s = [ss1_l_s, 1 - mean(mrate_v1_s(i,:))^2 ./ mean(mrate_v1_s(i,:).^2)];
    cv1_l_s = [cv1_l_s, std(mrate_v1_s(i,:)) ./ mean(mrate_v1_s(i,:))];
    [n, ~] = histcounts(mrate_v1_s(i,:), 100);
    h = n / sum(n);
    entr = 0;
    for j = h
        if j > 0
            entr = entr - j * log2(j);
        end
    end
    entr1_ls = [entr1_ls, entr];
end

% V2 – spontaneous (25 neurons)
for i = 1:size(mrate_v2_s, 1)
    ss2_l_s = [ss2_l_s, 1 - mean(mrate_v2_s(i,:))^2 ./ mean(mrate_v2_s(i,:).^2)];
    cv2_l_s = [cv2_l_s, std(mrate_v2_s(i,:)) ./ mean(mrate_v2_s(i,:))];
    [n, ~] = histcounts(mrate_v2_s(i,:), 100);
    h = n / sum(n);
    entr = 0;
    for j = h
        if j > 0
            entr = entr - j * log2(j);
        end
    end
    entr2_ls = [entr2_ls, entr];
end

%% -----------------------------------------------------------------------
%% SECTION 3: Figure – information-cost efficiency (V1, population level)
%% -----------------------------------------------------------------------
x = 0:0.1:30;

figure;

subplot(1, 2, 1)
% ICE = H / (r * C); theoretical bound: 1/C
scatter(cost1_p,  entr1_p  ./ (dur_driven * cost1_p), 'ko'); hold on;
scatter(cost1_ps, entr1_ps ./ (dur_spon * cost1_ps), 'bo'); hold on;
y = 1 ./ x;
plot(x, y)

subplot(1, 2, 2)
% Entropy vs cost; theoretical bound: 1 + log(C)
scatter(cost1_p,  entr1_p,  'ko'); hold on;
scatter(cost1_ps, entr1_ps, 'bo'); hold on;
y = 1 + log(x);
plot(x, y)

%% -----------------------------------------------------------------------
%% SECTION 4: Figure – mean lifetime CV vs ICE
%% -----------------------------------------------------------------------
figure;
plot([mean(cv1_l),   mean(cv1_l_s);   mean(cv2_l),   mean(cv2_l_s)], ...
     [mean(entr1_l) /mean(cost1_p),  mean(entr1_ls)/mean(cost1_ps); ...
      mean(entr2_l) /mean(cost2_p),  mean(entr2_ls)/mean(cost2_ps)]);
xlabel('Mean CV_l')
ylabel('information / calibrated cost proxy')

%% -----------------------------------------------------------------------
%% SECTION 5: Figure – population mean rate vs CV / sparseness / CV-SS
%% -----------------------------------------------------------------------
figure;

subplot(1, 3, 1)
% Population CV vs mean firing rate
plot(mean(mrate_v1_s), cv_v1_s, 'b*'); hold on;
plot(mean(mrate_v2_s), cv_v2_s, 'b+'); hold on;
plot(mean(mrate_v1),   cv_v1,   'r*'); hold on;
plot(mean(mrate_v2),   cv_v2,   'r+'); hold on;
ylim([0.5 2]);  xlim([0 22])

subplot(1, 3, 2)
% Population sparseness vs mean firing rate
plot(mean(mrate_v1_s), ss_v1_s, 'b*'); hold on;
plot(mean(mrate_v2_s), ss_v2_s, 'b+'); hold on;
plot(mean(mrate_v1),   ss_v1,   'r*'); hold on;
plot(mean(mrate_v2),   ss_v2,   'r+'); hold on;
ylim([0.2 1]);  xlim([0 22])

subplot(1, 3, 3)
% CV vs sparseness; theoretical curve: SS = CV^2 / (CV^2 + 1)
x = 0:0.01:3;
y = x.^2 ./ (x.^2 + 1);
plot(cv_v1_s, ss_v1_s, 'b*'); hold on;
plot(cv_v2_s, ss_v2_s, 'b+'); hold on;
plot(cv_v1,   ss_v1,   'r*'); hold on;
plot(cv_v2,   ss_v2,   'r+'); hold on;
plot(x, y);  xlim([0 3])

%% -----------------------------------------------------------------------
%% SECTION 6: Figure – entropy vs cost (all trials, V1)
%% -----------------------------------------------------------------------
x = 0:0.1:30;

figure;

subplot(1, 2, 1)
% Entropy vs cost
scatter(cost1_p,  entr1_p,  'ko'); hold on;
scatter(cost1_ps, entr1_ps, 'bo'); hold on;
y = 1 + log(x);
plot(x, y)
xlim([0 30])

subplot(1, 2, 2)
% ICE vs cost
scatter(cost1_p,  entr1_p  ./ (dur_driven * cost1_p), 'ko'); hold on;
scatter(cost1_ps, entr1_ps ./ (dur_spon * cost1_ps), 'bo'); hold on;
y = 1 ./ x;
plot(x, y)
ylim([0 1])

%% -----------------------------------------------------------------------
%% SECTION 7: Figure – bar plots of mean CV and sparseness (V1)
%% -----------------------------------------------------------------------
figure;

subplot(1, 2, 1)
% Mean CV: evoked vs spontaneous
model_series = [mean(cv_v1); mean(cv_v1_s)];
model_error  = [std(cv_v1).^2; std(cv_v1_s).^2];
b = bar(model_series, 'grouped');
hold on
[ngroups, nbars] = size(model_series);
x = nan(nbars, ngroups);
for i = 1:nbars
    x(i,:) = b(i).XEndPoints;
end
errorbar(x', model_series, model_error, 'k', 'linestyle', 'none');
ylabel('cv')
xticklabels({'evoked', 'spontaneous'})
ylim([1 1.2])

subplot(1, 2, 2)
% Mean sparseness: evoked vs spontaneous
model_series = [mean(ss_v1); mean(ss_v1_s)];
model_error  = [std(ss_v1).^2; std(ss_v1_s).^2];
b = bar(model_series, 'grouped');
hold on
[ngroups, nbars] = size(model_series);
x = nan(nbars, ngroups);
for i = 1:nbars
    x(i,:) = b(i).XEndPoints;
end
errorbar(x', model_series, model_error, 'k', 'linestyle', 'none');
ylabel('sparseness')
xticklabels({'evoked', 'spontaneous'})
ylim([0.5 0.54])

%% -----------------------------------------------------------------------
%% SECTION 8: Statistical tests (Wilcoxon rank-sum)
%% -----------------------------------------------------------------------
% Test whether CV differs between driven and spontaneous conditions (V1)
ranksum(cv_v1, cv_v1_s)

%% -----------------------------------------------------------------------
%% SECTION 9: Per-Neuron MI and Entropy Visualization
%% -----------------------------------------------------------------------
% Separate analysis of single neurons:
%   - Distribution of MI across all neurons
%   - Per-neuron H(r) comparison: Spon vs Driven
%   - Per-neuron encoding efficiency

figure('Name', 'Per-Neuron MI Analysis', 'NumberTitle', 'off', 'Position', [100 100 1400 900]);

% V1: MI distribution
subplot(2, 3, 1)
histogram(mi_v1_perNeuron, 20, 'FaceColor', [0.2 0.6 0.9], 'EdgeColor', 'k');
xlabel('MI [bits]');
ylabel('# Neurons');
title(sprintf('V1: Per-Neuron MI Distribution (n=%d)', 159));
hold on
plot([0.01 0.01], ylim, 'r--', 'LineWidth', 2, 'DisplayName', 'Threshold=0.01');
legend;
grid on

% V1: H(r) comparison spon vs driven
subplot(2, 3, 2)
scatter(Hr_v1_perNeuron_spon, Hr_v1_perNeuron_driven, 50, 'o', ...
    'MarkerFaceColor', [0.2 0.6 0.9], 'MarkerEdgeColor', 'k', 'LineWidth', 1);
hold on
lim_max = max(max(Hr_v1_perNeuron_spon), max(Hr_v1_perNeuron_driven));
plot([0 lim_max], [0 lim_max], 'r--', 'LineWidth', 1);
xlabel('H(r) | Spontaneous [bits]');
ylabel('H(r) | Driven [bits]');
title('V1: Response Entropy (Spon vs Driven)');
axis square
grid on

% V1: H(r) driven vs H(r|s) driven (noise comparison)
subplot(2, 3, 3)
scatter(Hr_v1_perNeuron_driven, Hrs_v1_perNeuron, 50, 'o', ...
    'MarkerFaceColor', [0.8 0.4 0.2], 'MarkerEdgeColor', 'k', 'LineWidth', 1);
hold on
lim_max = max(max(Hr_v1_perNeuron_driven), max(Hrs_v1_perNeuron));
plot([0 lim_max], [0 lim_max], 'r--', 'LineWidth', 1);
xlabel('H(r) | Driven [bits]');
ylabel('H(r|s) | Driven [bits]');
title('V1: Response vs Noise Entropy');
axis square
grid on

% V2: MI distribution
subplot(2, 3, 4)
histogram(mi_v2_perNeuron, 15, 'FaceColor', [0.9 0.4 0.2], 'EdgeColor', 'k');
xlabel('MI [bits]');
ylabel('# Neurons');
title(sprintf('V2: Per-Neuron MI Distribution (n=%d)', 25));
hold on
plot([0.01 0.01], ylim, 'r--', 'LineWidth', 2, 'DisplayName', 'Threshold=0.01');
legend;
grid on

% V2: H(r) comparison spon vs driven
subplot(2, 3, 5)
scatter(Hr_v2_perNeuron_spon, Hr_v2_perNeuron_driven, 50, 'o', ...
    'MarkerFaceColor', [0.9 0.4 0.2], 'MarkerEdgeColor', 'k', 'LineWidth', 1);
hold on
lim_max = max(max(Hr_v2_perNeuron_spon), max(Hr_v2_perNeuron_driven));
plot([0 lim_max], [0 lim_max], 'r--', 'LineWidth', 1);
xlabel('H(r) | Spontaneous [bits]');
ylabel('H(r) | Driven [bits]');
title('V2: Response Entropy (Spon vs Driven)');
axis square
grid on

% V2: H(r) driven vs H(r|s) driven
subplot(2, 3, 6)
scatter(Hr_v2_perNeuron_driven, Hrs_v2_perNeuron, 50, 'o', ...
    'MarkerFaceColor', [0.8 0.4 0.2], 'MarkerEdgeColor', 'k', 'LineWidth', 1);
hold on
lim_max = max(max(Hr_v2_perNeuron_driven), max(Hrs_v2_perNeuron));
plot([0 lim_max], [0 lim_max], 'r--', 'LineWidth', 1);
xlabel('H(r) | Driven [bits]');
ylabel('H(r|s) | Driven [bits]');
title('V2: Response vs Noise Entropy');
axis square
grid on

sgtitle('Per-Neuron Entropy and MI Analysis');

% Additional detailed comparison figure
figure('Name', 'Per-Neuron Population Comparison', 'NumberTitle', 'off', 'Position', [100 100 1400 600]);

% V1 and V2 comparison: MI and entropy statistics
subplot(1, 3, 1)
% MI comparison
mi_stats_v1 = [mean(mi_v1_perNeuron), std(mi_v1_perNeuron), min(mi_v1_perNeuron), max(mi_v1_perNeuron)];
mi_stats_v2 = [mean(mi_v2_perNeuron), std(mi_v2_perNeuron), min(mi_v2_perNeuron), max(mi_v2_perNeuron)];
x_pos = [1 2];
y_data = [mi_stats_v1(1), mi_stats_v2(1)];
e_data = [mi_stats_v1(2), mi_stats_v2(2)];
bar(x_pos, y_data, 'FaceColor', [0.5 0.5 0.5]);
hold on
errorbar(x_pos, y_data, e_data, 'k', 'LineStyle', 'none', 'LineWidth', 2, 'CapSize', 8);
set(gca, 'XTickLabel', {'V1', 'V2'});
ylabel('Mean ±Std MI [bits]');
title('Per-Neuron MI (Mean)');
grid on

subplot(1, 3, 2)
% H(r) spon vs driven comparison
h_spon_v1 = mean(Hr_v1_perNeuron_spon);
h_driven_v1 = mean(Hr_v1_perNeuron_driven);
h_spon_v2 = mean(Hr_v2_perNeuron_spon);
h_driven_v2 = mean(Hr_v2_perNeuron_driven);

x_pos = [1 1.4 2 2.4];
y_data = [h_spon_v1, h_driven_v1, h_spon_v2, h_driven_v2];
colors = [0 0 0.5; 1 0 0; 0 0 0.5; 1 0 0];  % spon=blue, driven=red
bar(x_pos, y_data);
for i = 1:4
    h = bar(x_pos(i), y_data(i));
    if mod(i, 2) == 1
        h.FaceColor = [0.3 0.3 0.8];  % spon: blue
    else
        h.FaceColor = [0.8 0.3 0.3];  % driven: red
    end
end
set(gca, 'XTickLabel', {'', 'V1', '', 'V2', ''});
ylabel('Mean H(r) [bits]');
title('Per-Neuron H(r): Spon (blue) vs Driven (red)');
legend('Spon', 'Driven', 'Location', 'northeast');
grid on

subplot(1, 3, 3)
% H(r|s) noise entropy comparison
hrs_v1 = mean(Hrs_v1_perNeuron);
hrs_v2 = mean(Hrs_v2_perNeuron);
h_driven_v1 = mean(Hr_v1_perNeuron_driven);
h_driven_v2 = mean(Hr_v2_perNeuron_driven);
mi_mean_v1 = mean(mi_v1_perNeuron);
mi_mean_v2 = mean(mi_v2_perNeuron);

x_pos = [1 2];
y_noise = [hrs_v1, hrs_v2];
y_info = [mi_mean_v1, mi_mean_v2];

% Stacked bar chart: noise + info
data = [y_noise; y_info]';
h_bars = bar(x_pos, data, 'stacked');
h_bars(1).FaceColor = [0.3 0.8 0.3];
h_bars(1).DisplayName = 'H(r|s) Noise';
h_bars(2).FaceColor = [0.8 0.3 0.3];
h_bars(2).DisplayName = 'MI Info';
set(gca, 'XTickLabel', {'V1', 'V2'});
ylabel('Entropy [bits]');
title('Per-Neuron Decomposition: H(r)=Noise+Info');
legend;
grid on

sgtitle('Population Comparison: Per-Neuron Statistics');

fprintf('  Per-neuron MI summary:\n');
fprintf('    V1: mean=%.4f, std=%.4f, %% selective=%.1f%%\n', ...
    mean(mi_v1_perNeuron), std(mi_v1_perNeuron), 100*sum(mi_v1_perNeuron > 0.01)/159);
fprintf('    V2: mean=%.4f, std=%.4f, %% selective=%.1f%%\n\n', ...
    mean(mi_v2_perNeuron), std(mi_v2_perNeuron), 100*sum(mi_v2_perNeuron > 0.01)/25);

%% ===== Compute per-neuron metabolic cost for entropy-cost analysis =====
% For each neuron, compute metabolic cost = r + firing_rate
% This allows plotting per-neuron entropy vs cost curves

cost1_p_perNeuron = [];      % V1 driven per-neuron
cost1_ps_perNeuron = [];     % V1 spontaneous per-neuron
entr1_p_perNeuron = [];      % V1 driven entropy per-neuron
entr1_ps_perNeuron = [];     % V1 spontaneous entropy per-neuron

cost2_p_perNeuron = [];      % V2 driven per-neuron
cost2_ps_perNeuron = [];     % V2 spontaneous per-neuron
entr2_p_perNeuron = [];      % V2 driven entropy per-neuron
entr2_ps_perNeuron = [];     % V2 spontaneous entropy per-neuron

% V1 per-neuron cost and entropy
for i = 1:nt
    spikes_v1 = neuralData.spikeRasters{i, 1};

    if size(spikes_v1, 2) == 1500
        dur = 1.5;  % spontaneous

        for neuron = 1:min(size(spikes_v1, 1), 159)
            firing_rate = sum(spikes_v1(neuron, :)) / dur;
            cost_neuron = C1 + C2 * firing_rate;
            cost1_ps_perNeuron = [cost1_ps_perNeuron, cost_neuron];

            % Compute per-trial entropy for this neuron
            spike_times = find(spikes_v1(neuron, :)) * dur / size(spikes_v1, 2);
            if length(spike_times) >= 3
                isi = diff(spike_times);
                rate_isi = 1000 ./ isi;
                [n_tmp, ~] = histcounts(rate_isi, 0:1:(max(rate_isi)+1));
                p_tmp = n_tmp / sum(n_tmp);
                p_tmp = p_tmp(p_tmp > 0);
                entr1_ps_perNeuron = [entr1_ps_perNeuron, -sum(p_tmp .* log2(p_tmp))];
            else
                entr1_ps_perNeuron = [entr1_ps_perNeuron, 0];
            end
        end
    else
        dur = 1.28;  % driven

        for neuron = 1:min(size(spikes_v1, 1), 159)
            firing_rate = sum(spikes_v1(neuron, :)) / dur;
            cost_neuron = C1 + C2 * firing_rate;
            cost1_p_perNeuron = [cost1_p_perNeuron, cost_neuron];

            % Compute per-trial entropy for this neuron
            spike_times = find(spikes_v1(neuron, :)) * dur / size(spikes_v1, 2);
            if length(spike_times) >= 3
                isi = diff(spike_times);
                rate_isi = 1000 ./ isi;
                [n_tmp, ~] = histcounts(rate_isi, 0:1:(max(rate_isi)+1));
                p_tmp = n_tmp / sum(n_tmp);
                p_tmp = p_tmp(p_tmp > 0);
                entr1_p_perNeuron = [entr1_p_perNeuron, -sum(p_tmp .* log2(p_tmp))];
            else
                entr1_p_perNeuron = [entr1_p_perNeuron, 0];
            end
        end
    end
end

% V2 per-neuron cost and entropy
for i = 1:nt
    spikes_v2 = neuralData.spikeRasters{i, 2};

    if size(spikes_v2, 2) == 1500
        dur = 1.5;  % spontaneous

        for neuron = 1:min(size(spikes_v2, 1), 25)
            firing_rate = sum(spikes_v2(neuron, :)) / dur;
            cost_neuron = C1 + C2 * firing_rate;
            cost2_ps_perNeuron = [cost2_ps_perNeuron, cost_neuron];

            spike_times = find(spikes_v2(neuron, :)) * dur / size(spikes_v2, 2);
            if length(spike_times) >= 3
                isi = diff(spike_times);
                rate_isi = 1000 ./ isi;
                [n_tmp, ~] = histcounts(rate_isi, 0:1:(max(rate_isi)+1));
                p_tmp = n_tmp / sum(n_tmp);
                p_tmp = p_tmp(p_tmp > 0);
                entr2_ps_perNeuron = [entr2_ps_perNeuron, -sum(p_tmp .* log2(p_tmp))];
            else
                entr2_ps_perNeuron = [entr2_ps_perNeuron, 0];
            end
        end
    else
        dur = 1.28;  % driven

        for neuron = 1:min(size(spikes_v2, 1), 25)
            firing_rate = sum(spikes_v2(neuron, :)) / dur;
            cost_neuron = C1 + C2 * firing_rate;
            cost2_p_perNeuron = [cost2_p_perNeuron, cost_neuron];

            spike_times = find(spikes_v2(neuron, :)) * dur / size(spikes_v2, 2);
            if length(spike_times) >= 3
                isi = diff(spike_times);
                rate_isi = 1000 ./ isi;
                [n_tmp, ~] = histcounts(rate_isi, 0:1:(max(rate_isi)+1));
                p_tmp = n_tmp / sum(n_tmp);
                p_tmp = p_tmp(p_tmp > 0);
                entr2_p_perNeuron = [entr2_p_perNeuron, -sum(p_tmp .* log2(p_tmp))];
            else
                entr2_p_perNeuron = [entr2_p_perNeuron, 0];
            end
        end
    end
end

%% -----------------------------------------------------------------------
%% SECTION 9b: Spatial-Temporal Decomposition of MI
%% -----------------------------------------------------------------------
%
% This section compares three complementary analysis levels:
%   1. SPATIAL-ONLY: Population average rate per trial (1 value/trial)
%   2. TEMPORAL: Per-neuron ISI-based rates (individual neuron resolution)
%   3. SPATIAL+TEMPORAL: All neurons × all ISI time points (full resolution)
%
% This reveals whether temporal structure (ISI-based rates) contributes
% substantially to stimulus information, or if population average suffices.
%

fprintf('\n=== Spatial-Temporal MI Decomposition ===\n');

% V1 Summary
fprintf('V1 MI Summary:\n');
fprintf('  Spatial-only MI:     %.4f bits\n', MI_spatial_v1);
fprintf('  Per-neuron MI (mean):%.4f bits\n', mean(mi_v1_perNeuron(mi_v1_perNeuron > 0)));
fprintf('  Population MI (full):%.4f bits\n', MI_v1);
fprintf('  Ratio (Full/Spatial):%.2f x\n', MI_v1 / max(MI_spatial_v1, 1e-6));

% V2 Summary  
fprintf('V2 MI Summary:\n');
fprintf('  Spatial-only MI:     %.4f bits\n', MI_spatial_v2);
fprintf('  Per-neuron MI (mean):%.4f bits\n', mean(mi_v2_perNeuron(mi_v2_perNeuron > 0)));
fprintf('  Population MI (full):%.4f bits\n', MI_v2);
fprintf('  Ratio (Full/Spatial):%.2f x\n', MI_v2 / max(MI_spatial_v2, 1e-6));

% Visualization: Three-level MI comparison
figure('Name', 'Spatial-Temporal MI Decomposition', 'NumberTitle', 'off', 'Position', [100 100 1400 600]);

% V1: Three levels of MI
subplot(1, 2, 1)
mi_levels_v1 = [MI_spatial_v1, mean(mi_v1_perNeuron), MI_v1];
level_names = {'Spatial-Only', 'Per-Neuron\n(mean)', 'Spatial+\nTemporal'};
bars_v1 = bar(mi_levels_v1, 'FaceColor', [0.2 0.6 0.9], 'EdgeColor', 'k', 'LineWidth', 1.5);
set(gca, 'XTickLabel', level_names, 'FontSize', 11);
ylabel('MI [bits]', 'FontSize', 11);
title('V1: MI across Spatial-Temporal Levels', 'FontSize', 12);
grid on
ylim_max_v1 = max(mi_levels_v1(~isnan(mi_levels_v1) & ~isinf(mi_levels_v1)));
if isempty(ylim_max_v1) || ylim_max_v1 <= 0
    ylim_max_v1 = 1;
end
ylim([0, ylim_max_v1*1.2]);
% Add value labels on bars
for i = 1:length(mi_levels_v1)
    if ~isnan(mi_levels_v1(i)) && ~isinf(mi_levels_v1(i))
        text(i, mi_levels_v1(i) + ylim_max_v1*0.02, sprintf('%.3f', mi_levels_v1(i)), ...
            'HorizontalAlignment', 'center', 'FontSize', 10);
    end
end

% V2: Three levels of MI
subplot(1, 2, 2)
mi_levels_v2 = [MI_spatial_v2, mean(mi_v2_perNeuron), MI_v2];
bars_v2 = bar(mi_levels_v2, 'FaceColor', [0.9 0.4 0.2], 'EdgeColor', 'k', 'LineWidth', 1.5);
set(gca, 'XTickLabel', level_names, 'FontSize', 11);
ylabel('MI [bits]', 'FontSize', 11);
title('V2: MI across Spatial-Temporal Levels', 'FontSize', 12);
grid on
ylim_max_v2 = max(mi_levels_v2(~isnan(mi_levels_v2) & ~isinf(mi_levels_v2)));
if isempty(ylim_max_v2) || ylim_max_v2 <= 0
    ylim_max_v2 = 1;
end
ylim([0, ylim_max_v2*1.2]);
% Add value labels on bars
for i = 1:length(mi_levels_v2)
    if ~isnan(mi_levels_v2(i)) && ~isinf(mi_levels_v2(i))
        text(i, mi_levels_v2(i) + ylim_max_v2*0.02, sprintf('%.3f', mi_levels_v2(i)), ...
            'HorizontalAlignment', 'center', 'FontSize', 10);
    end
end

sgtitle('Spatial-Temporal Decomposition: MI Contribution at Each Level', 'FontSize', 13);

%% Compute Spatial-Only entropy and cost data
% Spatial-only: For each trial, compute entropy of the distribution of
% neuron firing rates (one rate per neuron from spike counts / duration).
% This captures the spatial diversity of activity across the neural population.

% Number of histogram bins (square root rule, appropriate for N neurons)
nBins_v1 = max(2, round(sqrt(159)));  % ~13 bins for 159 V1 neurons
nBins_v2 = max(2, round(sqrt(25)));   % ~5 bins for 25 V2 neurons

% V1 driven: spatial-only (per-trial entropy)
cost1_p_spatial = [];
entr1_p_spatial = [];
for trial = 1:length(rast_v1_d)
    spikes = rast_v1_d{trial};
    dur_trial = 1.28;
    rates = sum(spikes, 2)' / dur_trial;  % firing rate per neuron (1×159)
    cost1_p_spatial = [cost1_p_spatial, C1 + C2 * mean(rates)];
    if any(rates > 0)
        [n_r, ~] = histcounts(rates, nBins_v1);
        p_r = n_r / sum(n_r);
        p_r = p_r(p_r > 0);
        entr1_p_spatial = [entr1_p_spatial, -sum(p_r .* log2(p_r))];
    else
        entr1_p_spatial = [entr1_p_spatial, 0];
    end
end

% V1 spontaneous: spatial-only (per-trial entropy)
cost1_ps_spatial = [];
entr1_ps_spatial = [];
for trial = 1:length(rast_v1_s)
    spikes = rast_v1_s{trial};
    dur_trial = 1.5;
    rates = sum(spikes, 2)' / dur_trial;
    cost1_ps_spatial = [cost1_ps_spatial, C1 + C2 * mean(rates)];
    if any(rates > 0)
        [n_r, ~] = histcounts(rates, nBins_v1);
        p_r = n_r / sum(n_r);
        p_r = p_r(p_r > 0);
        entr1_ps_spatial = [entr1_ps_spatial, -sum(p_r .* log2(p_r))];
    else
        entr1_ps_spatial = [entr1_ps_spatial, 0];
    end
end

% V2 driven: spatial-only (per-trial entropy)
cost2_p_spatial = [];
entr2_p_spatial = [];
for trial = 1:length(rast_v2_d)
    spikes = rast_v2_d{trial};
    dur_trial = 1.28;
    rates = sum(spikes, 2)' / dur_trial;
    cost2_p_spatial = [cost2_p_spatial, C1 + C2 * mean(rates)];
    if any(rates > 0)
        [n_r, ~] = histcounts(rates, nBins_v2);
        p_r = n_r / sum(n_r);
        p_r = p_r(p_r > 0);
        entr2_p_spatial = [entr2_p_spatial, -sum(p_r .* log2(p_r))];
    else
        entr2_p_spatial = [entr2_p_spatial, 0];
    end
end

% V2 spontaneous: spatial-only (per-trial entropy)
cost2_ps_spatial = [];
entr2_ps_spatial = [];
for trial = 1:length(rast_v2_s)
    spikes = rast_v2_s{trial};
    dur_trial = 1.5;
    rates = sum(spikes, 2)' / dur_trial;
    cost2_ps_spatial = [cost2_ps_spatial, C1 + C2 * mean(rates)];
    if any(rates > 0)
        [n_r, ~] = histcounts(rates, nBins_v2);
        p_r = n_r / sum(n_r);
        p_r = p_r(p_r > 0);
        entr2_ps_spatial = [entr2_ps_spatial, -sum(p_r .* log2(p_r))];
    else
        entr2_ps_spatial = [entr2_ps_spatial, 0];
    end
end

% Additional detailed comparison: H(r) and H(r|s)
figure('Name', 'Entropy Components: 3-Level Comparison', 'NumberTitle', 'off', 'Position', [100 100 1200 800]);

% V1: H(r) comparison
subplot(2, 2, 1)
hr_levels_v1 = [mean(entr1_p_spatial), mean(Hr_v1_perNeuron_driven), Hr];
bars = bar(hr_levels_v1, 'FaceColor', [0.2 0.6 0.9], 'EdgeColor', 'k', 'LineWidth', 1.5);
set(gca, 'XTickLabel', level_names, 'FontSize', 10);
ylabel('H(r) [bits]', 'FontSize', 10);
title('V1: Response Entropy', 'FontSize', 11);
grid on
hr_max_v1 = max(hr_levels_v1(~isnan(hr_levels_v1) & ~isinf(hr_levels_v1)));
if isempty(hr_max_v1) || hr_max_v1 <= 0, hr_max_v1 = 1; end
for i = 1:length(hr_levels_v1)
    if ~isnan(hr_levels_v1(i)) && ~isinf(hr_levels_v1(i))
        text(i, hr_levels_v1(i) + hr_max_v1*0.02, sprintf('%.3f', hr_levels_v1(i)), ...
            'HorizontalAlignment', 'center', 'FontSize', 9);
    end
end

% V1: H(r|s) comparison
subplot(2, 2, 2)
hrs_levels_v1 = [Hrs_spatial_v1, mean(Hrs_v1_perNeuron), Hrs];
bars = bar(hrs_levels_v1, 'FaceColor', [0.6 0.2 0.9], 'EdgeColor', 'k', 'LineWidth', 1.5);
set(gca, 'XTickLabel', level_names, 'FontSize', 10);
ylabel('H(r|s) [bits]', 'FontSize', 10);
title('V1: Conditional Entropy (Noise)', 'FontSize', 11);
grid on
hrs_max_v1 = max(hrs_levels_v1(~isnan(hrs_levels_v1) & ~isinf(hrs_levels_v1)));
if isempty(hrs_max_v1) || hrs_max_v1 <= 0, hrs_max_v1 = 1; end
for i = 1:length(hrs_levels_v1)
    if ~isnan(hrs_levels_v1(i)) && ~isinf(hrs_levels_v1(i))
        text(i, hrs_levels_v1(i) + hrs_max_v1*0.02, sprintf('%.3f', hrs_levels_v1(i)), ...
            'HorizontalAlignment', 'center', 'FontSize', 9);
    end
end

% V2: H(r) comparison
subplot(2, 2, 3)
hr_levels_v2 = [mean(entr2_p_spatial), mean(Hr_v2_perNeuron_driven), Hr_v2];
bars = bar(hr_levels_v2, 'FaceColor', [0.9 0.4 0.2], 'EdgeColor', 'k', 'LineWidth', 1.5);
set(gca, 'XTickLabel', level_names, 'FontSize', 10);
ylabel('H(r) [bits]', 'FontSize', 10);
title('V2: Response Entropy', 'FontSize', 11);
grid on
hr_max_v2 = max(hr_levels_v2(~isnan(hr_levels_v2) & ~isinf(hr_levels_v2)));
if isempty(hr_max_v2) || hr_max_v2 <= 0, hr_max_v2 = 1; end
for i = 1:length(hr_levels_v2)
    if ~isnan(hr_levels_v2(i)) && ~isinf(hr_levels_v2(i))
        text(i, hr_levels_v2(i) + hr_max_v2*0.02, sprintf('%.3f', hr_levels_v2(i)), ...
            'HorizontalAlignment', 'center', 'FontSize', 9);
    end
end

% V2: H(r|s) comparison
subplot(2, 2, 4)
hrs_levels_v2 = [Hrs_spatial_v2, mean(Hrs_v2_perNeuron), Hrs_v2];
bars = bar(hrs_levels_v2, 'FaceColor', [0.6 0.2 0.9], 'EdgeColor', 'k', 'LineWidth', 1.5);
set(gca, 'XTickLabel', level_names, 'FontSize', 10);
ylabel('H(r|s) [bits]', 'FontSize', 10);
title('V2: Conditional Entropy (Noise)', 'FontSize', 11);
grid on
hrs_max_v2 = max(hrs_levels_v2(~isnan(hrs_levels_v2) & ~isinf(hrs_levels_v2)));
if isempty(hrs_max_v2) || hrs_max_v2 <= 0, hrs_max_v2 = 1; end
for i = 1:length(hrs_levels_v2)
    if ~isnan(hrs_levels_v2(i)) && ~isinf(hrs_levels_v2(i))
        text(i, hrs_levels_v2(i) + hrs_max_v2*0.02, sprintf('%.3f', hrs_levels_v2(i)), ...
            'HorizontalAlignment', 'center', 'FontSize', 9);
    end
end

sgtitle('Entropy Components at Each Spatial-Temporal Level', 'FontSize', 13);

%% -----------------------------------------------------------------------
%% SECTION 10: Entropy-Cost Tradeoff (Three-Level: Spatial-Only, Per-Neuron, Spatial+Temporal)
%% -----------------------------------------------------------------------
%
% This section expands the entropy-cost analysis to three complementary levels:
%   - Spatial-Only: Population average rate per trial
%   - Per-Neuron: Individual neuron ISI-based rates
%   - Spatial+Temporal: All neurons × all ISI points (full resolution)
%
% Theoretical relationship: H(r) = 1 + log(C) where C is metabolic cost
% Information-Cost Efficiency (ICE): ICE = MI/C = H(r)/C (for information efficiency)
%


%% V1: Three-Level Analysis (Spatial-Only, Per-Neuron, Spatial+Temporal)
figure('Name', 'Entropy-Cost Analysis (Three Spatial-Temporal Levels)', 'NumberTitle', 'off', 'Position', [100 100 1600 700]);
subplot(2, 3, 1)
% Level 1: Spatial-Only
scatter(cost1_p_spatial, entr1_p_spatial, 60, 's', 'filled', 'MarkerFaceColor', [0.5 0.5 0.5], 'DisplayName', 'Spatial-O Driv');
hold on;
scatter(cost1_ps_spatial, entr1_ps_spatial, 50, 's', 'filled', 'MarkerFaceColor', [0.8 0.8 0.8], 'DisplayName', 'Spatial-O Spon');

% Level 2: Per-Neuron (individual neurons)
scatter(cost1_p_perNeuron, entr1_p_perNeuron, 25, 'r^', 'filled', 'DisplayName', 'Per-Neuron Driv');
scatter(cost1_ps_perNeuron, entr1_ps_perNeuron, 20, 'c^', 'filled', 'DisplayName', 'Per-Neuron Spon');

% Level 3: Spatial+Temporal (population, all ISIs)
scatter(cost1_p, entr1_p, 80, 'ko', 'filled', 'DisplayName', 'Spatial+Temp Driv');
scatter(cost1_ps, entr1_ps, 65, 'bo', 'filled', 'DisplayName', 'Spatial+Temp Spon');

x = 0:0.1:30;
y = 1 + log(x);
plot(x, y, 'k--', 'LineWidth', 2, 'DisplayName', 'Theory H=1+log(C)');
xlabel('Metabolic Cost');
ylabel('Entropy [bits]');
title('V1: Entropy vs Cost (Three Levels)');
legend('Location', 'best', 'FontSize', 7);
% xlim([0 30]);
grid on

subplot(2, 3, 2)
% Level 1: Spatial-Only (ICE)
scatter(cost1_p_spatial, entr1_p_spatial./(dur_driven*cost1_p_spatial), 60, 's', 'filled', 'MarkerFaceColor', [0.5 0.5 0.5], 'DisplayName', 'Spatial-O Driv');
hold on;
scatter(cost1_ps_spatial, entr1_ps_spatial./(dur_spon*cost1_ps_spatial), 50, 's', 'filled', 'MarkerFaceColor', [0.8 0.8 0.8], 'DisplayName', 'Spatial-O Spon');

% Level 2: Per-Neuron (ICE)
scatter(cost1_p_perNeuron, entr1_p_perNeuron./(dur_driven*cost1_p_perNeuron), 25, 'r^', 'filled', 'DisplayName', 'Per-Neuron Driv');
scatter(cost1_ps_perNeuron, entr1_ps_perNeuron./(dur_spon*cost1_ps_perNeuron), 20, 'c^', 'filled', 'DisplayName', 'Per-Neuron Spon');

% Level 3: Spatial+Temporal (ICE)
scatter(cost1_p, entr1_p./(dur_driven*cost1_p), 80, 'ko', 'filled', 'DisplayName', 'Spatial+Temp Driv');
scatter(cost1_ps, entr1_ps./(dur_spon*cost1_ps), 65, 'bo', 'filled', 'DisplayName', 'Spatial+Temp Spon');

y = 1 ./ x;
plot(x, y, 'k--', 'LineWidth', 2, 'DisplayName', 'Theory ICE=1/C');
xlabel('Metabolic Cost');
ylabel('information / calibrated cost proxy');
title('V1: ICE vs Cost (Three Levels)');
legend('Location', 'best', 'FontSize', 7);
% xlim([0 30]);
% ylim([0 1]);
grid on

subplot(2, 3, 3)
% V1: Mean comparison (three levels)
% Driven: stacked H(r|s) + MI;  Spon: Hr only
hold on
% Driven stacked bars (positions 1, 1.25, 1.5)
x_driv = [1, 1.25, 1.5];
hrs_driv_v1 = [Hrs_spatial_v1, mean(Hrs_v1_perNeuron), Hrs];
mi_driv_v1  = [MI_spatial_v1,  mean(mi_v1_perNeuron),  MI_v1];
colors_hrs = {[0.5 0.5 0.8], [0.8 0.5 0.5], [0.4 0.4 0.4]};
colors_mi  = {[0.3 0.3 1],   [1 0.3 0.3],   [0 0 0]};
for bi = 1:3
    bar(x_driv(bi), hrs_driv_v1(bi), 0.2, 'FaceColor', colors_hrs{bi}, 'EdgeColor', 'k');
    bar(x_driv(bi), mi_driv_v1(bi), 0.2, 'FaceColor', colors_mi{bi}, 'EdgeColor', 'k', 'BaseValue', hrs_driv_v1(bi));
end
% Spon bars (positions 2, 2.25, 2.5)
x_spon = [2, 2.25, 2.5];
hr_spon_v1 = [mean(entr1_ps_spatial), mean(entr1_ps_perNeuron), mean(entr1_ps)];
colors_spon = {[0.8 0.8 0.8], [0.7 0.7 1], [0.3 0.3 1]};
for bi = 1:3
    bar(x_spon(bi), hr_spon_v1(bi), 0.2, 'FaceColor', colors_spon{bi}, 'EdgeColor', 'k');
end
set(gca, 'XTick', [x_driv, x_spon]);
set(gca, 'XTickLabel', {'S-O', 'P-N', 'S+T', 'S-O', 'P-N', 'S+T'});
set(gca, 'XTickLabelRotation', 45);
ylabel('Entropy [bits]');
title('V1: Three-Level (Driven=H(r|s)+MI, Spon=Hr)');
yl = ylim;
plot([1.75 1.75], yl, 'k-', 'LineWidth', 2);
text(1.25, yl(2)*0.95, 'Driven', 'FontSize', 9, 'HorizontalAlignment', 'center');
text(2.25, yl(2)*0.95, 'Spon', 'FontSize', 9, 'HorizontalAlignment', 'center');
grid on

%% V2: Three-Level Analysis
subplot(2, 3, 4)
% Level 1: Spatial-Only
scatter(cost2_p_spatial, entr2_p_spatial, 60, 's', 'filled', 'MarkerFaceColor', [0.5 0.5 0.5], 'DisplayName', 'Spatial-O Driv');
hold on;
scatter(cost2_ps_spatial, entr2_ps_spatial, 50, 's', 'filled', 'MarkerFaceColor', [0.8 0.8 0.8], 'DisplayName', 'Spatial-O Spon');

% Level 2: Per-Neuron
scatter(cost2_p_perNeuron, entr2_p_perNeuron, 25, 'r^', 'filled', 'DisplayName', 'Per-Neuron Driv');
scatter(cost2_ps_perNeuron, entr2_ps_perNeuron, 20, 'c^', 'filled', 'DisplayName', 'Per-Neuron Spon');

% Level 3: Spatial+Temporal
scatter(cost2_p, entr2_p, 80, 'ko', 'filled', 'DisplayName', 'Spatial+Temp Driv');
scatter(cost2_ps, entr2_ps, 65, 'bo', 'filled', 'DisplayName', 'Spatial+Temp Spon');

x = 0:0.1:30;
y = 1 + log(x);
plot(x, y, 'k--', 'LineWidth', 2, 'DisplayName', 'Theory H=1+log(C)');
xlabel('Metabolic Cost');
ylabel('Entropy [bits]');
title('V2: Entropy vs Cost (Three Levels)');
legend('Location', 'best', 'FontSize', 7);
xlim([0 30]);
grid on

subplot(2, 3, 5)
% Level 1: Spatial-Only (ICE)
scatter(cost2_p_spatial, entr2_p_spatial./(dur_driven*cost2_p_spatial), 60, 's', 'filled', 'MarkerFaceColor', [0.5 0.5 0.5], 'DisplayName', 'Spatial-O Driv');
hold on;
scatter(cost2_ps_spatial, entr2_ps_spatial./(dur_spon*cost2_ps_spatial), 50, 's', 'filled', 'MarkerFaceColor', [0.8 0.8 0.8], 'DisplayName', 'Spatial-O Spon');

% Level 2: Per-Neuron (ICE)
scatter(cost2_p_perNeuron, entr2_p_perNeuron./(dur_driven*cost2_p_perNeuron), 25, 'r^', 'filled', 'DisplayName', 'Per-Neuron Driv');
scatter(cost2_ps_perNeuron, entr2_ps_perNeuron./(dur_spon*cost2_ps_perNeuron), 20, 'c^', 'filled', 'DisplayName', 'Per-Neuron Spon');

% Level 3: Spatial+Temporal (ICE)
scatter(cost2_p, entr2_p./(dur_driven*cost2_p), 80, 'ko', 'filled', 'DisplayName', 'Spatial+Temp Driv');
scatter(cost2_ps, entr2_ps./(dur_spon*cost2_ps), 65, 'bo', 'filled', 'DisplayName', 'Spatial+Temp Spon');

y = 1 ./ x;
plot(x, y, 'k--', 'LineWidth', 2, 'DisplayName', 'Theory ICE=1/C');
xlabel('Metabolic Cost');
ylabel('information / calibrated cost proxy');
title('V2: ICE vs Cost (Three Levels)');
legend('Location', 'best', 'FontSize', 7);
xlim([0 30]);
ylim([0 1]);
grid on

subplot(2, 3, 6)
% V2: Mean comparison (three levels)
% Driven: stacked H(r|s) + MI;  Spon: Hr only
hold on
% Driven stacked bars
x_driv = [1, 1.25, 1.5];
hrs_driv_v2 = [Hrs_spatial_v2, mean(Hrs_v2_perNeuron), Hrs_v2];
mi_driv_v2  = [MI_spatial_v2,  mean(mi_v2_perNeuron),  MI_v2];
colors_hrs = {[0.5 0.5 0.8], [0.8 0.5 0.5], [0.4 0.4 0.4]};
colors_mi  = {[0.3 0.3 1],   [1 0.3 0.3],   [0 0 0]};
for bi = 1:3
    bar(x_driv(bi), hrs_driv_v2(bi), 0.2, 'FaceColor', colors_hrs{bi}, 'EdgeColor', 'k');
    bar(x_driv(bi), mi_driv_v2(bi), 0.2, 'FaceColor', colors_mi{bi}, 'EdgeColor', 'k', 'BaseValue', hrs_driv_v2(bi));
end
% Spon bars
x_spon = [2, 2.25, 2.5];
hr_spon_v2 = [mean(entr2_ps_spatial), mean(entr2_ps_perNeuron), mean(entr2_ps)];
colors_spon = {[0.8 0.8 0.8], [0.7 0.7 1], [0.3 0.3 1]};
for bi = 1:3
    bar(x_spon(bi), hr_spon_v2(bi), 0.2, 'FaceColor', colors_spon{bi}, 'EdgeColor', 'k');
end
set(gca, 'XTick', [x_driv, x_spon]);
set(gca, 'XTickLabel', {'S-O', 'P-N', 'S+T', 'S-O', 'P-N', 'S+T'});
set(gca, 'XTickLabelRotation', 45);
ylabel('Entropy [bits]');
title('V2: Three-Level (Driven=H(r|s)+MI, Spon=Hr)');
yl = ylim;
plot([1.75 1.75], yl, 'k-', 'LineWidth', 2);
text(1.25, yl(2)*0.95, 'Driven', 'FontSize', 9, 'HorizontalAlignment', 'center');
text(2.25, yl(2)*0.95, 'Spon', 'FontSize', 9, 'HorizontalAlignment', 'center');
grid on

sgtitle('Entropy-Cost Tradeoff: Three Spatial-Temporal Levels (Squares=Spatial-Only, Triangles=Per-Neuron, Circles=Spatial+Temporal)', 'FontSize', 12);

fprintf('  Entropy-cost analysis (Three Spatial-Temporal Levels):\n');
fprintf('  V1 Driven:   Spatial-O=%.4f±%.4f, Per-Neuron=%.4f±%.4f, Spatial+T=%.4f±%.4f\n', ...
    mean(entr1_p_spatial), std(entr1_p_spatial), mean(entr1_p_perNeuron), std(entr1_p_perNeuron), mean(entr1_p), std(entr1_p));
fprintf('  V1 Spon:     Spatial-O=%.4f±%.4f, Per-Neuron=%.4f±%.4f, Spatial+T=%.4f±%.4f\n', ...
    mean(entr1_ps_spatial), std(entr1_ps_spatial), mean(entr1_ps_perNeuron), std(entr1_ps_perNeuron), mean(entr1_ps), std(entr1_ps));
fprintf('  V2 Driven:   Spatial-O=%.4f±%.4f, Per-Neuron=%.4f±%.4f, Spatial+T=%.4f±%.4f\n', ...
    mean(entr2_p_spatial), std(entr2_p_spatial), mean(entr2_p_perNeuron), std(entr2_p_perNeuron), mean(entr2_p), std(entr2_p));
fprintf('  V2 Spon:     Spatial-O=%.4f±%.4f, Per-Neuron=%.4f±%.4f, Spatial+T=%.4f±%.4f\n\n', ...
    mean(entr2_ps_spatial), std(entr2_ps_spatial), mean(entr2_ps_perNeuron), std(entr2_ps_perNeuron), mean(entr2_ps), std(entr2_ps));

%% -----------------------------------------------------------------------
%% SECTION 11: Encoding Comparison (Spontaneous vs Driven)
%% -----------------------------------------------------------------------
% Meaningful comparison of encoding capacity:
%   Spon H(r):      Baseline complexity (no stimulus)
%   Driven H(r|s):  Noise under stimulus (for comparison with Spon H(r))
%   Driven H(r):    Total driven complexity
%   Driven MI:      Information about stimulus
%
% Interpretation:
%   - If Spon H(r) ≈ Driven H(r|s): Stimulus doesn't add noise
%   - Driven H(r) = Driven H(r|s) + Driven MI
%   - MI/H(r) = fraction of response coding stimulus

figure;

subplot(2, 3, 1)
% V1: Response entropy comparison
conditions_labels = {'Spon H(r)', 'Driv H(r|s)', 'Driv H(r)'};
h_vals_v1 = [Hr_v1_spon, Hrs, Hr];
bar(h_vals_v1, 'FaceColor', [0.2 0.6 0.9]);
set(gca, 'XTickLabel', conditions_labels);
ylabel('Entropy [bits]');
title('V1: Entropy Comparison');
grid on
for i = 1:3
    text(i, h_vals_v1(i)+0.01, sprintf('%.3f', h_vals_v1(i)), 'HorizontalAlignment', 'center', 'FontSize', 9);
end

subplot(2, 3, 2)
% V2: Response entropy comparison
h_vals_v2 = [Hr_v2_spon, Hrs_v2, Hr_v2];
bar(h_vals_v2, 'FaceColor', [0.9 0.4 0.2]);
set(gca, 'XTickLabel', conditions_labels);
ylabel('Entropy [bits]');
title('V2: Entropy Comparison');
grid on
for i = 1:3
    text(i, h_vals_v2(i)+0.01, sprintf('%.3f', h_vals_v2(i)), 'HorizontalAlignment', 'center', 'FontSize', 9);
end

subplot(2, 3, 3)
% V1 & V2: MI in driven condition
mi_vals = [MI_v1, MI_v2];
bar(mi_vals, 'FaceColor', [0.2 0.8 0.3]);
set(gca, 'XTickLabel', {'V1', 'V2'});
ylabel('Mutual Information [bits]');
title('Stimulus Information (Driven Only)');
grid on
for i = 1:2
    text(i, mi_vals(i)+0.005, sprintf('%.4f', mi_vals(i)), 'HorizontalAlignment', 'center', 'FontSize', 9);
end

subplot(2, 3, 4)
% V1: Decomposition in Driven condition
driven_decomp_v1 = [Hrs, MI_v1];
labels_decomp = {'H(r|s)\n(Noise)', 'MI\n(Stimulus)'};
bar(driven_decomp_v1, 'FaceColor', [0.4 0.4 0.8]);
set(gca, 'XTickLabel', labels_decomp);
ylabel('Entropy [bits]');
title(sprintf('V1 Driven: H(r)=%.3f = Noise+Info', Hr));
grid on
hold on
% Overlay baseline
plot(xlim, [Hr_v1_spon Hr_v1_spon], '--r', 'LineWidth', 2, 'DisplayName', sprintf('Spon baseline=%.3f', Hr_v1_spon));

subplot(2, 3, 5)
% V2: Decomposition in Driven condition
driven_decomp_v2 = [Hrs_v2, MI_v2];
bar(driven_decomp_v2, 'FaceColor', [0.8 0.4 0.2]);
set(gca, 'XTickLabel', labels_decomp);
ylabel('Entropy [bits]');
title(sprintf('V2 Driven: H(r)=%.3f = Noise+Info', Hr_v2));
grid on
hold on
plot(xlim, [Hr_v2_spon Hr_v2_spon], '--r', 'LineWidth', 2, 'DisplayName', sprintf('Spon baseline=%.3f', Hr_v2_spon));

subplot(2, 3, 6)
% Efficiency: MI as fraction of H(r)
mi_fraction_v1 = MI_v1 / Hr;
mi_fraction_v2 = MI_v2 / Hr_v2;
mi_fractions = [mi_fraction_v1, mi_fraction_v2];
bars = bar(mi_fractions, 'FaceColor', [0.5 0.5 0.5]);
set(gca, 'XTickLabel', {'V1', 'V2'});
ylabel('MI / H(r)');
title('Fraction of Response Coding Stimulus');
ylim([0 1]);
grid on
for i = 1:2
    text(i, mi_fractions(i)+0.03, sprintf('%.1f%%', 100*mi_fractions(i)), 'HorizontalAlignment', 'center', 'FontSize', 9);
end

sgtitle('Encoding Comparison: Spontaneous vs Driven Conditions');

%% -----------------------------------------------------------------------
%% SECTION 12: Interpretation Guide (Text Summary)
%% -----------------------------------------------------------------------

figure('Position', [100, 100, 888, 888]);
axis off;

text_str = sprintf(['ENCODING CAPACITY ANALYSIS\n', ...
    '========================================\n\n', ...
    'INTERPRETATION:\n\n', ...
    '1. Response Entropy (H):\n', ...
    '   Spon H(r):     %.4f bits  →  Baseline activity complexity\n', ...
    '   Driv H(r|s):   %.4f bits  →  Noise when stimulus is shown\n', ...
    '   Driv H(r):     %.4f bits  →  Total response complexity\n\n', ...
    '2. Stimulus Information (MI):\n', ...
    '   V1: %.4f bits\n', ...
    '   V2: %.4f bits\n', ...
    '   → How much neural response codes stimulus\n\n', ...
    '3. Decomposition:\n', ...
    '   Driv H(r) = H(r|s) + MI\n', ...
    '   V1: %.4f = %.4f + %.4f\n', ...
    '   V2: %.4f = %.4f + %.4f\n\n', ...
    '4. Efficiency (%%  Information):\n', ...
    '   V1: %.1f%% of response variance codes stimulus\n', ...
    '   V2: %.1f%% of response variance codes stimulus\n\n', ...
    '5. Noise Stability:\n', ...
    '   V1: Spon H(r)=%.4f vs Driv H(r|s)=%.4f  (Δ=%.4f)\n', ...
    '   V2: Spon H(r)=%.4f vs Driv H(r|s)=%.4f  (Δ=%.4f)\n', ...
    '   → Check if stimulus presentation changes baseline noise\n'], ...
    Hr_v1_spon, Hrs, Hr, ...
    MI_v1, MI_v2, ...
    Hr, Hrs, MI_v1, Hr_v2, Hrs_v2, MI_v2, ...
    100*mi_fraction_v1, 100*mi_fraction_v2, ...
    Hr_v1_spon, Hrs, abs(Hr_v1_spon - Hrs), ...
    Hr_v2_spon, Hrs_v2, abs(Hr_v2_spon - Hrs_v2));

text(0.05, 0.95, text_str, 'Units', 'normalized', 'VerticalAlignment', 'top', ...
    'FontName', 'Monospaced', 'FontSize', 10, 'Parent', gca);

sgtitle('Encoding Analysis: Understanding the Metrics');

%% -----------------------------------------------------------------------
%% SECTION 13: Per-Stimulus Analysis (V1 driven trials, showing stimulus selectivity)
%% -----------------------------------------------------------------------
% Analyze how information is distributed across different stimulus classes

figure;

if ~isempty(stim_driven)
    unique_stim_list = unique(stim_driven);
    n_unique_stim = length(unique_stim_list);
    
    % Compute per-stimulus entropy
    per_stim_entropy = zeros(1, n_unique_stim);
    per_stim_count = zeros(1, n_unique_stim);
    
    for idx = 1:n_unique_stim
        s = unique_stim_list(idx);
        stim_mask = (stim_driven == s);
        rates_this_stim = [];
        for trial = find(stim_mask)
            if ~isempty(rate_v1_driven_pooled{trial})
                rates_this_stim = [rates_this_stim, rate_v1_driven_pooled{trial}];
            end
        end
        per_stim_count(idx) = sum(stim_mask);
        
        if ~isempty(rates_this_stim)
            binEdges_s = 0:1:(max(rates_this_stim)+1);
            if length(binEdges_s) < 2
                binEdges_s = [0 1];
            end
            [n_s, ~] = histcounts(rates_this_stim, binEdges_s);
            p_s = n_s / sum(n_s);
            p_s = p_s(p_s > 0);
            per_stim_entropy(idx) = -sum(p_s .* log2(p_s));
        end
    end
    
    subplot(2, 2, 1)
    bar(unique_stim_list, per_stim_entropy, 'FaceColor', [0.2 0.6 0.9]);
    xlabel('Stimulus Identity');
    ylabel('H(r|s) [bits]');
    title('Entropy Per Stimulus Class (V1 Driven)');
    grid on
    
    subplot(2, 2, 2)
    bar(unique_stim_list, per_stim_count, 'FaceColor', [0.9 0.6 0.2]);
    xlabel('Stimulus Identity');
    ylabel('Trial Count');
    title('Number of Trials per Stimulus');
    grid on
    
    % Mutual information contribution by stimulus
    if ~isempty(all_rates_v1)
        binEdges_all = 0:1:(max(all_rates_v1)+1);
        if length(binEdges_all) < 2
            binEdges_all = [0 1];
        end
        [n_r, ~] = histcounts(all_rates_v1, binEdges_all);
        
        mi_per_stim = zeros(1, n_unique_stim);
        for idx = 1:n_unique_stim
            s = unique_stim_list(idx);
            stim_mask = (stim_driven == s);
            rates_this_stim = [];
            for trial = find(stim_mask)
                if ~isempty(rate_v1_driven_pooled{trial})
                    rates_this_stim = [rates_this_stim, rate_v1_driven_pooled{trial}];
                end
            end
            
            if ~isempty(rates_this_stim)
                [n_s, ~] = histcounts(rates_this_stim, binEdges_all);
                p_s = n_s / sum(n_s);
                p_r_overall = (n_r / sum(n_r));
                
                % MI contribution ≈ KL divergence from stimulus-specific to overall
                p_s = p_s(p_s > 0);
                if ~isempty(p_s)
                    valid_idx = (p_r_overall > 0);
                    kl = sum(p_s(valid_idx) .* log2(p_s(valid_idx) ./ p_r_overall(valid_idx)));
                    mi_per_stim(idx) = max(0, kl);  % KL divergence
                end
            end
        end
        
        subplot(2, 2, 3)
        bar(unique_stim_list, mi_per_stim, 'FaceColor', [0.9 0.2 0.6]);
        xlabel('Stimulus Identity');
        ylabel('Information Contribution [bits]');
        title('Per-Stimulus Information');
        grid on
    end
    
    subplot(2, 2, 4)
    % Driven H(r) decomposition: stacked bar = H(r|s) + MI
    data_decomp = [Hrs, MI_v1; Hrs_v2, MI_v2];
    b = bar(data_decomp, 'stacked');
    b(1).FaceColor = [0.5 0.5 0.8]; b(1).DisplayName = 'H(r|s) Noise';
    b(2).FaceColor = [0.9 0.3 0.3]; b(2).DisplayName = 'MI';
    set(gca, 'XTickLabel', {'V1', 'V2'});
    ylabel('H(r) [bits]');
    title('Driven: H(r) = H(r|s) + MI');
    legend('Location', 'best');
    grid on
    % Add value labels
    for idx = 1:2
        text(idx, data_decomp(idx,1)/2, sprintf('%.3f', data_decomp(idx,1)), ...
            'HorizontalAlignment', 'center', 'FontSize', 9, 'Color', 'w');
        text(idx, data_decomp(idx,1) + data_decomp(idx,2)/2, sprintf('%.3f', data_decomp(idx,2)), ...
            'HorizontalAlignment', 'center', 'FontSize', 9, 'Color', 'w');
    end
end

sgtitle('Per-Stimulus Analysis (V1 Driven Trials)');

%% Save MI results (both spontaneous and driven)
save('MI_analysis_figure2.mat', ...
    'MI_v1', 'Hr', 'Hrs', 'MI_v2', 'Hr_v2', 'Hrs_v2', ...
    'MI_spatial_v1', 'Hr_spatial_v1', 'Hrs_spatial_v1', ...
    'MI_spatial_v2', 'Hr_spatial_v2', 'Hrs_spatial_v2', ...
    'MI_v1_spon', 'Hr_v1_spon', 'Hrs_v1_spon', 'MI_v2_spon', 'Hr_v2_spon', 'Hrs_v2_spon', ...
    'Hr_spatial_v1_spon', 'Hr_spatial_v2_spon', ...
    'mi_v1_perNeuron', 'mi_v2_perNeuron', ...
    'Hr_v1_perNeuron_spon', 'Hr_v1_perNeuron_driven', 'Hrs_v1_perNeuron', ...
    'Hr_v2_perNeuron_spon', 'Hr_v2_perNeuron_driven', 'Hrs_v2_perNeuron', ...
    'stim_driven', ...
    'rate_v1_driven_pooled', 'rate_v2_driven_pooled', ...
    'rate_v1_driven_byNeuron', 'rate_v2_driven_byNeuron', ...
    'rate_v1_spon_pooled', 'rate_v2_spon_pooled', ...
    'rate_v1_spon_byNeuron', 'rate_v2_spon_byNeuron', ...
    'all_spatial_rates_v1_driven', 'all_spatial_rates_v2_driven', ...
    'all_spatial_rates_v1_spon', 'all_spatial_rates_v2_spon', ...
    'all_rates_v1', 'all_rates_v2', 'all_rates_v1_spon', 'all_rates_v2_spon')

fprintf('\n=== MI Analysis Complete ===\n');
fprintf('Results saved to: MI_analysis_figure2.mat\n');
fprintf('\n==================== MI Comparison ====================\n');
fprintf('V1 Driven:  MI = %.4f bits, H(r) = %.4f bits\n', MI_v1, Hr);
fprintf('V1 Spon:    MI = %.4f bits, H(r) = %.4f bits (no stimulus)\n', MI_v1_spon, Hr_v1_spon);
fprintf('\nV2 Driven:  MI = %.4f bits, H(r) = %.4f bits\n', MI_v2, Hr_v2);
fprintf('V2 Spon:    MI = %.4f bits, H(r) = %.4f bits (no stimulus)\n', MI_v2_spon, Hr_v2_spon);
fprintf('=========================================================\n');
