%% figure3_main_dev.m  (earlier dev branch, writes MI_results_direct_1.mat)
%  Direct method (Strong et al. 1998) version of figure4_MI.m
%
%  Key difference from figure4_MI.m:
%    - MI computed using compute_entropy_toolbox (Information 1)
%      instead of histogram-based method (Information 3).
%    - Spike times stored per (stim, rep, window) and converted
%      to binary spike trains at 1 ms resolution.
%    - Output MI is directly in bits/s; ICE = MI / Cost [bits/J].
%
%  Requires: compute_entropy_toolbox.m on path
%
%% Add toolbox to path
addpath('/Users/mia/Library/Mobile Documents/com~apple~CloudDocs/compute_entropy');

%% Initialization
close all

%% Simulation Structure Parameters
NStim   = 20;       % Number of stimulus templates (outer loop)
NReps   = 10;       % Number of repetitions per stimulus

%% Model Parameters
% --- Learning rule ---
eta           = 0.0012;        % Learning rate
alpha         = 0.25 * eta;    % Decay term (≈ 2*rho0*tauSTDP*eta, rho0≈5Hz)
tauPlasticity = 30;            % Time window of the learning rule (ms)

% --- Simulation timing ---
duration = 1500;               % Duration of a single run (ms)
dt       = 0.1;                % Simulation time step (ms)
NRuns    = 80;                 % Number of consecutive runs
tRef     = 5;                  % Refractory period (ms)
winsize  = 4;                  % Window size for analysis (runs)
NWindows = NRuns / winsize;    % Number of analysis windows (= 20)

% --- Synaptic conductances (in units of g_leak = 10 nS) ---
gBarEx = 0.014;                % Excitatory conductance scaling factor
gBarIn = 0.035;                % Inhibitory conductance scaling factor

% --- Membrane properties ---
VRest  = -60;                  % Resting potential (mV)
Vth    = -50;                  % Spike threshold (mV)
taumem = 20;                   % Membrane time constant (ms)

% --- Reversal potentials ---
EAMPA = 0;                     % Excitatory (AMPA) reversal potential (mV)
EGABA = -80;                   % Inhibitory (GABA) reversal potential (mV)

% --- Synaptic time constants ---
tauEx = 5;                     % Excitatory synaptic time constant (ms)
tauIn = 10;                    % Inhibitory synaptic time constant (ms)

%% Input Parameters
noisetau       = 50;           % Filter time constant for input (ms)
Backgroundrate = 5 * dt / 1000;
ApproximateNormalizingFactor = 0.03;
Maxrate = 500 * dt / 1000;
NSigs   = 8;                   % Number of input signal channels
NCells  = 1000;                % Number of input spike trains
ExFrac  = 0.8;                 % Fraction of excitatory spike trains
ExGroupsize = (NCells * ExFrac) / NSigs;
InGroupsize = round((NCells * (1 - ExFrac)) / NSigs);

% --- Pre-calculated exponential decay factors ---
expGEx        = exp(-dt / tauEx);
expGIn        = exp(-dt / tauIn);
expPlasticity = exp(-dt / tauPlasticity);
expnoise      = exp(-dt / noisetau);

%% Time Vector
Timevector   = (0.1:dt:duration);
nStepsPerRun = length(Timevector) - 1;   % Time steps per run (t = 2:end)

%% Binary spike train parameters
dt_bin    = 10;                            % bin width for binary spike train [ms]
win_ms    = winsize * duration;            % window duration [ms] = 6000
nBins_win = win_ms / dt_bin;              % number of bins per window = 600

%% Assign Input Groups (deterministic, computed once)
InputGroup = zeros(1, NCells);
temptype   = 0;
for i = 1:NCells
    if (i <= NCells * ExFrac)
        if (mod(i - 1, ExGroupsize) == 0)
            temptype = temptype + 1;
        end
        InputGroup(i) = temptype;
    else
        if (mod(i, InGroupsize) == 0)
            temptype = temptype - 1;
        end
        InputGroup(i) = -temptype;
    end
end
InputGroup(1000) = InputGroup(999);

%% Generate Initial Synaptic Weights (computed once, reused for all stim & rep)
Synapse_init = ones(1, NCells);
for i = 1:800
    Synapse_init(i) = 0.3 + (1.1 / (1 + (InputGroup(i) - 5)^4)) + rand * 0.1;
end
for i = 801:NCells
    Synapse_init(i) = 0.1;
end

%% Allocate Global Storage
% Cell array storing spike times for each (stim, rep, window)
Spike_cell = cell(NStim, NReps, NWindows);

% Per-window statistics: (NStim x NReps x NWindows)
ss1_store   = zeros(NStim, NReps, NWindows);
cv1_store   = zeros(NStim, NReps, NWindows);
ent1_store  = zeros(NStim, NReps, NWindows);
ra1_store   = zeros(NStim, NReps, NWindows);
cost1_store = zeros(NStim, NReps, NWindows);
cvISI_store = zeros(NStim, NReps, NWindows);
mISI_store  = zeros(NStim, NReps, NWindows);
EI_store    = zeros(NStim, NReps, NWindows);
ss_store    = zeros(NStim, NReps, NWindows);
cv_store    = zeros(NStim, NReps, NWindows);
ent_store   = zeros(NStim, NReps, NWindows);
ra_store    = zeros(NStim, NReps, NWindows);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%       Main Simulation Loop           %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for stim = 1:NStim
    fprintf('=== Stimulus %d / %d ===\n', stim, NStim);

    %% Pre-generate Stimulus Template
    InputTemplate = zeros(NSigs, nStepsPerRun * NRuns);
    FWN_temp = zeros(1, NSigs);
    for step = 1:nStepsPerRun * NRuns
        for ch = 1:NSigs
            re = rand - 0.5;
            FWN_temp(ch) = re - (re - FWN_temp(ch)) * expnoise;
            InputTemplate(ch, step) = Backgroundrate + ...
                max(0, Maxrate * FWN_temp(ch)) / ApproximateNormalizingFactor;
        end
    end

    %% Repetition Loop
    for rep = 1:NReps
        fprintf('  Rep %d / %d\n', rep, NReps);

        %% Initialize Per-Repetition State
        plot_all  = 0;
        stopnow   = 0;
        id        = 0;
        spiketime = [];
        rateall   = [];
        v_rec     = [];
        ex_rec    = [];
        in_rec    = [];
        input_record = zeros(8, 2);
        Input = zeros(1, NSigs);
        signal = zeros(NSigs, length(Timevector), 2, 3);
        Exkeep = Timevector * 0;
        Inkeep = Timevector * 0;
        FilteredWhiteNoise = zeros(1, 8);
        InputSpikeRefr     = zeros(1, NCells);
        tolos = 0;

        Synapse   = Synapse_init;
        sgEx      = zeros(1, NCells);
        sgIn      = zeros(1, NCells);
        AveExCurr = zeros(1, NSigs);
        AveInCurr = zeros(1, NSigs);
        pre       = zeros(1, NCells);

        Time     = (1:NRuns);
        Rate     = zeros(1, NRuns);
        Rate_isi = zeros(1, NRuns);
        Rate_ms  = zeros(1, NRuns * duration / 100);

        % --- Windowed analysis vectors ---
        ss   = zeros(1, NWindows);
        cv   = zeros(1, NWindows);
        ent  = zeros(1, NWindows);
        ss1  = zeros(1, NWindows);
        cv1  = zeros(1, NWindows);
        ent1 = zeros(1, NWindows);
        ra   = zeros(1, NWindows);
        ra1  = zeros(1, NWindows);
        mISI_vec  = zeros(1, NWindows);
        cvISI_vec = zeros(1, NWindows);
        EI_vec    = zeros(1, NWindows);
        E_vec     = zeros(1, NWindows);
        I_vec     = zeros(1, NWindows);
        cost1_vec = zeros(1, NWindows);

        ss_t = ss;
        gEx   = 0;
        gIn   = 0;
        gLeak = 1;
        V(1)  = VRest;
        post  = 0;
        InputSpikeCount  = 0;
        OutputSpikeCount = 0;
        AveCurrCounter   = 0;
        tRunning = 0;
        runcount = 0;
        going    = 1;
        ExCurr = [];
        InCurr = [];

        %% Simulation Loop (while)
        while (going > 0)
            runcount  = runcount + 1;
            spike_run = [];
            OutputSpikeCount = 0;
            for i = 1:NSigs
                AveExCurr(i) = 0;
                AveInCurr(i) = 0;
            end
            AveCurrCounter = 0;

            % --- Plot synapse strengths (last rep of last stim only) ---
            if stim == NStim && rep == NReps
                figure(2)
                clf(2)
                plot(gBarEx * Synapse(1:800) * 10000, 'r.')
                hold on
                plot(-(-796:4:1), gBarIn * Synapse(801:NCells) * 10000, 'g.')
            end

            %% Time Loop
            for t = 2:length(Timevector)
                tRunning = tRunning + dt;

                % --- Exponential decay of synaptic conductances ---
                gEx = gEx * expGEx;
                gIn = gIn * expGIn;
                for i = 1:NSigs
                    sgEx(i) = sgEx(i) * expGEx;
                    sgIn(i) = sgIn(i) * expGIn;
                end

                % --- Exponential decay of learning traces ---
                for i = 801:NCells
                    pre(i) = pre(i) * expPlasticity;
                end
                post = post * expPlasticity;

                %% Input Signals (from pre-generated template)
                globalStep = (runcount - 1) * nStepsPerRun + (t - 1);
                Input = InputTemplate(:, globalStep)';
                input_record(:, t-1) = Input;

                %% Presynaptic Spike Trains
                for i = 1:NCells
                    if (rand < Input(abs(InputGroup(i))) && InputSpikeRefr(i) <= 0)
                        if (InputGroup(i) > 0)
                            gEx = gEx + (gBarEx * Synapse(i));
                            sgEx(abs(InputGroup(i))) = sgEx(abs(InputGroup(i))) + gBarEx * Synapse(i);
                        else
                            gIn = gIn + gBarIn * Synapse(i);
                            sgIn(abs(InputGroup(i))) = sgIn(abs(InputGroup(i))) + gBarIn * Synapse(i);
                            pre(i) = pre(i) + eta;
                            Synapse(i) = Synapse(i) + post - alpha;
                            if (Synapse(i) <= 0)
                                Synapse(i) = 0;
                            end
                        end
                        InputSpikeRefr(i) = tRef;
                        InputSpikeCount = InputSpikeCount + 1;
                    else
                        InputSpikeRefr(i) = InputSpikeRefr(i) - dt;
                    end
                end

                %% Membrane Potential and Postsynaptic Spikes
                if ((tRunning - tolos) < tRef)
                    V(t) = VRest;
                else
                    gTot   = gLeak + gEx + gIn;
                    tauEff = taumem / gTot;
                    VInf   = (gLeak*VRest + gEx*EAMPA + gIn*EGABA) / gTot;
                    V(t)   = VInf + (V(t-1) - VInf) * exp(-dt / tauEff);

                    for i = 1:NSigs
                        AveExCurr(i) = AveExCurr(i) + sgEx(i) * (V(t) - EAMPA);
                        AveInCurr(i) = AveInCurr(i) + ...
                            sgIn(i) * (V(t) - EGABA) + (gLeak * (V(t) - VRest)) / NSigs;
                        if runcount == 2
                            signal(i, t, 1, 1) = sgEx(i) * (V(t) - EAMPA);
                            signal(i, t, 2, 1) = sgIn(i) * (V(t) - EGABA) + (gLeak * (V(t) - VRest)) / NSigs;
                        end
                        if runcount == 8
                            signal(i, t, 1, 2) = sgEx(i) * (V(t) - EAMPA);
                            signal(i, t, 2, 2) = sgIn(i) * (V(t) - EGABA) + (gLeak * (V(t) - VRest)) / NSigs;
                        end
                        if runcount == 60
                            signal(i, t, 1, 3) = sgEx(i) * (V(t) - EAMPA);
                            signal(i, t, 2, 3) = sgIn(i) * (V(t) - EGABA) + (gLeak * (V(t) - VRest)) / NSigs;
                        end
                    end
                    AveCurrCounter = AveCurrCounter + 1;
                    ExCurr = [ExCurr abs(mean(AveExCurr))];
                    InCurr = [InCurr abs(mean(AveInCurr))];
                end

                if (V(t) > Vth)
                    tolos  = tRunning;
                    V(t-1) = 0;
                    V(t)   = VRest;
                    OutputSpikeCount = OutputSpikeCount + 1;
                    spiketime = [spiketime duration*(runcount-1) + Timevector(t)];
                    spike_run = [spike_run duration*(runcount-1) + Timevector(t)];
                    Rate_ms((runcount-1)*duration/100 + ceil(t/1000)) = ...
                        Rate_ms((runcount-1)*duration/100 + ceil(t/1000)) + 10;
                    post = post + eta;
                    for i = 801:NCells
                        Synapse(i) = Synapse(i) + pre(i);
                    end
                end
                Exkeep(t) = gEx * (V(t) - EAMPA);
                Inkeep(t) = gIn * (V(t) - EGABA);
            end
            % --- End of time loop ---

            %% Post-Run Analysis
            isi_run            = diff(spike_run);
            Rate_isi(runcount) = mean(1000 ./ isi_run);
            EIr(runcount)      = abs(mean(AveExCurr)) / abs(mean(AveInCurr));
            Rate(runcount)     = OutputSpikeCount / duration * 1000;
            Time(runcount)     = runcount * duration / 1000;

            %% Windowed Statistics
            if mod(runcount, winsize) == 0 && runcount >= winsize
                w = runcount / winsize;

                % Window start run for spike time offset
                win_start_ms = (w-1) * winsize * duration;

                rate = Rate_ms((runcount - winsize) * duration/100 + 1 : runcount * duration/100);
                ss(w) = sparseness_mrate(rate);
                cv(w) = std(rate) / mean(rate);
                [n, ~] = histcounts(rate, 100);
                h = n / sum(n);
                for j = h
                    if j > 0
                        ent(w) = ent(w) - j * log2(j);
                    end
                end
                ra(w) = mean(rate);

                isi = diff(spiketime);
                mISI_vec(w) = mean(diff(spiketime));
                if length(isi) >= 2
                    rate1 = 2000 ./ (isi(1:end-1) + isi(2:end));
                else
                    rate1 = [];
                end

                if ~isempty(rate1)
                    ra1(w)       = mean(rate1);
                    ss1(w)       = sparseness_mrate(rate1);
                    cv1(w)       = std(rate1) / mean(rate1);
                    cost1_vec(w) = 2.4 + 0.37 * mean(rate1);
                    [n, ~] = histcounts(rate1, 0:0.1:(max(rate1)+0.1));
                    h = n / sum(n);
                    for j = h
                        if j > 0
                            ent1(w) = ent1(w) - j * log2(j);
                        end
                    end
                end

                cvISI_vec(w) = std(diff(spiketime)) / mean(diff(spiketime));

                % === Store spike times (relative to window start) ===
                Spike_cell{stim, rep, w} = spiketime - win_start_ms;

                spiketime = [];
                rateall   = [rateall rate1];

                if ~isempty(InCurr) && ~isempty(ExCurr)
                    EI_vec(w) = mean(ExCurr) / mean(InCurr);
                    E_vec(w)  = mean(ExCurr);
                    I_vec(w)  = mean(InCurr);
                end
                ExCurr = [];
                InCurr = [];

                % === Store per-window statistics ===
                ss1_store(stim, rep, w)   = ss1(w);
                cv1_store(stim, rep, w)   = cv1(w);
                ent1_store(stim, rep, w)  = ent1(w);
                ra1_store(stim, rep, w)   = ra1(w);
                cost1_store(stim, rep, w) = cost1_vec(w);
                cvISI_store(stim, rep, w) = cvISI_vec(w);
                mISI_store(stim, rep, w)  = mISI_vec(w);
                EI_store(stim, rep, w)    = EI_vec(w);
                ss_store(stim, rep, w)    = ss(w);
                cv_store(stim, rep, w)    = cv(w);
                ent_store(stim, rep, w)   = ent(w);
                ra_store(stim, rep, w)    = ra(w);
            end

            %% Record Snapshots (last rep only)
            if rep == NReps
                id = id + 1;
                if runcount == 2 || runcount == 8 || runcount == 80
                    record(id).sigs     = [(-AveExCurr / AveCurrCounter) / 100; ...
                                            (AveInCurr / AveCurrCounter) / 100];
                    record(id).currents = [Exkeep / 100; Inkeep / 100];
                    record(id).spikes   = Timevector(V == 0);
                end
            end

            %% Termination
            if (runcount >= NRuns)
                going = 0;
            end
            if (stopnow == 1)
                going = 0;
            end

            % --- Accumulate full traces (last rep of last stim only) ---
            if stim == NStim && rep == NReps
                v_rec  = [v_rec  V];
                ex_rec = [ex_rec Exkeep];
                in_rec = [in_rec Inkeep];
            end
        end
        % --- End of simulation loop (while) ---
    end
    % --- End of repetition loop ---

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%  Per-Stimulus Diagnostic Plots       %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    x = linspace(0, 3, 100);

    cv1_mean  = squeeze(mean(cv1_store(stim, :, :), 2));
    ss1_mean  = squeeze(mean(ss1_store(stim, :, :), 2));
    ent1_mean = squeeze(mean(ent1_store(stim, :, :), 2));
    ra1_mean  = squeeze(mean(ra1_store(stim, :, :), 2));
    cost1_mean = squeeze(mean(cost1_store(stim, :, :), 2));
    cvISI_mean = squeeze(mean(cvISI_store(stim, :, :), 2));
    mISI_mean  = squeeze(mean(mISI_store(stim, :, :), 2));
    EI_mean    = squeeze(mean(EI_store(stim, :, :), 2));
    cv_mean    = squeeze(mean(cv_store(stim, :, :), 2));
    ent_mean   = squeeze(mean(ent_store(stim, :, :), 2));
    ra_mean    = squeeze(mean(ra_store(stim, :, :), 2));

    figure;
    plot(cv1_mean, ss1_mean, 'r*', x, x.^2 ./ (1 + x.^2), 'k-'); hold on;
    xlabel('Coefficient of Variance');
    ylabel('Sparseness');
    title(sprintf('Stim %d: ISI Rate (mean over %d reps)', stim, NReps));

    figure;
    subplot(2, 2, 1);
    plot(cv1_mean, cvISI_mean, 'k*'); hold on;
    ylabel('cv ISI'); xlabel('cv Rate'); title('CV');
    subplot(2, 2, 2);
    plot(ra1_mean, mISI_mean, 'k*'); hold on;
    xlabel('Mean Firing Rate(Hz)'); ylabel('Mean ISI(ms)'); title('Mean Value');
    subplot(2, 2, 3);
    plot(cv1_mean .* ra1_mean, cvISI_mean .* mISI_mean, 'k*'); hold on;
    ylabel('std ISI'); xlabel('std Rate'); title('Standard Deviation');
    subplot(2, 2, 4);
    plot(mISI_mean, cvISI_mean, 'k*'); hold on;
    ylabel('CV ISI'); xlabel('Mean ISI(ms)'); title('Mean-CV-ISI');

    %% Signal Correlation Analysis (from last rep)
    insim = zeros(8, 8, 3);
    E_I   = zeros(8, 3);
    st    = [2 8 80];
    figure;
    for k = 1:3
        r = st(k);
        for i = 1:8
            for j = 1:8
                insim(i, j, k) = 1 - corr(signal(i,:,1,k)', signal(j,:,1,k)');
                insim(j, i, k) = 1 - corr(signal(i,:,2,k)', signal(j,:,2,k)');
            end
            E_I(i, k) = 1 - corr(signal(i,:,1,k)', signal(i,:,2,k)');
        end
        unsim = insim(:, :, k);
        subplot(1, 3, k)
        imagesc(unsim)
        colormap(gray)
        colorbar
        caxis([0 1.5])
    end

    %% Entropy Analysis
    figure;
    plot(cv_mean, ent_mean, '*');
    xlabel('cv-lifetime'); ylabel('entropy-lifetime')
    figure;
    plot(cv_mean, ent_mean ./ ra_mean, '*');
    xlabel('cv-lifetime'); ylabel('entropy-lifetime/hz')

    %% Entropy vs Cost Analysis
    figure;
    subplot(2, 3, 1); plot(ent1_mean);
    subplot(2, 3, 2); plot(cost1_mean);
    subplot(2, 3, 3); plot(ent1_mean ./ cost1_mean);
    save(sprintf('ent%d.mat', stim), ...
        'ent1_mean', 'cost1_mean', 'cv1_mean', 'ss1_mean', ...
        'cvISI_mean', 'mISI_mean', 'EI_mean')
    save(sprintf('rec%d.mat', stim), 'record', 'insim')
    subplot(2, 3, 4); plot(cv1_mean, ent1_mean, '*');
    subplot(2, 3, 5); plot(cv1_mean, cost1_mean, '*');
    subplot(2, 3, 6); plot(cv1_mean, ent1_mean ./ cost1_mean, '*');
end
% --- End of stimulus loop ---

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  MI via Direct Method (Info 1)       %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\n=== Computing MI via direct method ===\n');

%% Convert spike times to binary spike trains
% Binary_cell{stim, rep, w} = [1 x nBins_win] binary vector
Binary_cell = cell(NStim, NReps, NWindows);
for s = 1:NStim
    for r = 1:NReps
        for w = 1:NWindows
            bin_train = zeros(1, nBins_win);
            st = Spike_cell{s, r, w};
            if ~isempty(st)
                bin_idx = ceil(st / dt_bin);
                bin_idx(bin_idx < 1) = 1;
                bin_idx(bin_idx > nBins_win) = nBins_win;
                bin_train(bin_idx) = 1;
            end
            Binary_cell{s, r, w} = bin_train;
        end
    end
end

%% ── Per-template MI via direct method ────────────────────────────────
% Each of the NStim=20 frozen templates is analysed independently.
% For template s, window w:
%   spike_repeat = [NReps × nBins_win] (10 reps of the same frozen stimulus)
%   compute_entropy_toolbox computes:
%     Hr_s(w) = total entropy rate  (word distribution pooled over time)
%               → includes stimulus-driven temporal modulation + noise
%     Hn_s(w) = noise entropy rate  (word distribution at each time t, across trials)
%               → trial-to-trial noise only
%     MI_s(w) = Hr_s(w) - Hn_s(w)  → information from temporal modulation
%
% Final curves: mean ± SEM across 20 templates (shaded region).

dt_sec = dt_bin / 1000;   % bin width in seconds (10 ms -> 0.01 s)

% --- Cross-stim MI via direct method (V1-comparable, matches figure2_new.m).
% Pool ALL (stim, rep) trials of each window into a single spike_long
% matrix to estimate H_total over the marginal P(word) across the entire
% stimulus ensemble; estimate H_noise WITHIN each stim across reps and
% take the P(s)-weighted average.  MI = H_total - H_noise_avg.
%
% Why we changed this: the previous version called
%   compute_entropy_toolbox(spike_repeat, spike_repeat, ...)
% with spike_repeat = one stim's reps in BOTH slots, which collapses the
% stimulus ensemble.  The MI it returned was within-one-frozen-template
% temporal information, which saturates near the spike-train entropy
% bound (~30 bits/s for a 10 Hz neuron at 10 ms bins) and is not the
% I(s; r) used by the framework's R_0 anchor in main-text Fig. 6.

T_list_mi = [4 5 6 7 8 10 11 12 13 15 17 20 22 25 28 30 40 50 60 70 80 90 100];
T_list_mi = T_list_mi(T_list_mi <= nBins_win);
n_fit_mi  = min(10, length(T_list_mi));

Hr_vec = zeros(1, NWindows);
Hn_vec = zeros(1, NWindows);
MI_vec = zeros(1, NWindows);

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
    if sum(spike_long_all(:)) == 0
        % No spikes in this window: leave Hr/Hn/MI at 0 (matching the
        % original behaviour and keeping downstream fit() calls happy).
        continue;
    end

    nT = length(T_list_mi);
    H_total_T = zeros(1, nT);
    H_noise_T = zeros(1, nT);

    for ti = 1:nT
        T = T_list_mi(ti);
        H_total_T(ti) = total_entropy_pool(spike_long_all, T);
    end

    for s = 1:NStim
        spike_rep_s = zeros(NReps, nBins_win);
        for r = 1:NReps
            spike_rep_s(r, :) = Binary_cell{s, r, w};
        end
        if sum(spike_rep_s(:)) == 0, continue; end
        for ti = 1:nT
            T = T_list_mi(ti);
            H_noise_T(ti) = H_noise_T(ti) + (1/NStim) * noise_entropy_avg(spike_rep_s, T);
        end
    end

    T_sec = T_list_mi * dt_sec;
    inv_T = 1 ./ T_sec;
    rate_total = H_total_T ./ T_sec;
    rate_noise = H_noise_T ./ T_sec;
    fit_idx = 1 : n_fit_mi;
    p_t = polyfit(inv_T(fit_idx), rate_total(fit_idx), 1);
    p_n = polyfit(inv_T(fit_idx), rate_noise(fit_idx), 1);

    Hr_vec(w) = p_t(2);
    Hn_vec(w) = p_n(2);
    MI_vec(w) = max(0, p_t(2) - p_n(2));
end

% Per-stim arrays kept (NaN-filled) for backwards-compatible save() below.
Hr_per_stim = nan(NStim, NWindows);
Hn_per_stim = nan(NStim, NWindows);
MI_per_stim = nan(NStim, NWindows);

% SEM placeholders: cross-stim Hr is a single estimate per window, so the
% per-stim spread is undefined.  Set to zero for now; switch to a
% bootstrap-over-stim resample if error bars are needed.
Hr_sem = zeros(1, NWindows);
Hn_sem = zeros(1, NWindows);
MI_sem = zeros(1, NWindows);

%% Compute Mean Statistics Across All Stimuli and Reps
cv1_grand   = squeeze(mean(mean(cv1_store, 1), 2));
ss1_grand   = squeeze(mean(mean(ss1_store, 1), 2));
ent1_grand  = squeeze(mean(mean(ent1_store, 1), 2));
ra1_grand   = squeeze(mean(mean(ra1_store, 1), 2));
cost1_grand = squeeze(mean(mean(cost1_store, 1), 2));
EI_grand    = squeeze(mean(mean(EI_store, 1), 2));

%% SEM for statistics (across stim, averaged over reps)
cv1_sem   = squeeze(std(mean(cv1_store, 2), 0, 1))   / sqrt(NStim);
ss1_sem   = squeeze(std(mean(ss1_store, 2), 0, 1))   / sqrt(NStim);
ent1_sem  = squeeze(std(mean(ent1_store, 2), 0, 1))  / sqrt(NStim);
ra1_sem   = squeeze(std(mean(ra1_store, 2), 0, 1))   / sqrt(NStim);
cost1_sem = squeeze(std(mean(cost1_store, 2), 0, 1)) / sqrt(NStim);
EI_sem    = squeeze(std(mean(EI_store, 2), 0, 1))    / sqrt(NStim);

%% ICE = MI / Cost  [bits/s / W = bits/J]
% Use the per-window cross-stim MI (MI_vec) directly: MI_per_stim is no
% longer populated under the cross-stim formulation.
ICE_vec     = MI_vec(:)' ./ cost1_grand(:)';
ICE_sem     = zeros(1, NWindows);
ICE_per_stim = nan(NStim, NWindows);    % kept for backwards-compatible save()

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%          MI Summary Plots            %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
timewindow = (1:NWindows) * winsize * duration / 1000;   % Time axis [s]

%% Plot MI, H(r), H(r|s) vs Time — with SEM
figure;
x2 = [timewindow, fliplr(timewindow)];

% --- Subplot 1: H(r), H_noise, MI ---
subplot(2, 2, 1);
fill(x2, [Hr_vec + Hr_sem, fliplr(Hr_vec - Hr_sem)], ...
     'k', 'FaceAlpha', 0.15, 'EdgeColor', 'none'); hold on;
plot(timewindow, Hr_vec, 'k-', 'LineWidth', 1.5);
fill(x2, [Hn_vec + Hn_sem, fliplr(Hn_vec - Hn_sem)], ...
     'b', 'FaceAlpha', 0.15, 'EdgeColor', 'none');
plot(timewindow, Hn_vec, 'b-', 'LineWidth', 1.5);
fill(x2, [MI_vec + MI_sem, fliplr(MI_vec - MI_sem)], ...
     'r', 'FaceAlpha', 0.15, 'EdgeColor', 'none');
plot(timewindow, MI_vec, 'r-', 'LineWidth', 1.5);
legend('', 'H_r', '', 'H_n', '', 'MI');
xlabel('Time [s]'); ylabel('[bits/s]');
title('MI via Direct Method');

% --- Subplot 2: MI only ---
subplot(2, 2, 2);
fill(x2, [MI_vec + MI_sem, fliplr(MI_vec - MI_sem)], ...
     'r', 'FaceAlpha', 0.15, 'EdgeColor', 'none'); hold on;
plot(timewindow, MI_vec, 'r-', 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('MI [bits/s]');
title('MI = H_r - H_n');

% --- Subplot 3: ICE ---
subplot(2, 2, 3);
fill(x2, [ICE_vec + ICE_sem, fliplr(ICE_vec - ICE_sem)], ...
     'm', 'FaceAlpha', 0.15, 'EdgeColor', 'none'); hold on;
plot(timewindow, ICE_vec, 'm-', 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('ICE [bits/J]');
title('Information-Cost Efficiency');

% --- Subplot 4: CV & Sparseness ---
subplot(2, 2, 4);
fill(x2, [cv1_grand'+cv1_sem', fliplr(cv1_grand'-cv1_sem')], ...
     'b', 'FaceAlpha', 0.15, 'EdgeColor', 'none'); hold on;
plot(timewindow, cv1_grand, 'b-', 'LineWidth', 1.5);
fill(x2, [ss1_grand'+ss1_sem', fliplr(ss1_grand'-ss1_sem')], ...
     'k', 'FaceAlpha', 0.15, 'EdgeColor', 'none');
plot(timewindow, ss1_grand, 'k-', 'LineWidth', 1.5);
legend('', 'CV', '', 'Sparseness');
xlabel('Time [s]'); ylabel('Value');
title('CV & Sparseness over Learning');

%% Plot Cost vs MI / ICE — with error bars
figure;
subplot(1, 2, 1)
errorbar(cost1_grand, MI_vec', MI_sem', MI_sem', cost1_sem, cost1_sem, ...
    'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 4); hold on;
xlabel('Cost [W]'); ylabel('MI [bits/s]'); title('Cost vs MI');

subplot(1, 2, 2)
errorbar(cost1_grand, ICE_vec', ICE_sem', ICE_sem', cost1_sem, cost1_sem, ...
    'bs', 'MarkerFaceColor', 'b', 'MarkerSize', 4); hold on;
xlabel('Cost [W]'); ylabel('ICE [bits/J]'); title('Cost vs ICE');

%% Plot CV vs MI / H(r) / H_noise / ICE — with error bars
figure;
subplot(1, 4, 1)
errorbar(cv1_grand, Hr_vec', Hr_sem', Hr_sem', cv1_sem, cv1_sem, ...
    'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 4); hold on;
xlabel('CV'); ylabel('H_r [bits/s]'); title('CV vs H(r)');

subplot(1, 4, 2)
errorbar(cv1_grand, Hn_vec', Hn_sem', Hn_sem', cv1_sem, cv1_sem, ...
    'bs', 'MarkerFaceColor', 'b', 'MarkerSize', 4); hold on;
xlabel('CV'); ylabel('H_n [bits/s]'); title('CV vs H(r|s)');

subplot(1, 4, 3)
errorbar(cv1_grand, MI_vec', MI_sem', MI_sem', cv1_sem, cv1_sem, ...
    'r^', 'MarkerFaceColor', 'r', 'MarkerSize', 4); hold on;
xlabel('CV'); ylabel('MI [bits/s]'); title('CV vs MI');

subplot(1, 4, 4)
errorbar(cv1_grand, ICE_vec', ICE_sem', ICE_sem', cv1_sem, cv1_sem, ...
    'k.', 'MarkerSize', 12); hold on;

% --- Gaussian channel theoretical overlay (Fig 4G) ---
%  Signal r ~ N(mu, sigma^2), noise n ~ N(0, sigma_n^2)
%  MI  = 0.5 * log2(1 + sigma^2 / sigma_n^2)   [bits/s at unit rate]
%  Cost = 2.4 + 0.37 * mu                        [W, physiological]
%  CV  = sigma / mu  =>  mu = sigma / CV
%  ICE = MI / Cost                                [bits/J]
sigma_n_th  = 1.0;
R0_th       = 5;                             % observation rate [Hz]
sigma_set   = [1, 5, 10, 50, 100];
CV_th       = 0.05:0.02:8;
th_colors   = [0.75 0.75 0.75;   % gray     sigma=1
               0.00 0.50 0.90;   % blue     sigma=5
               0.00 0.70 0.70;   % teal     sigma=10
               0.85 0.33 0.10;   % orange   sigma=50
               0.64 0.08 0.18];  % crimson  sigma=100
for si = 1:length(sigma_set)
    sig = sigma_set(si);
    MI_rate = 0.5 * log2(1 + sig^2 / sigma_n_th^2) * R0_th;  % bits/s
    mu_th   = sig ./ CV_th;
    C_th    = 2.4 + 0.37 * mu_th;                              % W = J/s
    ICE_th  = MI_rate ./ C_th;                                  % bits/J
    plot(CV_th, ICE_th, '-', 'Color', th_colors(si,:), 'LineWidth', 1.5);
end

% Poisson reference (gray dashed): ICE = H_Pois / (2.4 + 0.37*mu)
mu_pois_ref = 0.5:0.5:100;
CV_pois_ref = 1 ./ sqrt(mu_pois_ref);
H_pois_ref  = zeros(size(mu_pois_ref));
for ip = 1:length(mu_pois_ref)
    mu_p = mu_pois_ref(ip);
    k_max = max(30, ceil(mu_p + 8*sqrt(mu_p)));
    k  = 0:k_max;
    pk = poisspdf(k, mu_p);
    pk = pk(pk > 1e-15);  pk = pk / sum(pk);
    H_pois_ref(ip) = -sum(pk .* log2(pk));
end
C_pois_ref   = 2.4 + 0.37 * mu_pois_ref;
ICE_pois_ref = H_pois_ref ./ C_pois_ref;
plot(CV_pois_ref, ICE_pois_ref, '--', 'Color', [0.6 0.6 0.6], 'LineWidth', 1.2);

xlabel('CV'); ylabel('ICE [bits/J]'); title('CV vs ICE');
legend_entries = arrayfun(@(s) sprintf('\\sigma = %d', s), sigma_set, 'UniformOutput', false);
legend(['Neuron model', legend_entries, {'Poisson ref.'}], ...
    'Location', 'northeast', 'FontSize', 8);

%% Plot EI ratio vs ICE & CV — with error bars + exponential fit
ft = fittype('A * exp(-B * (x - 1)) + C', 'independent', 'x');
opts = fitoptions(ft);
opts.StartPoint = [1, 1, 0];
opts.Lower      = [-Inf, 0, -Inf];

% Drop any windows with non-finite EI/CV/ICE before the nonlinear fit
% (fit() refuses NaN/Inf in X, Y, or WEIGHTS).
ei_x   = EI_grand(:);
cv_y   = cv1_grand(:);
ice_y  = ICE_vec(:);
mask_f = isfinite(ei_x) & isfinite(cv_y) & isfinite(ice_y);

[fitRes1, ~] = fit(ei_x(mask_f), cv_y(mask_f),  ft, opts);
[fitRes2, ~] = fit(ei_x(mask_f), ice_y(mask_f), ft, opts);

xFit = linspace(min(ei_x(mask_f)), max(ei_x(mask_f)), 200)';

figure;
subplot(1, 2, 1);
errorbar(EI_grand, cv1_grand, cv1_sem, cv1_sem, EI_sem, EI_sem, ...
    'b.', 'MarkerSize', 10); hold on;
plot(xFit, fitRes1(xFit), 'r-', 'LineWidth', 1.5);
xlabel('E/I Ratio'); ylabel('CV');
title(sprintf('A=%.3f B=%.3f C=%.3f', fitRes1.A, fitRes1.B, fitRes1.C));
legend('Data', 'Fit');

subplot(1, 2, 2);
errorbar(EI_grand, ICE_vec', ICE_sem', ICE_sem', EI_sem, EI_sem, ...
    'k.', 'MarkerSize', 10); hold on;
plot(xFit, fitRes2(xFit), 'r-', 'LineWidth', 1.5);
xlabel('E/I Ratio'); ylabel('ICE [bits/J]');
title(sprintf('A=%.3f B=%.3f C=%.3f', fitRes2.A, fitRes2.B, fitRes2.C));
legend('Data', 'Fit');

%% Save MI Results
save('MI_results_direct_1.mat', ...
    'Hr_vec', 'Hn_vec', 'MI_vec', 'ICE_vec', 'timewindow', ...
    'MI_sem', 'Hr_sem', 'Hn_sem', 'ICE_sem', ...
    'MI_per_stim', 'Hr_per_stim', 'Hn_per_stim', 'ICE_per_stim', ...
    'cv1_grand', 'ss1_grand', 'ent1_grand', 'ra1_grand', 'cost1_grand', 'EI_grand', ...
    'cv1_sem', 'ss1_sem', 'ent1_sem', 'ra1_sem', 'cost1_sem', 'EI_sem', ...
    'Spike_cell', 'Binary_cell', 'NStim', 'NReps', 'NWindows', ...
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

%% STA — last rep of last stim
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

fprintf('\n=== Done. MI results saved to MI_results_direct.mat ===\n');


% ============================ local helpers ============================
% Mirror compute_entropy_toolbox's total_entropy / noise_entropy semantics
% but expose them as separate calls so we can pool across stims for the
% total but stay within stim for the noise.

function H = total_entropy_pool(spike_matrix, T)
    % Concatenate all rows, slice into length-T words, Shannon entropy.
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
    % For repeats of the same stim: at each time position j,
    % gather the [trials x T] words across reps, compute their entropy,
    % then average across non-trivial time positions.
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
    if any(valid)
        H = mean(entropy_values(valid));
    else
        H = 0;
    end
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