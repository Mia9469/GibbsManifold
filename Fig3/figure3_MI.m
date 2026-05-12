%% figure4_MI.m
%  Modified simulation with Mutual Information (MI) computation.
%
%  Based on figure4_formatted.m, with the following additions:
%    - Each stimulus template is pre-generated and fixed across repetitions.
%    - 10 repetitions per stimulus (different spike noise realizations).
%    - Per-window statistics are averaged across repetitions.
%    - MI = H(r) - H(r|s) is computed from pooled instantaneous firing rates:
%        H(r)   : entropy of rate1 pooled across ALL stimuli and repetitions
%        H(r|s) : mean entropy of rate1 pooled across repetitions per stimulus
%
%  Structure:
%    for stim = 1:NStim        (20 stimulus templates)
%      pre-generate InputTemplate
%      for rep = 1:NReps       (10 repetitions)
%        run full simulation (NRuns = 80 runs with iSTDP)
%        store per-window rate1 in R_cell{stim, rep, window}
%      end
%    end
%    compute MI per window from R_cell
%% Initialization
close all
%% Simulation Structure Parameters
NStim   = 20;       % Number of stimulus templates (outer loop)
NReps   = 10;       % Number of repetitions per stimulus
%% Model Parameters
% --- Learning rule ---
eta           = 0.0012;        % Learning rate
alpha         = 0.25 * eta;    % Decay term
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
%% Allocate Global Storage for MI Computation
% Cell array storing rate1 vectors for each (stim, rep, window)
R_cell = cell(NStim, NReps, NWindows);
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
    % The input signal is generated from time-filtered white noise.
    % We pre-generate it for all NRuns runs so that it is identical
    % across repetitions. Only the Poisson spike generation differs.
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
        Synapse   = Synapse_init;    % Reset to initial tuning each rep
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
                % Instead of generating random input at each step,
                % we look up the pre-generated stimulus template.
                globalStep = (runcount - 1) * nStepsPerRun + (t - 1);
                Input = InputTemplate(:, globalStep)';
                input_record(:, t-1) = Input;
                %% Presynaptic Spike Trains
                for i = 1:NCells
                    if (rand < Input(abs(InputGroup(i))) && InputSpikeRefr(i) <= 0)
                        if (InputGroup(i) > 0)
                            % Excitatory synapse
                            gEx = gEx + (gBarEx * Synapse(i));
                            sgEx(abs(InputGroup(i))) = sgEx(abs(InputGroup(i))) + gBarEx * Synapse(i);
                        else
                            % Inhibitory synapse
                            gIn = gIn + gBarIn * Synapse(i);
                            sgIn(abs(InputGroup(i))) = sgIn(abs(InputGroup(i))) + gBarIn * Synapse(i);
                            % --- iSTDP update ---
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
                spiketime = [];
                rateall   = [rateall rate1];
                if ~isempty(InCurr) && ~isempty(ExCurr)
                    EI_vec(w) = mean(ExCurr) / mean(InCurr);
                    E_vec(w)  = mean(ExCurr);
                    I_vec(w)  = mean(InCurr);
                end
                ExCurr = [];
                InCurr = [];
                % === Store for MI computation ===
                R_cell{stim, rep, w} = rate1;
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
    %%  Per-Stimulus Analysis (last rep)    %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    x = linspace(0, 3, 100);
    %% Sparseness vs CV (ISI-based rate, mean across reps)
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
    %% CV and Mean Value Diagnostic Plots
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
        t_idx = r * 150;
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
        uns  = mean(unsim(unsim ~= 0))
        pair = mean(E_I(:, k))
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
    % --- Save per-stimulus data (mean across reps) ---
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
%%      Mutual Information (MI)         %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Compute MI per Window
% MI(w) = H(r, w) - H(r|s, w)
%   H(r, w)   : response entropy — entropy of rate1 pooled across
%                ALL stimuli and ALL repetitions for window w
%   H(r|s, w) : noise entropy — for each stimulus s, compute entropy
%                of rate1 pooled across repetitions, then average over s
Hr_vec  = zeros(1, NWindows);
Hrs_vec = zeros(1, NWindows);
MI_vec  = zeros(1, NWindows);
for w = 1:NWindows
    %% H(r|s) — Noise Entropy
    H_noise = zeros(1, NStim);
    for s = 1:NStim
        % Pool rate1 across all repetitions for stimulus s
        pooled = [];
        for r = 1:NReps
            if ~isempty(R_cell{s, r, w})
                pooled = [pooled R_cell{s, r, w}];
            end
        end
        if ~isempty(pooled)
            binEdges = 0:0.1:(max(pooled)+0.1);
            if length(binEdges) < 2
                binEdges = [0 0.1];
            end
            [n, ~] = histcounts(pooled, binEdges);
            h_prob = n / sum(n);
            H_noise(s) = -sum(h_prob(h_prob > 0) .* log2(h_prob(h_prob > 0)));
        end
    end
    Hrs_vec(w) = mean(H_noise);
    %% H(r) — Response Entropy
    pooled_all = [];
    for s = 1:NStim
        for r = 1:NReps
            if ~isempty(R_cell{s, r, w})
                pooled_all = [pooled_all R_cell{s, r, w}];
            end
        end
    end
    if ~isempty(pooled_all)
        binEdges = 0:0.1:(max(pooled_all)+0.1);
        if length(binEdges) < 2
            binEdges = [0 0.1];
        end
        [n, ~] = histcounts(pooled_all, binEdges);
        h_prob = n / sum(n);
        Hr_vec(w) = -sum(h_prob(h_prob > 0) .* log2(h_prob(h_prob > 0)));
    end
    %% MI
    MI_vec(w) = Hr_vec(w) - Hrs_vec(w);
end
%% Compute Mean Statistics Across All Stimuli and Reps
cv1_grand   = squeeze(mean(mean(cv1_store, 1), 2));   % (NWindows x 1)
ss1_grand   = squeeze(mean(mean(ss1_store, 1), 2));
ent1_grand  = squeeze(mean(mean(ent1_store, 1), 2));
ra1_grand   = squeeze(mean(mean(ra1_store, 1), 2));
cost1_grand = squeeze(mean(mean(cost1_store, 1), 2));
EI_grand    = squeeze(mean(mean(EI_store, 1), 2));

% SEM across reps (average over stim first, then SEM over reps)
% _store is [NStim x NReps x NWindows]
cv1_sem   = squeeze(std(mean(cv1_store, 1), 0, 2))   / sqrt(NReps);
ss1_sem   = squeeze(std(mean(ss1_store, 1), 0, 2))   / sqrt(NReps);
ent1_sem  = squeeze(std(mean(ent1_store, 1), 0, 2))  / sqrt(NReps);
ra1_sem   = squeeze(std(mean(ra1_store, 1), 0, 2))   / sqrt(NReps);
cost1_sem = squeeze(std(mean(cost1_store, 1), 0, 2)) / sqrt(NReps);
EI_sem    = squeeze(std(mean(EI_store, 1), 0, 2))    / sqrt(NReps);

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
x2 = [timewindow, fliplr(timewindow)];

%% Plot MI, H(r), H(r|s) vs Time — with shaded SEM
figure;
% --- Subplot 1: H(r), H(r|s), MI ---
subplot(2, 2, 1);
% H(r)
fill(x2, [Hr_vec + Hr_sem, fliplr(Hr_vec - Hr_sem)], ...
     'k', 'FaceAlpha', 0.15, 'EdgeColor', 'none'); hold on;
plot(timewindow, Hr_vec, 'k-', 'LineWidth', 1.5);
% H(r|s)
fill(x2, [Hrs_vec + Hrs_sem, fliplr(Hrs_vec - Hrs_sem)], ...
     'b', 'FaceAlpha', 0.15, 'EdgeColor', 'none');
plot(timewindow, Hrs_vec, 'b-', 'LineWidth', 1.5);
% MI
fill(x2, [MI_vec + MI_sem, fliplr(MI_vec - MI_sem)], ...
     'r', 'FaceAlpha', 0.15, 'EdgeColor', 'none');
plot(timewindow, MI_vec, 'r-', 'LineWidth', 1.5);
legend('', 'H(r)', '', 'H(r|s)', '', 'MI');
xlabel('Time [s]'); ylabel('Entropy / MI [bits]');
title('Mutual Information over Learning');

% --- Subplot 2: MI only ---
subplot(2, 2, 2);
fill(x2, [MI_vec + MI_sem, fliplr(MI_vec - MI_sem)], ...
     'r', 'FaceAlpha', 0.15, 'EdgeColor', 'none'); hold on;
plot(timewindow, MI_vec, 'r-', 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('MI [bits]');
title('MI = H(r) - H(r|s)');

% --- Subplot 3: ICE ---
subplot(2, 2, 3);
fill(x2, [ICE_vec + ICE_sem, fliplr(ICE_vec - ICE_sem)], ...
     'm', 'FaceAlpha', 0.15, 'EdgeColor', 'none'); hold on;
plot(timewindow, ICE_vec, 'm-', 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('ICE [bits/J]');
title('Information-Cost Efficiency (ICE)');

% --- Subplot 4: CV & Sparseness with SEM ---
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

%% Plot cost vs MI / ICE — with error bars on both axes
figure;
subplot(1, 2, 1)
errorbar(cost1_grand, MI_vec', MI_sem', MI_sem', cost1_sem, cost1_sem, ...
    'ko', 'CapSize', 3, 'LineWidth', 1); hold on;
xlabel('Cost [W]'); ylabel('MI [bits]'); title('Cost vs MI');

subplot(1, 2, 2)
errorbar(cost1_grand, ICE_vec', ICE_sem', ICE_sem', cost1_sem, cost1_sem, ...
    'bs', 'CapSize', 3, 'LineWidth', 1); hold on;
xlabel('Cost [W]'); ylabel('ICE [bits/J]'); title('Cost vs ICE');

%% Plot CV vs MI / H(r) / H(r|s) / ICE — with error bars on both axes
figure;
subplot(1, 4, 1)
errorbar(cv1_grand, Hr_vec', Hr_sem', Hr_sem', cv1_sem, cv1_sem, ...
    'ko', 'CapSize', 3, 'LineWidth', 1); hold on;
xlabel('CV'); ylabel('H(r) [bits]'); title('CV vs H(r)');

subplot(1, 4, 2)
errorbar(cv1_grand, Hrs_vec', Hrs_sem', Hrs_sem', cv1_sem, cv1_sem, ...
    'bs', 'CapSize', 3, 'LineWidth', 1); hold on;
xlabel('CV'); ylabel('H(r|s) [bits]'); title('CV vs H(r|s)');

subplot(1, 4, 3)
errorbar(cv1_grand, MI_vec', MI_sem', MI_sem', cv1_sem, cv1_sem, ...
    'r^', 'CapSize', 3, 'LineWidth', 1); hold on;
xlabel('CV'); ylabel('MI [bits]'); title('CV vs MI');

subplot(1, 4, 4)
errorbar(cv1_grand, ICE_vec', ICE_sem', ICE_sem', cv1_sem, cv1_sem, ...
    'r.', 'CapSize', 3, 'MarkerSize', 12, 'LineWidth', 1); hold on;
xlabel('CV'); ylabel('ICE [bits/J]'); title('CV vs ICE');

%% Plot EIratio vs ICE & CV — with error bars + exponential fit
% Define the fit type
ft = fittype('A * exp(-B * (x - 1)) + C', 'independent', 'x');
% Set start points and options
opts = fitoptions(ft);
opts.StartPoint = [1, 1, 0];  % [A, B, C] initial guesses
opts.Lower      = [-Inf, 0, -Inf];
% Fit subplot 1: cv1_grand vs EI_grand
[fitRes1, gof1] = fit(EI_grand(:), cv1_grand(:), ft, opts);
% Fit subplot 2: ICE_vec vs EI_grand
[fitRes2, gof2] = fit(EI_grand(:), ICE_vec(:), ft, opts);
% Plot
xFit = linspace(min(EI_grand), max(EI_grand), 200)';
figure;
subplot(1,2,1);
errorbar(EI_grand, cv1_grand, cv1_sem, cv1_sem, EI_sem, EI_sem, ...
    'b.', 'CapSize', 3, 'MarkerSize', 10, 'LineWidth', 1); hold on;
plot(xFit, fitRes1(xFit), 'r-', 'LineWidth', 1.5);
xlabel('E/I Ratio'); ylabel('CV');
title(sprintf('A=%.3f B=%.3f C=%.3f', fitRes1.A, fitRes1.B, fitRes1.C));
legend('Data', 'Fit');

subplot(1,2,2);
errorbar(EI_grand, ICE_vec', ICE_sem', ICE_sem', EI_sem, EI_sem, ...
    'k.', 'CapSize', 3, 'MarkerSize', 10, 'LineWidth', 1); hold on;
plot(xFit, fitRes2(xFit), 'r-', 'LineWidth', 1.5);
xlabel('E/I Ratio'); ylabel('ICE [bits/J]');
title(sprintf('A=%.3f B=%.3f C=%.3f', fitRes2.A, fitRes2.B, fitRes2.C));
legend('Data', 'Fit');

%% Save MI Results (including jackknife and SEM)
save('MI_results.mat', ...
    'Hr_vec', 'Hrs_vec', 'MI_vec', 'ICE_vec', 'EI_vec', 'timewindow', ...
    'MI_sem', 'Hr_sem', 'Hrs_sem', 'ICE_sem', ...
    'MI_jack', 'Hr_jack', 'Hrs_jack', 'ICE_jack', ...
    'cv1_grand', 'ss1_grand', 'ent1_grand', 'ra1_grand', 'cost1_grand', 'EI_grand', ...
    'cv1_sem', 'ss1_sem', 'ent1_sem', 'ra1_sem', 'cost1_sem', 'EI_sem', ...
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
