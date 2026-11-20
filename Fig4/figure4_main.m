% figure4_main.m  (LEGACY; retained for provenance, not current inference)
%
% IMPORTANT: the current manuscript uses recompute_network_information.py.
% The historical entropy helper mismatched total/noise window starts and
% excluded valid zero-entropy positions. Those are implementation defects.
% The replacement ALSO changes pooled to neuron-wise readout, 0.1-ms to 1-ms
% bins, and the input/repeat protocol. These are analysis choices, not all
% bug fixes. Pooled MIE_ST is neither joint-population MI nor mean-neuron MI.
% Do not interpret a trend change as the effect of the two helper fixes alone.
% The saved weights remain unchanged. See the controlled endpoint audit in
% outputs/network_MI_change_explanation_20260907/MI_changes_CN.md.
%
% Historical Direct-Method (Strong et al. 1998) version of Figure5.m
%
% Key difference from Figure5.m:
%   - Uses FROZEN stimulus paradigm: one stimulus per learning stage,
%     repeated across Trials to estimate noise entropy via trial-to-trial
%     variability.
%   - Temporal and Spatial+Temporal MI computed with compute_entropy_toolbox
%     (direct method with 1/T extrapolation) instead of histogram binning.
%   - Spatial-only MI still uses histogram method (spike counts, not binary words).
%
% Requires: checkpoint.mat, runnet(), compute_entropy_toolbox()
load checkpoint.mat
addpath('/Users/mia/Library/Mobile Documents/com~apple~CloudDocs/compute_entropy');
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%  Computing Decoding Error, rates through Learning %%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%
%%%%% Frozen stimulus paradigm: for each learning stage, generate ONE input
%%%%% and present it Trials times. This enables the direct method for MI
%%%%% estimation (noise entropy from repeated trials).
%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Computing decoding errors and rates over learning (direct method)\n')
TimeT = 10000;   % size of the test input (time steps)
Trials = 10;     % number of repeated presentations of the frozen stimulus
% --- Basic statistics arrays ---
MeanPrateE  = zeros(1,T);
MeanPrateI  = zeros(1,T);
ErrorE      = zeros(1,T);
ErrorI      = zeros(1,T);
MembraneVarE = zeros(1,T);
MembraneVarI = zeros(1,T);
% --- Population entropy / cost arrays [Trials x T] ---
PentrE  = zeros(Trials,T);
PentrEl = zeros(Trials,T);
PentrIl = zeros(Trials,T);
PcostE  = zeros(Trials,T);
PentrI  = zeros(Trials,T);
PcostI  = zeros(Trials,T);
% --- Three-level MI arrays [Trials x T] ---
MIE_spatial  = zeros(Trials,T);  MII_spatial  = zeros(Trials,T);
MIE_temporal = zeros(Trials,T);  MII_temporal = zeros(Trials,T);
MIE_ST       = zeros(Trials,T);  MII_ST       = zeros(Trials,T);
% --- Corresponding Hr arrays ---
HrE_spatial  = zeros(Trials,T);  HrI_spatial  = zeros(Trials,T);
HrE_temporal = zeros(Trials,T);  HrI_temporal = zeros(Trials,T);
HrE_ST       = zeros(Trials,T);  HrI_ST       = zeros(Trials,T);
% --- Corresponding Hn (noise entropy) arrays for direct method ---
HnE_temporal = zeros(1,T);  HnI_temporal = zeros(1,T);
HnE_ST       = zeros(1,T);  HnI_ST       = zeros(1,T);
% --- Per-neuron MI storage for temporal error bars ---
mi_E_neurons_all = zeros(T, NneuronE);   % [stages x neurons]
mi_I_neurons_all = zeros(T, NneuronI);
hr_E_neurons_all = zeros(T, NneuronE);
hr_I_neurons_all = zeros(T, NneuronI);
% --- ST jackknife arrays ---
MIE_ST_jack = zeros(Trials, T);   % [leave-one-out x stages]
MII_ST_jack = zeros(Trials, T);
xT = zeros(Nx, TimeT);
% Spatial MI parameters (histogram method, same as original)
nbins_stim_spatial = 5;
nBins_resp = 30;
W_spatial  = 100;
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%  Main Computation Loop  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:T  % for each learning stage (OUTER loop)
    fprintf('  Learning stage %d / %d\n', i, T);
    % === Generate ONE frozen input for this learning stage ===
    InputT_frozen = A*(mvnrnd(zeros(1,Nx), eye(Nx), TimeT))';
    for k = 1:Nx
        InputT_frozen(k,:) = conv(InputT_frozen(k,:), w, 'same');
        InputT_frozen(k,:) = InputT_frozen(k,:) + rand(size(InputT_frozen(k,:)));
    end
    xT = zeros(Nx, TimeT);
    for t = 2:TimeT
        xT(:,t) = (1-lambda*dt)*xT(:,t-1) + dt*InputT_frozen(:,t-1);
    end
    % === Run network Trials times with the SAME frozen input ===
    OET_trials = cell(1, Trials);
    OIT_trials = cell(1, Trials);
    for r = 1:Trials
        [rOET, OET, VET, rOIT, OIT, VIT, ~, ~] = runnet(dt, TimeT, lambda, ...
            NneuronE, NneuronI, ThreshE, ThreshI, InputT_frozen, ...
            squeeze(FsE(i,:,:)), squeeze(CsEE(i,:,:)), squeeze(CsEI(i,:,:)), ...
            squeeze(CsII(i,:,:)), squeeze(CsIE(i,:,:)));
        OET_trials{r} = OET;  % [NneuronE x TimeT] binary spike matrix
        OIT_trials{r} = OIT;  % [NneuronI x TimeT] binary spike matrix
        % --- Decode and accumulate basic statistics (same as original) ---
        xestE = squeeze(DecsE(i,:,:)) * rOET;
        xestI = squeeze(DecsI(i,:,:)) * rOIT;
        if (i == 1 && r == Trials)
            xestEB = xestE;
            xestIB = xestI;
            OEB = OET;
            OIB = OIT;
        end
        ErrorE(1,i)      = ErrorE(1,i)      + sum(var(xT-xestE,0,2)) / (sum(var(xT,0,2))*Trials);
        ErrorI(1,i)      = ErrorI(1,i)      + sum(var(xT-xestI,0,2)) / (sum(var(xT,0,2))*Trials);
        MeanPrateE(1,i)  = MeanPrateE(1,i)  + sum(sum(OET)) / (TimeT*dt*NneuronE*Trials);
        MeanPrateI(1,i)  = MeanPrateI(1,i)  + sum(sum(OIT)) / (TimeT*dt*NneuronI*Trials);
        MembraneVarE(1,i)= MembraneVarE(1,i)+ sum(var(VET,0,2)) / (NneuronE*Trials);
        MembraneVarI(1,i)= MembraneVarI(1,i)+ sum(var(VIT,0,2)) / (NneuronI*Trials);
        % --- Population entropy and cost (histogram-based, same as original) ---
        % NOTE: PrateE/PrateI are per-neuron spike COUNTS over TimeT bins.
        % Histogram-based PentrE/PentrI use the count distribution (units
        % don't matter for entropy of a discrete distribution), but cost
        % needs firing rate in Hz: r_Hz = count / (TimeT*dt).  Without the
        % T_obs division, cost is inflated by a factor TimeT*dt.
        T_obs_loop  = TimeT * dt;
        PrateE      = sum(OET, 2);                             % counts (for histogram)
        PrateE_Hz   = PrateE / T_obs_loop;                     % Hz       (for cost)
        [counts, ~] = histcounts(PrateE, 'Normalization', 'probability');
        counts(counts == 0) = [];
        PentrE(r,i) = -sum(counts .* log2(counts));
        PcostE(r,i) = 2.4 + 0.37 * mean(PrateE_Hz);

        PrateI      = sum(OIT, 2);                             % counts (for histogram)
        PrateI_Hz   = PrateI / T_obs_loop;                     % Hz       (for cost)
        [counts, ~] = histcounts(PrateI, 'Normalization', 'probability');
        counts(counts == 0) = [];
        PentrI(r,i) = -sum(counts .* log2(counts));
        PcostI(r,i) = 2.4 + 0.37 * mean(PrateI_Hz);
        % Per-neuron filtered-rate entropy (histogram, for reference)
        entr = zeros(1, NneuronE);
        for k = 1:NneuronE
            [counts, ~] = histcounts(rOET(k,:), 'Normalization', 'probability');
            counts(counts == 0) = [];
            entr(k) = -sum(counts .* log2(counts));
        end
        PentrEl(r,i) = mean(entr);
        entr = zeros(1, NneuronI);
        for k = 1:NneuronI
            [counts, ~] = histcounts(rOIT(k,:), 'Normalization', 'probability');
            counts(counts == 0) = [];
            entr(k) = -sum(counts .* log2(counts));
        end
        PentrIl(r,i) = mean(entr);
        % === LEVEL 1: SPATIAL-ONLY MI (histogram method, same as original) ===
        input_signal = mean(InputT_frozen, 1);  % 1 x TimeT
        nWin = floor(TimeT / W_spatial);
        spatial_counts_E = zeros(NneuronE, nWin);
        spatial_counts_I = zeros(NneuronI, nWin);
        input_win = zeros(1, nWin);
        for w_idx = 1:nWin
            tidx = (w_idx-1)*W_spatial + 1 : w_idx*W_spatial;
            spatial_counts_E(:, w_idx) = sum(OET(:, tidx), 2);
            spatial_counts_I(:, w_idx) = sum(OIT(:, tidx), 2);
            input_win(w_idx) = mean(input_signal(tidx));
        end
        stim_win = discretize(input_win, nbins_stim_spatial);
        valid_win = ~isnan(stim_win);
        % E spatial-only
        pool_E = spatial_counts_E(:, valid_win);
        pool_E = pool_E(:)';
        stim_exp_E = repmat(stim_win(valid_win), NneuronE, 1);
        stim_exp_E = stim_exp_E(:)';
        if ~isempty(pool_E) && length(unique(pool_E)) > 1
            [~, edges_sp_E] = histcounts(pool_E, nBins_resp);
            [n_r, ~] = histcounts(pool_E, edges_sp_E);
            p_r = n_r/sum(n_r); p_r = p_r(p_r>0);
            Hr_sp_E = -sum(p_r .* log2(p_r));
            Hrs_sp_E = 0;
            for s = unique(stim_exp_E)
                mask = (stim_exp_E == s);
                P_s = sum(mask)/length(stim_exp_E);
                [n_s, ~] = histcounts(pool_E(mask), edges_sp_E);
                p_s = n_s/sum(n_s); p_s = p_s(p_s>0);
                if ~isempty(p_s)
                    Hrs_sp_E = Hrs_sp_E + P_s*(-sum(p_s .* log2(p_s)));
                end
            end
            MIE_spatial(r,i) = max(0, Hr_sp_E - Hrs_sp_E);
            HrE_spatial(r,i) = Hr_sp_E;
        end
        % I spatial-only
        pool_I = spatial_counts_I(:, valid_win);
        pool_I = pool_I(:)';
        stim_exp_I = repmat(stim_win(valid_win), NneuronI, 1);
        stim_exp_I = stim_exp_I(:)';
        if ~isempty(pool_I) && length(unique(pool_I)) > 1
            [~, edges_sp_I] = histcounts(pool_I, nBins_resp);
            [n_r, ~] = histcounts(pool_I, edges_sp_I);
            p_r = n_r/sum(n_r); p_r = p_r(p_r>0);
            Hr_sp_I = -sum(p_r .* log2(p_r));
            Hrs_sp_I = 0;
            for s = unique(stim_exp_I)
                mask = (stim_exp_I == s);
                P_s = sum(mask)/length(stim_exp_I);
                [n_s, ~] = histcounts(pool_I(mask), edges_sp_I);
                p_s = n_s/sum(n_s); p_s = p_s(p_s>0);
                if ~isempty(p_s)
                    Hrs_sp_I = Hrs_sp_I + P_s*(-sum(p_s .* log2(p_s)));
                end
            end
            MII_spatial(r,i) = max(0, Hr_sp_I - Hrs_sp_I);
            HrI_spatial(r,i) = Hr_sp_I;
        end
    end  % end Trials loop
    % =====================================================================
    % === LEVEL 2: PER-NEURON TEMPORAL MI (Direct Method) =================
    % =====================================================================
    % For each neuron, stack spike trains from all Trials as rows.
    % spike_mat = [Trials x TimeT] binary matrix.
    % Pass as both spike_long (total entropy) and spike_repeat (noise entropy).
    % The toolbox handles them differently:
    %   - total_entropy: concatenates all rows, cuts into words -> full repertoire
    %   - noise_entropy: at each time bin, looks across trials -> variability
    % Returns MI in bits/s directly.
    mi_E_neurons = zeros(1, NneuronE);
    hr_E_neurons = zeros(1, NneuronE);
    hn_E_neurons = zeros(1, NneuronE);
    for k = 1:NneuronE
        spike_mat = zeros(Trials, TimeT);
        for r = 1:Trials
            spike_mat(r,:) = OET_trials{r}(k,:);
        end
        % Skip neurons with no spikes at all
        if sum(spike_mat(:)) == 0
            continue;
        end
        res = compute_entropy_toolbox(spike_mat, spike_mat, 'dt', dt, 'refractory', 0);
        mi_E_neurons(k) = max(0, res.MI);
        hr_E_neurons(k) = res.Hr;
        hn_E_neurons(k) = res.Hn;
    end
    % Store per-trial placeholders as mean across neurons (one value per stage)
    % We replicate across the Trials dimension for compatibility with plotting code
    MIE_temporal(:,i) = mean(mi_E_neurons);
    HrE_temporal(:,i) = mean(hr_E_neurons);
    HnE_temporal(i)   = mean(hn_E_neurons);
    % Save per-neuron MI for cross-neuron SEM
    mi_E_neurons_all(i,:) = mi_E_neurons;
    hr_E_neurons_all(i,:) = hr_E_neurons;
    mi_I_neurons = zeros(1, NneuronI);
    hr_I_neurons = zeros(1, NneuronI);
    hn_I_neurons = zeros(1, NneuronI);
    for k = 1:NneuronI
        spike_mat = zeros(Trials, TimeT);
        for r = 1:Trials
            spike_mat(r,:) = OIT_trials{r}(k,:);
        end
        if sum(spike_mat(:)) == 0
            continue;
        end
        res = compute_entropy_toolbox(spike_mat, spike_mat, 'dt', dt, 'refractory', 0);
        mi_I_neurons(k) = max(0, res.MI);
        hr_I_neurons(k) = res.Hr;
        hn_I_neurons(k) = res.Hn;
    end
    MII_temporal(:,i) = mean(mi_I_neurons);
    HrI_temporal(:,i) = mean(hr_I_neurons);
    HnI_temporal(i)   = mean(hn_I_neurons);
    % Save per-neuron MI for cross-neuron SEM
    mi_I_neurons_all(i,:) = mi_I_neurons;
    hr_I_neurons_all(i,:) = hr_I_neurons;
    % =====================================================================
    % === LEVEL 3: SPATIAL+TEMPORAL MI (Direct Method) ====================
    % =====================================================================
    % Stack ALL neurons from ALL trials: [Trials*NneuronE x TimeT]
    % This captures both spatial diversity across neurons and temporal
    % patterns within each neuron.
    spike_pop_E = zeros(Trials * NneuronE, TimeT);
    for r = 1:Trials
        spike_pop_E((r-1)*NneuronE+1 : r*NneuronE, :) = OET_trials{r};
    end
    if sum(spike_pop_E(:)) > 0
        res_pop_E = compute_entropy_toolbox(spike_pop_E, spike_pop_E, 'dt', dt, 'refractory', 0);
        MIE_ST(:,i) = max(0, res_pop_E.MI);
        HrE_ST(:,i) = res_pop_E.Hr;
        HnE_ST(i)   = res_pop_E.Hn;
    end
    spike_pop_I = zeros(Trials * NneuronI, TimeT);
    for r = 1:Trials
        spike_pop_I((r-1)*NneuronI+1 : r*NneuronI, :) = OIT_trials{r};
    end
    if sum(spike_pop_I(:)) > 0
        res_pop_I = compute_entropy_toolbox(spike_pop_I, spike_pop_I, 'dt', dt, 'refractory', 0);
        MII_ST(:,i) = max(0, res_pop_I.MI);
        HrI_ST(:,i) = res_pop_I.Hr;
        HnI_ST(i)   = res_pop_I.Hn;
    end
    % --- ST Jackknife: leave-one-trial-out ---
    for jk = 1:Trials
        idx = setdiff(1:Trials, jk);
        % E population
        spike_jk_E = zeros((Trials-1)*NneuronE, TimeT);
        for rr = 1:length(idx)
            spike_jk_E((rr-1)*NneuronE+1:rr*NneuronE, :) = OET_trials{idx(rr)};
        end
        if sum(spike_jk_E(:)) > 0
            res_jk = compute_entropy_toolbox(spike_jk_E, spike_jk_E, 'dt', dt, 'refractory', 0);
            MIE_ST_jack(jk,i) = max(0, res_jk.MI);
        end
        % I population
        spike_jk_I = zeros((Trials-1)*NneuronI, TimeT);
        for rr = 1:length(idx)
            spike_jk_I((rr-1)*NneuronI+1:rr*NneuronI, :) = OIT_trials{idx(rr)};
        end
        if sum(spike_jk_I(:)) > 0
            res_jk = compute_entropy_toolbox(spike_jk_I, spike_jk_I, 'dt', dt, 'refractory', 0);
            MII_ST_jack(jk,i) = max(0, res_jk.MI);
        end
    end
end  % end learning stages loop
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%  Plotting Decoding Error, rates through Learning  %%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
orange = [1 0.5 0.2];
%% Convert spatial MI to rate [bits/s] for ICE computation
T_obs = TimeT * dt;                   % [s] total test observation duration
% Population entropy rate [bits/s]
PentrE_rate = PentrE / T_obs;
PentrI_rate = PentrI / T_obs;
% Spatial MI rate [bits/s] — only spatial needs conversion (histogram gives raw bits)
MIrateE_spatial = MIE_spatial / T_obs;
MIrateI_spatial = MII_spatial / T_obs;
% Temporal and S+T MI are ALREADY in bits/s from compute_entropy_toolbox
MIrateE_temporal = MIE_temporal;
MIrateI_temporal = MII_temporal;
MIrateE_ST       = MIE_ST;
MIrateI_ST       = MII_ST;
% PcostE, PcostI are calibrated affine cost-proxy values
% ICE = MI_rate / calibrated cost proxy
%% Compute SEM for each level
% Spatial: SEM across trials (per-trial values exist)
semE_sp = std(MIE_spatial, 0, 1) / sqrt(Trials);
semI_sp = std(MII_spatial, 0, 1) / sqrt(Trials);
semHrE_sp = std(HrE_spatial, 0, 1) / sqrt(Trials);
semHrI_sp = std(HrI_spatial, 0, 1) / sqrt(Trials);
% Temporal: SEM across neurons (per-neuron MI saved in mi_E/I_neurons_all)
% mi_E_neurons_all is [T x NneuronE] → std over dim 2, transpose to [1 x T]
semE_tp = std(mi_E_neurons_all, 0, 2)' / sqrt(NneuronE);
semI_tp = std(mi_I_neurons_all, 0, 2)' / sqrt(NneuronI);
semHrE_tp = std(hr_E_neurons_all, 0, 2)' / sqrt(NneuronE);
semHrI_tp = std(hr_I_neurons_all, 0, 2)' / sqrt(NneuronI);
% S+T: Jackknife SEM across trials
N_jk = Trials;
semE_st = sqrt((N_jk-1)/N_jk * sum((MIE_ST_jack - mean(MIE_ST_jack,1)).^2, 1));
semI_st = sqrt((N_jk-1)/N_jk * sum((MII_ST_jack - mean(MII_ST_jack,1)).^2, 1));
% ICE SEM
% Spatial ICE per trial → direct SEM
ICEE_sp = MIrateE_spatial ./ PcostE;
ICEI_sp = MIrateI_spatial ./ PcostI;
semICEE_sp = std(ICEE_sp, 0, 1) / sqrt(Trials);
semICEI_sp = std(ICEI_sp, 0, 1) / sqrt(Trials);
% Temporal ICE: neuron-level SEM / mean cost
meanPcostE = mean(PcostE, 1);
meanPcostI = mean(PcostI, 1);
semICEE_tp = semE_tp ./ meanPcostE;
semICEI_tp = semI_tp ./ meanPcostI;
% ST ICE: jackknife SEM / mean cost
semICEE_st = semE_st ./ meanPcostE;
semICEI_st = semI_st ./ meanPcostI;
% Population entropy SEM
semPentrE = std(PentrE, 0, 1) / sqrt(Trials);
semPentrI = std(PentrI, 0, 1) / sqrt(Trials);
semPcostE = std(PcostE, 0, 1) / sqrt(Trials);
semPcostI = std(PcostI, 0, 1) / sqrt(Trials);
semICE_pop_E = std(PentrE_rate./PcostE, 0, 1) / sqrt(Trials);
semICE_pop_I = std(PentrI_rate./PcostI, 0, 1) / sqrt(Trials);
%% Helper: shaded error band
shade_err = @(x, y, sem, col, alph) fill([x, fliplr(x)], ...
    [y+sem, fliplr(y-sem)], col, 'EdgeColor','none','FaceAlpha',alph);
x_time = (2.^(1:T(1,1)))*dt;
x2 = [x_time, fliplr(x_time)];
%% Figure 1: Evolution through learning (Cost, Entropy, ICE) + MI panels
figure
set(gcf,'Units','centimeters')
xSize = 50;  ySize = 34;
xLeft = (21-xSize)/4; yTop = (30-ySize)/4;
set(gcf,'Position',[xLeft yTop xSize ySize]);
set(gcf, 'Color', 'w');
lines = 6;
fsize = 11;
% --- Cost with SEM ---
h = subplot(lines,3,[1 4]);
ax = get(h,'Position');
set(gca, 'XScale', 'log'); hold on;
shade_err(x_time, mean(PcostE,1), semPcostE, orange, 0.15);
shade_err(x_time, mean(PcostI,1), semPcostI, 'b', 0.15);
plot(x_time, mean(PcostE,1), 'Color', orange, 'LineWidth', 1.5);
plot(x_time, mean(PcostI,1), 'b', 'LineWidth', 1.5);
set(gca,'FontSize',fsize,'FontName','Helvetica','TickDir','out')
ylabel('Calibrated cost proxy'); xlabel('time (s)');
title('Evolution Through Learning')
xlim([10^(-1) 10^6])
set(gca,'ticklength',[0.01 0.01]/ax(3)); box off
% --- Entropy with SEM ---
h = subplot(lines,3,[7 10]);
ax = get(h,'Position');
set(gca, 'XScale', 'log'); hold on;
shade_err(x_time, mean(PentrE,1), semPentrE, orange, 0.15);
shade_err(x_time, mean(PentrI,1), semPentrI, 'b', 0.15);
plot(x_time, mean(PentrE,1), 'Color', orange, 'LineWidth', 1.5);
plot(x_time, mean(PentrI,1), 'b', 'LineWidth', 1.5);
set(gca,'FontSize',fsize,'FontName','Helvetica','TickDir','out')
ylabel('Entropy [bits]'); xlabel('time (s)');
xlim([10^(-1) 10^6])
set(gca,'ticklength',[0.01 0.01]/ax(3)); box off
% --- Population ICE with SEM ---
h = subplot(lines,3,[13 16]);
ax = get(h,'Position');
set(gca, 'XScale', 'log'); hold on;
shade_err(x_time, mean(PentrE_rate./PcostE,1), semICE_pop_E, orange, 0.15);
shade_err(x_time, mean(PentrI_rate./PcostI,1), semICE_pop_I, 'b', 0.15);
plot(x_time, mean(PentrE_rate./PcostE,1), 'Color', orange, 'LineWidth', 1.5);
plot(x_time, mean(PentrI_rate./PcostI,1), 'b', 'LineWidth', 1.5);
set(gca,'FontSize',fsize,'FontName','Helvetica','TickDir','out')
ylabel('Information / calibrated cost proxy'); xlabel('time (s)');
xlim([10^(-1) 10^6])
set(gca,'ticklength',[0.01 0.01]/ax(3)); box off
% --- Three-level MI evolution with SEM ---
h = subplot(lines,3,[2 5]);
ax = get(h,'Position');
set(gca, 'XScale', 'log'); hold on;
% E population SEM bands
shade_err(x_time, mean(MIE_spatial,1), semE_sp, [0.85 0.85 0.85], 0.4);
shade_err(x_time, mean(MIE_temporal,1), semE_tp, [1 0.85 0.7], 0.4);
shade_err(x_time, mean(MIE_ST,1), semE_st, [1 0.7 0.4], 0.4);
% E mean lines
plot(x_time, mean(MIE_spatial,1), '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.5);
plot(x_time, mean(MIE_temporal,1), ':', 'Color', orange, 'LineWidth', 1.5);
plot(x_time, mean(MIE_ST,1), '-', 'Color', orange, 'LineWidth', 2);
% I population SEM bands
shade_err(x_time, mean(MII_spatial,1), semI_sp, [0.7 0.8 1], 0.3);
shade_err(x_time, mean(MII_temporal,1), semI_tp, [0.6 0.7 1], 0.3);
shade_err(x_time, mean(MII_ST,1), semI_st, [0.5 0.7 1], 0.3);
% I mean lines
plot(x_time, mean(MII_spatial,1), '--b', 'LineWidth', 1);
plot(x_time, mean(MII_temporal,1), ':b', 'LineWidth', 1.5);
plot(x_time, mean(MII_ST,1), '-b', 'LineWidth', 2);
set(gca,'FontSize',fsize,'FontName','Helvetica','TickDir','out')
ylabel('MI [bits/s]'); xlabel('time (s)');
title('MI Evolution (3 Levels, Direct Method)')
legend('','','','E Spatial','E Temporal','E S+T','','','','I Spatial','I Temporal','I S+T', ...
    'Location','best','FontSize',7)
xlim([10^(-1) 10^6])
set(gca,'ticklength',[0.01 0.01]/ax(3)); box off
% --- Three-level ICE with SEM ---
h = subplot(lines,3,[8 11]);
ax = get(h,'Position');
set(gca, 'XScale', 'log'); hold on;
% E population
shade_err(x_time, mean(ICEE_sp,1), semICEE_sp, [0.85 0.85 0.85], 0.4);
shade_err(x_time, mean(MIrateE_temporal./PcostE,1), semICEE_tp, [1 0.85 0.7], 0.4);
shade_err(x_time, mean(MIrateE_ST./PcostE,1), semICEE_st, [1 0.7 0.4], 0.4);
plot(x_time, mean(ICEE_sp,1), '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.5);
plot(x_time, mean(MIrateE_temporal./PcostE,1), ':', 'Color', orange, 'LineWidth', 1.5);
plot(x_time, mean(MIrateE_ST./PcostE,1), '-', 'Color', orange, 'LineWidth', 2);
% I population
shade_err(x_time, mean(ICEI_sp,1), semICEI_sp, [0.7 0.8 1], 0.3);
shade_err(x_time, mean(MIrateI_temporal./PcostI,1), semICEI_tp, [0.6 0.7 1], 0.3);
shade_err(x_time, mean(MIrateI_ST./PcostI,1), semICEI_st, [0.5 0.7 1], 0.3);
plot(x_time, mean(ICEI_sp,1), '--b', 'LineWidth', 1);
plot(x_time, mean(MIrateI_temporal./PcostI,1), ':b', 'LineWidth', 1.5);
plot(x_time, mean(MIrateI_ST./PcostI,1), '-b', 'LineWidth', 2);
set(gca,'FontSize',fsize,'FontName','Helvetica','TickDir','out')
ylabel('Information / calibrated cost proxy'); xlabel('time (s)');
title('MI per Cost (3 Levels)')
xlim([10^(-1) 10^6])
set(gca,'ticklength',[0.01 0.01]/ax(3)); box off
%% Figure 2: MI vs Cost scatter plots, with theoretical Gibbs frontier
% Frontier:  I(C)  = (R_0 / ln 2) * [1 + ln((C - C_0) / C_1)]
%            beta  = R_0 / ln 2 / (C - C_0)             (= dI/dC)
%            ICE   = I / C
% Per-system anchor: R_0 fixed once by requiring the minimum-rate
% empirical operating point to lie ON the frontier (free-trajectory rule
% used for all systems in main-text Fig. 6).
figure;
C_0 = 2.4;     % calibrated proxy baseline
C_1 = 0.37;    % calibrated proxy slope

% --- Min-rate anchor: smallest non-zero (PcostE, MIE_ST) operating point.
% Average across trials/jackknife for stability; require the point to be
% above C_0 so the ln in the frontier formula is defined.
PcostE_mean = mean(PcostE, 1, 'omitnan');                % [1 x T]
MIE_ST_mean = mean(MIE_ST, 1, 'omitnan');                % [1 x T]
valid_anchor = (PcostE_mean > C_0 + 1e-3) & (MIE_ST_mean > 0);
if any(valid_anchor)
    [~, ia] = min(PcostE_mean(valid_anchor));
    valid_idx = find(valid_anchor);
    a_idx = valid_idx(ia);
    C_anchor = PcostE_mean(a_idx);
    I_anchor = MIE_ST_mean(a_idx);
    R_0_rn   = I_anchor * log(2) / (1 + log((C_anchor - C_0) / C_1));
else
    warning('No valid anchor point; defaulting R_0 = 1.');
    R_0_rn = 1;
end
fprintf('Fig 5 frontier anchor:  Cproxy=%.2f, MI=%.2f bits/s  ->  R_0 = %.3f\n', ...
    C_anchor, I_anchor, R_0_rn);

x_ref  = C_0 + (0.001:0.1:max(15, max(PcostE_mean)));         % calibrated cost-proxy grid
r_vec  = (x_ref - C_0) / C_1;                                 % firing rate [Hz]
y1_ref = (R_0_rn / log(2)) * (1 + log(r_vec));                % I_frontier [bits/s]
y2_ref = (R_0_rn / log(2)) ./ (x_ref - C_0);                  % beta = dI/dCproxy
ICE    = y1_ref ./ x_ref;                                     % information / calibrated cost proxy

PcostE1 = PcostE;
PcostI1 = PcostI;
subplot(2,2,1); plot(x_ref, y1_ref, 'k-', 'LineWidth', 1.4); hold on
xlim([2 max(x_ref)])
subplot(2,2,3); plot(x_ref, y2_ref, 'k--', 'LineWidth', 1.0); hold on
plot(x_ref, ICE, 'k-', 'LineWidth', 1.4); hold on
xlim([2 max(x_ref)])
% for ii = 1:size(PentrE,2)
%     subplot(2,3,1)
%     scatter(PcostE1(:,ii), PentrE(:,ii), 36, orange, 'o'); hold on;
%     ylim([1 4])
%     subplot(2,3,4)
%     scatter(PcostE1(:,ii), PentrE_rate(:,ii)./PcostE1(:,ii), 36, orange, 'o'); hold on;
%     ylim([0 1])
% end
% MI vs Cost scatter plots (3 levels)
subplot(2,2,1)
for ii = 1:size(MIE_ST,2)
%     scatter(PcostE1(:,ii), MIE_spatial(:,ii), 20, [0.7 0.7 0.7], 'o'); hold on;
%     scatter(PcostE1(:,ii), MIE_temporal(:,ii), 20, [1 0.8 0.5], '^'); hold on;
    scatter(PcostE1(:,ii), MIE_ST(:,ii), 36, orange, 's', 'filled'); hold on;
end
xlabel('Calibrated cost proxy (E)'); ylabel('MI [bits/s]')
title('MI vs Cost (E, 3 Levels)')
ylim([0 6])
% legend('Spatial','Temporal','S+T','Location','best','FontSize',7); box off
subplot(2,2,3)
for ii = 1:size(MIE_ST,2)
%     scatter(PcostE1(:,ii), MIrateE_spatial(:,ii)./PcostE1(:,ii), 20, [0.7 0.7 0.7], 'o'); hold on;
%     scatter(PcostE1(:,ii), MIrateE_temporal(:,ii)./PcostE1(:,ii), 20, [1 0.8 0.5], '^'); hold on;
    scatter(PcostE1(:,ii), MIrateE_ST(:,ii)./PcostE1(:,ii), 36, orange, 's', 'filled'); hold on;
end
xlabel('Calibrated cost proxy (E)'); ylabel('Information / calibrated cost proxy')
ylim([0 1.2 ])
title('ICE vs Cost (E, 3 Levels)'); box off
% subplot(2,2,2)
% for ii = 1:size(MIE_ST,2)
% %     scatter(PentrE(:,ii), MIE_spatial(:,ii), 20, [0.7 0.7 0.7], 'o'); hold on;
% %     scatter(PentrE(:,ii), MIE_temporal(:,ii), 20, [1 0.8 0.5], '^'); hold on;
%     scatter(PentrE(:,ii), MIE_ST(:,ii), 36, orange, 's', 'filled'); hold on;
% end
% xlabel('Entropy E [bits]'); ylabel('MI [bits/s]')
% title('MI vs Entropy (E, 3 Levels)'); box off
% subplot(2,2,4)
% for ii = 1:size(MIE_ST,2)
% %     scatter(PentrE(:,ii), MIE_spatial(:,ii)./PentrE(:,ii), 20, [0.7 0.7 0.7], 'o'); hold on;
% %     scatter(PentrE(:,ii), MIE_temporal(:,ii)./PentrE(:,ii), 20, [1 0.8 0.5], '^'); hold on;
%     scatter(PentrE(:,ii), MIE_ST(:,ii)./PentrE(:,ii), 36, orange, 's', 'filled'); hold on;
% end
% xlabel('Entropy E [bits]'); ylabel('MI/Entropy')
% title('MI Efficiency (E, 3 Levels)'); box off
%% Figure 3: Entropy/MI time evolution with shaded SEM regions
figure;
subplot(1,3,1)
y = mean(PentrE,1);
set(gca, 'XScale', 'log'); hold on;
shade_err(x_time, y, semPentrE, [0.5 0.5 0.5], 0.4);
plot(x_time, y, 'k', 'LineWidth', 2);
xlim([10^(-1) 10^6]); ylabel('Entropy [bits]');
subplot(1,3,2)
y = mean(PcostE,1);
set(gca, 'XScale', 'log'); hold on;
shade_err(x_time, y, semPcostE, [0.5 0.5 0.5], 0.4);
plot(x_time, y, 'k', 'LineWidth', 2);
xlim([10^(-1) 10^6]); ylabel('Calibrated cost proxy');
subplot(1,3,3)
y = mean(PentrE_rate./PcostE,1);
set(gca, 'XScale', 'log'); hold on;
shade_err(x_time, y, semICE_pop_E, [0.5 0.5 0.5], 0.4);
plot(x_time, y, 'k', 'LineWidth', 2);
xlim([10^(-1) 10^6]); ylabel('Information / calibrated cost proxy');
%% Figure 4: Three-Level Mutual Information Analysis (with SEM)
figure;
% ---- E Population: MI Evolution with SEM ----
subplot(2,3,1)
set(gca, 'XScale', 'log'); hold on;
y_sp = mean(MIE_spatial,1); y_tp = mean(MIE_temporal,1); y_st = mean(MIE_ST,1);
shade_err(x_time, y_sp, semE_sp, [0.85 0.85 0.85], 0.5);
shade_err(x_time, y_tp, semE_tp, [1 0.85 0.7], 0.5);
shade_err(x_time, y_st, semE_st, [1 0.7 0.4], 0.5);
plot(x_time, y_sp, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.5);
plot(x_time, y_tp, ':', 'Color', orange, 'LineWidth', 1.5);
plot(x_time, y_st, '-', 'Color', orange, 'LineWidth', 2);
xlabel('time (s)'); ylabel('MI [bits/s]')
title('MI Evolution - E Pop (Direct)')
legend('','','','Spatial','Temporal','S+T','Location','best','FontSize',7)
xlim([10^(-1) 10^6]); grid on; box off
% ---- E Population: ICE with SEM ----
subplot(2,3,2)
set(gca, 'XScale', 'log'); hold on;
shade_err(x_time, mean(ICEE_sp,1), semICEE_sp, [0.85 0.85 0.85], 0.5);
shade_err(x_time, mean(MIrateE_temporal./PcostE,1), semICEE_tp, [1 0.85 0.7], 0.5);
shade_err(x_time, mean(MIrateE_ST./PcostE,1), semICEE_st, [1 0.7 0.4], 0.5);
plot(x_time, mean(ICEE_sp,1), '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.5);
plot(x_time, mean(MIrateE_temporal./PcostE,1), ':', 'Color', orange, 'LineWidth', 1.5);
plot(x_time, mean(MIrateE_ST./PcostE,1), '-', 'Color', orange, 'LineWidth', 2);
xlabel('time (s)'); ylabel('Information / calibrated cost proxy')
title('MI Efficiency - E Pop')
xlim([10^(-1) 10^6]); grid on; box off
% ---- E Population: H(r) decomposition with SEM ----
subplot(2,3,3)
set(gca, 'XScale', 'log'); hold on;
hr_sp_E = mean(HrE_spatial,1); hrs_sp_E = hr_sp_E - mean(MIE_spatial,1);
hr_tp_E = mean(HrE_temporal,1); hrs_tp_E = hr_tp_E - mean(MIE_temporal,1);
hr_st_E = mean(HrE_ST,1);       hrs_st_E = hr_st_E - mean(MIE_ST,1);
% SEM bands on Hr
shade_err(x_time, hr_sp_E, semHrE_sp, [0.85 0.85 0.85], 0.3);
shade_err(x_time, hr_tp_E, semHrE_tp, [1 0.85 0.7], 0.3);
% MI region fills (between Hr and Hn)
fill(x2, [hr_sp_E, fliplr(hrs_sp_E)], [0.85 0.85 0.85], 'EdgeColor','none','FaceAlpha',0.2);
fill(x2, [hr_tp_E, fliplr(hrs_tp_E)], [1 0.85 0.7], 'EdgeColor','none','FaceAlpha',0.2);
fill(x2, [hr_st_E, fliplr(hrs_st_E)], [1 0.7 0.4], 'EdgeColor','none','FaceAlpha',0.2);
% Hr (thick) and H(r|s) (thin) lines
plot(x_time, hr_sp_E, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.5);
plot(x_time, hrs_sp_E, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 0.5);
plot(x_time, hr_tp_E, ':', 'Color', orange, 'LineWidth', 1.5);
plot(x_time, hrs_tp_E, ':', 'Color', orange, 'LineWidth', 0.5);
plot(x_time, hr_st_E, '-', 'Color', orange, 'LineWidth', 2);
plot(x_time, hrs_st_E, '-', 'Color', orange, 'LineWidth', 0.5);
xlabel('time (s)'); ylabel('Entropy [bits/s]')
title('E: Hr (thick) & Hn (thin), shaded=MI')
legend('','','','','','Spatial Hr','Spatial Hn','Temporal Hr','Temporal Hn','S+T Hr','S+T Hn', ...
    'Location','best','FontSize',6)
xlim([10^(-1) 10^6]); grid on; box off
% ---- I Population: MI Evolution with SEM ----
subplot(2,3,4)
set(gca, 'XScale', 'log'); hold on;
y_sp = mean(MII_spatial,1); y_tp = mean(MII_temporal,1); y_st = mean(MII_ST,1);
shade_err(x_time, y_sp, semI_sp, [0.7 0.8 1], 0.5);
shade_err(x_time, y_tp, semI_tp, [0.6 0.7 1], 0.5);
shade_err(x_time, y_st, semI_st, [0.5 0.7 1], 0.5);
plot(x_time, y_sp, '--', 'Color', [0.5 0.5 0.8], 'LineWidth', 1.5);
plot(x_time, y_tp, ':', 'Color', 'b', 'LineWidth', 1.5);
plot(x_time, y_st, '-b', 'LineWidth', 2);
xlabel('time (s)'); ylabel('MI [bits/s]')
title('MI Evolution - I Pop (Direct)')
legend('','','','Spatial','Temporal','S+T','Location','best','FontSize',7)
xlim([10^(-1) 10^6]); grid on; box off
% ---- I Population: ICE with SEM ----
subplot(2,3,5)
set(gca, 'XScale', 'log'); hold on;
shade_err(x_time, mean(ICEI_sp,1), semICEI_sp, [0.7 0.8 1], 0.5);
shade_err(x_time, mean(MIrateI_temporal./PcostI,1), semICEI_tp, [0.6 0.7 1], 0.5);
shade_err(x_time, mean(MIrateI_ST./PcostI,1), semICEI_st, [0.5 0.7 1], 0.5);
plot(x_time, mean(ICEI_sp,1), '--b', 'LineWidth', 1);
plot(x_time, mean(MIrateI_temporal./PcostI,1), ':b', 'LineWidth', 1.5);
plot(x_time, mean(MIrateI_ST./PcostI,1), '-b', 'LineWidth', 2);
xlabel('time (s)'); ylabel('Information / calibrated cost proxy')
title('MI Efficiency - I Pop')
xlim([10^(-1) 10^6]); grid on; box off
% ---- I Population: H(r) decomposition with SEM ----
subplot(2,3,6)
set(gca, 'XScale', 'log'); hold on;
hr_sp_I = mean(HrI_spatial,1); hrs_sp_I = hr_sp_I - mean(MII_spatial,1);
hr_tp_I = mean(HrI_temporal,1); hrs_tp_I = hr_tp_I - mean(MII_temporal,1);
hr_st_I = mean(HrI_ST,1);       hrs_st_I = hr_st_I - mean(MII_ST,1);
% SEM bands on Hr
shade_err(x_time, hr_sp_I, semHrI_sp, [0.7 0.8 1], 0.3);
shade_err(x_time, hr_tp_I, semHrI_tp, [0.6 0.7 1], 0.3);
% MI region fills
fill(x2, [hr_sp_I, fliplr(hrs_sp_I)], [0.7 0.8 1], 'EdgeColor','none','FaceAlpha',0.2);
fill(x2, [hr_tp_I, fliplr(hrs_tp_I)], [0.6 0.7 1], 'EdgeColor','none','FaceAlpha',0.2);
fill(x2, [hr_st_I, fliplr(hrs_st_I)], [0.5 0.7 1], 'EdgeColor','none','FaceAlpha',0.2);
% Lines
plot(x_time, hr_sp_I, '--', 'Color', [0.5 0.5 0.8], 'LineWidth', 1.5);
plot(x_time, hrs_sp_I, '--', 'Color', [0.5 0.5 0.8], 'LineWidth', 0.5);
plot(x_time, hr_tp_I, ':', 'Color', 'b', 'LineWidth', 1.5);
plot(x_time, hrs_tp_I, ':', 'Color', 'b', 'LineWidth', 0.5);
plot(x_time, hr_st_I, '-b', 'LineWidth', 2);
plot(x_time, hrs_st_I, '-b', 'LineWidth', 0.5);
xlabel('time (s)'); ylabel('Entropy [bits/s]')
title('I: Hr (thick) & Hn (thin), shaded=MI')
legend('','','','','','Spatial Hr','Spatial Hn','Temporal Hr','Temporal Hn','S+T Hr','S+T Hn', ...
    'Location','best','FontSize',6)
xlim([10^(-1) 10^6]); grid on; box off
