%% ====================================================================
%% figure4_errbar_snippet.m  (NOT standalone; splice into figure4_main.m at the indicated lines) — Error bar patch
%%
%% 这个文件包含两部分修改：
%%   PART A: 计算循环内的小改动（保存 per-neuron MI 用于 error bar）
%%   PART B: 替换整个画图 section（从 line 311 "Plotting" 到文件末尾）
%% ====================================================================

%% ====================================================================
%% PART A: 计算循环内的改动
%% ====================================================================
%
% 1) 在 line 59 之后（数组初始化区域），添加：
%
%    % --- Per-neuron MI storage for temporal error bars ---
%    mi_E_neurons_all = zeros(T, NneuronE);   % [stages x neurons]
%    mi_I_neurons_all = zeros(T, NneuronI);
%    hr_E_neurons_all = zeros(T, NneuronE);
%    hr_I_neurons_all = zeros(T, NneuronI);
%
%    % --- ST jackknife arrays ---
%    MIE_ST_jack = zeros(Trials, T);   % [leave-one-out x stages]
%    MII_ST_jack = zeros(Trials, T);
%
% 2) 在 line 254 (MIE_temporal(:,i) = ...) 之后，添加：
%
%    mi_E_neurons_all(i,:) = mi_E_neurons;
%    hr_E_neurons_all(i,:) = hr_E_neurons;
%
% 3) 在 line 276 (MII_temporal(:,i) = ...) 之后，添加：
%
%    mi_I_neurons_all(i,:) = mi_I_neurons;
%    hr_I_neurons_all(i,:) = hr_I_neurons;
%
% 4) 在 line 307 (ST for I population) 之后，end 之前，添加 ST jackknife：
%
%    % --- ST Jackknife: leave-one-trial-out ---
%    for jk = 1:Trials
%        idx = setdiff(1:Trials, jk);
%        % E population
%        spike_jk_E = zeros((Trials-1)*NneuronE, TimeT);
%        for rr = 1:length(idx)
%            spike_jk_E((rr-1)*NneuronE+1:rr*NneuronE, :) = OET_trials{idx(rr)};
%        end
%        if sum(spike_jk_E(:)) > 0
%            res_jk = compute_entropy_toolbox(spike_jk_E, spike_jk_E, 'dt', dt, 'refractory', 0);
%            MIE_ST_jack(jk,i) = max(0, res_jk.MI);
%        end
%        % I population
%        spike_jk_I = zeros((Trials-1)*NneuronI, TimeT);
%        for rr = 1:length(idx)
%            spike_jk_I((rr-1)*NneuronI+1:rr*NneuronI, :) = OIT_trials{idx(rr)};
%        end
%        if sum(spike_jk_I(:)) > 0
%            res_jk = compute_entropy_toolbox(spike_jk_I, spike_jk_I, 'dt', dt, 'refractory', 0);
%            MII_ST_jack(jk,i) = max(0, res_jk.MI);
%        end
%    end
%
% ====================================================================

%% ====================================================================
%% PART B: 替换从 line 311 到文件末尾的画图 section
%% ====================================================================

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

%% Compute SEM for each level
% Spatial: SEM across trials (per-trial values exist)
semE_sp = std(MIE_spatial, 0, 1) / sqrt(Trials);
semI_sp = std(MII_spatial, 0, 1) / sqrt(Trials);

% Temporal: SEM across neurons (per-neuron MI saved in mi_E_neurons_all)
semE_tp = std(mi_E_neurons_all, 0, 2)' / sqrt(NneuronE);  % [1 x T]
semI_tp = std(mi_I_neurons_all, 0, 2)' / sqrt(NneuronI);

% S+T: Jackknife SEM across trials
N_jk = Trials;
MIE_ST_jkmean = mean(MIE_ST_jack, 1);
MII_ST_jkmean = mean(MII_ST_jack, 1);
semE_st = sqrt((N_jk-1)/N_jk * sum((MIE_ST_jack - MIE_ST_jkmean).^2, 1));
semI_st = sqrt((N_jk-1)/N_jk * sum((MII_ST_jack - MII_ST_jkmean).^2, 1));

% ICE SEM: propagate through division by cost
% Spatial ICE per trial exists → direct SEM
ICEE_sp = MIrateE_spatial ./ PcostE;
ICEI_sp = MIrateI_spatial ./ PcostI;
semICEE_sp = std(ICEE_sp, 0, 1) / sqrt(Trials);
semICEI_sp = std(ICEI_sp, 0, 1) / sqrt(Trials);

% Temporal ICE: use neuron-level SEM / mean cost
meanPcostE = mean(PcostE, 1);
meanPcostI = mean(PcostI, 1);
semICEE_tp = semE_tp ./ meanPcostE;
semICEI_tp = semI_tp ./ meanPcostI;

% ST ICE: jackknife SEM / mean cost
semICEE_st = semE_st ./ meanPcostE;
semICEI_st = semI_st ./ meanPcostI;

%% Spatial Hr SEM (for decomposition plots)
semHrE_sp = std(HrE_spatial, 0, 1) / sqrt(Trials);
semHrI_sp = std(HrI_spatial, 0, 1) / sqrt(Trials);
% Temporal Hr: cross-neuron SEM
semHrE_tp = std(hr_E_neurons_all, 0, 2)' / sqrt(NneuronE);
semHrI_tp = std(hr_I_neurons_all, 0, 2)' / sqrt(NneuronI);

%% ---- Helper: shaded error bar on semilog x ----
% Usage: shade_err(x, y, sem, color, alpha)
%   Plots mean line + shaded ± SEM region on current axes (must have hold on)
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

% --- Cost ---
h = subplot(lines,3,[1 4]);
ax = get(h,'Position');
semilogx(x_time, PcostE, 'Color', orange); hold on
semilogx(x_time, PcostI, 'b');
set(gca,'FontSize',fsize,'FontName','Helvetica','TickDir','out')
ylabel('Cost [W]'); xlabel('time (s)');
title('Evolution Through Learning')
xlim([10^(-1) 10^6])
set(gca,'ticklength',[0.01 0.01]/ax(3)); box off

% --- Entropy ---
h = subplot(lines,3,[7 10]);
ax = get(h,'Position');
semilogx(x_time, PentrE, 'Color', orange); hold on
semilogx(x_time, PentrI, 'b');
set(gca,'FontSize',fsize,'FontName','Helvetica','TickDir','out')
ylabel('Entropy [bits]'); xlabel('time (s)');
xlim([10^(-1) 10^6])
set(gca,'ticklength',[0.01 0.01]/ax(3)); box off

% --- Population ICE ---
h = subplot(lines,3,[13 16]);
ax = get(h,'Position');
semilogx(x_time, PentrE_rate./PcostE, 'Color', orange); hold on
semilogx(x_time, PentrI_rate./PcostI, 'b');
set(gca,'FontSize',fsize,'FontName','Helvetica','TickDir','out')
ylabel('ICE [bits/J]'); xlabel('time (s)');
xlim([10^(-1) 10^6])
set(gca,'ticklength',[0.01 0.01]/ax(3)); box off

% --- Three-level MI evolution (with shaded SEM) ---
h = subplot(lines,3,[2 5]);
ax = get(h,'Position');
set(gca, 'XScale', 'log'); hold on;
% Shaded SEM regions
shade_err(x_time, mean(MIE_spatial,1), semE_sp, [0.85 0.85 0.85], 0.4);
shade_err(x_time, mean(MIE_temporal,1), semE_tp, [1 0.85 0.7], 0.4);
shade_err(x_time, mean(MIE_ST,1), semE_st, [1 0.7 0.4], 0.4);
% Mean lines
plot(x_time, mean(MIE_spatial,1), '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.5);
plot(x_time, mean(MIE_temporal,1), ':', 'Color', orange, 'LineWidth', 1.5);
plot(x_time, mean(MIE_ST,1), '-', 'Color', orange, 'LineWidth', 2);
% I population
shade_err(x_time, mean(MII_spatial,1), semI_sp, [0.7 0.8 1], 0.3);
shade_err(x_time, mean(MII_temporal,1), semI_tp, [0.6 0.7 1], 0.3);
shade_err(x_time, mean(MII_ST,1), semI_st, [0.5 0.7 1], 0.3);
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

% --- Three-level ICE (with shaded SEM) ---
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
ylabel('ICE [bits/J]'); xlabel('time (s)');
title('MI per Cost (3 Levels)')
xlim([10^(-1) 10^6])
set(gca,'ticklength',[0.01 0.01]/ax(3)); box off

%% Figure 2: MI vs Cost scatter plots

figure;
x_ref = 0:0.1:30;
y1_ref = 1+log(x_ref);
y2_ref = 1./x_ref;
PcostE1 = PcostE;
PcostI1 = PcostI;

subplot(2,3,1); plot(x_ref, y1_ref); hold on
subplot(2,3,4); plot(x_ref, y2_ref); hold on
for ii = 1:size(PentrE,2)
    subplot(2,3,1)
    scatter(PcostE1(:,ii), PentrE(:,ii), 36, orange, 'o'); hold on;
    ylim([1 4])
    subplot(2,3,4)
    scatter(PcostE1(:,ii), PentrE_rate(:,ii)./PcostE1(:,ii), 36, orange, 'o'); hold on;
    ylim([0 1])
end

% MI vs Cost scatter plots (3 levels)
subplot(2,3,2)
for ii = 1:size(MIE_ST,2)
    scatter(PcostE1(:,ii), MIE_spatial(:,ii), 20, [0.7 0.7 0.7], 'o'); hold on;
    scatter(PcostE1(:,ii), MIE_temporal(:,ii), 20, [1 0.8 0.5], '^'); hold on;
    scatter(PcostE1(:,ii), MIE_ST(:,ii), 36, orange, 's', 'filled'); hold on;
end
xlabel('Cost E [W]'); ylabel('MI [bits/s]')
title('MI vs Cost (E, 3 Levels)')
legend('Spatial','Temporal','S+T','Location','best','FontSize',7); box off

subplot(2,3,5)
for ii = 1:size(MIE_ST,2)
    scatter(PcostE1(:,ii), MIrateE_spatial(:,ii)./PcostE1(:,ii), 20, [0.7 0.7 0.7], 'o'); hold on;
    scatter(PcostE1(:,ii), MIrateE_temporal(:,ii)./PcostE1(:,ii), 20, [1 0.8 0.5], '^'); hold on;
    scatter(PcostE1(:,ii), MIrateE_ST(:,ii)./PcostE1(:,ii), 36, orange, 's', 'filled'); hold on;
end
xlabel('Cost E [W]'); ylabel('ICE [bits/J]')
title('ICE vs Cost (E, 3 Levels)'); box off

subplot(2,3,3)
for ii = 1:size(MIE_ST,2)
    scatter(PentrE(:,ii), MIE_spatial(:,ii), 20, [0.7 0.7 0.7], 'o'); hold on;
    scatter(PentrE(:,ii), MIE_temporal(:,ii), 20, [1 0.8 0.5], '^'); hold on;
    scatter(PentrE(:,ii), MIE_ST(:,ii), 36, orange, 's', 'filled'); hold on;
end
xlabel('Entropy E [bits]'); ylabel('MI [bits/s]')
title('MI vs Entropy (E, 3 Levels)'); box off

subplot(2,3,6)
for ii = 1:size(MIE_ST,2)
    scatter(PentrE(:,ii), MIE_spatial(:,ii)./PentrE(:,ii), 20, [0.7 0.7 0.7], 'o'); hold on;
    scatter(PentrE(:,ii), MIE_temporal(:,ii)./PentrE(:,ii), 20, [1 0.8 0.5], '^'); hold on;
    scatter(PentrE(:,ii), MIE_ST(:,ii)./PentrE(:,ii), 36, orange, 's', 'filled'); hold on;
end
xlabel('Entropy E [bits]'); ylabel('MI/Entropy')
title('MI Efficiency (E, 3 Levels)'); box off

%% Figure 3: Entropy/MI time evolution with shaded std regions

figure;
subplot(1,3,1)
y = mean(PentrE,1);
std_dev = std(PentrE,0,1);
semilogx(x_time, y, 'k', 'LineWidth', 2); hold on;
shade_err(x_time, y, std_dev, [0.5 0.5 0.5], 0.4);
xlim([10^(-1) 10^6]); ylabel('Entropy [bits]');

subplot(1,3,2)
y = mean(PcostE,1);
std_dev = std(PcostE,0,1);
semilogx(x_time, y, 'k', 'LineWidth', 2); hold on;
shade_err(x_time, y, std_dev, [0.5 0.5 0.5], 0.4);
xlim([10^(-1) 10^6]); ylabel('Cost [W]');

subplot(1,3,3)
y = mean(PentrE_rate./PcostE,1);
std_dev = std(PentrE_rate./PcostE,0,1);
semilogx(x_time, y, 'k', 'LineWidth', 2); hold on;
shade_err(x_time, y, std_dev, [0.5 0.5 0.5], 0.4);
xlim([10^(-1) 10^6]); ylabel('ICE [bits/J]');

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
y_sp = mean(ICEE_sp,1); y_tp = mean(MIrateE_temporal./PcostE,1); y_st = mean(MIrateE_ST./PcostE,1);
shade_err(x_time, y_sp, semICEE_sp, [0.85 0.85 0.85], 0.5);
shade_err(x_time, y_tp, semICEE_tp, [1 0.85 0.7], 0.5);
shade_err(x_time, y_st, semICEE_st, [1 0.7 0.4], 0.5);
plot(x_time, y_sp, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.5);
plot(x_time, y_tp, ':', 'Color', orange, 'LineWidth', 1.5);
plot(x_time, y_st, '-', 'Color', orange, 'LineWidth', 2);
xlabel('time (s)'); ylabel('ICE [bits/J]')
title('MI Efficiency - E Pop')
xlim([10^(-1) 10^6]); grid on; box off

% ---- E Population: H(r) decomposition with SEM ----
subplot(2,3,3)
set(gca, 'XScale', 'log'); hold on;
hr_sp_E = mean(HrE_spatial,1); hrs_sp_E = hr_sp_E - mean(MIE_spatial,1);
hr_tp_E = mean(HrE_temporal,1); hrs_tp_E = hr_tp_E - mean(MIE_temporal,1);
hr_st_E = mean(HrE_ST,1);       hrs_st_E = hr_st_E - mean(MIE_ST,1);
% Shade MI region (between Hr and Hn) + SEM band on Hr
shade_err(x_time, hr_sp_E, semHrE_sp, [0.85 0.85 0.85], 0.3);
shade_err(x_time, hr_tp_E, semHrE_tp, [1 0.85 0.7], 0.3);
shade_err(x_time, hr_st_E, zeros(size(hr_st_E)), [1 0.7 0.4], 0.3); % ST Hr no per-neuron SEM
% MI region fills
fill(x2, [hr_sp_E, fliplr(hrs_sp_E)], [0.85 0.85 0.85], 'EdgeColor','none','FaceAlpha',0.2);
fill(x2, [hr_tp_E, fliplr(hrs_tp_E)], [1 0.85 0.7], 'EdgeColor','none','FaceAlpha',0.2);
fill(x2, [hr_st_E, fliplr(hrs_st_E)], [1 0.7 0.4], 'EdgeColor','none','FaceAlpha',0.2);
% Hr (thick) and H(r|s) (thin)
plot(x_time, hr_sp_E, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.5);
plot(x_time, hrs_sp_E, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 0.5);
plot(x_time, hr_tp_E, ':', 'Color', orange, 'LineWidth', 1.5);
plot(x_time, hrs_tp_E, ':', 'Color', orange, 'LineWidth', 0.5);
plot(x_time, hr_st_E, '-', 'Color', orange, 'LineWidth', 2);
plot(x_time, hrs_st_E, '-', 'Color', orange, 'LineWidth', 0.5);
xlabel('time (s)'); ylabel('Entropy [bits/s]')
title('E: Hr (thick) & Hn (thin), shaded=MI')
legend('','','','','','','Spatial Hr','Spatial Hn','Temporal Hr','Temporal Hn','S+T Hr','S+T Hn', ...
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
y_sp = mean(ICEI_sp,1); y_tp = mean(MIrateI_temporal./PcostI,1); y_st = mean(MIrateI_ST./PcostI,1);
shade_err(x_time, y_sp, semICEI_sp, [0.7 0.8 1], 0.5);
shade_err(x_time, y_tp, semICEI_tp, [0.6 0.7 1], 0.5);
shade_err(x_time, y_st, semICEI_st, [0.5 0.7 1], 0.5);
plot(x_time, y_sp, '--', 'Color', [0.5 0.5 0.8], 'LineWidth', 1.5);
plot(x_time, y_tp, ':', 'Color', 'b', 'LineWidth', 1.5);
plot(x_time, y_st, '-b', 'LineWidth', 2);
xlabel('time (s)'); ylabel('ICE [bits/J]')
title('MI Efficiency - I Pop')
xlim([10^(-1) 10^6]); grid on; box off

% ---- I Population: H(r) decomposition with SEM ----
subplot(2,3,6)
set(gca, 'XScale', 'log'); hold on;
hr_sp_I = mean(HrI_spatial,1); hrs_sp_I = hr_sp_I - mean(MII_spatial,1);
hr_tp_I = mean(HrI_temporal,1); hrs_tp_I = hr_tp_I - mean(MII_temporal,1);
hr_st_I = mean(HrI_ST,1);       hrs_st_I = hr_st_I - mean(MII_ST,1);
% SEM bands
shade_err(x_time, hr_sp_I, semHrI_sp, [0.7 0.8 1], 0.3);
shade_err(x_time, hr_tp_I, semHrI_tp, [0.6 0.7 1], 0.3);
shade_err(x_time, hr_st_I, zeros(size(hr_st_I)), [0.5 0.7 1], 0.3);
% MI region
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
legend('','','','','','','Spatial Hr','Spatial Hn','Temporal Hr','Temporal Hn','S+T Hr','S+T Hn', ...
    'Location','best','FontSize',6)
xlim([10^(-1) 10^6]); grid on; box off
