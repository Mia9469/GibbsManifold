load checkpoint.mat
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%  Computing Decoding Error, rates through Learning %%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%
%%%%% In this part we run the different instances of the network using a
%%%%% new test input and we measure the evolution of the dedocding error
%%%%% through learning using the decoders that we computed preciously. We also
%%%%% measure the evolution of the mean firing rate anf the variance of the
%%%%% membrane potential.
%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Computing decoding errors and rates over learning\n')
TimeT=10000; % size of the test input
MeanPrateE=zeros(1,T);%array of the mean rates over learning
PentrE=zeros(10,T);%array of the mean rates over learning
PentrEl=zeros(10,T);%array of the mean rates over learning
PentrIl=zeros(10,T);%array of the mean rates over learning
PcostE=zeros(10,T);%array of the mean rates over learning
PentrI=zeros(10,T);%array of the mean rates over learning
PcostI=zeros(10,T);%array of the mean rates over learning
MeanPrateI=zeros(1,T);%array of the mean rates over learning
ErrorE=zeros(1,T);%array of the decoding error over learning
ErrorI=zeros(1,T);%array of the decoding error over learning
MembraneVarE=zeros(1,T);%mean membrane potential variance over learning
MembraneVarI=zeros(1,T);%mean membrane potential variance over learning
% Three-level MI arrays (Trials × T)
MIE_spatial=zeros(10,T);   MII_spatial=zeros(10,T);
MIE_temporal=zeros(10,T);  MII_temporal=zeros(10,T);
MIE_ST=zeros(10,T);        MII_ST=zeros(10,T);
% Corresponding Hr arrays for decomposition
HrE_spatial=zeros(10,T);   HrI_spatial=zeros(10,T);
HrE_temporal=zeros(10,T);  HrI_temporal=zeros(10,T);
HrE_ST=zeros(10,T);        HrI_ST=zeros(10,T);
xT=zeros(Nx,TimeT);%target ouput
Fanos=zeros(Nx,T);
CVs=zeros(Nx,T);
Fanosimple=zeros(NneuronE,Nx);
CVsimple=zeros(NneuronE,Nx);
Counts=zeros(NneuronE,50);
ISIall=zeros(NneuronE,800);

NbISI=zeros(NneuronE,1);


Trials=10; %number of trials

for r=1:Trials %for each trial
    InputT=A*(mvnrnd(zeros(1,Nx),eye(Nx),TimeT))'; % we genrate a new input
    
    for k=1:Nx
        InputT(k,:)=conv(InputT(k,:),w,'same'); % we wmooth it
        InputT(k,:) = InputT(k,:)+rand(size(InputT(k,:)));
    end
%     InputT(InputT<0)=0;
    
    for t=2:TimeT
        xT(:,t)= (1-lambda*dt)*xT(:,t-1)+ dt*InputT(:,t-1); % ans we comput the target output by leaky inegration of the input
    end
    
    for i=1:T %for each instance of the network
        
        [ rOET, OET,VET, rOIT, OIT,VIT,~,~] = runnet(dt,TimeT, lambda,NneuronE,NneuronI,ThreshE,ThreshI,InputT,squeeze(FsE(i,:,:)),squeeze(CsEE(i,:,:)),squeeze(CsEI(i,:,:)),squeeze(CsII(i,:,:)),squeeze(CsIE(i,:,:)));%we run the network with current input InputT
        
        xestE=squeeze(DecsE(i,:,:))*rOET; %we deocode the ouptut using the optinal decoders previously computed
        xestI=squeeze(DecsI(i,:,:))*rOIT; %we deocode the ouptut using the optinal decoders previously computed
        
        if (i==1 && r==Trials)
            
            xestEB=xestE;
            xestIB=xestI;
            OEB=OET;
            OIB=OIT;
            
        end
        ErrorE(1,i)=ErrorE(1,i)+sum(var(xT-xestE,0,2))/(sum(var(xT,0,2))*Trials);%we comput the variance of the error normalized by the variance of the target E population
        ErrorI(1,i)=ErrorI(1,i)+sum(var(xT-xestI,0,2))/(sum(var(xT,0,2))*Trials);%we comput the variance of the error normalized by the variance of the target for I population
        MeanPrateE(1,i)=MeanPrateE(1,i)+sum(sum(OET))/(TimeT*dt*NneuronE*Trials);%we comput the average firing rate per neuron E population
        MeanPrateI(1,i)=MeanPrateI(1,i)+sum(sum(OIT))/(TimeT*dt*NneuronI*Trials);%we comput the average firing rate per neuron for I population
        MembraneVarE(1,i)=MembraneVarE(1,i)+sum(var(VET,0,2))/(NneuronE*Trials);% we compute the average membrane potential variance per neuron for E population
        MembraneVarI(1,i)=MembraneVarI(1,i)+sum(var(VIT,0,2))/(NneuronI*Trials);% we compute the average membrane potential variance per neuron for I population
        

        PrateE = sum(OET,2);
        [counts, ~] = histcounts(PrateE, 'Normalization', 'probability');
        counts(counts == 0) = [];
        PentrE(r,i) = -sum(counts .* log2(counts));
        PcostE(r,i) = 2.4+0.37*mean(PrateE);
        PrateI = sum(OIT,2);
        [counts, ~] = histcounts(PrateI, 'Normalization', 'probability');
        counts(counts == 0) = [];
        PentrI(r,i) = -sum(counts .* log2(counts));
        PcostI(r,i) = 2.4+0.37*mean(PrateI);
        
        entr = zeros(1,300);
        for k = 1:300
        [counts, ~] = histcounts(rOET(k,:), 'Normalization', 'probability');
        counts(counts == 0) = [];
        entr(k) =  -sum(counts .* log2(counts));
        end
        PentrEl(r,i) = mean(entr);

        entr = zeros(1,75);
        for k = 1:75
        [counts, ~] = histcounts(rOIT(k,:), 'Normalization', 'probability');
        counts(counts == 0) = [];
        entr(k) =  -sum(counts .* log2(counts));
        end
        PentrIl(r,i) = mean(entr);
        
        % ==================== THREE-LEVEL MI COMPUTATION ====================
        % Stimulus: discretized input signal (averaged across dims)
        % MI = H(r) - H(r|s) at each level
        %   Level 1 — Spatial-Only:     per-neuron spike count in windows
        %   Level 2 — Per-Neuron (Temporal): per-neuron filtered rate time series
        %   Level 3 — Spatial+Temporal:  all neurons × all time points

        nbins_stim_spatial  = 5;   % stimulus bins for spatial (100 windows → ~20/bin)
        nbins_stim_temporal = 50;  % stimulus bins for temporal (10000 steps → ~200/bin)
        nbins_stim_ST       = 50;  % stimulus bins for ST (abundant data)
        nBins_resp = 30;     % bins for response histograms
        W_spatial  = 100;    % window size for spatial-only (time steps)

        input_signal = mean(InputT, 1);  % 1 × TimeT

        % === LEVEL 1: SPATIAL-ONLY MI ===
        % Divide time into windows; per-neuron spike count per window
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

        % === LEVEL 2: PER-NEURON (TEMPORAL) MI ===
        stim_t = discretize(input_signal, nbins_stim_temporal);
        valid_t = ~isnan(stim_t);
        stim_valid = stim_t(valid_t);

        mi_neurons_E = zeros(1, NneuronE);
        hr_neurons_E = zeros(1, NneuronE);
        for k = 1:NneuronE
            rates_k = rOET(k, valid_t);
            if length(unique(rates_k)) < 2, continue; end

            [~, edges_k] = histcounts(rates_k, nBins_resp);
            [n_r, ~] = histcounts(rates_k, edges_k);
            p_r = n_r/sum(n_r); p_r = p_r(p_r>0);
            Hr_k = -sum(p_r .* log2(p_r));
            hr_neurons_E(k) = Hr_k;

            Hrs_k = 0;
            for s = unique(stim_valid)
                mask = (stim_valid == s);
                P_s = sum(mask)/length(stim_valid);
                [n_s, ~] = histcounts(rates_k(mask), edges_k);
                p_s = n_s/sum(n_s); p_s = p_s(p_s>0);
                if ~isempty(p_s)
                    Hrs_k = Hrs_k + P_s*(-sum(p_s .* log2(p_s)));
                end
            end
            mi_neurons_E(k) = max(0, Hr_k - Hrs_k);
        end
        MIE_temporal(r,i) = mean(mi_neurons_E);
        HrE_temporal(r,i) = mean(hr_neurons_E);

        mi_neurons_I = zeros(1, NneuronI);
        hr_neurons_I = zeros(1, NneuronI);
        for k = 1:NneuronI
            rates_k = rOIT(k, valid_t);
            if length(unique(rates_k)) < 2, continue; end

            [~, edges_k] = histcounts(rates_k, nBins_resp);
            [n_r, ~] = histcounts(rates_k, edges_k);
            p_r = n_r/sum(n_r); p_r = p_r(p_r>0);
            Hr_k = -sum(p_r .* log2(p_r));
            hr_neurons_I(k) = Hr_k;

            Hrs_k = 0;
            for s = unique(stim_valid)
                mask = (stim_valid == s);
                P_s = sum(mask)/length(stim_valid);
                [n_s, ~] = histcounts(rates_k(mask), edges_k);
                p_s = n_s/sum(n_s); p_s = p_s(p_s>0);
                if ~isempty(p_s)
                    Hrs_k = Hrs_k + P_s*(-sum(p_s .* log2(p_s)));
                end
            end
            mi_neurons_I(k) = max(0, Hr_k - Hrs_k);
        end
        MII_temporal(r,i) = mean(mi_neurons_I);
        HrI_temporal(r,i) = mean(hr_neurons_I);

        % === LEVEL 3: SPATIAL+TEMPORAL MI ===
        % Pool all neuron rates from all time points
        % Reuses stim_valid from Level 2 (same nbins_stim_ST = nbins_stim_temporal)
        all_rates_ST_E = rOET(:, valid_t);
        all_rates_ST_E = all_rates_ST_E(:)';
        stim_ST_E = repmat(stim_valid, NneuronE, 1);
        stim_ST_E = stim_ST_E(:)';

        if length(unique(all_rates_ST_E)) > 1
            [~, edges_ST_E] = histcounts(all_rates_ST_E, nBins_resp);
            [n_r, ~] = histcounts(all_rates_ST_E, edges_ST_E);
            p_r = n_r/sum(n_r); p_r = p_r(p_r>0);
            Hr_ST_E = -sum(p_r .* log2(p_r));

            Hrs_ST_E = 0;
            for s = unique(stim_ST_E)
                mask = (stim_ST_E == s);
                P_s = sum(mask)/length(stim_ST_E);
                [n_s, ~] = histcounts(all_rates_ST_E(mask), edges_ST_E);
                p_s = n_s/sum(n_s); p_s = p_s(p_s>0);
                if ~isempty(p_s)
                    Hrs_ST_E = Hrs_ST_E + P_s*(-sum(p_s .* log2(p_s)));
                end
            end
            MIE_ST(r,i) = max(0, Hr_ST_E - Hrs_ST_E);
            HrE_ST(r,i) = Hr_ST_E;
        end

        all_rates_ST_I = rOIT(:, valid_t);
        all_rates_ST_I = all_rates_ST_I(:)';
        stim_ST_I = repmat(stim_valid, NneuronI, 1);
        stim_ST_I = stim_ST_I(:)';

        if length(unique(all_rates_ST_I)) > 1
            [~, edges_ST_I] = histcounts(all_rates_ST_I, nBins_resp);
            [n_r, ~] = histcounts(all_rates_ST_I, edges_ST_I);
            p_r = n_r/sum(n_r); p_r = p_r(p_r>0);
            Hr_ST_I = -sum(p_r .* log2(p_r));

            Hrs_ST_I = 0;
            for s = unique(stim_ST_I)
                mask = (stim_ST_I == s);
                P_s = sum(mask)/length(stim_ST_I);
                [n_s, ~] = histcounts(all_rates_ST_I(mask), edges_ST_I);
                p_s = n_s/sum(n_s); p_s = p_s(p_s>0);
                if ~isempty(p_s)
                    Hrs_ST_I = Hrs_ST_I + P_s*(-sum(p_s .* log2(p_s)));
                end
            end
            MII_ST(r,i) = max(0, Hr_ST_I - Hrs_ST_I);
            HrI_ST(r,i) = Hr_ST_I;
        end
    end
    
    
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%  Plotting Decoding Error, rates through Learning  %%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

orange = [1 0.5 0.2];

%% Convert MI/entropy to information rate [bits/s] for ICE computation
% Use unified observation duration = full test window for all levels
T_obs = TimeT * dt;                   % [s] total test observation duration
% Population entropy rate [bits/s]
PentrE_rate = PentrE / T_obs;
PentrI_rate = PentrI / T_obs;
% MI rate arrays [bits/s] — unified T_obs for all three levels
MIrateE_spatial = MIE_spatial / T_obs;
MIrateE_temporal = MIE_temporal / T_obs;
MIrateE_ST = MIE_ST / T_obs;
MIrateI_spatial = MII_spatial / T_obs;
MIrateI_temporal = MII_temporal / T_obs;
MIrateI_ST = MII_ST / T_obs;
% PcostE, PcostI are already in Watts
% ICE = MI_rate / Power [bits/J]


figure
set(gcf,'Units','centimeters')
xSize = 50;  ySize =34;
xLeft = (21-xSize)/4; yTop = (30-ySize)/4;
set(gcf,'Position',[xLeft yTop xSize ySize]); %centers on A4 paper
set(gcf, 'Color', 'w');

lines=6;
fsize=11;

h=subplot(lines,3,[1 4]);
ax=get(h,'Position');
semilogx((2.^(1:T(1,1)))*dt,PcostE,'Color',orange);
hold on
semilogx((2.^(1:T(1,1)))*dt,PcostI,'b');
set(gca,'FontSize',fsize,'FontName','Helvetica')
set(gca,'TickDir','out')
ylabel('Cost')
xlabel('time (s)')
title('Evolution Through Learning')
xlim([10^(-1) 10^6])
set(gca,'ticklength',[0.01 0.01]/ax(3))
box off


h=subplot(lines,3,[7 10]);
ax=get(h,'Position');
semilogx((2.^(1:T(1,1)))*dt,PentrE,'Color',orange);
hold on
semilogx((2.^(1:T(1,1)))*dt,PentrI,'b');
set(gca,'FontSize',fsize,'FontName','Helvetica')
set(gca,'TickDir','out')
ylabel('Entropy')
xlabel('time (s)')
xlim([10^(-1) 10^6])
set(gca,'ticklength',[0.01 0.01]/ax(3))
box off


h=subplot(lines,3,[13 16]);
ax=get(h,'Position');
semilogx((2.^(1:T(1,1)))*dt,PentrE_rate./PcostE,'Color',orange);
hold on
semilogx((2.^(1:T(1,1)))*dt,PentrI_rate./PcostI,'b');
set(gca,'FontSize',fsize,'FontName','Helvetica')
set(gca,'TickDir','out')
ylabel('ICE [bits/J]')
xlabel('time (s)')
xlim([10^(-1) 10^6])
% plot(sort(firingRatesE(:,end),'descend'),'ko-')
% set(gca,'FontSize',fsize,'FontName','Helvetica')
% set(gca,'TickDir','out')
% ylabel('FiringRate(Hz)');xlabel("Neuron")
% ylim([0 100])
set(gca,'ticklength',[0.01 0.01]/ax(3))
box off

% Three-level MI evolution plots
h=subplot(lines,3,[2 5]);
ax=get(h,'Position');
semilogx((2.^(1:T(1,1)))*dt, mean(MIE_spatial,1), '--', 'Color', orange, 'LineWidth', 1); hold on
semilogx((2.^(1:T(1,1)))*dt, mean(MIE_temporal,1), ':', 'Color', orange, 'LineWidth', 1.5);
semilogx((2.^(1:T(1,1)))*dt, mean(MIE_ST,1), '-', 'Color', orange, 'LineWidth', 2);
semilogx((2.^(1:T(1,1)))*dt, mean(MII_spatial,1), '--b', 'LineWidth', 1);
semilogx((2.^(1:T(1,1)))*dt, mean(MII_temporal,1), ':b', 'LineWidth', 1.5);
semilogx((2.^(1:T(1,1)))*dt, mean(MII_ST,1), '-b', 'LineWidth', 2);
set(gca,'FontSize',fsize,'FontName','Helvetica')
set(gca,'TickDir','out')
ylabel('MI (bits)')
xlabel('time (s)')
title('MI Evolution (3 Levels)')
legend('E Spatial','E Temporal','E S+T','I Spatial','I Temporal','I S+T','Location','best','FontSize',7)
xlim([10^(-1) 10^6])
set(gca,'ticklength',[0.01 0.01]/ax(3))
box off

h=subplot(lines,3,[8 11]);
ax=get(h,'Position');
semilogx((2.^(1:T(1,1)))*dt, mean(MIrateE_spatial./PcostE,1), '--', 'Color', orange, 'LineWidth', 1); hold on
semilogx((2.^(1:T(1,1)))*dt, mean(MIrateE_temporal./PcostE,1), ':', 'Color', orange, 'LineWidth', 1.5);
semilogx((2.^(1:T(1,1)))*dt, mean(MIrateE_ST./PcostE,1), '-', 'Color', orange, 'LineWidth', 2);
semilogx((2.^(1:T(1,1)))*dt, mean(MIrateI_spatial./PcostI,1), '--b', 'LineWidth', 1);
semilogx((2.^(1:T(1,1)))*dt, mean(MIrateI_temporal./PcostI,1), ':b', 'LineWidth', 1.5);
semilogx((2.^(1:T(1,1)))*dt, mean(MIrateI_ST./PcostI,1), '-b', 'LineWidth', 2);
set(gca,'FontSize',fsize,'FontName','Helvetica')
set(gca,'TickDir','out')
ylabel('MI/Cost')
xlabel('time (s)')
title('MI per Cost (3 Levels)')
xlim([10^(-1) 10^6])
set(gca,'ticklength',[0.01 0.01]/ax(3))
box off


% h=subplot(lines,3,2);
% plot(dt:dt:TimeT*dt,xestEB(1,:),'Color',orange);
% hold on
% plot(dt:dt:TimeT*dt,xestIB(1,:),'b');
% plot(dt:dt:TimeT*dt,xT(1,:),'k')
% set(gca,'FontSize',fsize,'FontName','Helvetica')
% set(gca,'TickDir','out')
% ylabel('x,x_hat Dim1')
% title('Before Learning')
% box off
% 
% 
% h=subplot(lines,3,3);
% plot(dt:dt:TimeT*dt,xestE(1,:),'Color',orange);
% hold on
% plot(dt:dt:TimeT*dt,xestI(1,:),'b');
% plot(dt:dt:TimeT*dt,xT(1,:),'k')
% set(gca,'FontSize',fsize,'FontName','Helvetica')
% set(gca,'TickDir','out')
% title('After Learning')
% box off
% 
% h=subplot(lines,3,5);
% plot(dt:dt:TimeT*dt,xestEB(2,:),'Color',orange);
% hold on
% plot(dt:dt:TimeT*dt,xestIB(2,:),'b');
% plot(dt:dt:TimeT*dt,xT(2,:),'k')
% set(gca,'FontSize',fsize,'FontName','Helvetica')
% set(gca,'TickDir','out')
% ylabel('x,x_hat Dim2')
% box off
% 
% h=subplot(lines,3,6);
% plot(dt:dt:TimeT*dt,xestE(2,:),'Color',orange);
% hold on
% plot(dt:dt:TimeT*dt,xestI(2,:),'b');
% plot(dt:dt:TimeT*dt,xT(2,:),'k')
% set(gca,'FontSize',fsize,'FontName','Helvetica')
% set(gca,'TickDir','out')
% box off
% 
% h=subplot(lines,3,8);
% plot(dt:dt:TimeT*dt,xestEB(3,:),'Color',orange);
% hold on
% plot(dt:dt:TimeT*dt,xestIB(3,:),'b');
% plot(dt:dt:TimeT*dt,xT(3,:),'k')
% set(gca,'FontSize',fsize,'FontName','Helvetica')
% set(gca,'TickDir','out')
% ylabel('x,x_hat Dim2')
% box off
% 
% h=subplot(lines,3,9);
% plot(dt:dt:TimeT*dt,xestE(3,:),'Color',orange);
% hold on
% plot(dt:dt:TimeT*dt,xestI(3,:),'b');
% plot(dt:dt:TimeT*dt,xT(3,:),'k')
% set(gca,'FontSize',fsize,'FontName','Helvetica')
% set(gca,'TickDir','out')
% box off
% %
% h=subplot(lines,3,[11 14 17]);
% plot(OEB.*(ones(NneuronE,1)*(1:TimeT)),OEB.*((1:NneuronE)'*ones(1,TimeT)),'.','Color',orange);
% hold on
% plot(OIB.*(ones(NneuronI,1)*(1:TimeT)),OIB.*(((NneuronE+1):(NneuronE+NneuronI))'*ones(1,TimeT)),'.b');
% set(gca,'FontSize',fsize,'FontName','Helvetica')
% set(gca,'TickDir','out')
% xticklabels([0:0.2:1])
% ylim([0 375])
% xlabel('time (s)')
% box off
% 
% h=subplot(lines,3,[12 15 18]);
% plot(OET.*(ones(NneuronE,1)*(1:TimeT)),OET.*((1:NneuronE)'*ones(1,TimeT)),'.','Color',orange);
% hold on
% plot(OIT.*(ones(NneuronI,1)*(1:TimeT)),OIT.*(((NneuronE+1):(NneuronE+NneuronI))'*ones(1,TimeT)),'.b');
% set(gca,'FontSize',fsize,'FontName','Helvetica')
% set(gca,'TickDir','out')
% xticklabels([0:0.2:1])
% ylim([0 375])
% xlabel('time (s)')
% box 
% saveas(h,['test\FigS16.pdf']); set(gcf,'Color',[1,1,1])
% 

%% save data

% save learned_o_2025_20.mat
figure;
x = 0:0.1:30;
y1 = 1+log(x);
y2 = 1./x;
PcostE1 = PcostE;
PcostI1 = PcostI;
subplot(2,3,1);plot(x,y1);hold on 
subplot(2,3,4);plot(x,y2);hold on
for i =1:size(PentrE,2)
    subplot(2,3,1)
    scatter(PcostE1(:,i),PentrE(:,i),36,orange,'o');hold on;%scatter(mean(PcostI1(:,i)),mean(PentrI(:,i)),'bo');
    ylim([1 4])
    subplot(2,3,4)
    scatter(PcostE1(:,i),PentrE_rate(:,i)./PcostE1(:,i),36,orange,'o');hold on;%scatter(mean(PcostI1(:,i)),mean(PentrI(:,i)/PcostI1(:,i)),'bo');
    pause(0.1)
    ylim([0 1])
end

% MI vs Cost scatter plots (3 levels)
subplot(2,3,2)
for ii = 1:size(MIE_ST,2)
    scatter(PcostE1(:,ii), MIE_spatial(:,ii), 20, [0.7 0.7 0.7], 'o'); hold on;
    scatter(PcostE1(:,ii), MIE_temporal(:,ii), 20, [1 0.8 0.5], '^'); hold on;
    scatter(PcostE1(:,ii), MIE_ST(:,ii), 36, orange, 's', 'filled'); hold on;
end
xlabel('Cost E'); ylabel('MI (bits)')
title('MI vs Cost (E, 3 Levels)')
legend('Spatial','Temporal','S+T','Location','best','FontSize',7)
box off

subplot(2,3,5)
for ii = 1:size(MIE_ST,2)
    scatter(PcostE1(:,ii), MIrateE_spatial(:,ii)./PcostE1(:,ii), 20, [0.7 0.7 0.7], 'o'); hold on;
    scatter(PcostE1(:,ii), MIrateE_temporal(:,ii)./PcostE1(:,ii), 20, [1 0.8 0.5], '^'); hold on;
    scatter(PcostE1(:,ii), MIrateE_ST(:,ii)./PcostE1(:,ii), 36, orange, 's', 'filled'); hold on;
end
xlabel('Cost E'); ylabel('MI/Cost')
title('MI per Cost (E, 3 Levels)')
box off

subplot(2,3,3)
for ii = 1:size(MIE_ST,2)
    scatter(PentrE(:,ii), MIE_spatial(:,ii), 20, [0.7 0.7 0.7], 'o'); hold on;
    scatter(PentrE(:,ii), MIE_temporal(:,ii), 20, [1 0.8 0.5], '^'); hold on;
    scatter(PentrE(:,ii), MIE_ST(:,ii), 36, orange, 's', 'filled'); hold on;
end
xlabel('Entropy E'); ylabel('MI (bits)')
title('MI vs Entropy (E, 3 Levels)')
box off

subplot(2,3,6)
for ii = 1:size(MIE_ST,2)
    scatter(PentrE(:,ii), MIE_spatial(:,ii)./PentrE(:,ii), 20, [0.7 0.7 0.7], 'o'); hold on;
    scatter(PentrE(:,ii), MIE_temporal(:,ii)./PentrE(:,ii), 20, [1 0.8 0.5], '^'); hold on;
    scatter(PentrE(:,ii), MIE_ST(:,ii)./PentrE(:,ii), 36, orange, 's', 'filled'); hold on;
end
xlabel('Entropy E'); ylabel('MI/Entropy')
title('MI Efficiency (E, 3 Levels)')
box off

figure;
subplot(1,3,1)
y =mean(PentrE,1); % your mean vector;
x = (2.^(1:T(1,1)))*dt;
std_dev = std(PentrE,1);
curve1 = y + std_dev;
curve2 = y - std_dev;
x2 = [x, fliplr(x)];
inBetween = [curve1, fliplr(curve2)];
semilogx(x, y, 'k', 'LineWidth', 2);hold on;fill(x2, inBetween, [0.5 0.5 0.5]);
xlim([10^(-1) 10^6])
% set(gca,'ticklength',[0.01 0.01]/ax(3))
% box off

subplot(1,3,2)
y =mean(PcostE,1); % your mean vector;
x = (2.^(1:T(1,1)))*dt;
std_dev = std(PcostE,1);
curve1 = y + std_dev;
curve2 = y - std_dev;
x2 = [x, fliplr(x)];
inBetween = [curve1, fliplr(curve2)];
semilogx(x, y, 'k', 'LineWidth', 2);hold on;fill(x2, inBetween, [0.5 0.5 0.5]);
xlim([10^(-1) 10^6])
% set(gca,'ticklength',[0.01 0.01]/ax(3))
% box off

subplot(1,3,3)
y =mean(PentrE_rate./PcostE,1); % your mean vector;
x = (2.^(1:T(1,1)))*dt;
std_dev = std(PentrE_rate./PcostE,1);
curve1 = y + std_dev;
curve2 = y - std_dev;
x2 = [x, fliplr(x)];
inBetween = [curve1, fliplr(curve2)];
semilogx(x, y, 'k', 'LineWidth', 2);hold on;fill(x2, inBetween, [0.5 0.5 0.5]);
hold on;

xlim([10^(-1) 10^6])
% box off
% set(gca,'ticklength',[0.01 0.01]/ax(3))
% box off

%% Three-Level Mutual Information Analysis
figure;
x = (2.^(1:T(1,1)))*dt;
x2 = [x, fliplr(x)];

% ---- E Population: MI Evolution ----
subplot(2,3,1)
set(gca, 'XScale', 'log'); hold on;
y_sp = mean(MIE_spatial,1); std_sp = std(MIE_spatial,0,1);
y_tp = mean(MIE_temporal,1); std_tp = std(MIE_temporal,0,1);
y_st = mean(MIE_ST,1); std_st = std(MIE_ST,0,1);
fill(x2, [y_sp+std_sp, fliplr(y_sp-std_sp)], [0.85 0.85 0.85], 'EdgeColor','none','FaceAlpha',0.5);
fill(x2, [y_tp+std_tp, fliplr(y_tp-std_tp)], [1 0.85 0.7], 'EdgeColor','none','FaceAlpha',0.5);
fill(x2, [y_st+std_st, fliplr(y_st-std_st)], [1 0.7 0.4], 'EdgeColor','none','FaceAlpha',0.5);
plot(x, y_sp, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.5);
plot(x, y_tp, ':', 'Color', orange, 'LineWidth', 1.5);
plot(x, y_st, '-', 'Color', orange, 'LineWidth', 2);
xlabel('time (s)'); ylabel('MI (bits)')
title('MI Evolution - E Pop')
legend('','','','Spatial','Temporal','S+T','Location','best','FontSize',7)
xlim([10^(-1) 10^6]); grid on; box off

% ---- E Population: MI/Cost ----
subplot(2,3,2)
set(gca, 'XScale', 'log'); hold on;
y_sp = mean(MIrateE_spatial./PcostE,1); std_sp = std(MIrateE_spatial./PcostE,0,1);
y_tp = mean(MIrateE_temporal./PcostE,1); std_tp = std(MIrateE_temporal./PcostE,0,1);
y_st = mean(MIrateE_ST./PcostE,1); std_st = std(MIrateE_ST./PcostE,0,1);
fill(x2, [y_sp+std_sp, fliplr(y_sp-std_sp)], [0.85 0.85 0.85], 'EdgeColor','none','FaceAlpha',0.5);
fill(x2, [y_tp+std_tp, fliplr(y_tp-std_tp)], [1 0.85 0.7], 'EdgeColor','none','FaceAlpha',0.5);
fill(x2, [y_st+std_st, fliplr(y_st-std_st)], [1 0.7 0.4], 'EdgeColor','none','FaceAlpha',0.5);
plot(x, y_sp, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.5);
plot(x, y_tp, ':', 'Color', orange, 'LineWidth', 1.5);
plot(x, y_st, '-', 'Color', orange, 'LineWidth', 2);
xlabel('time (s)'); ylabel('MI/Cost')
title('MI Efficiency - E Pop')
xlim([10^(-1) 10^6]); grid on; box off

% ---- E Population: H(r) decomposition over learning ----
subplot(2,3,3)
set(gca, 'XScale', 'log'); hold on;
hr_sp_E = mean(HrE_spatial,1); hrs_sp_E = hr_sp_E - mean(MIE_spatial,1);
hr_tp_E = mean(HrE_temporal,1); hrs_tp_E = hr_tp_E - mean(MIE_temporal,1);
hr_st_E = mean(HrE_ST,1); hrs_st_E = hr_st_E - mean(MIE_ST,1);
% Shade MI region (between Hr and Hrs) for each level
fill(x2, [hr_sp_E, fliplr(hrs_sp_E)], [0.85 0.85 0.85], 'EdgeColor','none','FaceAlpha',0.4);
fill(x2, [hr_tp_E, fliplr(hrs_tp_E)], [1 0.85 0.7], 'EdgeColor','none','FaceAlpha',0.4);
fill(x2, [hr_st_E, fliplr(hrs_st_E)], [1 0.7 0.4], 'EdgeColor','none','FaceAlpha',0.4);
% Hr (thick) and H(r|s) (thin) lines
plot(x, hr_sp_E, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.5);
plot(x, hrs_sp_E, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 0.5);
plot(x, hr_tp_E, ':', 'Color', orange, 'LineWidth', 1.5);
plot(x, hrs_tp_E, ':', 'Color', orange, 'LineWidth', 0.5);
plot(x, hr_st_E, '-', 'Color', orange, 'LineWidth', 2);
plot(x, hrs_st_E, '-', 'Color', orange, 'LineWidth', 0.5);
xlabel('time (s)'); ylabel('Entropy [bits]')
title('E: Hr (thick) & H(r|s) (thin), shaded=MI')
legend('','','','Spatial Hr','Spatial H(r|s)','Temporal Hr','Temporal H(r|s)','S+T Hr','S+T H(r|s)','Location','best','FontSize',6)
xlim([10^(-1) 10^6]); grid on; box off

% ---- I Population: MI Evolution ----
subplot(2,3,4)
set(gca, 'XScale', 'log'); hold on;
y_sp = mean(MII_spatial,1); std_sp = std(MII_spatial,0,1);
y_tp = mean(MII_temporal,1); std_tp = std(MII_temporal,0,1);
y_st = mean(MII_ST,1); std_st = std(MII_ST,0,1);
fill(x2, [y_sp+std_sp, fliplr(y_sp-std_sp)], [0.7 0.8 1], 'EdgeColor','none','FaceAlpha',0.5);
fill(x2, [y_tp+std_tp, fliplr(y_tp-std_tp)], [0.6 0.7 1], 'EdgeColor','none','FaceAlpha',0.5);
fill(x2, [y_st+std_st, fliplr(y_st-std_st)], [0.5 0.7 1], 'EdgeColor','none','FaceAlpha',0.5);
plot(x, y_sp, '--', 'Color', [0.5 0.5 0.8], 'LineWidth', 1.5);
plot(x, y_tp, ':', 'Color', 'b', 'LineWidth', 1.5);
plot(x, y_st, '-b', 'LineWidth', 2);
xlabel('time (s)'); ylabel('MI (bits)')
title('MI Evolution - I Pop')
legend('','','','Spatial','Temporal','S+T','Location','best','FontSize',7)
xlim([10^(-1) 10^6]); grid on; box off

% ---- I Population: MI/Cost ----
subplot(2,3,5)
set(gca, 'XScale', 'log'); hold on;
y_sp = mean(MIrateI_spatial./PcostI,1); std_sp = std(MIrateI_spatial./PcostI,0,1);
y_tp = mean(MIrateI_temporal./PcostI,1); std_tp = std(MIrateI_temporal./PcostI,0,1);
y_st = mean(MIrateI_ST./PcostI,1); std_st = std(MIrateI_ST./PcostI,0,1);
fill(x2, [y_sp+std_sp, fliplr(y_sp-std_sp)], [0.7 0.8 1], 'EdgeColor','none','FaceAlpha',0.5);
fill(x2, [y_tp+std_tp, fliplr(y_tp-std_tp)], [0.6 0.7 1], 'EdgeColor','none','FaceAlpha',0.5);
fill(x2, [y_st+std_st, fliplr(y_st-std_st)], [0.5 0.7 1], 'EdgeColor','none','FaceAlpha',0.5);
plot(x, y_sp, '--', 'Color', [0.5 0.5 0.8], 'LineWidth', 1.5);
plot(x, y_tp, ':', 'Color', 'b', 'LineWidth', 1.5);
plot(x, y_st, '-b', 'LineWidth', 2);
xlabel('time (s)'); ylabel('MI/Cost')
title('MI Efficiency - I Pop')
xlim([10^(-1) 10^6]); grid on; box off

% ---- I Population: H(r) decomposition over learning ----
subplot(2,3,6)
set(gca, 'XScale', 'log'); hold on;
hr_sp_I = mean(HrI_spatial,1); hrs_sp_I = hr_sp_I - mean(MII_spatial,1);
hr_tp_I = mean(HrI_temporal,1); hrs_tp_I = hr_tp_I - mean(MII_temporal,1);
hr_st_I = mean(HrI_ST,1); hrs_st_I = hr_st_I - mean(MII_ST,1);
% Shade MI region
fill(x2, [hr_sp_I, fliplr(hrs_sp_I)], [0.7 0.8 1], 'EdgeColor','none','FaceAlpha',0.4);
fill(x2, [hr_tp_I, fliplr(hrs_tp_I)], [0.6 0.7 1], 'EdgeColor','none','FaceAlpha',0.4);
fill(x2, [hr_st_I, fliplr(hrs_st_I)], [0.5 0.7 1], 'EdgeColor','none','FaceAlpha',0.4);
% Hr (thick) and H(r|s) (thin) lines
plot(x, hr_sp_I, '--', 'Color', [0.5 0.5 0.8], 'LineWidth', 1.5);
plot(x, hrs_sp_I, '--', 'Color', [0.5 0.5 0.8], 'LineWidth', 0.5);
plot(x, hr_tp_I, ':', 'Color', 'b', 'LineWidth', 1.5);
plot(x, hrs_tp_I, ':', 'Color', 'b', 'LineWidth', 0.5);
plot(x, hr_st_I, '-b', 'LineWidth', 2);
plot(x, hrs_st_I, '-b', 'LineWidth', 0.5);
xlabel('time (s)'); ylabel('Entropy [bits]')
title('I: Hr (thick) & H(r|s) (thin), shaded=MI')
legend('','','','Spatial Hr','Spatial H(r|s)','Temporal Hr','Temporal H(r|s)','S+T Hr','S+T H(r|s)','Location','best','FontSize',6)
xlim([10^(-1) 10^6]); grid on; box off