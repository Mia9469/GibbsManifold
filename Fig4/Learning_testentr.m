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
MIE=zeros(10,T);%array of mutual information between E population and input over learning
MII=zeros(10,T);%array of mutual information between I population and input over learning
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
        PentrE(r,i) = -nansum(counts .* log(counts));
        PcostE(r,i) = 2.4+0.37*mean(PrateE);
        PrateI = sum(OIT,2);
        [counts, ~] = histcounts(PrateI, 'Normalization', 'probability');
        PentrI(r,i) = -nansum(counts .* log(counts));
        PcostI(r,i) = 2.4+0.37*mean(PrateI);
        
        entr = zeros(1,300);
        for k = 1:300
        [counts, ~] = histcounts(rOET(k,:), 'Normalization', 'probability');
        entr(k) =  -nansum(counts .* log(counts));
        end
        PentrEl(r,i) = mean(entr);
        
        entr = zeros(1,75);
        for k = 1:75
        [counts, ~] = histcounts(rOIT(k,:), 'Normalization', 'probability');
        entr(k) =  -nansum(counts .* log(counts));
        end
        PentrIl(r,i) = mean(entr);
        
        % Compute Mutual Information using Hr-Hnoise method
        % Hr = Response Entropy = H(Y) - total entropy of neural response
        % Hnoise = Noise Entropy = H(Y|X) - conditional entropy (stimulus-dependent noise)
        % MI = Hr - Hnoise
        
        nbins_input = 10;
        nbins_neural = 10;
        
        % Average input signal across dimensions to match neural time series length
        input_signal = mean(InputT, 1);  % 1 x TimeT
        input_binned = discretize(input_signal, nbins_input);  % 1 x TimeT
        
        % Remove NaN values
        valid_idx = ~isnan(input_binned);
        input_binned_clean = input_binned(valid_idx);
        
        % Compute MI for E population using Hr-Hnoise
        MI_E_temp = 0;
        for k = 1:min(300, NneuronE)
            neural_binned = discretize(rOET(k,:), nbins_neural);
            neural_binned_clean = neural_binned(valid_idx);
            
            if isempty(neural_binned_clean)
                continue;
            end
            
            % Compute Hr (Response Entropy) - total entropy of neural response
            p_neural = histcounts(neural_binned_clean, nbins_neural, 'Normalization', 'probability');
            p_neural(p_neural == 0) = [];
            H_r = -nansum(p_neural .* log(p_neural + eps));
            
            % Compute Hnoise (Noise Entropy) - conditional entropy H(Y|X)
            H_noise = 0;
            unique_inputs = unique(input_binned_clean);
            for input_state = unique_inputs'
                idx = input_binned_clean == input_state;
                p_input_state = sum(idx) / length(input_binned_clean);
                
                % Distribution of neural response given this input state
                neural_response_given_input = neural_binned_clean(idx);
                p_neural_given_input = histcounts(neural_response_given_input, nbins_neural, 'Normalization', 'probability');
                p_neural_given_input(p_neural_given_input == 0) = [];
                
                % Entropy of neural response for this input state
                H_cond_state = -nansum(p_neural_given_input .* log(p_neural_given_input + eps));
                
                % Add weighted contribution to total noise entropy
                H_noise = H_noise + p_input_state * H_cond_state;
            end
            
            % MI = Hr - Hnoise
            MI_E_temp = MI_E_temp + (H_r - H_noise);
        end
        MIE(r,i) = MI_E_temp / min(300, NneuronE);
        
        % Compute MI for I population using Hr-Hnoise
        MI_I_temp = 0;
        for k = 1:min(75, NneuronI)
            neural_binned = discretize(rOIT(k,:), nbins_neural);
            neural_binned_clean = neural_binned(valid_idx);
            
            if isempty(neural_binned_clean)
                continue;
            end
            
            % Compute Hr (Response Entropy) - total entropy of neural response
            p_neural = histcounts(neural_binned_clean, nbins_neural, 'Normalization', 'probability');
            p_neural(p_neural == 0) = [];
            H_r = -nansum(p_neural .* log(p_neural + eps));
            
            % Compute Hnoise (Noise Entropy) - conditional entropy H(Y|X)
            H_noise = 0;
            unique_inputs = unique(input_binned_clean);
            for input_state = unique_inputs'
                idx = input_binned_clean == input_state;
                p_input_state = sum(idx) / length(input_binned_clean);
                
                % Distribution of neural response given this input state
                neural_response_given_input = neural_binned_clean(idx);
                p_neural_given_input = histcounts(neural_response_given_input, nbins_neural, 'Normalization', 'probability');
                p_neural_given_input(p_neural_given_input == 0) = [];
                
                % Entropy of neural response for this input state
                H_cond_state = -nansum(p_neural_given_input .* log(p_neural_given_input + eps));
                
                % Add weighted contribution to total noise entropy
                H_noise = H_noise + p_input_state * H_cond_state;
            end
            
            % MI = Hr - Hnoise
            MI_I_temp = MI_I_temp + (H_r - H_noise);
        end
        MII(r,i) = MI_I_temp / min(75, NneuronI);
    end
    
    
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%  Plotting Decoding Error, rates through Learning  %%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

orange = [1 0.5 0.2];


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
T_obs_population = TimeT * dt;  % observation duration [s]
PentrE_rate = PentrE / T_obs_population;  % [nats/s]
PentrI_rate = PentrI / T_obs_population;  % [nats/s]
semilogx((2.^(1:T(1,1)))*dt,PentrE_rate./PcostE,'Color',orange);
hold on
semilogx((2.^(1:T(1,1)))*dt,PentrI_rate./PcostE,'b');
set(gca,'FontSize',fsize,'FontName','Helvetica')
set(gca,'TickDir','out')
ylabel('information / calibrated cost proxy')
xlabel('time (s)')
xlim([10^(-1) 10^6])
% plot(sort(firingRatesE(:,end),'descend'),'ko-')
% set(gca,'FontSize',fsize,'FontName','Helvetica')
% set(gca,'TickDir','out')
% ylabel('FiringRate(Hz)');xlabel("Neuron")
% ylim([0 100])
set(gca,'ticklength',[0.01 0.01]/ax(3))
box off

% Add Mutual Information plots
h=subplot(lines,3,[2]);
ax=get(h,'Position');
semilogx((2.^(1:T(1,1)))*dt,mean(MIE,1),'Color',orange);
hold on
semilogx((2.^(1:T(1,1)))*dt,mean(MII,1),'b');
set(gca,'FontSize',fsize,'FontName','Helvetica')
set(gca,'TickDir','out')
ylabel('MI (nats)')
xlabel('time (s)')
title('Mutual Information Evolution')
xlim([10^(-1) 10^6])
set(gca,'ticklength',[0.01 0.01]/ax(3))
box off

h=subplot(lines,3,[3]);
ax=get(h,'Position');
semilogx((2.^(1:T(1,1)))*dt,mean(MIE,1)./mean(PcostE,1),'Color',orange);
hold on
semilogx((2.^(1:T(1,1)))*dt,mean(MII,1)./mean(PcostI,1),'b');
set(gca,'FontSize',fsize,'FontName','Helvetica')
set(gca,'TickDir','out')
ylabel('MI/Cost')
xlabel('time (s)')
title('MI per Cost')
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

% Add MI vs Cost scatter plots
subplot(2,3,2)
for i =1:size(MIE,2)
    scatter(PcostE1(:,i),MIE(:,i),36,orange,'o');hold on;
end
xlabel('Cost E')
ylabel('MI E (nats)')
title('MI vs Cost (E population)')
box off

subplot(2,3,5)
for i =1:size(MIE,2)
    scatter(PcostE1(:,i),MIE(:,i)./PcostE1(:,i),36,orange,'o');hold on;
end
xlabel('Cost E')
ylabel('MI/Cost E')
title('MI per Cost (E population)')
box off

% Add MI vs Entropy scatter plot
subplot(2,3,3)
for i =1:size(MIE,2)
    scatter(PentrE(:,i),MIE(:,i),36,orange,'o');hold on;
end
xlabel('Entropy E')
ylabel('MI E (nats)')
title('MI vs Entropy (E population)')
box off

subplot(2,3,6)
for i =1:size(MIE,2)
    scatter(PentrE(:,i),MIE(:,i)./PentrE(:,i),36,orange,'o');hold on;
end
xlabel('Entropy E')
ylabel('MI/Entropy E')
title('MI Efficiency (E population)')
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

%% Mutual Information Analysis
figure;
subplot(2,2,1)
y =mean(MIE,1); % MI E mean
x = (2.^(1:T(1,1)))*dt;
std_dev = std(MIE,1);
curve1 = y + std_dev;
curve2 = y - std_dev;
x2 = [x, fliplr(x)];
inBetween = [curve1, fliplr(curve2)];
semilogx(x, y, 'Color', orange, 'LineWidth', 2);hold on;fill(x2, inBetween, [1 0.7 0.4]);
xlabel('time (s)')
ylabel('MI (nats)')
title('MI Evolution - E Population')
xlim([10^(-1) 10^6])
grid on
box off

subplot(2,2,2)
y =mean(MIE./PcostE,1); % MI/Cost ratio for E
x = (2.^(1:T(1,1)))*dt;
std_dev = std(MIE./PcostE,1);
curve1 = y + std_dev;
curve2 = y - std_dev;
x2 = [x, fliplr(x)];
inBetween = [curve1, fliplr(curve2)];
semilogx(x, y, 'Color', orange, 'LineWidth', 2);hold on;fill(x2, inBetween, [1 0.7 0.4]);
xlabel('time (s)')
ylabel('MI/Cost')
title('MI Efficiency - E Population')
xlim([10^(-1) 10^6])
grid on
box off

subplot(2,2,3)
y =mean(MII,1); % MI I mean
x = (2.^(1:T(1,1)))*dt;
std_dev = std(MII,1);
curve1 = y + std_dev;
curve2 = y - std_dev;
x2 = [x, fliplr(x)];
inBetween = [curve1, fliplr(curve2)];
semilogx(x, y, 'b', 'LineWidth', 2);hold on;fill(x2, inBetween, [0.5 0.7 1]);
xlabel('time (s)')
ylabel('MI (nats)')
title('MI Evolution - I Population')
xlim([10^(-1) 10^6])
grid on
box off

subplot(2,2,4)
y =mean(MII./PcostI,1); % MI/Cost ratio for I
x = (2.^(1:T(1,1)))*dt;
std_dev = std(MII./PcostI,1);
curve1 = y + std_dev;
curve2 = y - std_dev;
x2 = [x, fliplr(x)];
inBetween = [curve1, fliplr(curve2)];
semilogx(x, y, 'b', 'LineWidth', 2);hold on;fill(x2, inBetween, [0.5 0.7 1]);
xlabel('time (s)')
ylabel('MI/Cost')
title('MI Efficiency - I Population')
xlim([10^(-1) 10^6])
grid on
box off
