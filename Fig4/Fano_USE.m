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
clear
close all
for noisestd=[100]
load('learned_o_1023.mat')
% load('learned_n_1104.mat')
fprintf('Computing decoding errors and rates over learning\n')
TimeT=10000; % size of the test input
Trials=1; %number of trials
MeanPrateE=zeros(Trials,T);%array of the mean rates over learning
MeanPrateI=zeros(Trials,T);%array of the mean rates over learning
ErrorE=zeros(Trials,T);%array of the decoding error over learning
ErrorI=zeros(Trials,T);%array of the decoding error over learning
MembraneVarE=zeros(Trials,T);%mean membrane potential variance over learning
MembraneVarI=zeros(Trials,T);%mean membrane potential variance over learning
xT=zeros(Nx,TimeT);%target ouput
Fanos=zeros(Trials,T);
CVp=zeros(Trials,T);
CVl=zeros(Trials,T);
Sp=zeros(Trials,T);
Sl=zeros(Trials,T);
mmp=zeros(Trials,T);
sta_rec_E = zeros(1201,T,Trials);
sta_rec_I = zeros(1201,T,Trials);
ei = zeros(T,Trials);


% InputT=A*(mvnrnd(zeros(1,Nx),eye(Nx),TimeT))'; % we generate a new input
% 
% for k=1:Nx
%     InputT(k,:)=conv(InputT(k,:),w,'same'); % we wmooth it
% %     InputT(k,:) = InputT(k,:)-min(InputT(k,:));
% end 
% % InputT(InputT<0)=0;
% save net_input_long.mat InputT

load('net_input_long.mat')


InputT = InputT+noisestd*rand(size(InputT));

for t=2:TimeT
    xT(:,t)= (1-lambda*dt)*xT(:,t-1)+ dt*InputT(:,t-1); % ans we comput the target output by leaky inegration of the input
end

% figure;
for i=1:T %for each instance of the network
    rec_raster=cell(1,Trials);
    for r=1:Trials %for each trial

        ISIall=zeros(NneuronE,1);
        NbISI=zeros(NneuronE,1);
        ir=zeros(NneuronE,1);
        
        
        [ rOET, OET,VET, rOIT, OIT,VIT,sc2E, sc2I] = runnet(dt,TimeT, lambda,NneuronE,NneuronI,ThreshE,ThreshI,InputT,squeeze(FsE(i,:,:)),squeeze(CsEE(i,:,:)),squeeze(CsEI(i,:,:)),squeeze(CsII(i,:,:)),squeeze(CsIE(i,:,:)));%we run the network with current input InputT
        Counts=sum(OET,2); %for each trial we register the number of spikes emitted bt each neuron
        rec_raster{r}=[OET;OIT];
        for g=1:NneuronE
            
            if(Counts(g)>1) % if the neuron g has more than one spike
                U=diff(find(OET(g,:)>0));
                ISIall(g,(NbISI(g)+1):(NbISI(g)+length(U)))=U/10; % we conpute the ISIs (inter-spike interval)
                NbISI(g)=NbISI(g)+length(U);  % and the number of spikes  per trial
                ir(g,(NbISI(g)+1):(NbISI(g)+length(U)))=10000./U;
                
                [n,p] = hist(ir(g,:),10:10:500);
                h = n/sum(n);
                entr(g) = 0;
                for j=h
                    if j>0
                    entr(g) = entr(g)-j*log(j);
                    end
                end 
            
            end
        end
        
        
        % Define the Gaussian kernel
        kernel_width = 0.005; % Width of Gaussian kernel in milliseconds
        kernel_size = 5 * kernel_width; % Define the size of the kernel (adjust as needed)
        t_kernel = -kernel_size:dt:kernel_size; % Time vector for the kernel

        % Gaussian kernel formula
        sigma = kernel_width; % Standard deviation of Gaussian (in milliseconds)
        gaussian_kernel = exp(-t_kernel.^2 / (2 * sigma^2));
        gaussian_kernel = gaussian_kernel / sum(gaussian_kernel); % Normalize the kernel

        % Convolve the delta pulses with the Gaussian kernel
        smoothed_signal_E = conv(squeeze(sc2E(2,:,1)), gaussian_kernel, 'same');
        smoothed_signal_I = conv(squeeze(sc2E(2,:,2)), gaussian_kernel, 'same');

        [staE,~] = sta_mia(smoothed_signal_E,OET(2,:),0.1,100,20,1);
        [staI,~] = sta_mia(smoothed_signal_I,OET(2,:),0.1,100,20,1);
        
        ei(i,r) = mean(abs(smoothed_signal_E))/mean(abs(smoothed_signal_I));
        
        if staE~=0
            sta_rec_E(:,i,r) =   sta_rec_E(:,i,r) +staE';
            sta_rec_I(:,i,r) =   sta_rec_I(:,i,r) +staI';
        end
            
        xestE=squeeze(DecsE(i,:,:))*rOET; %we deocode the ouptut using the optinal decoders previously computed
        xestI=squeeze(DecsI(i,:,:))*rOIT; %we deocode the ouptut using the optinal decoders previously computed
        
        if (i==1 && r==Trials)
            
            xestEB=xestE;
            xestIB=xestI;
            smoothed_signal_EB =  smoothed_signal_E;
            smoothed_signal_IB = smoothed_signal_I;
            OEB=OET;
            OIB=OIT;
            
        end
        ErrorE(r,i)=sum(var(xT-xestE,0,2))/(sum(var(xT,0,2))*Trials);%we comput the variance of the error normalized by the variance of the target E population
        ErrorI(r,i)=sum(var(xT-xestI,0,2))/(sum(var(xT,0,2))*Trials);%we comput the variance of the error normalized by the variance of the target for I population
        MeanPrateE(r,i)=sum(sum(OET))/(TimeT*dt*NneuronE*Trials);%we comput the average firing rate per neuron E population
        MeanPrateI(r,i)=sum(sum(OIT))/(TimeT*dt*NneuronI*Trials);%we comput the average firing rate per neuron for I population
        MembraneVarE(r,i)=sum(var(VET,0,2))/(NneuronE*Trials);% we compute the average membrane potential variance per neuron for E population
        MembraneVarI(r,i)=sum(var(VIT,0,2))/(NneuronI*Trials);% we compute the average membrane potential variance per neuron for I population
        id = mean(ir');
        id = id>0;
        CVp(r,i)=(std(sum(OET,2))/mean(sum(OET,2)));%we comput the average firing rate per neuron E population
        Sp(r,i)=(1-mean(sum(OET,2))^2/mean(sum(OET,2).^2));%we comput the average firing rate per neuron for I population
        CVl(r,i)=mean(std(ir(id,:)')./mean(ir(id,:)'));%we comput the average firing rate per neuron for I population
        Sl(r,i)=mean(ones(1,sum(id))-mean(ir(id,:)').^2./mean(ir(id,:)'.^2));%we comput the average firing rate per neuron for I population
        Fanos(r,i)=mean(var(OET(id,:)')./mean(OET(id,:)'));%we comput the average firing rate per neuron for I population
        mmp(r,i)=mean(sum(OET,2).^2);%we comput the average firing rate per neuron for I population
        entrl(r,i)= mean(entr);
        [n,p] = hist(sum(OET,2),1:1:20);
        h = n/sum(n);
        entrp(r,i) = 0;
        for j=h
            if j>0
            entrp(r,i) = entrp(r,i)-j*log(j);
            end
        end 
        
%         if r==1
%         subplot(3,10,i);histogram(sum(OET,2),1:1:60)
%         end
    end
    
%     save(['recorded_spikes_' num2str(i) '_' num2str(noisestd) '_rec_long.mat'], "rec_raster","sc2E","sc2I","xestI","xestE")
   
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%  Plotting Decoding Error, rates through Learning  %%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
orange = [1 0.5 0.2];


figure
set(gcf,'Units','centimeters')
xSize = 50;  ySize =34;
xLeft = (21-xSize)/4; yTop = (30-ySize)/4;
set(gcf,'Position',[xLeft yTop xSize ySize]); %centers on A4 paper
set(gcf, 'Color', 'w');

lines=6;
fsize=11;

h=subplot(lines,3,13);
ax=get(h,'Position');
semilogx((2.^(1:T(1,1)))*dt,mean(ErrorE),'Color',orange);
hold on
semilogx((2.^(1:T(1,1)))*dt,mean(ErrorI),'b');
set(gca,'FontSize',fsize,'FontName','Helvetica')
set(gca,'TickDir','out')
ylabel('Decoding Error')
xlim([10^(-1) 10^5])
set(gca,'ticklength',[0.01 0.01]/ax(3))
box off


% h=subplot(lines,3,[7 10]);
h=subplot(lines,3,16);
ax=get(h,'Position');
semilogx((2.^(1:T(1,1)))*dt,mean(MeanPrateE),'Color',orange);
hold on
semilogx((2.^(1:T(1,1)))*dt,mean(MeanPrateI),'b');
hold on
% semilogx((2.^(1:T(1,1)))*dt,mmp,'k');
set(gca,'FontSize',fsize,'FontName','Helvetica')
set(gca,'TickDir','out')
ylabel('Mean rate per neuron')
xlim([10^(-1) 10^5])
set(gca,'ticklength',[0.01 0.01]/ax(3))
box off

h=subplot(lines,3,1);
ax=get(h,'Position');
semilogx((2.^(1:T(1,1)))*dt,mean(MeanPrateE)./mean(MeanPrateI));
set(gca,'FontSize',fsize,'FontName','Helvetica')
set(gca,'TickDir','out')
xlabel('time')
ylabel('f_E/f_I')
title('Evolution Through Learning')
xlim([10^(-1) 10^5])
set(gca,'ticklength',[0.01 0.01]/ax(3))
box off

h=subplot(lines,3,7);
ax=get(h,'Position');
semilogx((2.^(1:T(1,1)))*dt,mean(ei,2));
set(gca,'FontSize',fsize,'FontName','Helvetica')
set(gca,'TickDir','out')
xlabel('time')
ylabel('EPSC/IPSC')
xlim([10^(-1) 10^5])
set(gca,'ticklength',[0.01 0.01]/ax(3))
box off
% 
% h=subplot(lines,3,3);
% ax=get(h,'Position');
% semilogx((2.^(1:T(1,1)))*dt,CVp);
% set(gca,'FontSize',fsize,'FontName','Helvetica')
% set(gca,'TickDir','out')
% ylabel('CV_P')
% xlim([10^(-1) 10^5])
% set(gca,'ticklength',[0.01 0.01]/ax(3))
% box off
% h=subplot(lines,3,4);
% ax=get(h,'Position');
% semilogx((2.^(1:T(1,1)))*dt,CVl);
% set(gca,'FontSize',fsize,'FontName','Helvetica')
% set(gca,'TickDir','out')
% ylabel('CV_L')
% xlim([10^(-1) 10^5])
% set(gca,'ticklength',[0.01 0.01]/ax(3))
% box off
% hold on;
% semilogx((2.^(1:T(1,1)))*dt,Sp);
% hold on;
% semilogx((2.^(1:T(1,1)))*dt,Sl);
% hold on;
% semilogx((2.^(1:T(1,1)))*dt,Fanos);
% hold on;


h=subplot(lines,3,4);
semilogx((2.^(1:T(1,1)))*dt,mean(CVp));
set(gca,'FontSize',fsize,'FontName','Helvetica')
set(gca,'TickDir','out')
ylabel('CVp')
xlim([10^(-1) 10^5])
box off


h=subplot(lines,3,10);
semilogx((2.^(1:T(1,1)))*dt,mean(CVl));
set(gca,'FontSize',fsize,'FontName','Helvetica')
set(gca,'TickDir','out')
ylabel('CVl')
xlim([10^(-1) 10^5])
box off

h=subplot(lines,3,[2 5]);
plot(dt:dt:TimeT*dt,xestEB(3,:),'Color',orange);
hold on
plot(dt:dt:TimeT*dt,xestIB(3,:),'b');
plot(dt:dt:TimeT*dt,xT(3,:),'k')
set(gca,'FontSize',fsize,'FontName','Helvetica')
set(gca,'TickDir','out')
ylabel('x,x_h')
title('Before Learning')
box off

h=subplot(lines,3,[3 6]);
plot(dt:dt:TimeT*dt,xestE(3,:),'Color',orange);
hold on
plot(dt:dt:TimeT*dt,xestI(3,:),'b');
plot(dt:dt:TimeT*dt,xT(3,:),'k')
set(gca,'FontSize',fsize,'FontName','Helvetica')
set(gca,'TickDir','out')
title('After Learning')
xlabel('time')
box off

h=subplot(lines,3,[8 11]);
plot(dt:dt:TimeT*dt,smoothed_signal_EB,'Color',orange);
hold on
plot(dt:dt:TimeT*dt,-smoothed_signal_IB,'b');
set(gca,'FontSize',fsize,'FontName','Helvetica')
set(gca,'TickDir','out')
ylabel('Synaptic Current')
ylim([-0.05,0.05])
box off

h=subplot(lines,3,[9 12]);
plot(dt:dt:TimeT*dt,smoothed_signal_E,'Color',orange);
hold on
plot(dt:dt:TimeT*dt,-smoothed_signal_I,'b');
set(gca,'FontSize',fsize,'FontName','Helvetica')
set(gca,'TickDir','out')
ylim([-0.05,0.05])
xlabel('time')
box off
%
h=subplot(lines,3,[14 17]);
plot(OEB.*(ones(NneuronE,1)*(1:TimeT)),OEB.*((1:NneuronE)'*ones(1,TimeT)),'|k');
hold on
plot(OIB.*(ones(NneuronI,1)*(1:TimeT)),OIB.*(((NneuronE+1):(NneuronE+NneuronI))'*ones(1,TimeT)),'|','Color',[0.8 0.8 0.8]);
set(gca,'FontSize',fsize,'FontName','Helvetica')
set(gca,'TickDir','out')
xlabel('time')
ylabel('Neuron ID')
box off

h=subplot(lines,3,[15 18]);
plot(OET.*(ones(NneuronE,1)*(1:TimeT)),OET.*((1:NneuronE)'*ones(1,TimeT)),'|k');
hold on
plot(OIT.*(ones(NneuronI,1)*(1:TimeT)),OIT.*(((NneuronE+1):(NneuronE+NneuronI))'*ones(1,TimeT)),'|','Color',[0.8 0.8 0.8]);
set(gca,'FontSize',fsize,'FontName','Helvetica')
set(gca,'TickDir','out')
xlabel('time')
ylabel('Neuron ID')
box off


staE = mean(sta_rec_E,3);
staI = mean(sta_rec_I,3);
figure;
k=0;
for i = 1:2:size(staE,3)
    k=k+1;
    subplot(3,5,k)
    plot(-100:0.1:20,staE(:,i),'Color',orange);hold on;
    plot(-100:0.1:20,staI(:,i),'b');hold on;
    plot(-100:0.1:20,-staI(:,i)+staE(:,i),'k');hold on;
    xlabel('Time')
    ylabel('Current')
    ylim([-0.025 0.025])
  
end


figure;
subplot(2,2,1)
semilogx((2.^(1:T(1,1)))*dt,mean(entrl))
xlim([10^(-1) 10^5])
ylabel('Entropy/neuron')
subplot(2,2,2)
semilogx((2.^(1:T(1,1)))*dt,mean(entrp))
ylabel('Entropy of population')
xlim([10^(-1) 10^5])
subplot(2,2,3)
semilogx((2.^(1:T(1,1)))*dt,mean(entrl)./mean(MeanPrateE))
ylabel('ICE/neuron')
xlim([10^(-1) 10^5])
subplot(2,2,4)
semilogx((2.^(1:T(1,1)))*dt,mean(entrp)./mean(MeanPrateE))
ylabel('ICE')
xlim([10^(-1) 10^5])

% clearvars rec_raster
% save([num2str(noisestd) '_rec_long.mat'],"CVl","CVp","ei","entrl","entrp","ErrorE",...
%     "ErrorI","Fanos","MeanPrateE","MeanPrateI","MembraneVarE","MembraneVarI","mmp",...
%     "Sp","Sl","sta_rec_E","sta_rec_I") 

end