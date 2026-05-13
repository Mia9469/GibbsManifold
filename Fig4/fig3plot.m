%% CVL,SL
clear;close all
load('0_rec.mat')
figure;
subplot(3,2,1)

model_series =[mean(CVl(:,[1 29]),1);mean(Sl(:,[1 29]),1)]'; 
model_error =[std(CVl(:,[1 29]),0,1);std(Sl(:,[1 29]),0,1)]'; 
b = bar(model_series, 'grouped');

hold on
% Calculate the number of groups and number of bars in each group
[ngroups,nbars] = size(model_series);

% Get the x coordinate of the bars
x = nan(nbars, ngroups);
for i = 1:nbars
 x(i,:) = b(i).XEndPoints;
end

% Plot the errorbars
errorbar(x',model_series,model_error,'k','linestyle','none');
hold off
xticklabels(["before","after",])
legend(["CVl","Sl"])

%% CVP,SP
subplot(3,2,2)
% Example data 
model_series =[mean(CVp(:,[1 29]),1);mean(Sp(:,[1 29]),1)]'; 
model_error =[std(CVp(:,[1 29]),0,1);std(Sp(:,[1 29]),0,1)]'; 
b = bar(model_series, 'grouped');

hold on
% Calculate the number of groups and number of bars in each group
[ngroups,nbars] = size(model_series);

% Get the x coordinate of the bars
x = nan(nbars, ngroups);
for i = 1:nbars
 x(i,:) = b(i).XEndPoints;
end

% Plot the errorbars
errorbar(x',model_series,model_error,'k','linestyle','none');
hold off
xticklabels(["before","after",])
legend(["CVp","Sp"])

% %% Hr,Hn
% load('Info_all_100.mat')
% va1 = [Hr{1,1};Hr{2,1}];
% va11 = [Hn{1,1}; Hn{2,1}];
% 
% subplot(3,2,3)
% 
% % Example data 
% model_series =[mean(va1,2),mean(va11,2)]; 
% model_error =[std(va1,0,2),std(va11,0,2)];
% b = bar(model_series, 'grouped');
% 
% hold on
% % Calculate the number of groups and number of bars in each group
% [ngroups,nbars] = size(model_series);
% 
% % Get the x coordinate of the bars
% x = nan(nbars, ngroups);
% for i = 1:nbars
%  x(i,:) = b(i).XEndPoints;
% end
% 
% % Plot the errorbars
% errorbar(x',model_series,model_error,'k','linestyle','none');
% hold off
% xticklabels(["before","after",])
% legend(["Hr","Hn"])

%% SNR,ICE
load('Info_all_500.mat')
va1 = [Hr{1,1};Hr{2,1}];
va11 = [Hn{1,1}; Hn{2,1}];
v1 = va1(~isnan(va1));

load('500_rec_long.mat', 'MeanPrateE')
subplot(3,2,3)
% Example data
model_series =mean(va1./va11,2);
model_error =std(va1./va11,0,2);
b = bar(model_series, 'grouped');

hold on
% Calculate the number of groups and number of bars in each group
[ngroups,nbars] = size(model_series);

% Get the x coordinate of the bars
x = nan(nbars, ngroups);
for i = 1:nbars
 x(i,:) = b(i).XEndPoints;
end

% Plot the errorbars
errorbar(x',model_series,model_error,'k','linestyle','none');
hold off
xticklabels(["before","after",])
title("SNR")

subplot(3,2,4)
va1 = [Hr{1,1};Hr{2,1}];
va11 = [Hn{1,1}; Hn{2,1}];
v1 = va1(~isnan(va1));

model_series =mean([va1-va11]./(ones(10,1)*4.03+ones(10,1)*MeanPrateE([1 29]))',2);

model_error =std([va1-va11]./(4.03*ones(10,1)+ones(10,1)*MeanPrateE([1 29]))',0,2);
b = bar(model_series, 'grouped');

hold on
% Calculate the number of groups and number of bars in each group
[ngroups,nbars] = size(model_series);

% Get the x coordinate of the bars
x = nan(nbars, ngroups);
for i = 1:nbars
 x(i,:) = b(i).XEndPoints;
end

% Plot the errorbars
errorbar(x',model_series,model_error,'k','linestyle','none');
hold off
xticklabels(["before","after",])

title("ICE")



%% Reproducibility
load  reliability_array_final.mat
subplot(3,2,5)

% Example data 
model_series =mean(ssim_sp1([1 29],2,:),3); 
model_error =std(ssim_sp1([1 29],2,:),0,3); 
b = bar(model_series, 'grouped');

hold on
% Calculate the number of groups and number of bars in each group
[ngroups,nbars] = size(model_series);

% Get the x coordinate of the bars
x = nan(nbars, ngroups);
for i = 1:nbars
 x(i,:) = b(i).XEndPoints;
end

% Plot the errorbars
errorbar(x',model_series,model_error,'k','linestyle','none');
hold off
ylabel("Reliability (SSIM RAST)")
xticklabels(["before","after",])

ylim([0.3 0.5])

%% Error


load 10_rec.mat
subplot(3,2,6)

% Example data 
model_series =mean(ErrorE(:,[1 29]),1)'; 
model_error =std(ErrorE(:,[1 29]),0,1); 
b = bar(model_series);

hold on
% Calculate the number of groups and number of bars in each group
[ngroups,nbars] = size(model_series);

% Get the x coordinate of the bars
x = nan(nbars, ngroups);
for i = 1:nbars
 x(i,:) = b(i).XEndPoints;
end

% Plot the errorbars
errorbar(x',model_series,model_error,'k','linestyle','none');
hold off

ylabel("Robustness")
xticklabels(["before","after",])

figure
%% Reproducibility
load  reliability_array_final.mat

model_series =mean(ssim_sp1([1 29],:,:),3); 
model_error =std(ssim_sp1([1 29],:,:),0,3); 


subplot(3,2,1)
y =model_series(1,:); % your mean vector;
x = 0:100:1000;
std_dev = model_error(1,:);
curve1 = y + std_dev;
curve2 = y - std_dev;
x2 = [x, fliplr(x)];
inBetween = [curve1, fliplr(curve2)];
fill(x2, inBetween, [0.5 0.5 0.5]);
hold on;
plot(x, y, 'k', 'LineWidth', 2);
title("Before")
ylabel("Reliability (SSIM RAST)")
xlabel("noise std (pA)")
ylim([0.3 0.5])

subplot(3,2,2)
y =model_series(2,:); % your mean vector;
std_dev = model_error(2,:);
curve1 = y + std_dev;
curve2 = y - std_dev;
x2 = [x, fliplr(x)];
inBetween = [curve1, fliplr(curve2)];
fill(x2, inBetween, [0.5 0.5 0.5]);
hold on;
plot(x, y, 'k', 'LineWidth', 2);
title("After")
ylabel("Reliability (SSIM RAST)")
xlabel("noise std (pA)")
ylim([0.3 0.5])

%% SNR


load('Info_all_0.mat')
va1 = Hr{1,1};
va2 = Hr{2,1};
va11 = Hn{1,1};
va22 = Hn{2,1};
% figure;
% bar([0 1],[mean(va1(va1>0),'all'),mean(va2(va2>0),'all');mean(va11(va11>0),'all'),mean(va22(va22>0),'all')]');


for i = 1:10
load(['Info_all_' num2str(i) '00.mat'])
va1 = [va1;Hr{1,1}];
va2 = [va2;Hr{2,1}];
va11 = [va11;Hn{1,1}];
va22 = [va22;Hn{2,1}];
end


% Example data 
model_series =[mean(va1./va11,2),mean(va2./va22,2)]'; 
model_error =[std(va1./va11,0,2),std(va2./va22,0,2)]';

subplot(3,2,3)
y =model_series(1,:); % your mean vector;
x = 0:100:1000;
std_dev = model_error(1,:);
curve1 = y + std_dev;
curve2 = y - std_dev;
x2 = [x, fliplr(x)];
inBetween = [curve1, fliplr(curve2)];
fill(x2, inBetween, [0.5 0.5 0.5]);
hold on;
plot(x, y, 'k', 'LineWidth', 2);
title("Before")
ylabel("SNR")
xlabel("noise std (pA)")
ylim([0 10])

subplot(3,2,4)
y =model_series(2,:); % your mean vector;
std_dev = model_error(2,:);
curve1 = y + std_dev;
curve2 = y - std_dev;
x2 = [x, fliplr(x)];
inBetween = [curve1, fliplr(curve2)];
fill(x2, inBetween, [0.5 0.5 0.5]);
hold on;
plot(x, y, 'k', 'LineWidth', 2);
title("After")
ylabel("SNR")
xlabel("noise std (pA)")
ylim([0 10])

%% Error

subplot(1,3,3)


load('0_rec.mat')
va1 = ErrorE(:,1); 
va2 = ErrorE(:,29); 
for i = 1:10
load([num2str(i) '00_rec.mat'])
va1 = [va1,ErrorE(:,1)];
va2 = [va2,ErrorE(:,29)];
end



% Example data 
model_series =[mean(va1,1);mean(va2,1)]; 
model_error =[std(va1,0,1);std(va2,0,1)];
b = bar(model_series, 'grouped');

hold on
% Calculate the number of groups and number of bars in each group
[ngroups,nbars] = size(model_series);

% Get the x coordinate of the bars
x = nan(nbars, ngroups);
for i = 1:nbars
 x(i,:) = b(i).XEndPoints;
end

% Plot the errorbars
errorbar(x',model_series,model_error,'k','linestyle','none');
hold on

ylabel("ERROR")
xticklabels(["before","after",])
xlabel("noise std (pA)")

figure;
xx = 0:0.1:1;
plot(xx,model_series(1,:),'*');hold on;
plot(xx,model_series(2,:),'*');hold on;
c1 = polyfit(xx,model_series(1,:),1);
c2 = polyfit(xx,model_series(2,:),1);
plot(xx,c1(2)+c1(1)*xx);hold on;
plot(xx,c2(2)+c2(1)*xx);hold on;

%% l1 l2
load 100_rec.mat
figure;
subplot(3,4,[1 2])

plot(mean(MeanPrateE))
subplot(3,4,[5 6]);
plot(mean(sqrt(mmp-MeanPrateE.^2)));
subplot(3,4,[9 10]);
plot(mean(sqrt(mmp-MeanPrateE.^2))./MeanPrateE);
subplot(3,4,[3 4 7 8 11 12])
plot(MeanPrateE,mean(sqrt(mmp-MeanPrateE.^2)));
xlabel("1st-order moment");
ylabel("2nd-order cumulant");