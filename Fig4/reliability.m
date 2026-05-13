%% EInet ANALYSIS
clear;%close all;

noise = [50 500];
n_trials = 50;
batch = [10 30];
T=29;
corec = cell(T,length(noise),n_trials);
pattern_rep = zeros(T,length(noise),n_trials);
ssim_sp = cell(T,length(noise),batch(1));
ssim_corr = cell(T,length(noise),batch(1));
corr_sp = cell(T,length(noise),batch(1));
for i = 1:batch(1)
for kk = 1:T
for no = 1:length(noise)
    file  = dir(['recorded_spikes_' num2str(kk) '_' num2str(noise(no)) '_rec.mat']);
    load(file.name)
    trials = randperm(n_trials,batch(2));
    sz = length(trials);
    ssim_spike=zeros(sz,sz);
    corr2_spike=zeros(sz,sz);
    pattern_rep=zeros(sz,sz);
    for tr = 1:sz
         for trr = tr+1:sz
         ssim_spike(tr,trr) = ssim(rec_raster{trials(tr)},rec_raster{trials(trr)});
         corr2_spike(tr,trr) = corr2(rec_raster{trials(tr)},rec_raster{trials(trr)});
         end
        cc = corr(rec_raster{trials(tr)}');
        cc(isnan(cc)) =0;
        selfcorr(kk,no,tr) = mean(cc(:));
        corec{kk,no,tr} = cc;
    end
    for tr = 1:sz
        for trr = tr+1:sz
         pattern_rep(tr,trr) = ssim(corec{kk,no,tr},corec{kk,no,trr});
        end
    end
    ssim_sp{kk,no,i} = ssim_spike;
    ssim_corr{kk,no,i} = pattern_rep;
    corr_sp{kk,no,i} = corr2_spike;
end
end
end



save reliability_final.mat ssim_corr selfcorr ssim_sp corr_sp

ssim_sp1 = zeros(T,length(noise),batch(1));
ssim_corr1 = zeros(T,length(noise),batch(1));
corr_sp1 = zeros(T,length(noise),batch(1));

for i = 1:batch(1)
for kk = 1:T
for no = 1:length(noise)
    ssim_sp1(kk,no,i) = mean(ssim_sp{kk,no,i},'all');
    ssim_corr1(kk,no,i) = mean(ssim_corr{kk,no,i},'all');
    corr_sp1(kk,no,i) = mean(corr_sp{kk,no,i},'all');
end
end
end

save reliability_array_final.mat ssim_corr1 selfcorr ssim_sp1 corr_sp1


figure;

% Example data 
model_series =mean(ssim_sp1([1 29],:,:),3)'; 
model_error =std(ssim_sp1([1 29],:,:),0,3)'; 
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
xticklabels([50 500])

ylabel("Reliability (SSIM RAST)")
legend(["before","after",])
xlabel("noise std (pA)")
ylim([0.3 0.5])