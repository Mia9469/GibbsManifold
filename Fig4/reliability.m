%% reliability.m  —  Trial-to-trial spike-raster reliability for Fig. 4K
%
% Computes pairwise trial-to-trial similarity of spike rasters for the
% recurrent E–I network, before vs. after learning, at each tested
% noise level. Three similarity metrics are reported. Pearson correlation
% and SSIM can be negative; they are not probabilities:
%
%   (1)  ssim_sp(kk, no, batch_i)   — SSIM between full spike rasters
%                                      (image-similarity on the binary
%                                       neuron×time matrix)
%   (2)  corr_sp(kk, no, batch_i)   — corr2 between spike rasters
%                                      (Pearson correlation of the
%                                       flattened raster)
%   (3)  ssim_corr(kk, no, batch_i) — SSIM between per-trial
%                                      neuron-pair correlation matrices
%                                      (assembly-level reproducibility)
%
% The first metric is what Fig. 4K reports as "Reliability (SSIM RAST)".
%
% Inputs (must be on the MATLAB path or in cwd):
%   recorded_spikes_<kk>_<noise>_rec.mat   produced by figure4_main.m /
%                                          train_network.m, containing
%                                          rec_raster{trial}: neuron × time
%
% Outputs:
%   reliability_final.mat         raw similarity cells per (kk, noise, batch)
%   reliability_array_final.mat   batch-averaged scalars (T × noise × batch)
%
% Loop structure:
%   T learning stages × length(noise) noise levels × batch(1) outer batches
%   In each inner cell, batch(2) trials are randomly drawn from n_trials
%   and all pairwise (sz × sz) similarities are computed.
%
% Notes:
%   - Only distinct unordered trial pairs are averaged. Zero placeholders
%     in the diagonal/lower triangle are excluded, but valid zero-valued
%     upper-triangle similarities are retained. For sz=30, the historical
%     whole-matrix mean was 29/60 of the correct 435-pair mean.
%   - Binary raster SSIM uses DynamicRange=1 for every pair and state.
%   - The ten batches overlap in trials from one network trajectory. Their
%     SD is descriptive resampling spread, not independent-run uncertainty.
%     Batch IDs do not pair trial draws between stages or noise levels.
%   - The two noise levels [50 500] (pA) bracket the empirical noise range
%     used in Fig. 4 panel C–E.
%
% Reproduction:
%   matlab -batch "reliability"
% then re-run fig3plot.m for the bar panel.

clear; %close all;

% ----- analysis parameters ---------------------------------------------
noise    = [50 500];      % noise stds (pA) recorded as suffixes in filenames
n_trials = 50;            % total trials saved per (stage, noise) recording
batch    = [10 30];       % batch(1) outer reps; batch(2) trials drawn per rep
T        = 29;            % number of learning stages saved by training

% pre-allocate
corec     = cell(T, length(noise), n_trials);  %#ok<NASGU>  (kept for backwards compat)
selfcorr  = zeros(T, length(noise), batch(2));
ssim_sp   = cell(T, length(noise), batch(1));
ssim_corr = cell(T, length(noise), batch(1));
corr_sp   = cell(T, length(noise), batch(1));
trial_indices = cell(T, length(noise), batch(1));
metric_protocol = struct('raster_ssim_dynamic_range', 1, ...
    'pair_aggregation', 'strict upper triangle, excluding diagonal', ...
    'replication', 'overlapping trial-resampling batches; descriptive only');

% ----- pairwise similarity loop -----------------------------------------
for i = 1:batch(1)
    for kk = 1:T
        for no = 1:length(noise)
            file = dir(['recorded_spikes_' num2str(kk) '_' num2str(noise(no)) '_rec.mat']);
            load(file.name)                                                 %#ok<LOAD>
            trials = randperm(n_trials, batch(2));
            sz     = length(trials);
            trial_indices{kk, no, i} = trials;
            for tr = 1:sz
                raster = rec_raster{trials(tr)};
                assert(all(isfinite(raster(:))) && all(raster(:) == 0 | raster(:) == 1), ...
                    'Reliability SSIM requires finite binary rasters');
            end

            ssim_spike  = zeros(sz, sz);
            corr2_spike = zeros(sz, sz);
            pair_rep    = zeros(sz, sz);                                   % renamed from duplicate `pattern_rep`
            corec_kkno  = cell(sz, 1);                                     % per-trial correlation matrices

            for tr = 1:sz
                for trr = tr+1:sz
                    ssim_spike(tr, trr)  = ssim(double(rec_raster{trials(tr)}), ...
                        double(rec_raster{trials(trr)}), 'DynamicRange', 1);
                    corr2_spike(tr, trr) = corr2(rec_raster{trials(tr)}, rec_raster{trials(trr)});
                end
                cc = corr(rec_raster{trials(tr)}');
                cc(isnan(cc)) = 0;
                selfcorr(kk, no, tr) = mean(cc(:));
                corec_kkno{tr}       = cc;
            end
            for tr = 1:sz
                for trr = tr+1:sz
                    pair_rep(tr, trr) = ssim(corec_kkno{tr}, corec_kkno{trr});
                end
            end

            ssim_sp{kk, no, i}   = ssim_spike;
            ssim_corr{kk, no, i} = pair_rep;
            corr_sp{kk, no, i}   = corr2_spike;
        end
    end
end

save reliability_final.mat ssim_corr selfcorr ssim_sp corr_sp trial_indices metric_protocol noise batch

% ----- collapse to (T × noise × batch) scalar arrays --------------------
ssim_sp1   = zeros(T, length(noise), batch(1));
ssim_corr1 = zeros(T, length(noise), batch(1));
corr_sp1   = zeros(T, length(noise), batch(1));

for i = 1:batch(1)
    for kk = 1:T
        for no = 1:length(noise)
            ssim_sp1(kk, no, i)   = mean_unordered_pairs(ssim_sp{kk, no, i});
            ssim_corr1(kk, no, i) = mean_unordered_pairs(ssim_corr{kk, no, i});
            corr_sp1(kk, no, i)   = mean_unordered_pairs(corr_sp{kk, no, i});
        end
    end
end

save reliability_array_final.mat ssim_corr1 selfcorr ssim_sp1 corr_sp1 trial_indices metric_protocol noise batch

% ----- quick before-vs-after bar plot at each noise level ---------------
% (full bar panel for Fig. 4 is in fig3plot.m; this is a sanity check.)
figure;
stages_plot = [1 29];   % first/final stored checkpoints, not independent runs
mu          = mean(ssim_sp1(stages_plot, :, :), 3)';                        % [noise × stage]
sd          = std (ssim_sp1(stages_plot, :, :), 0, 3)';
b = bar(mu, 'grouped'); hold on
[ngroups, nbars] = size(mu);
xx = nan(nbars, ngroups);
for k = 1:nbars
    xx(k, :) = b(k).XEndPoints;
end
errorbar(xx', mu, sd, 'k', 'linestyle', 'none');
hold off
xticklabels(string(noise))
xlabel('noise std (pA)')
ylabel('Raster similarity (SSIM)')
legend(["first stored checkpoint", "final stored checkpoint"], 'Location', 'best')
ylim([0 1])
title('Raster SSIM: descriptive resampling mean and SD')
