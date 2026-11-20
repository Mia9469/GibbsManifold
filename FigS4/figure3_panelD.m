%% figure3_panelD.m — Gaussian channel ICE vs CV (for Fig. 4G overlay)
%
%  Signal:  r ~ N(mu, sigma^2),  sigma fixed per curve
%  Channel: y = r + n,  n ~ N(0, sigma_n^2)
%  MI  = 0.5 * log2(1 + sigma^2 / sigma_n^2)   [bits]   — constant per sigma
%  Cost proxy = 2.4 + 0.37 * mu                 [calibrated proxy units]
%  CV  = sigma / mu  =>  mu = sigma / CV
%  ICE = MI / calibrated cost proxy
%
%  Sweep sigma = {1, 5, 10, 50, 100}, sigma_n = 1

clear; clc; close all;

%% Parameters
sigma_n   = 1.0;
sigma_set = [1, 5, 10, 50, 100];
R0        = 5;                              % observation rate [Hz]
CV_vec    = 0.05:0.01:8;

nS  = length(sigma_set);
nCV = length(CV_vec);

%% Compute ICE curves
MI_vals    = zeros(nS, 1);                  % bits per channel use
MI_rate    = zeros(nS, 1);                  % bits/s
ICE_curves = zeros(nS, nCV);

for si = 1:nS
    sig = sigma_set(si);
    MI_vals(si) = 0.5 * log2(1 + sig^2 / sigma_n^2);
    MI_rate(si) = MI_vals(si) * R0;         % bits/s
    mu_vec = sig ./ CV_vec;
    C_vec  = 2.4 + 0.37 * mu_vec;
    ICE_curves(si, :) = MI_rate(si) ./ C_vec;
end

%% Print MI values
fprintf('\n  sigma   MI [bits]   MI_rate [bits/s]   ICE_max [proxy^-1 bits]\n');
fprintf('  -----   ---------   ---------------   ----------------\n');
for si = 1:nS
    fprintf('  %4d    %.4f      %.4f            %.4f\n', ...
        sigma_set(si), MI_vals(si), MI_rate(si), MI_rate(si) / 2.4);
end
fprintf('\n');

%% Plot
figure;
set(gcf, 'Color', 'w', 'Position', [200 200 600 450]);

cmap = [0.75 0.75 0.75;     % gray    sigma=1
        0.00 0.50 0.90;     % blue    sigma=5
        0.00 0.70 0.70;     % teal    sigma=10
        0.85 0.33 0.10;     % orange  sigma=50
        0.64 0.08 0.18];    % crimson sigma=100

hold on;
hleg = zeros(nS, 1);
for si = 1:nS
    hleg(si) = plot(CV_vec, ICE_curves(si,:), '-', ...
        'Color', cmap(si,:), 'LineWidth', 2);
end

xlabel('CV', 'FontSize', 13);
ylabel('MI / calibrated cost proxy', 'FontSize', 13);
title('Gaussian channel: ICE vs CV', 'FontSize', 14, 'FontWeight', 'bold');
set(gca, 'FontSize', 11);
xlim([0 8]);

legend(hleg, arrayfun(@(s) sprintf('\\sigma = %d', s), sigma_set, ...
    'UniformOutput', false), 'Location', 'northeast', 'FontSize', 10);

box off;
