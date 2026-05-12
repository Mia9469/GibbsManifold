%% figure3.m — Theoretical ICE vs CV
%
%  H    = entropy of firing-rate distribution  [bits]
%  Cost = E[r^2] = mu^2 + sigma^2  (signal energy)
%  ICE  = H / C
%
%  Parameterisation (consistent with Supplementary Section 2B proofs):
%    Poisson  — sigma = sqrt(mu) [inherent], CV = 1/sqrt(mu), C = mu^2 + mu
%    Gaussian — sigma = CONSTANT, CV = sigma/mu,  C = mu^2 + sigma^2, H = const
%    Uniform  — sigma = CONSTANT, CV = sigma/mu,  C = mu^2 + sigma^2, H = const
%
%  For Gaussian and Uniform the supp proof fixes sigma and varies mu.
%  This guarantees H = const and ICE = H/C is strictly increasing with CV.
%
%  Lines = analytical,  Dots = Monte-Carlo validation

clear; clc; close all;

%% ======================================================================
%% Parameters
%% ======================================================================
CV_vec  = logspace(-1, 1, 600);    % CV axis: 0.1 to 10
N_samp  = 100000;
n_bins  = 300;

% Fixed sigma for Gaussian and Uniform (supp proof: sigma = constant)
sigma_fixed = 1;

%% ======================================================================
%% Poisson  (sigma = sqrt(lambda) inherent, CV = 1/sqrt(lambda))
%% ======================================================================
lambda_p = 1 ./ CV_vec.^2;         % lambda = 1/CV^2
CVp      = CV_vec;
Cp       = lambda_p.^2 + lambda_p; % C = mu^2 + sigma^2 = lambda^2 + lambda

% Entropy: no closed form, evaluate numerically
Hp = zeros(size(lambda_p));
for i = 1:length(lambda_p)
    lam   = lambda_p(i);
    k_max = max(30, ceil(lam + 8*sqrt(lam)));
    k     = 0:k_max;
    pk    = poisspdf(k, lam);
    pk    = pk(pk > 1e-15);  pk = pk / sum(pk);
    Hp(i) = -sum(pk .* log2(pk));
end
ICEp = Hp ./ Cp;

%% ======================================================================
%% Gaussian  (sigma = sigma_fixed = constant, vary mu = sigma/CV)
%% ======================================================================
mu_g  = sigma_fixed ./ CV_vec;     % mu = sigma/CV
CVg   = CV_vec;
Cg    = mu_g.^2 + sigma_fixed^2;  % C = mu^2 + sigma^2
Hg    = 0.5 * log2(2*pi*exp(1)*sigma_fixed^2) * ones(size(CV_vec));  % H = const
ICEg  = Hg ./ Cg;                  % monotonically increasing by proof

%% ======================================================================
%% Uniform  (sigma = sigma_fixed = constant, half-width = sqrt(3)*sigma)
%% ======================================================================
mu_u  = sigma_fixed ./ CV_vec;     % mu = sigma/CV
CVu   = CV_vec;
Cu    = mu_u.^2 + sigma_fixed^2;  % C = mu^2 + sigma^2
Hu    = log2(2*sqrt(3)*sigma_fixed) * ones(size(CV_vec));  % H = const
ICEu  = Hu ./ Cu;                  % monotonically increasing by proof

%% ======================================================================
%% Monte-Carlo validation dots
%% ======================================================================
CV_dots = [0.15, 0.2:0.1:1, 1.2:0.2:2, 2.5:0.5:5, 6:2:10];
nD      = length(CV_dots);

CV_mc  = zeros(3, nD);
H_mc   = zeros(3, nD);
C_mc   = zeros(3, nD);

for i = 1:nD
    cv = CV_dots(i);

    % --- Poisson ---
    lam = 1 / cv^2;
    r   = double(poissrnd(lam, 1, N_samp));
    CV_mc(1,i) = std(r) / (mean(r) + 1e-12);
    C_mc(1,i)  = mean(r.^2);
    vals = unique(r);  cnt = histc(r, vals); %#ok
    p = cnt / sum(cnt);  p = p(p > 0);
    H_mc(1,i) = -sum(p .* log2(p));

    % --- Gaussian (fixed sigma) ---
    mu_val = sigma_fixed / cv;
    r = mu_val + sigma_fixed * randn(1, N_samp);
    CV_mc(2,i) = std(r) / (mean(r) + 1e-12);
    C_mc(2,i)  = mean(r.^2);
    H_mc(2,i)  = hist_diff_entropy(r, n_bins);

    % --- Uniform (fixed sigma) ---
    hw = sqrt(3) * sigma_fixed;     % half-width = sqrt(3)*sigma
    r  = (mu_val - hw) + 2*hw * rand(1, N_samp);
    CV_mc(3,i) = std(r) / (mean(r) + 1e-12);
    C_mc(3,i)  = mean(r.^2);
    H_mc(3,i)  = hist_diff_entropy(r, n_bins);
end

ICE_mc = H_mc ./ C_mc;

%% ======================================================================
%% Figure: 1×3
%% ======================================================================
figure;
set(gcf, 'Color', 'w', 'Position', [60 120 1250 380]);

col_p = [0.15 0.15 0.15];     % Poisson  — black
col_g = [0.00 0.70 0.70];     % Gaussian — teal
col_u = [0.55 0.30 0.80];     % Uniform  — purple
cols  = [col_p; col_g; col_u];
names = {'Poisson', 'Gaussian', 'Uniform'};
dot_sz = 20;

% ---- Panel A: C = E[r^2] vs CV ----
subplot(1, 3, 1); hold on;
scatter(CV_mc(1,:), C_mc(1,:), dot_sz, col_p, 'filled', 'MarkerFaceAlpha', 0.5);
scatter(CV_mc(2,:), C_mc(2,:), dot_sz, col_g, 'filled', 'MarkerFaceAlpha', 0.5);
scatter(CV_mc(3,:), C_mc(3,:), dot_sz, col_u, 'filled', 'MarkerFaceAlpha', 0.5);
plot(CVp, Cp, '-',  'Color', col_p, 'LineWidth', 2);
plot(CVg, Cg, '-',  'Color', col_g, 'LineWidth', 2);
plot(CVu, Cu, '--', 'Color', col_u, 'LineWidth', 2);
set(gca, 'XScale', 'log', 'YScale', 'log', 'FontSize', 11);
xlabel('CV', 'FontSize', 12);  ylabel('C = E[r^2]', 'FontSize', 12);
legend(names, 'Location', 'northeast', 'FontSize', 9);
title('A', 'FontSize', 14, 'FontWeight', 'bold');

% ---- Panel B: H vs CV ----
subplot(1, 3, 2); hold on;
scatter(CV_mc(1,:), H_mc(1,:), dot_sz, col_p, 'filled', 'MarkerFaceAlpha', 0.5);
scatter(CV_mc(2,:), H_mc(2,:), dot_sz, col_g, 'filled', 'MarkerFaceAlpha', 0.5);
scatter(CV_mc(3,:), H_mc(3,:), dot_sz, col_u, 'filled', 'MarkerFaceAlpha', 0.5);
plot(CVp, Hp, '-',  'Color', col_p, 'LineWidth', 1.8);
yline(Hg(1),  '-',  'Color', col_g, 'LineWidth', 1.8);   % constant
yline(Hu(1),  '--', 'Color', col_u, 'LineWidth', 1.8);   % constant
xlabel('CV', 'FontSize', 12);  ylabel('H  (bits)', 'FontSize', 12);
set(gca, 'FontSize', 11);
title('B', 'FontSize', 14, 'FontWeight', 'bold');

% ---- Panel C: ICE = H / E[r^2] vs CV ----
subplot(1, 3, 3); hold on;
scatter(CV_mc(1,:), ICE_mc(1,:), dot_sz, col_p, 'filled', 'MarkerFaceAlpha', 0.5);
scatter(CV_mc(2,:), ICE_mc(2,:), dot_sz, col_g, 'filled', 'MarkerFaceAlpha', 0.5);
scatter(CV_mc(3,:), ICE_mc(3,:), dot_sz, col_u, 'filled', 'MarkerFaceAlpha', 0.5);
plot(CVp, ICEp, '-',  'Color', col_p, 'LineWidth', 1.8);
plot(CVg, ICEg, '-',  'Color', col_g, 'LineWidth', 1.8);
plot(CVu, ICEu, '--', 'Color', col_u, 'LineWidth', 1.8);
xlabel('CV', 'FontSize', 12);
ylabel('ICE = H / E[r^2]  (bits)', 'FontSize', 12);
set(gca, 'FontSize', 11);
legend(names, 'Location', 'southeast', 'FontSize', 9);
title('C', 'FontSize', 14, 'FontWeight', 'bold');

sgtitle(sprintf('Fig. 3: Theoretical ICE vs CV  (\\sigma_{fixed} = %g for Gaussian/Uniform)', ...
    sigma_fixed), 'FontSize', 13, 'FontWeight', 'bold');

%% ======================================================================
function H = hist_diff_entropy(samples, n_bins)
    [counts, edges] = histcounts(samples, n_bins);
    delta = edges(2) - edges(1);
    p = counts / sum(counts);
    p = p(p > 0);
    H = -sum(p .* log2(p)) + log2(delta);
end
