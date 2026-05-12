%% figure3_HC_ICE.m
%  H-C and ICE-C curves for Poisson / Gaussian / Uniform
%  under the signal-energy cost  C = mu^2 + sigma^2
%
%  Parameterisation (consistent with Supplementary Section 2B):
%    Poisson  — sigma = sqrt(lambda), CV = 1/sqrt(lambda)
%    Gaussian — sigma = sigma_fixed (constant), vary mu = sigma/CV
%    Uniform  — sigma = sigma_fixed (constant), vary mu = sigma/CV
%
%  For Gaussian and Uniform, H is constant in sigma, so:
%    H-C  : horizontal line at H = const, for C in [sigma^2, inf)
%    ICE-C: strictly decreasing  ICE = H/C
%    ICE-CV: strictly increasing ICE = H/sigma^2 * CV^2/(1+CV^2)

clear; clc; close all;

%% ======================================================================
%% Parameters
%% ======================================================================
sigma_fixed = 1;                        % fixed sigma for Gaussian & Uniform
CV_vec      = logspace(-1, 1, 800);    % CV in [0.1, 10]

%% ======================================================================
%% Poisson  — parametric (lambda -> C, H)
%% ======================================================================
lambda_p = 1 ./ CV_vec.^2;            % lambda = 1/CV^2
Cp       = lambda_p.^2 + lambda_p;    % C = lambda^2 + lambda
Hp       = zeros(size(lambda_p));
for i = 1:length(lambda_p)
    lam   = lambda_p(i);
    k_max = max(30, ceil(lam + 8*sqrt(lam)));
    k     = 0:k_max;
    pk    = poisspdf(k, lam);
    pk    = pk(pk > 1e-15);  pk = pk / sum(pk);
    Hp(i) = -sum(pk .* log2(pk));
end
ICEp = Hp ./ Cp;

% Inversion: lambda(C) = (-1 + sqrt(1+4C)) / 2  (used for H-C plot)
C_p_range  = linspace(min(Cp), max(Cp), 800);
lam_inv    = (-1 + sqrt(1 + 4*C_p_range)) / 2;
H_p_of_C   = zeros(size(lam_inv));
for i = 1:length(lam_inv)
    lam   = lam_inv(i);
    k_max = max(30, ceil(lam + 8*sqrt(lam)));
    k     = 0:k_max;
    pk    = poisspdf(k, lam);
    pk    = pk(pk > 1e-15);  pk = pk / sum(pk);
    H_p_of_C(i) = -sum(pk .* log2(pk));
end
ICE_p_of_C = H_p_of_C ./ C_p_range;

%% ======================================================================
%% Gaussian  — closed form (sigma = const)
%% ======================================================================
H_g   = 0.5 * log2(2*pi*exp(1) * sigma_fixed^2);   % constant [bits]
mu_g  = sigma_fixed ./ CV_vec;
Cg    = mu_g.^2 + sigma_fixed^2;                    % C = sigma^2*(1 + 1/CV^2)
ICEg  = H_g ./ Cg;                                  % = H/sigma^2 * CV^2/(1+CV^2)

% As explicit function of C  (C >= sigma^2)
C_g_range  = linspace(sigma_fixed^2 * 1.001, max(Cg), 800);
ICE_g_of_C = H_g ./ C_g_range;                      % strictly decreasing

%% ======================================================================
%% Uniform  — closed form (sigma = const)
%% ======================================================================
H_u   = log2(2*sqrt(3) * sigma_fixed);              % constant [bits]
mu_u  = sigma_fixed ./ CV_vec;
Cu    = mu_u.^2 + sigma_fixed^2;                    % same C(CV) as Gaussian
ICEu  = H_u ./ Cu;

C_u_range  = C_g_range;
ICE_u_of_C = H_u ./ C_u_range;

%% ======================================================================
%% Print analytical expressions
%% ======================================================================
fprintf('\n--- Analytical expressions (sigma_fixed = %.4g) ---\n', sigma_fixed);
fprintf('Gaussian:  H  = %.4f bits  (const)\n', H_g);
fprintf('           C  = sigma^2 * (1 + 1/CV^2) = %.4g * (1 + 1/CV^2)\n', sigma_fixed^2);
fprintf('           ICE(CV) = H/sigma^2 * CV^2/(1+CV^2) = %.4f * CV^2/(1+CV^2)\n', H_g/sigma_fixed^2);
fprintf('           ICE(C)  = H/C  (strictly decreasing)\n\n');

fprintf('Uniform :  H  = %.4f bits  (const)\n', H_u);
fprintf('           C  = sigma^2 * (1 + 1/CV^2) = %.4g * (1 + 1/CV^2)\n', sigma_fixed^2);
fprintf('           ICE(CV) = H/sigma^2 * CV^2/(1+CV^2) = %.4f * CV^2/(1+CV^2)\n', H_u/sigma_fixed^2);
fprintf('           ICE(C)  = H/C  (strictly decreasing)\n\n');

fprintf('Poisson :  H(C) via lambda(C) = (-1+sqrt(1+4C))/2  [numerical]\n');
fprintf('           C = lambda^2 + lambda\n');
fprintf('           ICE(C) = H(C)/C\n\n');

%% ======================================================================
%% Figure 1: H-C plane
%% ======================================================================
col_p = [0.15 0.15 0.15];
col_g = [0.00 0.70 0.70];
col_u = [0.55 0.30 0.80];

figure('Color','w','Position',[60 120 1300 460]);

subplot(1,3,1); hold on;
plot(CV_vec, Hp, '-',  'Color', col_p, 'LineWidth', 2.0);
yline(H_g, '-',  'Color', col_g, 'LineWidth', 2.0, ...
    'Label', sprintf('H_G = %.2f bits', H_g), 'LabelHorizontalAlignment','left');
yline(H_u, '--', 'Color', col_u, 'LineWidth', 2.0, ...
    'Label', sprintf('H_U = %.2f bits', H_u), 'LabelHorizontalAlignment','left');
xlabel('CV', 'FontSize', 12);
ylabel('H  (bits)',  'FontSize', 12);
title('H–CV plane', 'FontSize', 13, 'FontWeight', 'bold');
legend({'Poisson','Gaussian','Uniform'}, 'Location','southeast','FontSize',10);
set(gca,'FontSize',11); box off;
xlim([0, 10]);

%% ======================================================================
%% Figure 2: ICE-C plane
%% ======================================================================
subplot(1,3,2); 
loglog(CV_vec,  Cp, '-',  'Color', col_p, 'LineWidth', 2.0);hold on;
loglog(CV_vec,  Cg, '-',  'Color', col_g, 'LineWidth', 2.0);hold on;
loglog(CV_vec,  Cu, '--', 'Color', col_u, 'LineWidth', 2.0);hold on;
ylabel('C = E[r^2]', 'FontSize', 12);
xlabel('CV', 'FontSize', 12);
title('C-CV plane', 'FontSize', 13, 'FontWeight', 'bold');
legend({'Poisson','Gaussian','Uniform'}, 'Location','northeast','FontSize',10);
% set(gca,'FontSize',11); box off;

%% ======================================================================
%% Figure 3: ICE-CV  (reference: should be monotonically increasing)
%% ======================================================================
subplot(1,3,3); hold on;
plot(CV_vec, ICEp, '-',  'Color', col_p, 'LineWidth', 2.0);
plot(CV_vec, ICEg, '-',  'Color', col_g, 'LineWidth', 2.0);
plot(CV_vec, ICEu, '--', 'Color', col_u, 'LineWidth', 2.0);
xlabel('CV', 'FontSize', 12);
ylabel('ICE = H/C  (bits)', 'FontSize', 12);
title('ICE–CV  (all monotone \uparrow)', 'FontSize', 13, 'FontWeight', 'bold');
legend({'Poisson','Gaussian','Uniform'}, 'Location','southeast','FontSize',10);
set(gca,'FontSize',11); box off;
xlim([0.1, 10]);

sgtitle(sprintf('H–C / ICE–C / ICE–CV  |  C = E[r^2], \\sigma_{fixed} = %g', sigma_fixed), ...
    'FontSize', 13, 'FontWeight', 'bold');
