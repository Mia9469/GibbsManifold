%% figure_HC_phase.m — H-C phase diagram on the Gibbs manifold
%
%  Gibbs-optimal frontier under calibrated cost proxy c(r) = 2.4 + 0.37*r.
%  All curves are expressed directly in terms of C (no explicit firing rate).
%  Consistent with the formula in the main text (Section 2.5 / result5):
%
%    H(C)   = R0/ln2 * [1 + ln((C-2.4)/0.37)]   [bits/s]
%    beta(C) = dH/dC = R0 / [(ln2)(C-2.4)]
%    ICE(C)  = H(C) / C
%
%  Free parameter: R0 (observation rate [Hz])

clear; clc; close all;

%% ======================================================================
%% Parameter
%% ======================================================================
R0    = 3;                          % observation rate [Hz]

%% ======================================================================
%% Cost-proxy axis  (C > 2.4, baseline excluded)
%% ======================================================================
C_vec = 2.401 : 0.01 : 50;         % calibrated cost proxy

%% ======================================================================
%% Gibbs-optimal frontier (no firing-rate variable needed)
%% ======================================================================
Hdot = (R0 / log(2)) .* (1 + log((C_vec - 2.4) / 0.37));   % bits/s
beta = R0 ./ (log(2) .* (C_vec - 2.4));
ICE  = Hdot ./ C_vec;

%% ======================================================================
%% Reference points (C values only; r = (C-2.4)/0.37 not needed)
%% ======================================================================
C_marks    = 2.4 + 0.37 * [1, 5, 10, 20, 50];   % selected proxy-cost levels
H_marks    = (R0 / log(2)) .* (1 + log((C_marks - 2.4) / 0.37));
beta_marks = R0 ./ (log(2) .* (C_marks - 2.4));
ICE_marks  = H_marks ./ C_marks;

%% ======================================================================
%% Figure
%% ======================================================================
figure;
set(gcf, 'Color', 'w', 'Position', [100 100 1000 800]);

% ---- Panel A: H vs C ----
subplot(2, 2, 1); hold on;
plot(C_vec, Hdot, 'k-', 'LineWidth', 2);
% Tangent lines (slope = beta) at selected C
for i = [1, 3, 5]
    C0 = C_marks(i);  H0 = H_marks(i);  s = beta_marks(i);
    C_tan = C0 + [-2, 2];
    H_tan = H0 + s * (C_tan - C0);
    plot(C_tan, H_tan, '--', 'Color', [0.6 0.6 0.6], 'LineWidth', 1);
end
scatter(C_marks, H_marks, 60, 'r', 'filled');
xlabel('calibrated cost proxy C', 'FontSize', 13);
ylabel('H  [bits/s]',  'FontSize', 13);
title('A.  H vs C  (Gibbs frontier)', 'FontSize', 13, 'FontWeight', 'bold');
set(gca, 'FontSize', 11);

% ---- Panel B: beta vs C ----
subplot(2, 2, 2); hold on;
plot(C_vec, beta, 'b-', 'LineWidth', 2);
scatter(C_marks, beta_marks, 60, 'r', 'filled');
xlabel('calibrated cost proxy C', 'FontSize', 13);
ylabel('\beta = dH/dC', 'FontSize', 13);
title('B.  Marginal efficiency \beta vs C', 'FontSize', 13, 'FontWeight', 'bold');
set(gca, 'FontSize', 11);

% ---- Panel C: ICE vs C ----
subplot(2, 2, 3); hold on;
plot(C_vec, ICE, '-', 'Color', [0.85 0.33 0.1], 'LineWidth', 2);
scatter(C_marks, ICE_marks, 60, 'r', 'filled');
xlabel('calibrated cost proxy C', 'FontSize', 13);
ylabel('ICE = H/C', 'FontSize', 13);
title('C.  Average efficiency ICE vs C', 'FontSize', 13, 'FontWeight', 'bold');
set(gca, 'FontSize', 11);

% ---- Panel D: beta and ICE together ----
subplot(2, 2, 4); hold on;
plot(C_vec, beta, 'b-',  'LineWidth', 2);
plot(C_vec, ICE,  '-',   'Color', [0.85 0.33 0.1], 'LineWidth', 2);
xlabel('calibrated cost proxy C', 'FontSize', 13);
ylabel('information / cost proxy', 'FontSize', 13);
title('D.  \beta vs ICE  (ICE > \beta everywhere)', 'FontSize', 13, 'FontWeight', 'bold');
legend({'\beta  (marginal)', 'ICE  (average)'}, 'Location', 'northeast', 'FontSize', 10);
set(gca, 'FontSize', 11);

sgtitle(sprintf('H–C Phase Diagram  (R_0 = %d Hz)', R0), ...
    'FontSize', 15, 'FontWeight', 'bold');

%% ======================================================================
%% Print key values
%% ======================================================================
fprintf('\n  cost proxy C   H [bits/s]   beta   ICE\n');
fprintf('  ------    ----------   -------------   ------------\n');
for i = 1:length(C_marks)
    fprintf('  %6.2f    %10.2f   %10.4f      %10.4f\n', ...
        C_marks(i), H_marks(i), beta_marks(i), ICE_marks(i));
end
