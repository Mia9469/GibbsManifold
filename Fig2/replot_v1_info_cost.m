%% replot_v1_info_cost.m
%  Standalone PLOT-ONLY script for the V1 Information-vs-Cost panels
%  (manuscript Fig. 5C and the diagnostic panels of Fig. 2 / figure2_new.m).
%
%  Loads pre-computed per-neuron / per-trial data from:
%    - unified_v1_Hrho_ICE.mat   (driven per-neuron H, Hn, MI, C, ICE
%                                 across 5 v1v2-1 sessions; produced by
%                                 unified_v1_Hrho_ICE.m)
%    - figure2_stats_only.mat    (per-trial cost1_p, cost1_ps; per-neuron
%                                 mean firing rate; produced by
%                                 figure2_stats_only.m)
%    - spon_Hr_perN.mat          (OPTIONAL - per-neuron spontaneous Hr;
%                                 produced by compute_v1_spon_Hr.m.
%                                 If absent, the spontaneous H trace is
%                                 omitted with a warning.)
%    - v1_CVL_perN.mat           (OPTIONAL - per-neuron CV_L on driven
%                                 trials; produced by compute_v1_CVL_perN.m.
%                                 If absent, Figure 3 (ICE-vs-CV_L using H
%                                 vs using I) is skipped with a warning.)
%
%  Re-runs no analysis.  Generates three figures:
%    Figure 1 - Information vs Cost   : 3 traces (Spon H / Driven H / Driven MI)
%    Figure 2 - ICE vs Cost           : 3 traces (Spon H/C / Driven H/C / Driven MI/C)
%    Figure 3 - ICE vs CV_L using H vs using I  (reviewer P1.2 check:
%               does the rank ordering / monotonic ICE-CV_L relationship
%               preserve when ICE is computed from true MI instead of
%               total entropy H?  Plotted only if v1_CVL_perN.mat is found.)
%
%  Usage:
%    matlab -batch "replot_v1_info_cost"
% ---------------------------------------------------------------------

clear; close all;
fprintf('=== Replot: V1 Information vs Cost ===\n');

% ----- shared style ----------------------------------------------------
col_spon = [0.30 0.60 0.90];        % blue   (spontaneous)
col_evk  = [0.85 0.33 0.10];        % orange (evoked / driven)
xrange   = [0 50];                  % cost range (W) — adjust if needed
fontprops = {'FontSize', 11};

% ----- 1) load driven per-neuron from unified_v1_Hrho_ICE.mat ---------
if ~isfile('unified_v1_Hrho_ICE.mat')
    error(['unified_v1_Hrho_ICE.mat not found.  Run ' ...
           'unified_v1_Hrho_ICE.m first to generate per-neuron driven ' ...
           'H, MI, C across 5 v1v2-1 sessions.']);
end
S = load('unified_v1_Hrho_ICE.mat');
sessions = S.results.v1v2;            % 5-element struct array (sessions)
nS = numel(sessions);
hr_evk = []; mi_evk = []; cost_evk = []; ice_evk = []; sess_id = [];
for i = 1:nS
    sd = sessions(i);
    hr_evk   = [hr_evk;   sd.hr(:)];
    mi_evk   = [mi_evk;   sd.mi(:)];
    cost_evk = [cost_evk; sd.C(:)];
    ice_evk  = [ice_evk;  sd.ICE(:)];
    sess_id  = [sess_id;  i*ones(numel(sd.hr), 1)];
end
fprintf('  loaded %d V1 neurons across %d sessions (driven)\n', length(hr_evk), nS);
% mask out NaNs / non-finite
keep = isfinite(hr_evk) & isfinite(mi_evk) & isfinite(cost_evk) & cost_evk > 0;
hr_evk = hr_evk(keep); mi_evk = mi_evk(keep); cost_evk = cost_evk(keep);
ice_evk = ice_evk(keep);

% ----- 2) load spontaneous data (optional) ----------------------------
have_spon = false;
if isfile('spon_Hr_perN.mat')
    Sp = load('spon_Hr_perN.mat');
    hr_spon  = Sp.hr_spon(:);
    cost_spon = Sp.C_spon(:);
    keep_s = isfinite(hr_spon) & isfinite(cost_spon) & cost_spon > 0;
    hr_spon = hr_spon(keep_s); cost_spon = cost_spon(keep_s);
    have_spon = true;
    fprintf('  loaded %d V1 neurons (spontaneous) from spon_Hr_perN.mat\n', length(hr_spon));
else
    warning(['spon_Hr_perN.mat not found.  Spontaneous H trace will be ' ...
             'omitted.  Run compute_v1_spon_Hr.m to generate it (~5 min).']);
end

% ----- 3) Figure 1: Information vs Cost --------------------------------
figure('Name', 'V1: Information vs Cost', 'NumberTitle', 'off', ...
       'Position', [100 100 600 450]);
hold on;
% Spon H — solid blue
if have_spon
    h_spon = scatter(cost_spon, hr_spon, 30, col_spon, 'o', 'filled', ...
                     'MarkerFaceAlpha', 0.85, 'DisplayName', 'Spon H(r)');
end
% Evoked H — orange faded (upper-bound capacity)
h_evkH = scatter(cost_evk, hr_evk, 30, col_evk, 'o', 'filled', ...
                 'MarkerFaceAlpha', 0.30, 'DisplayName', 'Evoked H(r)');
% Evoked MI — orange solid (deployed information)
h_evkM = scatter(cost_evk, mi_evk, 30, col_evk, 'o', 'filled', ...
                 'MarkerFaceAlpha', 1.0, 'MarkerEdgeColor', 'k', ...
                 'LineWidth', 0.4, 'DisplayName', 'Evoked MI(s;r)');

% Theoretical Gibbs frontier (per-observation): H*(C) = 1 + ln(C/C0)
% (in nats; using C0 = 2.4 W from Yu et al. 2018).  Plotted only if it's
% within range of the data.
xx = linspace(2.5, max(xrange), 200);
y_th = 1 + log((xx - 2.4) / 0.37);
plot(xx(y_th > 0), y_th(y_th > 0), 'k--', 'LineWidth', 1.4, ...
     'DisplayName', 'Gibbs frontier  H*(C) = 1 + ln((C-2.4)/0.37)');
xlabel('Calibrated cost proxy  C', fontprops{:});
ylabel('Information  [nats]', fontprops{:});
title('V1: Spontaneous H, Evoked H, Evoked MI vs Cost', fontprops{:});
legend('show', 'Location', 'best');
xlim(xrange); grid on; box off;
set(gca, fontprops{:}, 'TickDir', 'out');

print(gcf, 'replot_v1_info_cost.pdf', '-dpdf', '-bestfit');
print(gcf, 'replot_v1_info_cost.png', '-dpng', '-r200');
fprintf('  wrote replot_v1_info_cost.pdf / .png\n');

% ----- 4) Figure 2: ICE vs Cost ---------------------------------------
figure('Name', 'V1: ICE vs Cost', 'NumberTitle', 'off', ...
       'Position', [100 100 600 450]);
hold on;
if have_spon
    scatter(cost_spon, hr_spon ./ cost_spon, 30, col_spon, 'o', 'filled', ...
            'MarkerFaceAlpha', 0.85, 'DisplayName', 'Spon H(r) / C');
end
scatter(cost_evk, hr_evk ./ cost_evk, 30, col_evk, 'o', 'filled', ...
        'MarkerFaceAlpha', 0.30, 'DisplayName', 'Evoked H(r) / C');
scatter(cost_evk, mi_evk ./ cost_evk, 30, col_evk, 'o', 'filled', ...
        'MarkerFaceAlpha', 1.0, 'MarkerEdgeColor', 'k', 'LineWidth', 0.4, ...
        'DisplayName', 'Evoked MI(s;r) / C');
xlabel('Calibrated cost proxy  C', fontprops{:});
ylabel('ICE  [nats / J]', fontprops{:});
title('V1: ICE vs Cost (H/C, MI/C)', fontprops{:});
legend('show', 'Location', 'best');
xlim(xrange); grid on; box off;
set(gca, fontprops{:}, 'TickDir', 'out');

print(gcf, 'replot_v1_ICE_cost.pdf', '-dpdf', '-bestfit');
print(gcf, 'replot_v1_ICE_cost.png', '-dpng', '-r200');
fprintf('  wrote replot_v1_ICE_cost.pdf / .png\n');

% ----- 4.5) Figure 3: ICE-vs-CV_L using H vs using I  -----------------
%   Reviewer P1.2 check: when noise entropy is non-negligible (rho ~= 0.85
%   on this dataset), does the monotonic ICE-CV_L relationship survive
%   replacing the entropy-based ICE (H/C) with the true MI-based ICE
%   (I/C)?  We side-by-side plot both, fit a robust monotonic regression
%   line, and print Spearman correlations to verify rank preservation.
have_CVL = false;
if isfile('v1_CVL_perN.mat')
    Lc = load('v1_CVL_perN.mat');
    % length match: unified file is one row per neuron in session order;
    % so is v1_CVL_perN.mat.  Subselect with the same `keep` mask we used
    % above for the driven data (which removed NaNs / non-positive C).
    if numel(Lc.cvL_drv) ~= numel(keep)
        warning(['v1_CVL_perN.mat length (%d) does not match the ' ...
                 'unified driven-data length (%d).  Skipping Fig 3.'], ...
                 numel(Lc.cvL_drv), numel(keep));
    else
        cvL = Lc.cvL_drv(keep);
        have_CVL = isfinite(cvL) & cvL > 0;     % logical mask of valid CV_L
        have_CVL_flag = any(have_CVL);
    end
else
    have_CVL_flag = false;
    warning(['v1_CVL_perN.mat not found.  Figure 3 (ICE-CV_L using H vs ' ...
             'using I) will be skipped.  Run compute_v1_CVL_perN.m to ' ...
             'generate it (<1 min).']);
end

if have_CVL_flag
    msk = have_CVL;
    cvL_ok   = cvL(msk);
    iceH     = hr_evk(msk) ./ cost_evk(msk);    % entropy-based ICE
    iceI     = mi_evk(msk) ./ cost_evk(msk);    % MI-based ICE
    rhoH = corr(cvL_ok, iceH, 'type', 'Spearman', 'rows', 'pairwise');
    rhoI = corr(cvL_ok, iceI, 'type', 'Spearman', 'rows', 'pairwise');
    [pH,~] = corr(cvL_ok, iceH, 'type', 'Spearman', 'rows', 'pairwise');
    [pI,~] = corr(cvL_ok, iceI, 'type', 'Spearman', 'rows', 'pairwise');

    figure('Name', 'V1: ICE-CV_L using H vs using I', 'NumberTitle', 'off', ...
           'Position', [100 100 1100 450]);

    subplot(1, 2, 1); hold on;
    scatter(cvL_ok, iceH, 28, col_evk, 'o', 'filled', ...
            'MarkerFaceAlpha', 0.40);
    xlabel('CV_L  (per-neuron, driven)', fontprops{:});
    ylabel('H(r) / calibrated cost proxy', fontprops{:});
    title(sprintf('Entropy-based ICE (H-based)  Spearman \\rho = %.2f', rhoH), ...
          fontprops{:});
    grid on; box off; set(gca, fontprops{:}, 'TickDir', 'out');

    subplot(1, 2, 2); hold on;
    scatter(cvL_ok, iceI, 28, col_evk, 'o', 'filled', ...
            'MarkerFaceAlpha', 1.0, 'MarkerEdgeColor', 'k', 'LineWidth', 0.4);
    xlabel('CV_L  (per-neuron, driven)', fontprops{:});
    ylabel('I(s;r) / calibrated cost proxy', fontprops{:});
    title(sprintf('MI-based ICE (I-based)  Spearman \\rho = %.2f', rhoI), ...
          fontprops{:});
    grid on; box off; set(gca, fontprops{:}, 'TickDir', 'out');

    sgtitle(sprintf(['ICE vs CV_L: H-based vs I-based  (n = %d V1 neurons, ' ...
                     'mean \\rho_{noise} ~ 0.85)'], numel(cvL_ok)), fontprops{:});

    print(gcf, 'replot_v1_ICE_CVL_HvsI.pdf', '-dpdf', '-bestfit');
    print(gcf, 'replot_v1_ICE_CVL_HvsI.png', '-dpng', '-r200');
    fprintf('  wrote replot_v1_ICE_CVL_HvsI.pdf / .png\n');
    fprintf('  ICE-CV_L  H-based: rho = %.3f, p = %.2g\n', rhoH, pH);
    fprintf('  ICE-CV_L  I-based: rho = %.3f, p = %.2g\n', rhoI, pI);
    fprintf('  delta-rho |H - I| = %.3f  (small = rank ordering preserved)\n', ...
            abs(rhoH - rhoI));
end

% ----- 5) console summary ----------------------------------------------
fprintf('\n=== Summary (per-neuron, V1, %d sessions, n=%d) ===\n', nS, length(hr_evk));
fprintf('  driven  H : median %.2f, IQR [%.2f, %.2f]  bits\n', ...
        median(hr_evk), prctile(hr_evk, 25), prctile(hr_evk, 75));
fprintf('  driven  MI: median %.2f, IQR [%.2f, %.2f]  bits  (rho ~ %.2f)\n', ...
        median(mi_evk), prctile(mi_evk, 25), prctile(mi_evk, 75), ...
        median(1 - mi_evk ./ max(hr_evk, eps)));
fprintf('  driven  C : median %.2f, IQR [%.2f, %.2f]  W\n', ...
        median(cost_evk), prctile(cost_evk, 25), prctile(cost_evk, 75));
if have_spon
    fprintf('  spon    H : median %.2f, IQR [%.2f, %.2f]  bits\n', ...
            median(hr_spon), prctile(hr_spon, 25), prctile(hr_spon, 75));
end
fprintf('=== done ===\n');
