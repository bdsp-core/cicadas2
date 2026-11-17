% MAKE_PAPER_FIGURES
% Generates all 12 figures in CICADAS_II.pdf from cached simulation results.
%
% Inputs:  results/*.mat (cached outputs of the s0-s16 scripts)
% Outputs: Figures/paper_figNN_*.pdf
%
% Iteration is fast (~seconds) because no simulation is re-run.
% To regenerate the underlying cache, run the corresponding scripts/sX_*.m.

function make_paper_figures(varargin)

opts.figs = 1:12;          % which paper figures to (re)generate
opts.outdir = 'Figures';
opts.results_dir = 'results';
for k = 1:2:numel(varargin)
    opts.(varargin{k}) = varargin{k+1};
end

if ~exist(opts.outdir, 'dir'); mkdir(opts.outdir); end
addpath('functions');

dispatch = {
    1,  @() fig01_demographics(opts);
    2,  @() fig02_param_recovery(opts);
    3,  @() fig03_derived_recovery(opts);
    4,  @() fig04_confounding(opts);
    5,  @() fig05_te_by_severity(opts);
    6,  @() fig06_quadrant(opts);
    7,  @() fig07_threshold_sweep(opts);
    8,  @() fig08_personalized(opts);
    9,  @() fig09_uniform_vs_dynamic(opts);
    10, @() fig10_closed_loop(opts);
    11, @() fig11_ekf_sensitivity(opts);
    12, @() fig12_graded_misspec(opts);
};

for r = 1:size(dispatch, 1)
    fig_num = dispatch{r,1};
    if ~ismember(fig_num, opts.figs); continue; end
    fprintf('=== Paper figure %d ===\n', fig_num);
    try
        dispatch{r,2}();
    catch ME
        fprintf('  FAILED: %s\n', ME.message);
        fprintf('  %s\n', getReport(ME, 'extended'));
    end
end

close all;
fprintf('\nAll done. PDFs in %s/\n', opts.outdir);

end

% =========================================================================
% Figure 1: Synthetic patient demographics (Age + SOFA)
% =========================================================================
function fig01_demographics(opts)
    % s0 does not cache; regenerate inline (fast).
    N = 10000;
    dt = 5/60/24;
    T = 0:dt:4;
    rng(42);
    ages = zeros(N, 1);
    sofas = zeros(N, 1);
    pp.peak_rate = 10;
    for i = 1:N
        p = fcn_generate_patient(T, pp);
        ages(i) = p.age;
        sofas(i) = p.sofa;
    end

    fig = figure('Position', [100 100 1000 400], 'Color', 'w');
    subplot(1,2,1);
    [f, xi] = ksdensity(ages, 'Support', [18 100], 'BoundaryCorrection', 'reflection');
    plot(xi, f*100, 'b-', 'LineWidth', 2);
    xlabel('Age (years)'); ylabel('Percent of Patients (%)');
    title(sprintf('Age Distribution (N = %d)', N));
    xlim([18 100]); grid on;
    text(0.95, 0.95, sprintf('Mean: %.1f\nMedian: %.1f\nSD: %.1f', mean(ages), median(ages), std(ages)), ...
        'Units','normalized', 'HorizontalAlignment','right', 'VerticalAlignment','top', ...
        'FontSize', 10, 'BackgroundColor', 'w');

    subplot(1,2,2);
    sofa_vals = 0:24;
    counts = histcounts(sofas, -0.5:24.5);
    pct = counts / N * 100;
    plot(sofa_vals, pct, 'r-', 'LineWidth', 2); hold on;
    plot(sofa_vals, pct, 'ro', 'MarkerSize', 4, 'MarkerFaceColor', 'r');
    xlabel('SOFA Score'); ylabel('Percent of Patients (%)');
    title(sprintf('SOFA Distribution (N = %d)', N));
    xlim([-0.5 24.5]); grid on;
    text(0.95, 0.95, sprintf('Mean: %.1f\nMedian: %.1f\nSD: %.1f', mean(sofas), median(sofas), std(sofas)), ...
        'Units','normalized', 'HorizontalAlignment','right', 'VerticalAlignment','top', ...
        'FontSize', 10, 'BackgroundColor', 'w');

    sgtitle('Synthetic Patient Demographics');
    savepdf(fig, fullfile(opts.outdir, 'paper_fig01_demographics.pdf'));
end

% =========================================================================
% Figure 2: Individual-level parameter recovery (Ap, r, d × untreated, treated)
% =========================================================================
function fig02_param_recovery(opts)
    S = load(fullfile(opts.results_dir, 'parameter_estimation_results.mat'));
    idx_ut = ~S.is_treated; idx_tx = S.is_treated;
    cAp_u = corr(S.true_Ap(idx_ut), S.est_Ap(idx_ut));
    cAp_t = corr(S.true_Ap(idx_tx), S.est_Ap(idx_tx));
    cr_u  = corr(S.true_r(idx_ut),  S.est_r(idx_ut));
    cr_t  = corr(S.true_r(idx_tx),  S.est_r(idx_tx));
    cd_u  = corr(S.true_d(idx_ut),  S.est_d(idx_ut));
    cd_t  = corr(S.true_d(idx_tx),  S.est_d(idx_tx));

    lim_Ap = [0, max([S.true_Ap; S.est_Ap]) * 1.05];
    lim_r = [0, min(max([S.true_r; S.est_r]) * 1.05, 5)];
    lim_d = [0, min(max([S.true_d; S.est_d]) * 1.05, 2)];

    fig = figure('Position', [50 50 1200 800], 'Color', 'w');
    rows = {idx_ut, idx_tx};
    row_colors = {'b', 'r'};
    row_labels = {'untreated', 'treated'};
    corrs = {[cAp_u cr_u cd_u], [cAp_t cr_t cd_t]};

    for row = 1:2
        idx = rows{row}; col = row_colors{row}; lab = row_labels{row}; cs = corrs{row};
        subplot(2,3,(row-1)*3+1)
            scatter(S.true_Ap(idx), S.est_Ap(idx), 30, col, 'filled', 'MarkerFaceAlpha', 0.5);
            hold on; plot(lim_Ap, lim_Ap, 'k--', 'LineWidth', 1);
            xlabel('True A_p'); ylabel('Estimated A_p');
            title(sprintf('A_p (%s) r=%.2f', lab, cs(1)));
            grid on; xlim(lim_Ap); ylim(lim_Ap);
        subplot(2,3,(row-1)*3+2)
            scatter(S.true_r(idx), S.est_r(idx), 30, col, 'filled', 'MarkerFaceAlpha', 0.5);
            hold on; plot(lim_r, lim_r, 'k--', 'LineWidth', 1);
            xlabel('True r'); ylabel('Estimated r');
            title(sprintf('r (%s) r=%.2f', lab, cs(2)));
            grid on; xlim(lim_r); ylim(lim_r);
        subplot(2,3,(row-1)*3+3)
            scatter(S.true_d(idx), S.est_d(idx), 30, col, 'filled', 'MarkerFaceAlpha', 0.5);
            hold on; plot(lim_d, lim_d, 'k--', 'LineWidth', 1);
            xlabel('True d'); ylabel('Estimated d');
            title(sprintf('d (%s) r=%.2f', lab, cs(3)));
            grid on; xlim(lim_d); ylim(lim_d);
    end
    sgtitle('Parameter Recovery: True vs Estimated');
    savepdf(fig, fullfile(opts.outdir, 'paper_fig02_param_recovery.pdf'));
end

% =========================================================================
% Figure 3: Derived quantity recovery (Lcum, peak height, peak timing)
% =========================================================================
function fig03_derived_recovery(opts)
    S = load(fullfile(opts.results_dir, 'parameter_estimation_results.mat'));
    idx_ut = ~S.is_treated; idx_tx = S.is_treated;
    cLc = corr(S.true_Lcum, S.est_Lcum);
    cPh = corr(S.true_peak_height, S.est_peak_height);
    cPt = corr(S.true_peak_time, S.est_peak_time);
    maePt = mean(abs(S.true_peak_time - S.est_peak_time));

    fig = figure('Position', [50 50 1200 400], 'Color', 'w');
    lim_lc = [0, max([S.true_Lcum; S.est_Lcum]) * 1.05];
    lim_ph = [0, max([S.true_peak_height; S.est_peak_height]) * 1.05];
    lim_pt = [0, max([S.true_peak_time; S.est_peak_time]) * 1.1];

    subplot(1,3,1)
        scatter(S.true_Lcum(idx_ut), S.est_Lcum(idx_ut), 30, 'b', 'filled', 'MarkerFaceAlpha', 0.5); hold on;
        scatter(S.true_Lcum(idx_tx), S.est_Lcum(idx_tx), 30, 'r', 'filled', 'MarkerFaceAlpha', 0.5);
        plot(lim_lc, lim_lc, 'k--', 'LineWidth', 1);
        xlabel('True L_{cum}'); ylabel('Estimated L_{cum}');
        title(sprintf('L_{cum} (r=%.3f)', cLc));
        legend('Untreated', 'Treated', 'Location', 'best');
        grid on; xlim(lim_lc); ylim(lim_lc);
    subplot(1,3,2)
        scatter(S.true_peak_height(idx_ut), S.est_peak_height(idx_ut), 30, 'b', 'filled', 'MarkerFaceAlpha', 0.5); hold on;
        scatter(S.true_peak_height(idx_tx), S.est_peak_height(idx_tx), 30, 'r', 'filled', 'MarkerFaceAlpha', 0.5);
        plot(lim_ph, lim_ph, 'k--', 'LineWidth', 1);
        xlabel('True Peak Height'); ylabel('Estimated Peak Height');
        title(sprintf('Peak Height (r=%.3f)', cPh));
        legend('Untreated', 'Treated', 'Location', 'best');
        grid on; xlim(lim_ph); ylim(lim_ph);
    subplot(1,3,3)
        scatter(S.true_peak_time(idx_ut), S.est_peak_time(idx_ut), 30, 'b', 'filled', 'MarkerFaceAlpha', 0.5); hold on;
        scatter(S.true_peak_time(idx_tx), S.est_peak_time(idx_tx), 30, 'r', 'filled', 'MarkerFaceAlpha', 0.5);
        plot(lim_pt, lim_pt, 'k--', 'LineWidth', 1);
        xlabel('True Peak Time (days)'); ylabel('Estimated Peak Time (days)');
        title(sprintf('Peak Timing (r=%.3f, MAE=%.2f d)', cPt, maePt));
        legend('Untreated', 'Treated', 'Location', 'best');
        grid on; xlim(lim_pt); ylim(lim_pt);

    sgtitle('Derived Quantity Recovery: What Matters for Mortality');
    savepdf(fig, fullfile(opts.outdir, 'paper_fig03_derived_recovery.pdf'));
end

% =========================================================================
% Figure 4: Confounding by indication (RCT vs Observational)
% =========================================================================
function fig04_confounding(opts)
    S = load(fullfile(opts.results_dir, 'gformula_results.mat'));
    [peak_rates, ~] = regenerate_s9_covariates(S.Npat, S.use_cicadas_dgp);

    fig = figure('Position', [50 50 1000 400], 'Color', 'w');

    subplot(1,2,1);
        histogram(peak_rates(S.rct_assignment), 20, 'FaceColor', [0.3 0.8 0.3], 'FaceAlpha', 0.5, 'DisplayName', 'Treated');
        hold on;
        histogram(peak_rates(~S.rct_assignment), 20, 'FaceColor', [0.7 0.7 0.7], 'FaceAlpha', 0.5, 'DisplayName', 'Untreated');
        xlabel('Peak Seizure Rate (sz/day)'); ylabel('Count');
        title(sprintf('RCT: No Confounding\nmean treated=%.1f, untreated=%.1f', ...
            mean(peak_rates(S.rct_assignment)), mean(peak_rates(~S.rct_assignment))));
        legend('Location', 'best'); grid on;

    subplot(1,2,2);
        obs = logical(S.obs_assignment);
        histogram(peak_rates(obs), 20, 'FaceColor', [0.3 0.8 0.3], 'FaceAlpha', 0.5, 'DisplayName', 'Treated');
        hold on;
        histogram(peak_rates(~obs), 20, 'FaceColor', [0.7 0.7 0.7], 'FaceAlpha', 0.5, 'DisplayName', 'Untreated');
        xlabel('Peak Seizure Rate (sz/day)'); ylabel('Count');
        title(sprintf('Observational: Confounding by Indication\nmean treated=%.1f, untreated=%.1f', ...
            mean(peak_rates(obs)), mean(peak_rates(~obs))));
        legend('Location', 'best'); grid on;

    sgtitle('Treatment Assignment Bias');
    savepdf(fig, fullfile(opts.outdir, 'paper_fig04_confounding.pdf'));
end

% =========================================================================
% Figure 5: G-formula treatment effect by patient severity
% =========================================================================
function fig05_te_by_severity(opts)
    S = load(fullfile(opts.results_dir, 'gformula_results.mat'));
    [peak_rates, sofas] = regenerate_s9_covariates(S.Npat, S.use_cicadas_dgp);

    individual_te = (S.gform_surv_treated - S.gform_surv_untreated) * 100;

    fig = figure('Position', [50 50 800 500], 'Color', 'w');
    scatter(peak_rates, individual_te, 30, sofas, 'filled', 'MarkerFaceAlpha', 0.6);
    hold on;
    yline(0, 'k--', 'LineWidth', 1.5);
    yline(S.gform_ate, 'r-', sprintf('ATE=%.1f%%', S.gform_ate), 'LineWidth', 2);
    xlabel('Peak Seizure Rate (sz/day)');
    ylabel('Individual Treatment Effect (%)');
    title('G-formula: Treatment Effect by Patient Severity');
    colormap('hot');
    c = colorbar; c.Label.String = 'SOFA Score';
    grid on;
    savepdf(fig, fullfile(opts.outdir, 'paper_fig05_te_by_severity.pdf'));
end

% =========================================================================
% Figure 6: Optimal strategy depends on harm profile (quadrant 2x2)
% =========================================================================
function fig06_quadrant(opts)
    S = load(fullfile(opts.results_dir, 'quadrant_results.mat'));
    colors = [0.7 0.7 0.7; 0.3 0.3 0.8; 0.8 0.3 0.3; 0.3 0.8 0.3];

    fig = figure('Position', [50 50 1200 900], 'Color', 'w');
    n_strats = numel(S.strat_labels);
    for qi = 1:4
        subplot(2,2,qi);
        b = bar(S.surv_rates(qi,:), 'FaceColor', 'flat');
        for s = 1:n_strats; b.CData(s,:) = colors(s,:); end
        set(gca, 'XTickLabel', S.strat_labels);
        ylabel('Survival (%)'); ylim([0 100]);
        [bestVal, bestIdx] = max(S.surv_rates(qi,:));
        title(sprintf('%s\nBest: %s (%.0f%%)', S.regimes(qi).name, S.strat_labels{bestIdx}, bestVal), 'FontSize', 11);
        grid on;
    end
    sgtitle('Optimal Strategy Depends on Harm Profile', 'FontSize', 14, 'FontWeight', 'bold');
    savepdf(fig, fullfile(opts.outdir, 'paper_fig06_quadrant.pdf'));
end

% =========================================================================
% Figure 7: Threshold sweep (PRN and Adaptive)
% =========================================================================
function fig07_threshold_sweep(opts)
    S = load(fullfile(opts.results_dir, 'threshold_sweep_results.mat'));
    rp = S.results_prn; ro = S.results_omega; rb = S.results_baseline;
    [bestPrn,  iPrn]  = max(rp.surv_rate);
    [bestOm,   iOm]   = max(ro.surv_rate);

    yl = [min([rp.surv_rate; ro.surv_rate; rb.surv_rate])-5, ...
          max([rp.surv_rate; ro.surv_rate])+5];
    xl = [0 16];

    fig = figure('Position', [100 100 1000 400], 'Color', 'w');
    subplot(1,2,1)
        errorbar(S.prn_thresholds, rp.surv_rate, rp.surv_std, 'o-', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor','r');
        hold on; yline(rb.surv_rate, 'k--', 'No Treatment', 'LineWidth', 1.5);
        scatter(S.prn_thresholds(iPrn), bestPrn, 150, 'g', 'filled', 'MarkerEdgeColor','k');
        xlabel('PRN Threshold'); ylabel('Survival Rate (%)');
        title(sprintf('PRN: Optimal threshold = %.0f', S.prn_thresholds(iPrn)));
        grid on; legend('PRN','Baseline','Best','Location','best');
        ylim(yl); xlim(xl);

    subplot(1,2,2)
        errorbar(S.omega_values, ro.surv_rate, ro.surv_std, 'o-', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor','b');
        hold on; yline(rb.surv_rate, 'k--', 'No Treatment', 'LineWidth', 1.5);
        scatter(S.omega_values(iOm), bestOm, 150, 'g', 'filled', 'MarkerEdgeColor','k');
        xlabel('Adaptive Target Rate (\omega)'); ylabel('Survival Rate (%)');
        title(sprintf('Adaptive: Optimal \\omega = %.0f', S.omega_values(iOm)));
        grid on; legend('Adaptive','Baseline','Best','Location','best');
        ylim(yl); xlim(xl);

    sgtitle(sprintf('Threshold Sweep (N=%d patients)', S.Npat));
    savepdf(fig, fullfile(opts.outdir, 'paper_fig07_threshold_sweep.pdf'));
end

% =========================================================================
% Figure 8: Personalized threshold validation (uniform/true/estimated)
% =========================================================================
function fig08_personalized(opts)
    S = load(fullfile(opts.results_dir, 'personalized_validation_results.mat'));
    n_regimes = numel(S.regimes);
    bar_data = zeros(n_regimes, 3);
    [~, idx_uniform] = min(abs(S.omega_vals - 2.0));
    for reg = 1:n_regimes
        bar_data(reg, 1) = mean(S.surv_true_grid(:, idx_uniform, reg)) * 100;
        bar_data(reg, 2) = mean(S.surv_at_true_opt(:, reg)) * 100;
        bar_data(reg, 3) = mean(S.surv_at_est_opt(:, reg)) * 100;
    end

    fig = figure('Position', [50 50 900 500], 'Color', 'w');
    b = bar(bar_data);
    b(1).FaceColor = [0.5 0.5 0.5];
    b(2).FaceColor = [0.2 0.7 0.2];
    b(3).FaceColor = [0.2 0.4 0.8];
    set(gca, 'XTickLabel', {S.regimes.name});
    ylabel('Survival (%)');
    legend('Uniform (\omega=2)', 'Personalized (true \lambda)', 'Personalized (estimated \lambda)', 'Location', 'best');
    title('Can Estimated Parameters Recover Personalized Benefit?');
    ylim([0 100]); grid on;
    for reg = 1:n_regimes
        for col = 2:3
            improvement = bar_data(reg, col) - bar_data(reg, 1);
            x_pos = reg + (col - 2.5) * 0.22;
            text(x_pos, bar_data(reg, col) + 1, sprintf('%+.1f%%', improvement), ...
                'HorizontalAlignment', 'center', 'FontSize', 9, 'FontWeight', 'bold');
        end
    end
    savepdf(fig, fullfile(opts.outdir, 'paper_fig08_personalized.pdf'));
end

% =========================================================================
% Figure 9: Uniform vs Fixed-Optimal vs Dynamic Omega
% =========================================================================
function fig09_uniform_vs_dynamic(opts)
    S = load(fullfile(opts.results_dir, 'closed_loop_personalized_results.mat'));
    n_regimes = numel(S.regimes);
    fig = figure('Position', [50 50 900 500], 'Color', 'w');
    b = bar(S.surv_results);
    b(1).FaceColor = [0.5 0.5 0.5];
    b(2).FaceColor = [0.2 0.7 0.2];
    b(3).FaceColor = [0.2 0.4 0.8];
    set(gca, 'XTickLabel', {S.regimes.name});
    ylabel('Survival (%)');
    legend({'Uniform (\omega=2)', 'Fixed \omega^*', 'Dynamic \omega(t)'}, 'Location', 'best');
    title('Uniform vs Fixed-Optimal vs Dynamic Omega');
    ylim([0 100]); grid on;
    for reg = 1:n_regimes
        for si = 1:3
            improvement = S.surv_results(reg, si) - S.surv_results(reg, 1);
            if abs(improvement) > 0.5
                x_pos = reg + (si - 2) * 0.22;
                text(x_pos, S.surv_results(reg,si) + 1, sprintf('%+.1f', improvement), ...
                    'HorizontalAlignment', 'center', 'FontSize', 9, 'FontWeight', 'bold');
            end
        end
    end
    savepdf(fig, fullfile(opts.outdir, 'paper_fig09_uniform_vs_dynamic.pdf'));
end

% =========================================================================
% Figure 10: Closed-loop pipeline validation (True vs Estimated, 8 panels)
% =========================================================================
function fig10_closed_loop(opts)
    S = load(fullfile(opts.results_dir, 'closed_loop_results.mat'));
    colors = [0.7 0.7 0.7; 0.3 0.3 0.8; 0.8 0.3 0.3; 0.3 0.8 0.3];
    n_regimes = numel(S.regimes); n_strats = numel(S.strat_labels);

    fig = figure('Position', [50 50 1200 500], 'Color', 'w');
    for reg = 1:n_regimes
        subplot(2, n_regimes, reg);
            b = bar(S.surv_true(reg,:), 'FaceColor', 'flat');
            for s = 1:n_strats; b.CData(s,:) = colors(s,:); end
            set(gca, 'XTickLabel', S.strat_labels);
            ylabel('Survival (%)'); ylim([0 100]);
            [~, best] = max(S.surv_true(reg,:));
            title(sprintf('%s\nTRUE: %s (%.0f%%)', S.regimes(reg).name, S.strat_labels{best}, max(S.surv_true(reg,:))));
            grid on;

        subplot(2, n_regimes, n_regimes + reg);
            b = bar(S.surv_est(reg,:), 'FaceColor', 'flat');
            for s = 1:n_strats; b.CData(s,:) = colors(s,:); end
            set(gca, 'XTickLabel', S.strat_labels);
            ylabel('Survival (%)'); ylim([0 100]);
            [~, best] = max(S.surv_est(reg,:));
            title(sprintf('EST: %s (%.0f%%)', S.strat_labels{best}, max(S.surv_est(reg,:))));
            grid on;
    end
    sgtitle(sprintf('Closed-Loop Validation: True (top) vs Estimated (bottom), N=%d', S.Npat));
    savepdf(fig, fullfile(opts.outdir, 'paper_fig10_closed_loop.pdf'));
end

% =========================================================================
% Figure 11: EKF parameter sensitivity (6 panels)
% =========================================================================
function fig11_ekf_sensitivity(opts)
    S = load(fullfile(opts.results_dir, 'ekf_sensitivity_results.mat'));
    results = S.results; sweeps = S.sweeps;

    fig = figure('Position', [50 50 1400 800], 'Color', 'w');
    for s = 1:numel(results)
        subplot(2, 3, s);
        if strcmp(results(s).param, 'q')
            semilogx(results(s).values, results(s).survival, 'bo-', 'LineWidth', 2, 'MarkerSize', 8);
        else
            plot(results(s).values, results(s).survival, 'bo-', 'LineWidth', 2, 'MarkerSize', 8);
        end
        hold on;
        yline(S.baseline_none,  'r--', 'No treatment', 'LineWidth', 1.5);
        yline(S.baseline_fixed, 'k--', 'Fixed dosing', 'LineWidth', 1.5);

        default_idx = find(abs(results(s).values - sweeps(s).default) < 1e-10, 1);
        if ~isempty(default_idx)
            if strcmp(results(s).param, 'q')
                semilogx(results(s).values(default_idx), results(s).survival(default_idx), 'rs', 'MarkerSize', 14, 'LineWidth', 2);
            else
                plot(results(s).values(default_idx), results(s).survival(default_idx), 'rs', 'MarkerSize', 14, 'LineWidth', 2);
            end
        end

        xlabel(results(s).name); ylabel('7-day Survival (%)');
        title(results(s).name);
        grid on;
        yl = [max(0, min([results(s).survival; S.baseline_none; S.baseline_fixed]) - 5), ...
              min(100, max([results(s).survival; S.baseline_none; S.baseline_fixed]) + 5)];
        ylim(yl);
    end

    subplot(2, 3, 6);
    spreads = arrayfun(@(r) max(r.survival) - min(r.survival), results);
    barh(spreads);
    set(gca, 'YTickLabel', {results.param});
    xlabel('Survival spread (pp)');
    title('Parameter Sensitivity Ranking');
    grid on;
    sgtitle('EKF Parameter Sensitivity Analysis', 'FontSize', 14, 'FontWeight', 'bold');
    savepdf(fig, fullfile(opts.outdir, 'paper_fig11_ekf_sensitivity.pdf'));
end

% =========================================================================
% Figure 12: Graded trajectory misspecification (robustness curve)
% =========================================================================
function fig12_graded_misspec(opts)
    S = load(fullfile(opts.results_dir, 'misspecification_results.mat'));

    fig = figure('Position', [50 50 700 500], 'Color', 'w');
    yyaxis left
    plot(S.mix_fractions * 100, S.graded_bias, 'bo-', 'LineWidth', 2, 'MarkerSize', 8);
    ylabel('G-formula Residual Bias (pp)');
    ylim([0, max(S.graded_bias) * 1.3 + 1]);

    yyaxis right
    plot(S.mix_fractions * 100, S.graded_gform, 'rs--', 'LineWidth', 1.5, 'MarkerSize', 6);
    hold on;
    plot(S.mix_fractions * 100, S.graded_rct,   'k:+',  'LineWidth', 1.5, 'MarkerSize', 6);
    ylabel('ATE (%)');
    legend('Bias (left axis)', 'G-formula ATE', 'RCT ATE (truth)', 'Location', 'best');

    xlabel('Fraction of SDE patients (%)');
    title('Graded Trajectory Misspecification: Robustness Curve');
    grid on;
    savepdf(fig, fullfile(opts.outdir, 'paper_fig12_graded_misspec.pdf'));
end

% =========================================================================
% Helpers
% =========================================================================
function savepdf(fig, path)
    set(fig, 'PaperPositionMode', 'auto');
    try
        exportgraphics(fig, path, 'ContentType', 'vector');
    catch
        % Fallback for headless / non-vector
        print(fig, path, '-dpdf', '-bestfit');
    end
    fprintf('  saved %s\n', path);
end

function [peak_rates, sofas] = regenerate_s9_covariates(Npat, use_cicadas_dgp)
    % Deterministically regenerate the s9 patient cohort so we can recover
    % per-patient peak_rate and sofa for figures 4 and 5 without re-running
    % the slow ATE simulation. Must mirror s9's rng and generation order
    % exactly.
    [time_params, drug_params, ~, ~] = default_params();
    T = time_params.T;
    pp.peak_rate = 10;
    pp.drug_params = drug_params;

    rng(42);  % must match s9
    peak_rates = zeros(Npat, 1);
    sofas = zeros(Npat, 1);
    for p = 1:Npat
        if use_cicadas_dgp
            patient = fcn_generate_patient_cicadas(T, pp);
        else
            patient = fcn_generate_patient(T, pp);
        end
        peak_rates(p) = max(patient.lambda_base);
        sofas(p) = patient.sofa;
    end
end
