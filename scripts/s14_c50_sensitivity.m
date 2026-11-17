% S14_C50_SENSITIVITY
% Tests how C50 estimation error affects parameter recovery and
% downstream clinical conclusions.
%
% C50 = drug concentration for 50% effect. Currently given as pop-mean
% in s8/s12. This script sweeps C50 error and measures impact on
% L_cum recovery and winner matching.
%
% Structurally identical to s11 (half-life sweep).

clear; clc;
addpath('functions');

%% Load parameters
[time_params, drug_params, ctrl_params, mort_params] = default_params();
T = time_params.T;
Nt = time_params.Nt;
dt = time_params.dt;

%% Settings
Npat = 200;
Nruns = 100;
use_ekf = true;
true_C50 = drug_params.C50;  % 500

% C50 multipliers: 0.5 means we think C50 is 50% lower than truth
c50_multipliers = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3, 1.5];
n_levels = numel(c50_multipliers);

rng(42);
patient_params.peak_rate = 10;
patient_params.drug_params = drug_params;

%% Quadrant regimes
theta4_high = 1.25;
regimes = struct();
regimes(1).name = 'High Sz / Low Drug';
regimes(1).theta = [-5, 0.5, 3.0, 1.0, 0.05, 0.02];
regimes(2).name = 'High Sz / High Drug';
regimes(2).theta = [-5, 0.5, 3.0, 1.0, theta4_high, 0.3];
regimes(3).name = 'Low Sz / Low Drug';
regimes(3).theta = [-5, 0.5, 0.5, 0.2, 0.05, 0.02];
regimes(4).name = 'Low Sz / High Drug';
regimes(4).theta = [-5, 0.5, 0.5, 0.2, theta4_high, 0.3];
n_regimes = numel(regimes);
strategies = {'none', 'fixed', 'prn', 'adaptive'};
n_strats = numel(strategies);
strat_labels = {'None', 'Fixed', 'PRN', 'Adaptive'};

%% Generate patients and run fixed strategy with TRUE PKPD
fprintf('=== C50 SENSITIVITY ANALYSIS ===\n');
fprintf('Patients: %d | C50 levels: %d\n', Npat, n_levels);
fprintf('True C50: %.0f\n\n', true_C50);

patients = cell(Npat, 1);
true_Lcum = zeros(Npat, 1);
true_lambda_base = zeros(Npat, Nt);
fixed_results = cell(Npat, 1);

fprintf('Generating patients and running fixed strategy...\n');
tic;

for p = 1:Npat
    patient = fcn_generate_patient(T, patient_params);
    patients{p} = patient;
    true_lambda_base(p,:) = patient.lambda_base;
    true_Lcum(p) = sum(patient.lambda_base) * dt;
    
    dp = drug_params;
    dp.alpha = patient.alpha;
    if isfield(patient, 'C50')
        dp.patient_C50 = patient.C50;
        dp.patient_gamma = patient.gamma_hill;
    end
    r = fcn_run_strategy('fixed', patient.lambda_base, dp, ctrl_params, T, dt, false);
    fixed_results{p} = r;
end
fprintf('Done (%.1fs).\n\n', toc);

%% Sweep C50 error levels
fprintf('Sweeping C50 error levels...\n');

corr_Lcum = zeros(n_levels, 1);
corr_peak = zeros(n_levels, 1);
mape_Lcum = zeros(n_levels, 1);
mean_Lcum_bias = zeros(n_levels, 1);
est_lambda_all = zeros(Npat, Nt, n_levels);
tested_C50 = zeros(n_levels, 1);

for li = 1:n_levels
    c50_test = true_C50 * c50_multipliers(li);
    tested_C50(li) = c50_test;
    pct_err = (c50_multipliers(li) - 1) * 100;
    fprintf('  C50=%.0f (%+.0f%%)...', c50_test, pct_err);
    
    est_Lcum_level = zeros(Npat, 1);
    est_peak_level = zeros(Npat, 1);
    true_peak_level = zeros(Npat, 1);
    
    for p = 1:Npat
        r = fixed_results{p};
        
        [~, fit_info] = fcn_estimate_natural_history( ...
            r.seizure_counts, T, dt, ...
            'peak_rate', patient_params.peak_rate, ...
            'k_e', drug_params.k_e, ...
            'doses_given', r.doses_given, ...
            'pd_mode', 'hill', ...
            'C50', c50_test, ...
            'gamma_hill', drug_params.gamma_hill);
        
        est_lambda_all(p, :, li) = fit_info.lambda_base_hat;
        est_Lcum_level(p) = sum(fit_info.lambda_base_hat) * dt;
        est_peak_level(p) = max(fit_info.lambda_base_hat);
        true_peak_level(p) = max(true_lambda_base(p,:));
    end
    
    corr_Lcum(li) = corr(true_Lcum, est_Lcum_level);
    corr_peak(li) = corr(true_peak_level, est_peak_level);
    mape_Lcum(li) = mean(abs(true_Lcum - est_Lcum_level) ./ max(true_Lcum, 0.1)) * 100;
    mean_Lcum_bias(li) = mean(est_Lcum_level - true_Lcum);
    
    fprintf(' L_cum r=%.3f, peak r=%.3f, MAPE=%.1f%%, bias=%+.1f\n', ...
        corr_Lcum(li), corr_peak(li), mape_Lcum(li), mean_Lcum_bias(li));
end
fprintf('Sweep done (%.1fs).\n\n', toc);

%% Winner matching
fprintf('Computing true winners...\n');

true_strat = cell(Npat, n_strats);
for p = 1:Npat
    patient = patients{p};
    dp = drug_params;
    dp.alpha = patient.alpha;
    if isfield(patient, 'C50')
        dp.patient_C50 = patient.C50;
        dp.patient_gamma = patient.gamma_hill;
    end
    for s = 1:n_strats
        true_strat{p,s} = fcn_run_strategy(strategies{s}, ...
            true_lambda_base(p,:), dp, ctrl_params, T, dt, use_ekf);
    end
end

surv_true = zeros(n_regimes, n_strats);
for reg = 1:n_regimes
    for p = 1:Npat
        for s = 1:n_strats
            alive = 0;
            for run = 1:Nruns
                [a,~] = fcn_mortality(true_strat{p,s}.lambda_obs, ...
                    true_strat{p,s}.x, dt, regimes(reg).theta, ...
                    patients{p}.age, patients{p}.sofa);
                alive = alive + a;
            end
            surv_true(reg, s) = surv_true(reg, s) + alive / Nruns;
        end
    end
end
surv_true = surv_true / Npat * 100;
[~, true_winners] = max(surv_true, [], 2);

fprintf('True winners:\n');
for reg = 1:n_regimes
    fprintf('  %s: %s (%.1f%%)\n', regimes(reg).name, ...
        strat_labels{true_winners(reg)}, max(surv_true(reg,:)));
end

winner_matches = zeros(n_levels, 1);
surv_est_all = zeros(n_regimes, n_strats, n_levels);
max_surv_diff = zeros(n_levels, 1);

fprintf('\nRunning winner matching at each C50 level...\n');

for li = 1:n_levels
    pct_err = (c50_multipliers(li) - 1) * 100;
    fprintf('  C50 %+.0f%%: ', pct_err);
    
    est_strat = cell(Npat, n_strats);
    for p = 1:Npat
        patient = patients{p};
        dp = drug_params;
        dp.alpha = patient.alpha;
        if isfield(patient, 'C50')
            dp.patient_C50 = patient.C50;
            dp.patient_gamma = patient.gamma_hill;
        end
        for s = 1:n_strats
            est_strat{p,s} = fcn_run_strategy(strategies{s}, ...
                est_lambda_all(p,:,li), dp, ctrl_params, T, dt, use_ekf);
        end
    end
    
    surv_est = zeros(n_regimes, n_strats);
    for reg = 1:n_regimes
        for p = 1:Npat
            for s = 1:n_strats
                alive = 0;
                for run = 1:Nruns
                    [a,~] = fcn_mortality(est_strat{p,s}.lambda_obs, ...
                        est_strat{p,s}.x, dt, regimes(reg).theta, ...
                        patients{p}.age, patients{p}.sofa);
                    alive = alive + a;
                end
                surv_est(reg, s) = surv_est(reg, s) + alive / Nruns;
            end
        end
    end
    surv_est = surv_est / Npat * 100;
    surv_est_all(:, :, li) = surv_est;
    
    [~, est_winners] = max(surv_est, [], 2);
    matches = sum(est_winners == true_winners);
    winner_matches(li) = matches;
    max_surv_diff(li) = max(abs(surv_est(:) - surv_true(:)));
    
    fprintf('%d/4 match, max diff=%.1f%%\n', matches, max_surv_diff(li));
end
fprintf('Winner matching done (%.1fs).\n\n', toc);

%% Summary table
fprintf('=====================================================\n');
fprintf('C50 SENSITIVITY: Summary\n');
fprintf('=====================================================\n');
fprintf('%-8s %6s %10s %10s %10s %8s %8s\n', ...
    'C50 err', 'C50', 'L_cum r', 'Peak r', 'MAPE%', 'Bias', 'Winners');
fprintf('%s\n', repmat('-', 1, 70));
for li = 1:n_levels
    pct_err = (c50_multipliers(li) - 1) * 100;
    fprintf('%+6.0f%%  %5.0f %10.3f %10.3f %9.1f%% %+8.1f   %d/4', ...
        pct_err, tested_C50(li), corr_Lcum(li), corr_peak(li), ...
        mape_Lcum(li), mean_Lcum_bias(li), winner_matches(li));
    if winner_matches(li) == n_regimes
        fprintf('  [ALL MATCH]');
    end
    if c50_multipliers(li) == 1.0
        fprintf('  <-- true');
    end
    fprintf('\n');
end

%% FIGURE 1: Correlation and bias vs C50 error
figure('Position', [50 50 1000 400]); clf;

c50_pct = (c50_multipliers - 1) * 100;

subplot(1,2,1);
plot(c50_pct, corr_Lcum, 'b-o', 'LineWidth', 2, 'MarkerSize', 8); hold on;
plot(c50_pct, corr_peak, 'r-s', 'LineWidth', 2, 'MarkerSize', 8);
xline(0, 'k--', 'true', 'LineWidth', 1);
xlabel('C_{50} error (%)');
ylabel('Correlation with truth');
title('Estimation Quality vs C_{50} Error');
legend('L_{cum}', 'Peak height', 'Location', 'best');
grid on; ylim([0 1]);

subplot(1,2,2);
plot(c50_pct, mean_Lcum_bias, 'b-o', 'LineWidth', 2, 'MarkerSize', 8);
hold on;
yline(0, 'k--', 'LineWidth', 1);
xline(0, 'k--', 'LineWidth', 1);
xlabel('C_{50} error (%)');
ylabel('Mean L_{cum} bias (est - true)');
title('Systematic Bias vs C_{50} Error');
grid on;

sgtitle('C_{50} Sensitivity: Parameter Recovery');

%% FIGURE 2: Winner matches and max survival difference
figure('Position', [50 50 1000 400]); clf;

subplot(1,2,1);
bar(c50_pct, winner_matches, 'FaceColor', [0.3 0.7 0.3]);
hold on;
yline(4, 'r--', '4/4', 'LineWidth', 1.5);
xlabel('C_{50} error (%)');
ylabel('Regimes with matching winner');
title('Winner Agreement vs C_{50} Error');
ylim([0 5]); grid on;
set(gca, 'YTick', 0:4);

subplot(1,2,2);
plot(c50_pct, max_surv_diff, 'r-o', 'LineWidth', 2, 'MarkerSize', 8);
hold on;
xline(0, 'k--', 'LineWidth', 1);
yline(5, 'r--', '5% threshold', 'LineWidth', 1);
xlabel('C_{50} error (%)');
ylabel('Max survival difference (%)');
title('Worst-Case Error vs C_{50} Error');
grid on;

sgtitle('C_{50} Sensitivity: Clinical Conclusions');

%% FIGURE 3: Detailed survival at selected C50 errors
figure('Position', [50 50 1200 500]); clf;

[~, idx_neg30] = min(abs(c50_multipliers - 0.7));
[~, idx_zero] = min(abs(c50_multipliers - 1.0));
[~, idx_pos30] = min(abs(c50_multipliers - 1.3));
show_levels = [idx_neg30, idx_zero, idx_pos30];

for qi = 1:n_regimes
    subplot(1, n_regimes, qi);
    
    bar_data = zeros(n_strats, numel(show_levels) + 1);
    bar_data(:, 1) = surv_true(qi, :)';
    for si = 1:numel(show_levels)
        bar_data(:, si+1) = surv_est_all(qi, :, show_levels(si))';
    end
    
    b = bar(bar_data);
    b(1).FaceColor = [0.2 0.2 0.2];
    b(2).FaceColor = [0.8 0.5 0.5];
    b(3).FaceColor = [0.5 0.8 0.5];
    b(4).FaceColor = [0.5 0.5 0.8];
    
    set(gca, 'XTickLabel', strat_labels);
    ylabel('Survival (%)');
    ylim([0 100]); grid on;
    title(regimes(qi).name);
    if qi == 1
        legend('True', 'C50 -30%', 'C50 exact', 'C50 +30%', 'Location', 'best');
    end
end
sgtitle('Survival by Strategy at Different C_{50} Errors');

%% Save
save('c50_sensitivity_results.mat', 'c50_multipliers', 'tested_C50', ...
    'corr_Lcum', 'corr_peak', 'mape_Lcum', 'mean_Lcum_bias', ...
    'winner_matches', 'max_surv_diff', 'surv_true', 'surv_est_all', ...
    'true_C50', 'regimes', 'strat_labels', 'Npat', 'Nruns');
fprintf('\nResults saved to c50_sensitivity_results.mat\n');

%% Conclusion
fprintf('\n=== KEY FINDINGS ===\n');

match_mask = winner_matches == n_regimes;
if any(match_mask)
    safe_range = c50_multipliers(match_mask);
    fprintf('All 4/4 winners match for C50 in [%+.0f%%, %+.0f%%]\n', ...
        (min(safe_range)-1)*100, (max(safe_range)-1)*100);
    fprintf('C50 range: %.0f to %.0f (true=%.0f)\n', ...
        true_C50*min(safe_range), true_C50*max(safe_range), true_C50);
else
    fprintf('No C50 error achieves 4/4 match.\n');
    [best, idx] = max(winner_matches);
    fprintf('Best: %d/4 at C50=%+.0f%%\n', best, (c50_multipliers(idx)-1)*100);
end

fprintf('\nInterpretation:\n');
fprintf('  Underestimate C50 -> think drug more potent -> underestimate lambda_base\n');
fprintf('  Overestimate C50 -> think drug less potent -> overestimate lambda_base\n');
fprintf('  Published PD studies report C50 with ~20-40%% uncertainty.\n');
