% S3_POPULATION_SIM
% Runs simulation across N patients with stratified analysis by SOFA and age.

clear; clc;

% Add functions to path
addpath('../functions');

%% Load default parameters
[time_params, drug_params, ctrl_params, mort_params] = default_params();
T = time_params.T;
Nt = time_params.Nt;
dt = time_params.dt;

%% Simulation settings
Npat = 500;     % Number of patients
Nruns = 1000;    % Mortality runs per patient
use_ekf = true; % Use EKF estimates for treatment decisions

rng(42);  % For reproducibility

%% Storage
results = struct();
results.age = zeros(Npat, 1);
results.sofa = zeros(Npat, 1);
results.A_patient = zeros(Npat, 1);
results.peak_rate = zeros(Npat, 1);
results.alpha = zeros(Npat, 1);  % patient-specific drug sensitivity

% Survival outcomes (average over Nruns)
results.surv_none = zeros(Npat, 1);
results.surv_fixed = zeros(Npat, 1);
results.surv_prn = zeros(Npat, 1);
results.surv_adaptive = zeros(Npat, 1);

% Burden metrics (deterministic, only need 1 run)
results.burden_none = zeros(Npat, 1);
results.burden_fixed = zeros(Npat, 1);
results.burden_prn = zeros(Npat, 1);
results.burden_adaptive = zeros(Npat, 1);

% Drug exposure
results.drug_none = zeros(Npat, 1);
results.drug_fixed = zeros(Npat, 1);
results.drug_prn = zeros(Npat, 1);
results.drug_adaptive = zeros(Npat, 1);

%% Main simulation loop
fprintf('=== POPULATION SIMULATION ===\n');
fprintf('Patients: %d | Mortality runs per patient: %d | EKF: %s\n', ...
    Npat, Nruns, string(use_ekf));
fprintf('\n');

patient_params.peak_rate = 10;  % seizures/day at peak
patient_params.drug_params = drug_params;  % Pass drug params for alpha computation
tic;

for p = 1:Npat
    % Generate patient (each run uses same patient trajectory)
    patient = fcn_generate_patient(T, patient_params);
    
    % Store covariates
    results.age(p) = patient.age;
    results.sofa(p) = patient.sofa;
    results.A_patient(p) = patient.A_patient;
    results.peak_rate(p) = max(patient.lambda_base);
    results.alpha(p) = patient.alpha;
    
    % Use patient-specific drug sensitivity
    dp = drug_params;
    dp.alpha = patient.alpha;
    if isfield(patient, 'C50')
        dp.patient_C50 = patient.C50;
        dp.patient_gamma = patient.gamma_hill;
    end
    
    % Run strategies once to get burden/drug metrics
    r_none = fcn_run_strategy('none', patient.lambda_base, dp, ctrl_params, T, dt, use_ekf);
    r_fixed = fcn_run_strategy('fixed', patient.lambda_base, dp, ctrl_params, T, dt, use_ekf);
    r_prn = fcn_run_strategy('prn', patient.lambda_base, dp, ctrl_params, T, dt, use_ekf);
    r_adaptive = fcn_run_strategy('adaptive', patient.lambda_base, dp, ctrl_params, T, dt, use_ekf);
    
    % Store burden and drug exposure
    results.burden_none(p) = r_none.burden;
    results.burden_fixed(p) = r_fixed.burden;
    results.burden_prn(p) = r_prn.burden;
    results.burden_adaptive(p) = r_adaptive.burden;
    
    results.drug_none(p) = r_none.total_dose;
    results.drug_fixed(p) = r_fixed.total_dose;
    results.drug_prn(p) = r_prn.total_dose;
    results.drug_adaptive(p) = r_adaptive.total_dose;
    
    % Simulate mortality multiple times (due to random hazard)
    surv_temp = zeros(4, Nruns);
    for r = 1:Nruns
        [surv_temp(1,r), ~] = fcn_mortality(r_none.lambda_obs, r_none.x, dt, mort_params.theta, patient.age, patient.sofa);
        [surv_temp(2,r), ~] = fcn_mortality(r_fixed.lambda_obs, r_fixed.x, dt, mort_params.theta, patient.age, patient.sofa);
        [surv_temp(3,r), ~] = fcn_mortality(r_prn.lambda_obs, r_prn.x, dt, mort_params.theta, patient.age, patient.sofa);
        [surv_temp(4,r), ~] = fcn_mortality(r_adaptive.lambda_obs, r_adaptive.x, dt, mort_params.theta, patient.age, patient.sofa);
    end
    
    % Store average survival
    results.surv_none(p) = mean(surv_temp(1,:));
    results.surv_fixed(p) = mean(surv_temp(2,:));
    results.surv_prn(p) = mean(surv_temp(3,:));
    results.surv_adaptive(p) = mean(surv_temp(4,:));
    
    % Progress
    if mod(p, 50) == 0
        fprintf('  %d/%d patients (%.1f sec)\n', p, Npat, toc);
    end
end

fprintf('\nSimulation complete. Total time: %.1f seconds\n', toc);

%% Summary statistics
fprintf('\n========================================\n');
fprintf('POPULATION SUMMARY\n');
fprintf('========================================\n');
fprintf('Age:  mean=%.1f, std=%.1f, range=[%.0f, %.0f]\n', ...
    mean(results.age), std(results.age), min(results.age), max(results.age));
fprintf('SOFA: mean=%.1f, std=%.1f, range=[%.0f, %.0f]\n', ...
    mean(results.sofa), std(results.sofa), min(results.sofa), max(results.sofa));
fprintf('Peak rate: mean=%.1f seizures/day\n', mean(results.peak_rate));

fprintf('\n========================================\n');
fprintf('SURVIVAL RATES\n');
fprintf('========================================\n');
surv_rates = [mean(results.surv_none), mean(results.surv_fixed), ...
              mean(results.surv_prn), mean(results.surv_adaptive)] * 100;
fprintf('No treatment: %.1f%% (±%.1f)\n', surv_rates(1), 100*std(results.surv_none)/sqrt(Npat));
fprintf('Fixed:        %.1f%% (±%.1f)\n', surv_rates(2), 100*std(results.surv_fixed)/sqrt(Npat));
fprintf('PRN:          %.1f%% (±%.1f)\n', surv_rates(3), 100*std(results.surv_prn)/sqrt(Npat));
fprintf('Adaptive:     %.1f%% (±%.1f)\n', surv_rates(4), 100*std(results.surv_adaptive)/sqrt(Npat));

fprintf('\n========================================\n');
fprintf('BURDEN (L_cum) & DRUG USAGE\n');
fprintf('========================================\n');
fprintf('%-12s %12s %12s\n', 'Strategy', 'L_cum', 'Total Dose');
fprintf('%s\n', repmat('-', 1, 40));
fprintf('%-12s %12.1f %12.0f\n', 'None', mean(results.burden_none), mean(results.drug_none));
fprintf('%-12s %12.1f %12.0f\n', 'Fixed', mean(results.burden_fixed), mean(results.drug_fixed));
fprintf('%-12s %12.1f %12.0f\n', 'PRN', mean(results.burden_prn), mean(results.drug_prn));
fprintf('%-12s %12.1f %12.0f\n', 'Adaptive', mean(results.burden_adaptive), mean(results.drug_adaptive));

%% Statistical tests
fprintf('\n========================================\n');
fprintf('STATISTICAL TESTS (vs No Treatment)\n');
fprintf('========================================\n');

% Paired t-test: Fixed vs None
[~, p_fixed] = ttest(results.surv_fixed, results.surv_none);
fprintf('Fixed vs None:    p = %.4f %s\n', p_fixed, significance_stars(p_fixed));

% Paired t-test: PRN vs None
[~, p_prn] = ttest(results.surv_prn, results.surv_none);
fprintf('PRN vs None:      p = %.4f %s\n', p_prn, significance_stars(p_prn));

% Paired t-test: Adaptive vs None
[~, p_adaptive] = ttest(results.surv_adaptive, results.surv_none);
fprintf('Adaptive vs None: p = %.4f %s\n', p_adaptive, significance_stars(p_adaptive));

% Fixed vs Adaptive
[~, p_fa] = ttest(results.surv_fixed, results.surv_adaptive);
fprintf('Fixed vs Adaptive: p = %.4f %s\n', p_fa, significance_stars(p_fa));

%% FIGURE 1: Population Overview
figure('Position', [100 100 1200 400]); clf;

% Age distribution
subplot(1,4,1)
histogram(results.age, 15, 'FaceColor', [0.3 0.3 0.8]);
xlabel('Age (years)');
ylabel('Count');
title('Age Distribution');
xline(65, 'r--', 'Age 65', 'LineWidth', 1.5);

% SOFA distribution
subplot(1,4,2)
histogram(results.sofa, 0:15, 'FaceColor', [0.8 0.3 0.3]);
xlabel('SOFA Score');
ylabel('Count');
title('SOFA Distribution');
xline(5, 'r--', 'SOFA 5', 'LineWidth', 1.5);

% Survival by strategy
subplot(1,4,3)
bar(surv_rates, 'FaceColor', [0.3 0.7 0.3]);
set(gca, 'XTickLabel', {'None', 'Fixed', 'PRN', 'Adapt.'});
ylabel('Survival Rate (%)');
title('Overall Survival');
ylim([0 100]);
grid on;

% Burden reduction
subplot(1,4,4)
burden_means = [mean(results.burden_none), mean(results.burden_fixed), ...
                mean(results.burden_prn), mean(results.burden_adaptive)];
bar(burden_means, 'FaceColor', [0.8 0.5 0.2]);
set(gca, 'XTickLabel', {'None', 'Fixed', 'PRN', 'Adapt.'});
ylabel('L_{cum} (expected seizures)');
title('Seizure Burden');
grid on;

sgtitle(sprintf('Population Overview (N=%d)', Npat));

%% FIGURE 2: Survival by Subgroup
figure('Position', [100 100 1000 400]); clf;

% By SOFA
subplot(1,2,1)
sofa_low = results.sofa < 5;
sofa_high = results.sofa >= 5;

surv_sofa = [
    mean(results.surv_none(sofa_low)), mean(results.surv_fixed(sofa_low)), ...
    mean(results.surv_prn(sofa_low)), mean(results.surv_adaptive(sofa_low));
    mean(results.surv_none(sofa_high)), mean(results.surv_fixed(sofa_high)), ...
    mean(results.surv_prn(sofa_high)), mean(results.surv_adaptive(sofa_high))
] * 100;

b = bar(surv_sofa);
b(1).FaceColor = [0.7 0.7 0.7];
b(2).FaceColor = [0.3 0.3 0.8];
b(3).FaceColor = [0.8 0.3 0.3];
b(4).FaceColor = [0.3 0.8 0.3];

set(gca, 'XTickLabel', {sprintf('Low SOFA<5\n(n=%d)', sum(sofa_low)), ...
                        sprintf('High SOFA≥5\n(n=%d)', sum(sofa_high))});
ylabel('Survival Rate (%)');
title('Survival by SOFA Group');
legend('None', 'Fixed', 'PRN', 'Adaptive', 'Location', 'best');
ylim([0 100]);
grid on;

% By Age
subplot(1,2,2)
age_young = results.age < 65;
age_old = results.age >= 65;

surv_age = [
    mean(results.surv_none(age_young)), mean(results.surv_fixed(age_young)), ...
    mean(results.surv_prn(age_young)), mean(results.surv_adaptive(age_young));
    mean(results.surv_none(age_old)), mean(results.surv_fixed(age_old)), ...
    mean(results.surv_prn(age_old)), mean(results.surv_adaptive(age_old))
] * 100;

b = bar(surv_age);
b(1).FaceColor = [0.7 0.7 0.7];
b(2).FaceColor = [0.3 0.3 0.8];
b(3).FaceColor = [0.8 0.3 0.3];
b(4).FaceColor = [0.3 0.8 0.3];

set(gca, 'XTickLabel', {sprintf('Age<65\n(n=%d)', sum(age_young)), ...
                        sprintf('Age≥65\n(n=%d)', sum(age_old))});
ylabel('Survival Rate (%)');
title('Survival by Age Group');
legend('None', 'Fixed', 'PRN', 'Adaptive', 'Location', 'best');
ylim([0 100]);
grid on;

sgtitle('Survival by Patient Subgroup');

%% FIGURE 3: Treatment Benefit Analysis
figure('Position', [100 100 1000 400]); clf;

% Benefit = survival improvement (treatment - none)
benefit_fixed = results.surv_fixed - results.surv_none;
benefit_prn = results.surv_prn - results.surv_none;
benefit_adaptive = results.surv_adaptive - results.surv_none;

% Scatter: Benefit vs Patient Severity
subplot(1,2,1)
scatter(results.A_patient, benefit_fixed*100, 50, results.sofa, 'filled', 'MarkerFaceAlpha', 0.6);
xlabel('Patient Severity (A_{patient})');
ylabel('Survival Benefit (Fixed - None) %');
title('Fixed Dose: Benefit vs Severity');
colormap('hot');
c = colorbar; c.Label.String = 'SOFA Score';
yline(0, 'k--', 'LineWidth', 1);
grid on;

subplot(1,2,2)
scatter(results.burden_none, benefit_fixed*100, 50, results.age, 'filled', 'MarkerFaceAlpha', 0.6);
xlabel('Untreated Burden (L_{cum})');
ylabel('Survival Benefit (Fixed - None) %');
title('Fixed Dose: Benefit vs Untreated Burden');
colormap('cool');
c = colorbar; c.Label.String = 'Age';
yline(0, 'k--', 'LineWidth', 1);
grid on;

sgtitle('Treatment Benefit Analysis');

%% FIGURE 4: Burden Reduction
figure('Position', [100 100 600 400]); clf;

L_reduction_fixed = results.burden_none - results.burden_fixed;
L_reduction_adaptive = results.burden_none - results.burden_adaptive;

scatter(L_reduction_fixed, benefit_fixed*100, 50, results.sofa, 'filled', 'MarkerFaceAlpha', 0.6);
xlabel('L_{cum} Reduction (Seizures Prevented)');
ylabel('Survival Benefit (%)');
title('Survival Benefit vs Seizure Prevention');
colormap('hot');
c = colorbar; c.Label.String = 'SOFA Score';
yline(0, 'k--', 'LineWidth', 1);
xline(0, 'k--', 'LineWidth', 1);
grid on;

%% Save results
save('population_results.mat', 'results', 'Npat', 'Nruns', 'use_ekf', ...
     'time_params', 'drug_params', 'ctrl_params', 'mort_params');
fprintf('\nResults saved to population_results.mat\n');

%% Helper function
function s = significance_stars(p)
    if p < 0.001
        s = '***';
    elseif p < 0.01
        s = '**';
    elseif p < 0.05
        s = '*';
    else
        s = '';
    end
end
