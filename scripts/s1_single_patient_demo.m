% S1_SINGLE_PATIENT_DEMO
% Runs all treatment strategies for a single patient.

clear; clc;
addpath('functions');

%% Parameters and patient
[time_params, drug_params, ctrl_params, ~] = default_params();
T = time_params.T;
dt = time_params.dt;

patient_params.peak_rate = 15;  % sz/day

% rng(42);  % uncomment for reproducibility
rng('shuffle');

% patient_params.age = 70;   % override
% patient_params.sofa = 6;

patient_params.drug_params = drug_params;
patient = fcn_generate_patient(T, patient_params);

drug_params.alpha = patient.alpha;
if isfield(patient, 'C50')
    drug_params.patient_C50 = patient.C50;
    drug_params.patient_gamma = patient.gamma_hill;
end

fprintf('Patient: age=%d, SOFA=%d, peak=%.0f sz/day, alpha=%.4f\n\n', ...
    round(patient.age), patient.sofa, max(patient.lambda_base), patient.alpha);

%% Run all strategies
r_none = fcn_run_strategy('none', patient.lambda_base, drug_params, ctrl_params, T, dt, true);
r_fixed = fcn_run_strategy('fixed', patient.lambda_base, drug_params, ctrl_params, T, dt, true);
r_prn = fcn_run_strategy('prn', patient.lambda_base, drug_params, ctrl_params, T, dt, true);
r_adaptive = fcn_run_strategy('adaptive', patient.lambda_base, drug_params, ctrl_params, T, dt, true);
r_pi = fcn_run_strategy('pi', patient.lambda_base, drug_params, ctrl_params, T, dt, true);

%% Seizure event times
sz_none = T(r_none.seizure_counts > 0);
sz_fixed = T(r_fixed.seizure_counts > 0);
sz_prn = T(r_prn.seizure_counts > 0);
sz_adaptive = T(r_adaptive.seizure_counts > 0);
sz_pi = T(r_pi.seizure_counts > 0);

%% Print results
fprintf('%-15s %8s %8s %10s\n', 'Strategy', 'L_cum', 'Seizures', 'Drug');
fprintf('%-15s %8.1f %8d %10.0f\n', 'None', r_none.burden, sum(r_none.seizure_counts), r_none.total_dose);
fprintf('%-15s %8.1f %8d %10.0f\n', 'Fixed (pill)', r_fixed.burden, sum(r_fixed.seizure_counts), r_fixed.total_dose);
fprintf('%-15s %8.1f %8d %10.0f\n', 'PRN (pill)', r_prn.burden, sum(r_prn.seizure_counts), r_prn.total_dose);
fprintf('%-15s %8.1f %8d %10.0f\n', 'Adaptive', r_adaptive.burden, sum(r_adaptive.seizure_counts), r_adaptive.total_dose);
fprintf('%-15s %8.1f %8d %10.0f\n', 'PI Controller', r_pi.burden, sum(r_pi.seizure_counts), r_pi.total_dose);

%% FIGURE 1: Comparison
figure('Position', [100 100 1400 800]); clf;

rate_ylim = [0 max(patient.lambda_base)*1.1];

% Seizure rates with event markers
subplot(2,3,1);
plot(T, patient.lambda_base, 'k-', 'LineWidth', 2.5); hold on;
plot(T, r_fixed.lambda_obs, 'b-', 'LineWidth', 1.5);
plot(T, r_prn.lambda_obs, 'r-', 'LineWidth', 1.5);
plot(T, r_adaptive.lambda_obs, 'Color', [0.3 0.8 0.3], 'LineWidth', 1.5);
plot(T, r_pi.lambda_obs, 'm-', 'LineWidth', 1.5);
ylabel('Seizures/day'); xlabel('Days');
legend('None', 'Fixed', 'PRN', 'Adaptive', 'PI', 'Location', 'northeast');
title('Seizure Rate');
xlim([0 T(end)]); ylim(rate_ylim); grid on;

% Drug concentration
subplot(2,3,2);
plot(T, r_fixed.x, 'b-', 'LineWidth', 1.5); hold on;
plot(T, r_prn.x, 'r-', 'LineWidth', 1.5);
plot(T, r_adaptive.x, 'Color', [0.3 0.8 0.3], 'LineWidth', 1.5);
plot(T, r_pi.x, 'm-', 'LineWidth', 1.5);
ylabel('Drug Concentration'); xlabel('Days');
legend('Fixed', 'PRN', 'Adaptive', 'PI', 'Location', 'best');
title('Drug in Blood');
xlim([0 T(end)]); grid on;

% Adaptive detail
subplot(2,3,3);
plot(T, patient.lambda_base, 'k--', 'LineWidth', 1.5); hold on;
plot(T, r_adaptive.lambda_obs, 'Color', [0.3 0.8 0.3], 'LineWidth', 2);
plot(T, r_adaptive.lambda_hat, 'Color', [0.3 0.8 0.3], 'LineWidth', 1, 'LineStyle', ':');
yline(1.5*ctrl_params.omega, 'g--', '1.5\omega (skip)', 'LineWidth', 1);
yline(3*ctrl_params.omega, 'g:', '3\omega (full)', 'LineWidth', 1);
ylabel('Seizures/day'); xlabel('Days');
title('Adaptive Detail');
legend('Base', 'Observed', 'EKF est', 'Location', 'best');
xlim([0 T(end)]); ylim(rate_ylim); grid on;

% PI controller detail
subplot(2,3,4);
plot(T, patient.lambda_base, 'k--', 'LineWidth', 1.5); hold on;
plot(T, r_pi.lambda_obs, 'm-', 'LineWidth', 2);
yline(ctrl_params.omega, 'r:', '\omega', 'LineWidth', 1.5);
ylabel('Seizure Rate (sz/day)'); xlabel('Days');
title('PI Controller');
legend('Base rate', 'With PI', 'Target', 'Location', 'best');
xlim([0 T(end)]); ylim(rate_ylim); grid on;

% Bar chart: burden
subplot(2,3,5);
strat_names = categorical({'None', 'Fixed', 'PRN', 'Adapt.', 'PI'});
strat_names = reordercats(strat_names, {'None', 'Fixed', 'PRN', 'Adapt.', 'PI'});
seizures = [sum(r_none.seizure_counts), sum(r_fixed.seizure_counts), ...
            sum(r_prn.seizure_counts), sum(r_adaptive.seizure_counts), sum(r_pi.seizure_counts)];
bar(strat_names, seizures, 'FaceColor', [0.8 0.3 0.3]);
ylabel('Seizure Count'); title('Total Seizures');
grid on;

% Bar chart: drug
subplot(2,3,6);
drugs = [r_none.total_dose, r_fixed.total_dose, r_prn.total_dose, ...
         r_adaptive.total_dose, r_pi.total_dose];
bar(strat_names, drugs, 'FaceColor', [0.3 0.3 0.8]);
ylabel('Total Dose'); title('Drug Exposure');
grid on;

sgtitle(sprintf('Patient: age=%d, SOFA=%d, peak=%.0f sz/day', ...
    round(patient.age), patient.sofa, max(patient.lambda_base)));
