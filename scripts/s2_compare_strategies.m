% S2_COMPARE_STRATEGIES
% Compares all treatment strategies on one patient.
% Pill-based: none, fixed, prn, prn_ekf, adaptive
% Infusion-based: fixed_infusion, pi

clear; clc;
addpath('../functions');

%% Load parameters
[time_params, drug_params, ctrl_params, mort_params] = default_params();
T = time_params.T;
Nt = time_params.Nt;
dt = time_params.dt;

%% Settings
use_ekf = true;

%% Generate patient
patient_params.peak_rate = 12;  % sz/day
patient_params.drug_params = drug_params;

% rng(42);  % uncomment for reproducibility
rng('shuffle');

patient = fcn_generate_patient(T, patient_params);

drug_params.alpha = patient.alpha;
if isfield(patient, 'C50')
    drug_params.patient_C50 = patient.C50;
    drug_params.patient_gamma = patient.gamma_hill;
end

fprintf('=== Patient ===\n');
fprintf('Age: %.0f years\n', patient.age);
fprintf('SOFA: %d\n', patient.sofa);
fprintf('Alpha: %.4f\n', patient.alpha);
fprintf('Peak rate: %.1f sz/day\n', max(patient.lambda_base));
fprintf('omega (target): %.1f\n', ctrl_params.omega);
fprintf('PRN threshold: %.1f\n', ctrl_params.prn_threshold);
fprintf('\n');

%% Run all strategies
strategies = {'none', 'fixed', 'prn', 'prn_ekf', 'adaptive', 'pi'};
results = struct();

for i = 1:length(strategies)
    strat = strategies{i};
    r = fcn_run_strategy(strat, patient.lambda_base, drug_params, ctrl_params, T, dt, use_ekf);
    results.(strat) = r;
end

%% Print comparison table
fprintf('=== Strategy Comparison ===\n');
fprintf('%-12s %8s %10s %10s %8s\n', 'Strategy', 'L_cum', 'Seizures', 'Total Dose', 'Doses');
fprintf('%s\n', repmat('-', 1, 55));
for i = 1:length(strategies)
    strat = strategies{i};
    r = results.(strat);
    n_doses = sum(r.doses_given > 0);
    if r.is_infusion
        n_doses_str = 'infusion';
    else
        n_doses_str = sprintf('%d', n_doses);
    end
    fprintf('%-12s %8.1f %10d %10.0f %8s\n', strat, r.burden, sum(r.seizure_counts), r.total_dose, n_doses_str);
end

%% Mortality comparison
Nruns = 1000;
surv = zeros(length(strategies), 1);

for run = 1:Nruns
    for i = 1:length(strategies)
        strat = strategies{i};
        r = results.(strat);
        [alive, ~] = fcn_mortality(r.lambda_obs, r.x, dt, mort_params.theta, patient.age, patient.sofa);
        surv(i) = surv(i) + alive;
    end
end

fprintf('\n=== Survival (N=%d runs) ===\n', Nruns);
for i = 1:length(strategies)
    fprintf('%-12s: %.1f%%\n', strategies{i}, surv(i) / Nruns * 100);
end

%% FIGURE 1: Seizure Rate Comparison
figure('Position', [100 100 1400 900]); clf;

rate_ylim = [0 max(patient.lambda_base)*1.1];

% Row 1: Pill-based strategies
subplot(2,3,1);
plot(T, patient.lambda_base, 'k-', 'LineWidth', 2); hold on;
yline(ctrl_params.prn_threshold, 'm--', 'PRN thresh', 'LineWidth', 1);
title(sprintf('No Treatment (Sz=%d)', sum(results.none.seizure_counts)));
ylabel('Seizures/day'); xlabel('Days');
legend('Base rate', 'Location', 'best');
xlim([0 T(end)]); ylim(rate_ylim); grid on;

subplot(2,3,2);
plot(T, patient.lambda_base, 'k--', 'LineWidth', 1); hold on;
plot(T, results.fixed.lambda_obs, 'b-', 'LineWidth', 2);
title(sprintf('Fixed (Sz=%d, Doses=%d)', sum(results.fixed.seizure_counts), sum(results.fixed.doses_given>0)));
xlabel('Days');
xlim([0 T(end)]); ylim(rate_ylim); grid on;

subplot(2,3,3);
plot(T, patient.lambda_base, 'k--', 'LineWidth', 1); hold on;
plot(T, results.prn.lambda_obs, 'r-', 'LineWidth', 2);
yline(ctrl_params.prn_threshold, 'm--', 'LineWidth', 1);
title(sprintf('PRN-rolling (Sz=%d, Doses=%d)', sum(results.prn.seizure_counts), sum(results.prn.doses_given>0)));
xlabel('Days');
xlim([0 T(end)]); ylim(rate_ylim); grid on;

% Row 2: EKF-based strategies
subplot(2,3,4);
plot(T, patient.lambda_base, 'k--', 'LineWidth', 1); hold on;
plot(T, results.prn_ekf.lambda_obs, 'c-', 'LineWidth', 2);
plot(T, results.prn_ekf.lambda_hat, 'c:', 'LineWidth', 1);
yline(ctrl_params.prn_threshold, 'm--', 'LineWidth', 1);
title(sprintf('PRN-EKF (Sz=%d, Doses=%d)', sum(results.prn_ekf.seizure_counts), sum(results.prn_ekf.doses_given>0)));
ylabel('Seizures/day'); xlabel('Days');
legend('Base', 'Observed', 'EKF est', 'Location', 'best');
xlim([0 T(end)]); ylim(rate_ylim); grid on;

subplot(2,3,5);
plot(T, patient.lambda_base, 'k--', 'LineWidth', 1); hold on;
plot(T, results.adaptive.lambda_obs, 'Color', [0.5 0 0.5], 'LineWidth', 2);
plot(T, results.adaptive.lambda_hat, 'Color', [0.7 0.3 0.7], 'LineWidth', 1, 'LineStyle', ':');
yline(0.5*ctrl_params.omega, 'g:', '0.5\omega (skip)', 'LineWidth', 1);
yline(ctrl_params.omega, 'g--', '\omega (low)', 'LineWidth', 1);
yline(2*ctrl_params.omega, 'g-.', '2\omega (high)', 'LineWidth', 1);
title(sprintf('Adaptive (Sz=%d, Doses=%d)', sum(results.adaptive.seizure_counts), sum(results.adaptive.doses_given>0)));
xlabel('Days');
xlim([0 T(end)]); ylim(rate_ylim); grid on;

subplot(2,3,6);
plot(T, patient.lambda_base, 'k--', 'LineWidth', 1); hold on;
plot(T, results.pi.lambda_obs, 'g-', 'LineWidth', 2);
yline(ctrl_params.omega, 'g--', '\omega', 'LineWidth', 1);
title(sprintf('PI Controller (Sz=%d)', sum(results.pi.seizure_counts)));
xlabel('Days');
xlim([0 T(end)]); ylim(rate_ylim); grid on;

sgtitle(sprintf('Strategy Comparison: Age=%d, SOFA=%d, Peak=%.0f sz/day', ...
    round(patient.age), patient.sofa, max(patient.lambda_base)));

%% FIGURE 2: Drug Concentration
figure('Position', [100 100 1000 400]); clf;

subplot(1,2,1);
plot(T, results.fixed.x, 'b-', 'LineWidth', 1.5); hold on;
plot(T, results.prn.x, 'r-', 'LineWidth', 1.5);
plot(T, results.prn_ekf.x, 'c-', 'LineWidth', 1.5);
plot(T, results.adaptive.x, 'Color', [0.5 0 0.5], 'LineWidth', 1.5);
xlabel('Days'); ylabel('Drug Concentration');
title('Pill Strategies');
legend('Fixed', 'PRN-rolling', 'PRN-EKF', 'Adaptive', 'Location', 'best');
xlim([0 T(end)]); grid on;

subplot(1,2,2);
plot(T, results.pi.x, 'g-', 'LineWidth', 1.5);
xlabel('Days'); ylabel('Drug Concentration');
title('PI Controller (Infusion)');
xlim([0 T(end)]); grid on;

%% FIGURE 3: Bar Chart Summary
figure('Position', [100 100 800 400]); clf;

strat_names = categorical({'None', 'Fixed', 'PRN', 'PRN-EKF', 'Adaptive', 'PI'});
strat_names = reordercats(strat_names, {'None', 'Fixed', 'PRN', 'PRN-EKF', 'Adaptive', 'PI'});

burdens = [results.none.burden, results.fixed.burden, results.prn.burden, ...
           results.prn_ekf.burden, results.adaptive.burden, results.pi.burden];
survivals = surv / Nruns * 100;

subplot(1,2,1);
bar(strat_names, burdens, 'FaceColor', [0.8 0.2 0.2]);
ylabel('L_{cum} (expected seizures)');
title('Seizure Burden');
grid on;

subplot(1,2,2);
bar(strat_names, survivals, 'FaceColor', [0.2 0.6 0.2]);
ylabel('Survival %');
title(sprintf('Survival (%d runs)', Nruns));
ylim([0 100]);
grid on;

sgtitle('Strategy Performance Summary');
