% S4_THRESHOLD_SWEEP
% Sweeps PRN threshold and adaptive omega to find optimal values.

clear; clc;

% Add functions to path
addpath('../functions');

%% Load default parameters
[time_params, drug_params, ctrl_params, mort_params] = default_params();
T = time_params.T;
Nt = time_params.Nt;
dt = time_params.dt;

%% Sweep settings
Npat = 200;      % Patients per threshold
Nruns = 50;       % Mortality runs per patient
use_ekf = true;

% Threshold values to test
prn_thresholds = [2, 3, 5, 7, 10, 15];
omega_values = [1, 2, 3, 5, 8, 10];

rng(42);

%% Storage
n_prn = length(prn_thresholds);
n_omega = length(omega_values);

results_prn = struct();
results_prn.threshold = prn_thresholds;
results_prn.surv_rate = zeros(n_prn, 1);
results_prn.burden = zeros(n_prn, 1);
results_prn.drug = zeros(n_prn, 1);
results_prn.surv_std = zeros(n_prn, 1);

results_omega = struct();
results_omega.omega = omega_values;
results_omega.surv_rate = zeros(n_omega, 1);
results_omega.burden = zeros(n_omega, 1);
results_omega.drug = zeros(n_omega, 1);
results_omega.surv_std = zeros(n_omega, 1);

% Also get baseline (no treatment)
results_baseline = struct();
results_baseline.surv_rate = 0;
results_baseline.burden = 0;

%% Generate patient population once
fprintf('=== THRESHOLD SWEEP ===\n');
fprintf('Patients: %d | PRN thresholds: %d | Omega values: %d\n', ...
    Npat, n_prn, n_omega);
fprintf('\n');

patient_params.peak_rate = 10;
patient_params.drug_params = drug_params;  % For patient-specific alpha
patients = cell(Npat, 1);
for p = 1:Npat
    patients{p} = fcn_generate_patient(T, patient_params);
end
fprintf('Generated %d patients\n\n', Npat);

%% Baseline (no treatment)
fprintf('Running baseline (no treatment)...\n');
tic;
surv_baseline = zeros(Npat, 1);
burden_baseline = zeros(Npat, 1);
for p = 1:Npat
    patient = patients{p};
    dp = drug_params; dp.alpha = patient.alpha;
    if isfield(patient, 'C50')
        dp.patient_C50 = patient.C50;
        dp.patient_gamma = patient.gamma_hill;
    end
    r = fcn_run_strategy('none', patient.lambda_base, dp, ctrl_params, T, dt, use_ekf);
    burden_baseline(p) = r.burden;
    
    surv_temp = 0;
    for run = 1:Nruns
        [alive, ~] = fcn_mortality(r.lambda_obs, r.x, dt, mort_params.theta, patient.age, patient.sofa);
        surv_temp = surv_temp + alive;
    end
    surv_baseline(p) = surv_temp / Nruns;
end
results_baseline.surv_rate = mean(surv_baseline) * 100;
results_baseline.burden = mean(burden_baseline);
fprintf('  Baseline: %.1f%% survival, L_cum=%.1f (%.1f sec)\n', ...
    results_baseline.surv_rate, results_baseline.burden, toc);

%% Sweep PRN thresholds
fprintf('\nSweeping PRN thresholds...\n');
for i = 1:n_prn
    tic;
    ctrl_params_temp = ctrl_params;
    ctrl_params_temp.prn_threshold = prn_thresholds(i);
    
    surv_all = zeros(Npat, 1);
    burden_all = zeros(Npat, 1);
    drug_all = zeros(Npat, 1);
    
    for p = 1:Npat
        patient = patients{p};
        dp = drug_params; dp.alpha = patient.alpha;
    if isfield(patient, 'C50')
        dp.patient_C50 = patient.C50;
        dp.patient_gamma = patient.gamma_hill;
    end
        r = fcn_run_strategy('prn', patient.lambda_base, dp, ctrl_params_temp, T, dt, use_ekf);
        burden_all(p) = r.burden;
        drug_all(p) = r.total_dose;
        
        surv_temp = 0;
        for run = 1:Nruns
            [alive, ~] = fcn_mortality(r.lambda_obs, r.x, dt, mort_params.theta, patient.age, patient.sofa);
            surv_temp = surv_temp + alive;
        end
        surv_all(p) = surv_temp / Nruns;
    end
    
    results_prn.surv_rate(i) = mean(surv_all) * 100;
    results_prn.burden(i) = mean(burden_all);
    results_prn.drug(i) = mean(drug_all);
    results_prn.surv_std(i) = std(surv_all) * 100 / sqrt(Npat);
    
    fprintf('  PRN θ=%.0f: %.1f%% survival, L_cum=%.1f, Drug=%.0f (%.1f sec)\n', ...
        prn_thresholds(i), results_prn.surv_rate(i), results_prn.burden(i), ...
        results_prn.drug(i), toc);
end

%% Sweep omega (adaptive strategy)
fprintf('\nSweeping omega (adaptive)...\n');
for i = 1:n_omega
    tic;
    ctrl_params_temp = ctrl_params;
    ctrl_params_temp.omega = omega_values(i);
    
    surv_all = zeros(Npat, 1);
    burden_all = zeros(Npat, 1);
    drug_all = zeros(Npat, 1);
    
    for p = 1:Npat
        patient = patients{p};
        dp = drug_params; dp.alpha = patient.alpha;
    if isfield(patient, 'C50')
        dp.patient_C50 = patient.C50;
        dp.patient_gamma = patient.gamma_hill;
    end
        r = fcn_run_strategy('adaptive', patient.lambda_base, dp, ctrl_params_temp, T, dt, use_ekf);
        burden_all(p) = r.burden;
        drug_all(p) = r.total_dose;
        
        surv_temp = 0;
        for run = 1:Nruns
            [alive, ~] = fcn_mortality(r.lambda_obs, r.x, dt, mort_params.theta, patient.age, patient.sofa);
            surv_temp = surv_temp + alive;
        end
        surv_all(p) = surv_temp / Nruns;
    end
    
    results_omega.surv_rate(i) = mean(surv_all) * 100;
    results_omega.burden(i) = mean(burden_all);
    results_omega.drug(i) = mean(drug_all);
    results_omega.surv_std(i) = std(surv_all) * 100 / sqrt(Npat);
    
    fprintf('  Adaptive ω=%.0f: %.1f%% survival, L_cum=%.1f, Drug=%.0f (%.1f sec)\n', ...
        omega_values(i), results_omega.surv_rate(i), results_omega.burden(i), ...
        results_omega.drug(i), toc);
end

%% Find optimal thresholds
[best_prn_surv, best_prn_idx] = max(results_prn.surv_rate);
[best_omega_surv, best_omega_idx] = max(results_omega.surv_rate);

fprintf('\n========================================\n');
fprintf('OPTIMAL THRESHOLDS\n');
fprintf('========================================\n');
fprintf('Baseline (no treatment): %.1f%% survival\n', results_baseline.surv_rate);
fprintf('\n');
fprintf('Best PRN threshold:    θ = %.0f (%.1f%% survival, +%.1f%% vs baseline)\n', ...
    prn_thresholds(best_prn_idx), best_prn_surv, best_prn_surv - results_baseline.surv_rate);
fprintf('Best adaptive omega:   ω = %.0f (%.1f%% survival, +%.1f%% vs baseline)\n', ...
    omega_values(best_omega_idx), best_omega_surv, best_omega_surv - results_baseline.surv_rate);

%% FIGURE 1: Survival vs Threshold
figure('Position', [100 100 1000 400]); clf;

surv_ylim = [min([results_prn.surv_rate; results_omega.surv_rate; results_baseline.surv_rate])-5, ...
             max([results_prn.surv_rate; results_omega.surv_rate])+5];
x_lim = [0 16];

% PRN threshold sweep
subplot(1,2,1)
errorbar(prn_thresholds, results_prn.surv_rate, results_prn.surv_std, ...
    'o-', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'r');
hold on;
yline(results_baseline.surv_rate, 'k--', 'No Treatment', 'LineWidth', 1.5);
scatter(prn_thresholds(best_prn_idx), best_prn_surv, 150, 'g', 'filled', 'MarkerEdgeColor', 'k');
xlabel('PRN Threshold'); ylabel('Survival Rate (%)');
title(sprintf('PRN: Optimal threshold = %.0f', prn_thresholds(best_prn_idx)));
grid on; legend('PRN', 'Baseline', 'Best', 'Location', 'best');
ylim(surv_ylim); xlim(x_lim);

% Adaptive omega sweep
subplot(1,2,2)
errorbar(omega_values, results_omega.surv_rate, results_omega.surv_std, ...
    'o-', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'b');
hold on;
yline(results_baseline.surv_rate, 'k--', 'No Treatment', 'LineWidth', 1.5);
scatter(omega_values(best_omega_idx), best_omega_surv, 150, 'g', 'filled', 'MarkerEdgeColor', 'k');
xlabel('Adaptive Target Rate (\omega)'); ylabel('Survival Rate (%)');
title(sprintf('Adaptive: Optimal \\omega = %.0f', omega_values(best_omega_idx)));
grid on; legend('Adaptive', 'Baseline', 'Best', 'Location', 'best');
ylim(surv_ylim); xlim(x_lim);

sgtitle(sprintf('Threshold Sweep (N=%d patients)', Npat));

%% FIGURE 2: Trade-off: Survival vs Drug Usage
figure('Position', [100 100 1000 400]); clf;

drug_xlim = [0 max([results_prn.drug; results_omega.drug])*1.1];
trade_ylim = surv_ylim;

subplot(1,2,1)
scatter(results_prn.drug, results_prn.surv_rate, 100, prn_thresholds, 'filled');
hold on;
scatter(0, results_baseline.surv_rate, 150, 'k', 's', 'filled');
xlabel('Total Drug Given'); ylabel('Survival Rate (%)');
title('PRN: Survival vs Drug');
colormap(gca, 'hot');
c = colorbar; c.Label.String = 'Threshold';
for i = 1:n_prn
    text(results_prn.drug(i)+50, results_prn.surv_rate(i), sprintf('%d', prn_thresholds(i)), 'FontSize', 8);
end
text(50, results_baseline.surv_rate, 'None', 'FontSize', 8);
grid on; xlim(drug_xlim); ylim(trade_ylim);

subplot(1,2,2)
scatter(results_omega.drug, results_omega.surv_rate, 100, omega_values, 'filled');
hold on;
scatter(0, results_baseline.surv_rate, 150, 'k', 's', 'filled');
xlabel('Total Drug Given'); ylabel('Survival Rate (%)');
title('Adaptive: Survival vs Drug');
colormap(gca, 'cool');
c = colorbar; c.Label.String = '\omega';
for i = 1:n_omega
    text(results_omega.drug(i)+50, results_omega.surv_rate(i), sprintf('%d', omega_values(i)), 'FontSize', 8);
end
text(50, results_baseline.surv_rate, 'None', 'FontSize', 8);
grid on; xlim(drug_xlim); ylim(trade_ylim);

sgtitle('Efficiency Frontier: Survival vs Drug Exposure');

%% FIGURE 3: Trade-off: Survival vs Burden
figure('Position', [100 100 600 400]); clf;

scatter(results_prn.burden, results_prn.surv_rate, 100, 'r', 'filled', 'MarkerFaceAlpha', 0.7);
hold on;
scatter(results_omega.burden, results_omega.surv_rate, 100, 'b', 'filled', 'MarkerFaceAlpha', 0.7);
scatter(results_baseline.burden, results_baseline.surv_rate, 150, 'k', 's', 'filled');

xlabel('Seizure Burden (L_{cum})');
ylabel('Survival Rate (%)');
title('Survival vs Seizure Burden');
legend('PRN', 'Adaptive', 'No Treatment', 'Location', 'best');
grid on;

%% Save results
save('threshold_sweep_results.mat', 'results_prn', 'results_omega', 'results_baseline', ...
     'prn_thresholds', 'omega_values', 'Npat', 'Nruns');
fprintf('\nResults saved to threshold_sweep_results.mat\n');
