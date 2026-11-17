% S13_CLOSED_LOOP_PERSONALIZED
% Path B: Real-time adaptive omega based on accumulated harm.
%
% Instead of fixing omega at admission, the controller adjusts omega
% dynamically during the 7-day ICU stay based on:
%   - Accumulated seizure burden L_cum(t) so far
%   - Accumulated drug exposure X_cum(t) so far
%   - Patient covariates (age, SOFA)
%   - Harm profile theta
%
% The idea: if a patient has already accumulated a lot of drug,
% raise omega (back off). If seizures are accumulating, lower omega
% (get more aggressive). This is real-time cost minimization.
%
% Compares: uniform omega=2, fixed personalized omega*, dynamic omega(t)

clear; clc;
addpath('functions');

%% Load parameters
[time_params, drug_params, ctrl_params, mort_params] = default_params();
T = time_params.T;
Nt = time_params.Nt;
dt = time_params.dt;

%% Settings
Npat = 200;
Nruns = 200;
use_ekf = true;

rng(42);
patient_params.peak_rate = 10;
patient_params.drug_params = drug_params;

%% Regimes
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

fprintf('=== CLOSED-LOOP PERSONALIZED OMEGA ===\n');
fprintf('Patients: %d | Mortality runs: %d\n\n', Npat, Nruns);

%% Generate patients
patients = cell(Npat, 1);
for p = 1:Npat
    patients{p} = fcn_generate_patient(T, patient_params);
end
fprintf('Generated %d patients.\n\n', Npat);

%% Dynamic omega controller
% Runs the 5-tier adaptive strategy but updates omega every 24 hours
% based on the accumulated harm balance.
%
% Logic at each daily checkpoint:
%   marginal_sz_harm  = sz_coeff * mean_lambda_hat_today / L_norm
%   marginal_drug_harm = drug_coeff * mean_x_today / X_norm
%   ratio = marginal_drug_harm / (marginal_sz_harm + eps)
%
%   If ratio > 1.5: we're giving too much drug -> raise omega (back off)
%   If ratio < 0.5: seizures dominating -> lower omega (get aggressive)
%   Else: hold current omega
%
% omega is clamped to [0.5, 10]

function results = run_dynamic_adaptive(lambda_base, drug_params, ctrl_params, ...
        T, dt, theta, age, sofa)
    
    Nt = numel(lambda_base);
    k_e = drug_params.k_e;
    dosing_interval_days = drug_params.dosing_interval / 24;
    
    % Compute harm coefficients from theta
    sz_coeff = theta(3) + theta(4) * sofa / 10;
    drug_coeff = theta(5) + theta(6) * age / 65;
    L_norm = 35; X_norm = 5000;
    
    % Build PD struct
    pd_struct.mode = 'hill';
    pd_struct.C50 = drug_params.C50;
    pd_struct.gamma = drug_params.gamma_hill;
    if isfield(drug_params, 'patient_C50')
        pd_struct.C50 = drug_params.patient_C50;
    end
    if isfield(drug_params, 'patient_gamma')
        pd_struct.gamma = drug_params.patient_gamma;
    end
    
    % EKF parameters
    tau_x = 0.5; q = 1e-4;
    phi = exp(-dt / tau_x);
    ekf_x = 0; ekf_P = 1;
    
    % Population prior
    r_pop = 1.5; kappa_pop = 0.3; peak_pop = 5;
    shape_pop = exp(-kappa_pop*T) .* (1 - exp(-r_pop*T));
    mu_prior = max(0.1, peak_pop * shape_pop / max(shape_pop));
    
    % Find dosing times
    dose_indices = [];
    for i = 1:Nt
        if mod(T(i), dosing_interval_days) < dt/2 || ...
           (dosing_interval_days - mod(T(i), dosing_interval_days)) < dt/2
            dose_indices = [dose_indices, i];
        end
    end
    
    % Initialize
    x = zeros(1, Nt);
    lambda_obs = zeros(1, Nt);
    lambda_hat = zeros(1, Nt);
    doses_given = zeros(1, Nt);
    seizure_counts = zeros(1, Nt);
    omega_history = zeros(1, Nt);
    
    % Starting omega: use harm ratio to pick initial value
    % If drug harm >> sz harm, start conservative (high omega)
    % If sz harm >> drug harm, start aggressive (low omega)
    harm_ratio = drug_coeff / max(sz_coeff, 0.01);
    omega = max(0.5, min(10, 2 * harm_ratio));
    
    last_update_day = 0;
    daily_lambda_sum = 0;
    daily_x_sum = 0;
    daily_count = 0;
    
    %% Main loop
    for k = 1:Nt
        % Observed rate
        lambda_obs(k) = fcn_observed_rate(lambda_base(k), x(k), pd_struct);
        seizure_counts(k) = poissrnd(lambda_obs(k) * dt);
        
        % EKF
        ekf_x = phi * ekf_x;
        ekf_P = phi^2 * ekf_P + q;
        lam_pred = mu_prior(k) * exp(ekf_x);
        yhat = lam_pred * dt;
        H = yhat; R = max(yhat, 1e-3);
        K = (ekf_P * H) / (H * ekf_P * H + R);
        ekf_x = ekf_x + K * (seizure_counts(k) - yhat);
        ekf_P = (1 - K*H) * ekf_P * (1 - K*H)' + K * R * K';
        lambda_hat(k) = mu_prior(k) * exp(ekf_x);
        
        % Track daily stats for omega update
        daily_lambda_sum = daily_lambda_sum + lambda_hat(k);
        daily_x_sum = daily_x_sum + x(k);
        daily_count = daily_count + 1;
        
        % Update omega every 24 hours
        current_day = floor(T(k));
        if current_day > last_update_day && daily_count > 0
            mean_lambda_today = daily_lambda_sum / daily_count;
            mean_x_today = daily_x_sum / daily_count;
            
            marginal_sz = sz_coeff * mean_lambda_today / L_norm;
            marginal_drug = drug_coeff * mean_x_today / X_norm;
            
            ratio = marginal_drug / max(marginal_sz, 1e-6);
            
            if ratio > 1.5
                omega = min(10, omega * 1.3);   % back off
            elseif ratio < 0.5
                omega = max(0.5, omega * 0.7);  % get aggressive
            end
            % else: hold current omega
            
            last_update_day = current_day;
            daily_lambda_sum = 0;
            daily_x_sum = 0;
            daily_count = 0;
        end
        omega_history(k) = omega;
        
        % 5-tier dosing with current (dynamic) omega
        if ismember(k, dose_indices)
            if lambda_hat(k) < 0.5 * omega
                doses_given(k) = 0;
            elseif lambda_hat(k) < omega
                doses_given(k) = ctrl_params.dose_low;
            elseif lambda_hat(k) < 1.5 * omega
                doses_given(k) = ctrl_params.dose_intermediate;
            elseif lambda_hat(k) < 2 * omega
                doses_given(k) = ctrl_params.dose_medium;
            else
                doses_given(k) = ctrl_params.dose_high;
            end
        end
        
        % PK update
        if k < Nt
            x(k+1) = x(k) * k_e + doses_given(k);
        end
    end
    
    % Package
    results.lambda_obs = lambda_obs;
    results.lambda_hat = lambda_hat;
    results.x = x;
    results.doses_given = doses_given;
    results.omega_history = omega_history;
    results.expected_burden = sum(lambda_obs) * dt;
    results.drug_exposure = sum(x) * dt;
    results.total_dose = sum(doses_given);
end

%% Run three strategies per regime: uniform, fixed-optimal, dynamic
% Fixed-optimal omega per regime (from s10 results):
omega_optimal = [0.5, 1.5, 0.5, 8.0];  % R1, R2, R3, R4

fprintf('Running strategies per regime...\n');
tic;

% Storage: surv(regime, strategy) where strategy = 1:uniform, 2:fixed-opt, 3:dynamic
surv_results = zeros(n_regimes, 3);
strat_names = {'Uniform (\omega=2)', 'Fixed \omega^*', 'Dynamic \omega(t)'};

for reg = 1:n_regimes
    fprintf('  Regime %d: %s\n', reg, regimes(reg).name);
    theta = regimes(reg).theta;
    
    surv_pat = zeros(Npat, 3);
    
    for p = 1:Npat
        patient = patients{p};
        dp = drug_params;
        dp.alpha = patient.alpha;
        if isfield(patient, 'C50')
            dp.patient_C50 = patient.C50;
            dp.patient_gamma = patient.gamma_hill;
        end
        
        % Strategy 1: Uniform omega=2
        cp1 = ctrl_params; cp1.omega = 2;
        r1 = fcn_run_strategy('adaptive', patient.lambda_base, dp, cp1, T, dt, true);
        
        % Strategy 2: Fixed optimal omega for this regime
        cp2 = ctrl_params; cp2.omega = omega_optimal(reg);
        if isinf(omega_optimal(reg))
            r2 = fcn_run_strategy('none', patient.lambda_base, dp, cp2, T, dt, true);
        else
            r2 = fcn_run_strategy('adaptive', patient.lambda_base, dp, cp2, T, dt, true);
        end
        
        % Strategy 3: Dynamic omega controller
        r3 = run_dynamic_adaptive(patient.lambda_base, dp, ctrl_params, ...
            T, dt, theta, patient.age, patient.sofa);
        
        % Evaluate mortality for all three
        strats = {r1, r2, r3};
        for si = 1:3
            alive = 0;
            for run = 1:Nruns
                [a,~] = fcn_mortality(strats{si}.lambda_obs, strats{si}.x, ...
                    dt, theta, patient.age, patient.sofa);
                alive = alive + a;
            end
            surv_pat(p, si) = alive / Nruns;
        end
    end
    
    surv_results(reg, :) = mean(surv_pat) * 100;
    fprintf('    Done (%.1fs)\n', toc);
end

%% Print results
fprintf('\n=====================================================\n');
fprintf('CLOSED-LOOP PERSONALIZED: Uniform vs Fixed-Opt vs Dynamic\n');
fprintf('=====================================================\n');
fprintf('%-22s %12s %12s %12s\n', 'Regime', 'Uniform', 'Fixed w*', 'Dynamic w(t)');
fprintf('%s\n', repmat('-', 1, 60));
for reg = 1:n_regimes
    [best_val, best_idx] = max(surv_results(reg,:));
    fprintf('%-22s', regimes(reg).name);
    for si = 1:3
        if si == best_idx
            fprintf(' %9.1f%% *', surv_results(reg,si));
        else
            fprintf(' %10.1f%%', surv_results(reg,si));
        end
    end
    fprintf('\n');
end

%% FIGURE 1: Bar comparison
figure('Position', [50 50 900 500]); clf;

b = bar(surv_results);
b(1).FaceColor = [0.5 0.5 0.5];
b(2).FaceColor = [0.2 0.7 0.2];
b(3).FaceColor = [0.2 0.4 0.8];
set(gca, 'XTickLabel', {regimes.name});
ylabel('Survival (%)');
legend(strat_names, 'Location', 'best');
title('Uniform vs Fixed-Optimal vs Dynamic Omega');
ylim([0 100]); grid on;

for reg = 1:n_regimes
    for si = 1:3
        improvement = surv_results(reg, si) - surv_results(reg, 1);
        if abs(improvement) > 0.5
            x_pos = reg + (si - 2) * 0.22;
            text(x_pos, surv_results(reg,si) + 1, sprintf('%+.1f', improvement), ...
                'HorizontalAlignment', 'center', 'FontSize', 8, 'FontWeight', 'bold');
        end
    end
end

%% FIGURE 2: Example patient omega trajectory
figure('Position', [50 50 1200 400]); clf;

% Run dynamic on one example patient per regime
for reg = 1:n_regimes
    subplot(1, n_regimes, reg);
    
    patient = patients{1};
    dp = drug_params;
    dp.alpha = patient.alpha;
    if isfield(patient, 'C50')
        dp.patient_C50 = patient.C50;
        dp.patient_gamma = patient.gamma_hill;
    end
    
    r3 = run_dynamic_adaptive(patient.lambda_base, dp, ctrl_params, ...
        T, dt, regimes(reg).theta, patient.age, patient.sofa);
    
    yyaxis left;
    plot(T, patient.lambda_base, 'k-', 'LineWidth', 1.5); hold on;
    plot(T, r3.lambda_obs, 'b-', 'LineWidth', 1);
    ylabel('Seizure rate (sz/day)');
    
    yyaxis right;
    plot(T, r3.omega_history, 'r-', 'LineWidth', 2);
    ylabel('\omega(t)');
    ylim([0 12]);
    
    xlabel('Days');
    title(regimes(reg).name);
    grid on;
end
sgtitle('Dynamic \omega(t) Trajectory for Example Patient');

%% Save
save('closed_loop_personalized_results.mat', 'surv_results', ...
    'regimes', 'omega_optimal', 'Npat', 'Nruns');
fprintf('\nResults saved to closed_loop_personalized_results.mat\n');

fprintf('\n=== KEY FINDINGS ===\n');
fprintf('Three levels of treatment personalization:\n');
fprintf('  1. Uniform omega=2 (current standard)\n');
fprintf('  2. Fixed omega* per regime (s10 result)\n');
fprintf('  3. Dynamic omega(t) that adapts in real time (this script)\n\n');
for reg = 1:n_regimes
    fprintf('  %s: Uniform=%.1f%%, Fixed w*=%.1f%%, Dynamic=%.1f%%\n', ...
        regimes(reg).name, surv_results(reg,1), surv_results(reg,2), surv_results(reg,3));
end
fprintf('\nDynamic omega should match or slightly beat fixed omega*\n');
fprintf('because it can adapt to individual disease trajectory shape.\n');
