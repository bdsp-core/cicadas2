% S15_EKF_SENSITIVITY
% Sensitivity analysis for EKF tuning parameters.
%
% Sweeps each EKF parameter individually while holding others at defaults,
% measuring 7-day survival under the adaptive strategy for each value.
%
% Parameters swept:
%   1. Process noise q: controls bias-variance tradeoff of rate estimate
%   2. Mean-reversion time constant tau_x: how fast filter forgets deviations
%   3. Population prior peak P_pop: prior seizure rate scale
%   4. Prior floor: minimum rate the filter can estimate
%   5. Dose tier threshold omega: maps rate-to-dose breakpoints
%
% Output: survival curves and strategy comparison across parameter values.
%
% Uses: fcn_run_adaptive_custom (in functions/) for parameterized EKF runs.

clear; clc;
addpath('functions');

%% Load default parameters
[time_params, drug_params, ctrl_params, mort_params] = default_params();
T = time_params.T;
Nt = time_params.Nt;
dt = time_params.dt;

%% Settings
Npat = 300;          % patients per sweep point (balance speed vs precision)
Nruns_mort = 200;    % mortality MC runs per patient
rng(42);

patient_params.peak_rate = 10;
patient_params.drug_params = drug_params;
theta_mort = mort_params.theta;

%% Generate patient population (shared across all sweeps)
fprintf('=== EKF SENSITIVITY ANALYSIS ===\n');
fprintf('Generating %d patients...\n', Npat);
patients = cell(Npat, 1);
for p = 1:Npat
    patients{p} = fcn_generate_patient(T, patient_params);
end
fprintf('Done.\n\n');

%% Define sweep ranges
sweeps = struct();

sweeps(1).name = 'q (process noise)';
sweeps(1).param = 'q';
sweeps(1).values = [1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2];
sweeps(1).default = 1e-4;

sweeps(2).name = 'tau_x (mean-reversion, days)';
sweeps(2).param = 'tau_x';
sweeps(2).values = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0];
sweeps(2).default = 0.5;

sweeps(3).name = 'P_pop (prior peak, sz/day)';
sweeps(3).param = 'P_pop';
sweeps(3).values = [1, 2, 3, 5, 7, 10, 15, 20];
sweeps(3).default = 5;

sweeps(4).name = 'floor (prior minimum)';
sweeps(4).param = 'floor';
sweeps(4).values = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0];
sweeps(4).default = 0.1;

sweeps(5).name = 'omega (dose threshold, sz/day)';
sweeps(5).param = 'omega';
sweeps(5).values = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0];
sweeps(5).default = 2.0;

%% Compute baselines: no treatment and fixed dosing (invariant to EKF)
fprintf('Computing baseline strategies (no treatment, fixed dosing)...\n');
tic;
surv_none = zeros(Npat, 1);
surv_fixed = zeros(Npat, 1);

for p = 1:Npat
    patient = patients{p};
    dp = drug_params;
    dp.alpha = patient.alpha;
    if isfield(patient, 'C50')
        dp.patient_C50 = patient.C50;
        dp.patient_gamma = patient.gamma_hill;
    end

    % No treatment
    r_none = fcn_run_strategy('none', patient.lambda_base, dp, ctrl_params, T, dt, true);
    alive_sum = 0;
    for run = 1:Nruns_mort
        [a, ~] = fcn_mortality(r_none.lambda_obs, r_none.x, dt, theta_mort, patient.age, patient.sofa);
        alive_sum = alive_sum + a;
    end
    surv_none(p) = alive_sum / Nruns_mort;

    % Fixed dosing
    r_fixed = fcn_run_strategy('fixed', patient.lambda_base, dp, ctrl_params, T, dt, true);
    alive_sum = 0;
    for run = 1:Nruns_mort
        [a, ~] = fcn_mortality(r_fixed.lambda_obs, r_fixed.x, dt, theta_mort, patient.age, patient.sofa);
        alive_sum = alive_sum + a;
    end
    surv_fixed(p) = alive_sum / Nruns_mort;
end

baseline_none = mean(surv_none) * 100;
baseline_fixed = mean(surv_fixed) * 100;
fprintf('Baselines: None=%.1f%%, Fixed=%.1f%% (%.1fs)\n\n', ...
    baseline_none, baseline_fixed, toc);

%% Run sweeps
results = struct();

for s = 1:numel(sweeps)
    sweep = sweeps(s);
    nvals = numel(sweep.values);
    surv_adaptive = zeros(nvals, 1);

    fprintf('--- Sweep %d/%d: %s ---\n', s, numel(sweeps), sweep.name);
    tic;

    for v = 1:nvals
        val = sweep.values(v);

        % Set parameters: sweep one, hold others at default
        q_use = sweeps(1).default;
        tau_use = sweeps(2).default;
        Ppop_use = sweeps(3).default;
        floor_use = sweeps(4).default;
        omega_use = sweeps(5).default;

        switch sweep.param
            case 'q',     q_use = val;
            case 'tau_x', tau_use = val;
            case 'P_pop', Ppop_use = val;
            case 'floor', floor_use = val;
            case 'omega', omega_use = val;
        end

        surv_pat = zeros(Npat, 1);
        for p = 1:Npat
            patient = patients{p};
            dp = drug_params;
            dp.alpha = patient.alpha;
            if isfield(patient, 'C50')
                dp.patient_C50 = patient.C50;
                dp.patient_gamma = patient.gamma_hill;
            end

            surv_pat(p) = fcn_run_adaptive_custom(patient, dp, ctrl_params, T, dt, ...
                theta_mort, Nruns_mort, q_use, tau_use, Ppop_use, floor_use, omega_use);
        end

        surv_adaptive(v) = mean(surv_pat) * 100;
        fprintf('  %s = %.4g -> Survival = %.1f%%\n', sweep.param, val, surv_adaptive(v));
    end

    results(s).name = sweep.name;
    results(s).param = sweep.param;
    results(s).values = sweep.values;
    results(s).survival = surv_adaptive;
    results(s).elapsed = toc;

    fprintf('  Completed in %.1fs\n\n', results(s).elapsed);
end

%% Save results
save('ekf_sensitivity_results.mat', 'results', 'sweeps', ...
    'baseline_none', 'baseline_fixed', 'Npat', 'Nruns_mort');
fprintf('Results saved to ekf_sensitivity_results.mat\n');

%% Print summary table
fprintf('\n=====================================================\n');
fprintf('EKF SENSITIVITY ANALYSIS SUMMARY\n');
fprintf('=====================================================\n');
fprintf('Baselines: No treatment = %.1f%%, Fixed dosing = %.1f%%\n\n', ...
    baseline_none, baseline_fixed);

for s = 1:numel(results)
    fprintf('%s:\n', results(s).name);
    fprintf('  Range: [%.4g, %.4g]\n', min(results(s).values), max(results(s).values));
    fprintf('  Survival range: [%.1f%%, %.1f%%]\n', ...
        min(results(s).survival), max(results(s).survival));
    fprintf('  Spread: %.1f percentage points\n', ...
        max(results(s).survival) - min(results(s).survival));
    if all(results(s).survival > baseline_fixed)
        fprintf('  All above fixed? YES\n');
    else
        fprintf('  All above fixed? NO\n');
    end
    fprintf('\n');
end

%% FIGURE: Multi-panel sensitivity plot
figure('Position', [50 50 1400 800]); clf;

for s = 1:numel(results)
    subplot(2, 3, s);

    % Plot adaptive survival across sweep
    if strcmp(results(s).param, 'q')
        semilogx(results(s).values, results(s).survival, 'bo-', 'LineWidth', 2, 'MarkerSize', 8);
    else
        plot(results(s).values, results(s).survival, 'bo-', 'LineWidth', 2, 'MarkerSize', 8);
    end
    hold on;

    % Baseline reference lines
    yline(baseline_none, 'r--', 'No treatment', 'LineWidth', 1.5);
    yline(baseline_fixed, 'k--', 'Fixed dosing', 'LineWidth', 1.5);

    % Mark default value
    default_idx = find(abs(results(s).values - sweeps(s).default) < 1e-10);
    if ~isempty(default_idx)
        if strcmp(results(s).param, 'q')
            semilogx(results(s).values(default_idx), results(s).survival(default_idx), ...
                'rs', 'MarkerSize', 14, 'LineWidth', 2);
        else
            plot(results(s).values(default_idx), results(s).survival(default_idx), ...
                'rs', 'MarkerSize', 14, 'LineWidth', 2);
        end
    end

    xlabel(results(s).name, 'FontSize', 11);
    ylabel('7-day Survival (%)', 'FontSize', 11);
    title(sprintf('%s', results(s).name), 'FontSize', 12);
    grid on;

    % Set reasonable y-axis range
    all_vals = [results(s).survival; baseline_none; baseline_fixed];
    yl = [max(0, min(all_vals) - 5), min(100, max(all_vals) + 5)];
    ylim(yl);
end

% Summary in 6th panel
subplot(2, 3, 6);
spreads = arrayfun(@(r) max(r.survival) - min(r.survival), results);
names = {results.param};
barh(spreads);
set(gca, 'YTickLabel', names);
xlabel('Survival spread (pp)');
title('Parameter Sensitivity Ranking');
grid on;

sgtitle('EKF Parameter Sensitivity Analysis', 'FontSize', 14, 'FontWeight', 'bold');
