% S9_GFORMULA_CAUSAL
% Parametric g-formula for causal treatment effect estimation.
% Full CICADAS alignment:
%   - CICADAS SDE patients (Eq 12-14) when use_cicadas_dgp=true
%   - Full 8-feature treatment assignment (Eq 22)
%   - G-formula counterfactual simulation (Eq 8-9)
% Steps:
%   1. Simulate RCT (ground truth)
%   2. Simulate observational data (confounding by indication)
%   3. Naive analysis (biased)
%   4. G-formula correction (debiased)
%   5. Compare all three
% Reference: CICADAS Sections 3.3, 4.1, 4.4, 4.5, 5.7.

clear; clc;
addpath('functions');

%% Load parameters
[time_params, drug_params, ctrl_params, mort_params] = default_params();
T = time_params.T;
Nt = time_params.Nt;
dt = time_params.dt;

%% Settings
Npat = 1000;
Nruns_mort = 500;
use_ekf = true;
use_cicadas_dgp = false;   % true = CICADAS SDE patients, false = gamma-esque

rng(42);

patient_params.peak_rate = 10;
patient_params.drug_params = drug_params;

%% Mortality theta for this analysis
% Calibrated separately for CICADAS SDE vs gamma-esque patients.
% SDE patients have capped peak intensity (carrying capacity H) and
% mean-reverting decay, producing lower L_cum on average -> need stronger theta.
if use_cicadas_dgp
    theta_mort = [-4, 0.8, 3.0, 0.8, 0.8, 0.3];  % stronger harm for SDE
else
    theta_mort = [-5, 0.5, 1.5, 0.5, 0.5, 0.2];   % original for gamma-esque
end

%% Step 1: Generate patient population
fprintf('=== G-FORMULA CAUSAL INFERENCE ===\n');
fprintf('Patients: %d\n\n', Npat);

patients = cell(Npat, 1);
for p = 1:Npat
    if use_cicadas_dgp
        patients{p} = fcn_generate_patient_cicadas(T, patient_params);
    else
        patients{p} = fcn_generate_patient(T, patient_params);
    end
end
if use_cicadas_dgp
    fprintf('Generated %d CICADAS SDE patients.\n', Npat);
else
    fprintf('Generated %d gamma-esque patients.\n', Npat);
end

%% Step 2: Simulate RCT (ground truth)
% Random 50/50 assignment, no censoring, no confounding.
% This is what we'd see in a perfect randomized trial.
fprintf('\n--- SIMULATING RCT ---\n');
tic;

rng(100);
rct_assignment = rand(Npat, 1) > 0.5;  % 1 = treated, 0 = untreated

surv_rct_treated = zeros(Npat, 1);
surv_rct_untreated = zeros(Npat, 1);
n_rct_treated = sum(rct_assignment);
n_rct_untreated = sum(~rct_assignment);

for p = 1:Npat
    patient = patients{p};
    dp = drug_params;
    dp.alpha = patient.alpha;
    if isfield(patient, 'C50')
        dp.patient_C50 = patient.C50;
        dp.patient_gamma = patient.gamma_hill;
    end
    
    if rct_assignment(p)
        r = fcn_run_strategy('adaptive', patient.lambda_base, dp, ctrl_params, T, dt, use_ekf);
    else
        r = fcn_run_strategy('none', patient.lambda_base, dp, ctrl_params, T, dt, use_ekf);
    end
    
    alive_sum = 0;
    for run = 1:Nruns_mort
        [a, ~] = fcn_mortality(r.lambda_obs, r.x, dt, theta_mort, patient.age, patient.sofa);
        alive_sum = alive_sum + a;
    end
    
    if rct_assignment(p)
        surv_rct_treated(p) = alive_sum / Nruns_mort;
    else
        surv_rct_untreated(p) = alive_sum / Nruns_mort;
    end
end

rct_surv_treated = mean(surv_rct_treated(rct_assignment)) * 100;
rct_surv_untreated = mean(surv_rct_untreated(~rct_assignment)) * 100;
rct_ate = rct_surv_treated - rct_surv_untreated;

fprintf('RCT results: Treated=%.1f%%, Untreated=%.1f%%, ATE=%.1f%% (%.1fs)\n', ...
    rct_surv_treated, rct_surv_untreated, rct_ate, toc);

%% Step 3: Simulate observational data (confounding by indication)
% Sicker patients preferentially receive treatment.
% Treatment assignment based on early disease severity (CICADAS Section 4.5).
fprintf('\n--- SIMULATING OBSERVATIONAL DATA ---\n');
tic;

% Compute early disease features for treatment assignment
% CICADAS Eq 22: full 8-feature logistic model for confounding by indication
early_steps = min(round(1.0 / dt), Nt);  % first 1 day
half_steps = floor(early_steps / 2);
obs_assignment = false(Npat, 1);
obs_treat_prob = zeros(Npat, 1);

for p = 1:Npat
    patient = patients{p};
    L_vec = patient.lambda_base(1:early_steps);
    
    % --- All 8 CICADAS features (Eq 22) ---
    % All features normalized to O(1) range for stable logit
    
    % beta1: mean early severity (normalized by peak_rate)
    L_early_mean = mean(L_vec);
    feat_mean = L_early_mean / patient_params.peak_rate;  % ~0-2
    
    % beta2: max early severity (normalized by peak_rate)
    L_max = max(L_vec);
    feat_max = L_max / (2 * patient_params.peak_rate);    % ~0-2
    
    % beta3: relative growth rate (log-compressed to tame SDE explosions)
    raw_growth = (L_vec(end) - L_vec(1)) / (L_vec(1) + 0.1);
    feat_growth = log1p(max(0, raw_growth));               % ~0-5, compressed
    
    % beta4: linear slope (normalized by L_max to get fractional rate)
    p_fit = polyfit((1:early_steps)', L_vec(:), 1);
    L_slope = p_fit(1);
    feat_slope = L_slope * early_steps / max(L_max, 0.1); % ~0-2 (total rise / peak)
    
    % beta5: acceleration (normalized by L_max)
    if half_steps > 1
        p1 = polyfit((1:half_steps)', L_vec(1:half_steps)', 1);
        p2 = polyfit((half_steps+1:early_steps)', L_vec(half_steps+1:end)', 1);
        L_accel = p2(1) - p1(1);
        feat_accel = L_accel * early_steps / max(L_max, 0.1);  % ~-1 to 1
    else
        feat_accel = 0;
    end
    
    % beta6: SOFA score (0-2.4)
    sofa = patient.sofa;
    
    % beta7: Age (0-1.5)
    age = patient.age;
    
    % Logistic model (CICADAS Eq 22)
    % Coefficients scaled by DGP: SDE needs stronger severity terms
    % to overcome noisy early features
    if use_cicadas_dgp
        z_treat = -1.5 ...                            % beta0: lower baseline
            + 1.5  * feat_mean ...                    % beta1: mean severity (stronger)
            + 1.2  * feat_max ...                     % beta2: peak severity (stronger)
            + 0.6  * feat_growth ...                  % beta3: growth
            + 0.4  * feat_slope ...                   % beta4: slope
            + 0.3  * feat_accel ...                   % beta5: acceleration
            + 0.5  * (sofa / 10) ...                  % beta6: SOFA
            - 0.3  * (age / 65) ...                   % beta7: age
            + randn() * 0.3;                          % eta
    else
        z_treat = -0.5 ...                            % beta0: original baseline
            + 0.8  * feat_mean ...                    % beta1
            + 0.6  * feat_max ...                     % beta2
            + 0.4  * feat_growth ...                  % beta3
            + 0.3  * feat_slope ...                   % beta4
            + 0.3  * feat_accel ...                   % beta5
            + 0.5  * (sofa / 10) ...                  % beta6
            - 0.3  * (age / 65) ...                   % beta7
            + randn() * 0.3;                          % eta
    end
    
    obs_treat_prob(p) = 1 / (1 + exp(-z_treat));
    obs_assignment(p) = rand() < obs_treat_prob(p);
end

n_obs_treated = sum(obs_assignment);
n_obs_untreated = sum(~obs_assignment);
pct_treated = n_obs_treated / Npat * 100;

fprintf('Observational: %d treated (%.1f%%), %d untreated\n', ...
    n_obs_treated, pct_treated, n_obs_untreated);

% Safety: need both groups for valid naive comparison
if n_obs_untreated < 10 || n_obs_treated < 10
    warning('Treatment assignment too imbalanced (%.0f%% treated). Adjust beta0.', pct_treated);
end

% Verify confounding: treated patients should be sicker
mean_severity_treated = mean(cellfun(@(p) max(p.lambda_base), patients(obs_assignment)));
mean_severity_untreated = mean(cellfun(@(p) max(p.lambda_base), patients(~obs_assignment)));
severity_ratio = mean_severity_treated / mean_severity_untreated;
fprintf('Confounding check: mean peak severity treated=%.1f, untreated=%.1f (ratio=%.2f)\n', ...
    mean_severity_treated, mean_severity_untreated, severity_ratio);

% Run strategies and compute naive survival
surv_obs_treated = zeros(Npat, 1);
surv_obs_untreated = zeros(Npat, 1);

for p = 1:Npat
    patient = patients{p};
    dp = drug_params;
    dp.alpha = patient.alpha;
    if isfield(patient, 'C50')
        dp.patient_C50 = patient.C50;
        dp.patient_gamma = patient.gamma_hill;
    end
    
    if obs_assignment(p)
        r = fcn_run_strategy('adaptive', patient.lambda_base, dp, ctrl_params, T, dt, use_ekf);
    else
        r = fcn_run_strategy('none', patient.lambda_base, dp, ctrl_params, T, dt, use_ekf);
    end
    
    alive_sum = 0;
    for run = 1:Nruns_mort
        [a, ~] = fcn_mortality(r.lambda_obs, r.x, dt, theta_mort, patient.age, patient.sofa);
        alive_sum = alive_sum + a;
    end
    
    if obs_assignment(p)
        surv_obs_treated(p) = alive_sum / Nruns_mort;
    else
        surv_obs_untreated(p) = alive_sum / Nruns_mort;
    end
end

naive_surv_treated = mean(surv_obs_treated(obs_assignment)) * 100;
naive_surv_untreated = mean(surv_obs_untreated(~obs_assignment)) * 100;
naive_ate = naive_surv_treated - naive_surv_untreated;

fprintf('Naive observational: Treated=%.1f%%, Untreated=%.1f%%, ATE=%.1f%% (%.1fs)\n', ...
    naive_surv_treated, naive_surv_untreated, naive_ate, toc);
fprintf('  (Compare RCT ATE=%.1f%% -- bias = %.1f percentage points)\n', ...
    rct_ate, naive_ate - rct_ate);

%% Step 4: G-formula estimation
% The g-formula corrects confounding by:
%   1. Estimating natural history parameters for ALL patients (inverting PKPD for treated)
%   2. Fitting the population distribution of disease parameters
%   3. Simulating counterfactual outcomes for EVERYONE under both strategies
%   4. Comparing survival free from confounding
%
% This mirrors CICADAS Section 3.3, Equation 9.

fprintf('\n--- G-FORMULA ESTIMATION ---\n');
tic;

% Step 4a: Estimate disease parameters for every patient from observations
fprintf('Estimating disease parameters from observational data...\n');

est_Ap = zeros(Npat, 1);
est_r = zeros(Npat, 1);
est_kappa = zeros(Npat, 1);
est_lambda_base = zeros(Npat, Nt);

for p = 1:Npat
    patient = patients{p};
    dp = drug_params;
    dp.alpha = patient.alpha;
    if isfield(patient, 'C50')
        dp.patient_C50 = patient.C50;
        dp.patient_gamma = patient.gamma_hill;
    end
    
    if obs_assignment(p)
        % Treated: run strategy to get dosing, then invert PKPD
        r = fcn_run_strategy('adaptive', patient.lambda_base, dp, ctrl_params, T, dt, use_ekf);
        
        if strcmpi(drug_params.pd_mode, 'hill')
            [params_est, fit_info] = fcn_estimate_natural_history(r.seizure_counts, T, dt, ...
                'peak_rate', patient_params.peak_rate, ...
                'k_e', dp.k_e, ...
                'doses_given', r.doses_given, ...
                'pd_mode', 'hill', ...
                'C50', dp.patient_C50, ...
                'gamma_hill', dp.patient_gamma);
        else
            [params_est, fit_info] = fcn_estimate_natural_history(r.seizure_counts, T, dt, ...
                'peak_rate', patient_params.peak_rate, ...
                'k_e', dp.k_e, ...
                'doses_given', r.doses_given, ...
                'pd_mode', 'linear', ...
                'alpha', dp.alpha);
        end
    else
        % Untreated: observe directly
        counts = poissrnd(patient.lambda_base * dt);
        [params_est, fit_info] = fcn_estimate_natural_history(counts, T, dt, ...
            'peak_rate', patient_params.peak_rate);
    end
    
    est_Ap(p) = params_est.A_p;
    est_r(p) = params_est.r;
    est_kappa(p) = params_est.kappa;
    est_lambda_base(p,:) = fit_info.lambda_base_hat;
    
    if mod(p, 200) == 0
        fprintf('  %d/%d estimated (%.1fs)\n', p, Npat, toc);
    end
end
fprintf('Parameter estimation done (%.1fs).\n', toc);

% Step 4b: G-formula Monte Carlo simulation
% For EVERY patient, simulate outcomes under BOTH strategies using their
% estimated lambda_base. This breaks the confounding because we're no longer
% conditioning on the biased treatment assignment.
fprintf('\nRunning g-formula counterfactual simulation...\n');

gform_surv_treated = zeros(Npat, 1);
gform_surv_untreated = zeros(Npat, 1);

for p = 1:Npat
    patient = patients{p};
    dp = drug_params;
    dp.alpha = patient.alpha;
    if isfield(patient, 'C50')
        dp.patient_C50 = patient.C50;
        dp.patient_gamma = patient.gamma_hill;
    end
    
    % Use ESTIMATED lambda_base (not true)
    lam_base_est = est_lambda_base(p,:);

    % Counterfactual 1: what if this patient received treatment?
    r_treat = fcn_run_strategy('adaptive', lam_base_est, dp, ctrl_params, T, dt, use_ekf);
    
    % Counterfactual 2: what if this patient received no treatment?
    r_none = fcn_run_strategy('none', lam_base_est, dp, ctrl_params, T, dt, use_ekf);
    
    % Evaluate mortality for both counterfactuals
    alive_treat = 0;
    alive_none = 0;
    for run = 1:Nruns_mort
        [a, ~] = fcn_mortality(r_treat.lambda_obs, r_treat.x, dt, ...
            theta_mort, patient.age, patient.sofa);
        alive_treat = alive_treat + a;
        
        [a, ~] = fcn_mortality(r_none.lambda_obs, r_none.x, dt, ...
            theta_mort, patient.age, patient.sofa);
        alive_none = alive_none + a;
    end
    
    gform_surv_treated(p) = alive_treat / Nruns_mort;
    gform_surv_untreated(p) = alive_none / Nruns_mort;
    
    if mod(p, 200) == 0
        fprintf('  %d/%d simulated (%.1fs)\n', p, Npat, toc);
    end
end

gform_surv_treat_pct = mean(gform_surv_treated) * 100;
gform_surv_untreat_pct = mean(gform_surv_untreated) * 100;
gform_ate = gform_surv_treat_pct - gform_surv_untreat_pct;

fprintf('G-formula: Treated=%.1f%%, Untreated=%.1f%%, ATE=%.1f%% (%.1fs)\n', ...
    gform_surv_treat_pct, gform_surv_untreat_pct, gform_ate, toc);

%% Step 5: Compare all three analyses
fprintf('\n=====================================================\n');
fprintf('CAUSAL INFERENCE COMPARISON\n');
fprintf('=====================================================\n');
fprintf('%-20s %10s %10s %10s\n', 'Method', 'Treated', 'Untreated', 'ATE');
fprintf('%s\n', repmat('-', 1, 55));
fprintf('%-20s %9.1f%% %9.1f%% %+9.1f%%\n', 'RCT (truth)', rct_surv_treated, rct_surv_untreated, rct_ate);
fprintf('%-20s %9.1f%% %9.1f%% %+9.1f%%\n', 'Naive observational', naive_surv_treated, naive_surv_untreated, naive_ate);
fprintf('%-20s %9.1f%% %9.1f%% %+9.1f%%\n', 'G-formula', gform_surv_treat_pct, gform_surv_untreat_pct, gform_ate);

fprintf('\nBias in naive analysis: %.1f percentage points\n', naive_ate - rct_ate);
fprintf('Bias in g-formula:     %.1f percentage points\n', gform_ate - rct_ate);
naive_bias = abs(naive_ate - rct_ate);
gform_bias = abs(gform_ate - rct_ate);
if isnan(naive_bias) || naive_bias < 0.5
    fprintf('Bias reduction:        N/A (naive bias too small or NaN)\n');
else
    fprintf('Bias reduction:        %.1f%%\n', (1 - gform_bias / naive_bias) * 100);
end

%% FIGURE 1: ATE Comparison
figure('Position', [50 50 900 400]); clf;

subplot(1,2,1);
bar_data = [rct_surv_treated, rct_surv_untreated; ...
            naive_surv_treated, naive_surv_untreated; ...
            gform_surv_treat_pct, gform_surv_untreat_pct];
b = bar(bar_data);
b(1).FaceColor = [0.3 0.8 0.3];
b(2).FaceColor = [0.7 0.7 0.7];
set(gca, 'XTickLabel', {'RCT', 'Naive Obs', 'G-formula'});
ylabel('Survival (%)');
title('Survival by Method');
legend('Treated', 'Untreated', 'Location', 'best');
ylim([0 100]); grid on;

subplot(1,2,2);
ate_vals = [rct_ate, naive_ate, gform_ate];
colors_ate = [0.3 0.3 0.8; 0.8 0.3 0.3; 0.3 0.8 0.3];
b = bar(ate_vals);
b.FaceColor = 'flat';
for i = 1:3
    b.CData(i,:) = colors_ate(i,:);
end
set(gca, 'XTickLabel', {'RCT (truth)', 'Naive', 'G-formula'});
ylabel('ATE (percentage points)');
title('Average Treatment Effect');
yline(0, 'k--', 'LineWidth', 1);
grid on;

sgtitle('Causal Inference: RCT vs Naive vs G-formula');

%% FIGURE 2: Confounding demonstration
figure('Position', [50 50 1000 400]); clf;

peak_rates = cellfun(@(p) max(p.lambda_base), patients);
sofas = cellfun(@(p) p.sofa, patients);

% RCT: no confounding (distributions should overlap)
subplot(1,2,1);
histogram(peak_rates(rct_assignment), 20, 'FaceColor', [0.3 0.8 0.3], 'FaceAlpha', 0.5, 'DisplayName', 'Treated');
hold on;
histogram(peak_rates(~rct_assignment), 20, 'FaceColor', [0.7 0.7 0.7], 'FaceAlpha', 0.5, 'DisplayName', 'Untreated');
xlabel('Peak Seizure Rate (sz/day)');
ylabel('Count');
title(sprintf('RCT: No Confounding\nmean treated=%.1f, untreated=%.1f', ...
    mean(peak_rates(rct_assignment)), mean(peak_rates(~rct_assignment))));
legend('Location', 'best'); grid on;

% Observational: confounding (treated patients are sicker)
subplot(1,2,2);
histogram(peak_rates(obs_assignment), 20, 'FaceColor', [0.3 0.8 0.3], 'FaceAlpha', 0.5, 'DisplayName', 'Treated');
hold on;
histogram(peak_rates(~obs_assignment), 20, 'FaceColor', [0.7 0.7 0.7], 'FaceAlpha', 0.5, 'DisplayName', 'Untreated');
xlabel('Peak Seizure Rate (sz/day)');
ylabel('Count');
title(sprintf('Observational: Confounding by Indication\nmean treated=%.1f, untreated=%.1f', ...
    mean(peak_rates(obs_assignment)), mean(peak_rates(~obs_assignment))));
legend('Location', 'best'); grid on;

sgtitle('Treatment Assignment Bias');

%% FIGURE 3: G-formula individual treatment effects by severity
figure('Position', [50 50 800 400]); clf;

individual_te = (gform_surv_treated - gform_surv_untreated) * 100;

scatter(peak_rates, individual_te, 30, sofas, 'filled', 'MarkerFaceAlpha', 0.6);
hold on;
yline(0, 'k--', 'LineWidth', 1.5);
yline(gform_ate, 'r-', sprintf('ATE=%.1f%%', gform_ate), 'LineWidth', 2);
xlabel('Peak Seizure Rate (sz/day)');
ylabel('Individual Treatment Effect (%)');
title('G-formula: Treatment Effect by Patient Severity');
colormap('hot');
c = colorbar; c.Label.String = 'SOFA Score';
grid on;

%% FIGURE 4: Summary comparison (CICADAS-style)
figure('Position', [50 50 600 500]); clf;

methods = categorical({'RCT\n(truth)', 'Naive\nObs', 'G-formula'});
methods = reordercats(methods, {'RCT\n(truth)', 'Naive\nObs', 'G-formula'});

hold on;
% Treated survival
plot([1 2 3], [rct_surv_treated, naive_surv_treated, gform_surv_treat_pct], ...
    'go-', 'LineWidth', 2, 'MarkerSize', 12, 'MarkerFaceColor', 'g');
% Untreated survival
plot([1 2 3], [rct_surv_untreated, naive_surv_untreated, gform_surv_untreat_pct], ...
    'ks-', 'LineWidth', 2, 'MarkerSize', 12, 'MarkerFaceColor', [0.7 0.7 0.7]);

set(gca, 'XTick', [1 2 3], 'XTickLabel', {'RCT (truth)', 'Naive Obs', 'G-formula'});
ylabel('Survival (%)');
legend('Treated', 'Untreated', 'Location', 'best');
title('Causal Effect Recovery');
grid on;
xlim([0.5 3.5]);

% Add ATE annotations
for i = 1:3
    treat_vals = [rct_surv_treated, naive_surv_treated, gform_surv_treat_pct];
    untreat_vals = [rct_surv_untreated, naive_surv_untreated, gform_surv_untreat_pct];
    mid = (treat_vals(i) + untreat_vals(i)) / 2;
    ate_val = treat_vals(i) - untreat_vals(i);
    text(i + 0.15, mid, sprintf('ATE=%+.1f%%', ate_val), 'FontSize', 10, 'FontWeight', 'bold');
end

%% Save
save('gformula_results.mat', ...
    'rct_surv_treated', 'rct_surv_untreated', 'rct_ate', ...
    'naive_surv_treated', 'naive_surv_untreated', 'naive_ate', ...
    'gform_surv_treat_pct', 'gform_surv_untreat_pct', 'gform_ate', ...
    'obs_assignment', 'rct_assignment', 'obs_treat_prob', ...
    'est_Ap', 'est_r', 'est_kappa', ...
    'gform_surv_treated', 'gform_surv_untreated', ...
    'Npat', 'Nruns_mort', 'theta_mort', 'use_cicadas_dgp');
fprintf('\nResults saved to gformula_results.mat\n');

%% Conclusion
fprintf('\n=== CONCLUSION ===\n');
if isnan(naive_ate)
    fprintf('Naive analysis failed (NaN) -- treatment assignment too imbalanced.\n');
    fprintf('G-formula still recovers ATE=%.1f%% vs truth=%.1f%% (residual=%.1f%%).\n', ...
        gform_ate, rct_ate, gform_ate - rct_ate);
elseif abs(naive_ate - rct_ate) > 3
    fprintf('Naive analysis is biased by %.1f percentage points.\n', naive_ate - rct_ate);
    if abs(gform_ate - rct_ate) < abs(naive_ate - rct_ate) * 0.5
        fprintf('G-formula reduces bias by >50%%, recovering ATE=%.1f%% vs truth=%.1f%%.\n', ...
            gform_ate, rct_ate);
        fprintf('Pipeline validated for causal inference from confounded observational data.\n');
    else
        fprintf('G-formula partially corrects bias but significant residual remains.\n');
        fprintf('  G-formula ATE=%.1f%%, Truth=%.1f%%, Residual bias=%.1f%%\n', ...
            gform_ate, rct_ate, gform_ate - rct_ate);
    end
else
    fprintf('Confounding is weak (bias=%.1f%%). Increase treatment assignment coefficients.\n', ...
        naive_ate - rct_ate);
end
