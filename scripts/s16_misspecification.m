% S16_MISSPECIFICATION
% Model misspecification analysis with decomposed error sources.
%
% Three-layer decomposition:
%   Layer A: EKF misspecification only
%     - Rise-decay DGP (well-specified trajectory estimation)
%     - Misspecified EKF (wrong prior, wrong tau_x, wrong q)
%     - Measures: how do adaptive dosing decisions degrade?
%
%   Layer B: Trajectory estimation misspecification only
%     - CICADAS SDE DGP, estimated with rise-decay model
%     - Default EKF parameters
%     - Measures: how do g-formula counterfactuals degrade?
%
%   Layer C: Joint misspecification (realistic scenario)
%     - SDE DGP + rise-decay estimation + misspecified EKF
%
% Experiment 2: Graded trajectory misspecification
%   - Mixture of rise-decay + SDE patients, fraction 0% to 100%
%
% Experiment 3: Mortality model misspecification
%   - Generate with squared burden, estimate with linear model
%
% Uses:
%   fcn_run_adaptive_with_ekf  (for EKF-misspecified adaptive strategy)
%   fcn_mortality_squared       (for squared-burden mortality DGP)
%   fcn_generate_patient_cicadas (for SDE disease trajectories)

clear; clc;
addpath('functions');

%% Load parameters
[time_params, drug_params, ctrl_params, mort_params] = default_params();
T = time_params.T;
Nt = time_params.Nt;
dt = time_params.dt;

%% Settings
Npat = 500;
Nruns_mort = 300;
rng(42);

patient_params.peak_rate = 10;
patient_params.drug_params = drug_params;

% G-formula mortality parameters (same as s9)
theta_mort = [-5, 0.5, 1.5, 0.5, 0.5, 0.2];

% Misspecified EKF parameters
ekf_misspec.P_pop = 20;       % 4x too high
ekf_misspec.tau_x = 2.0;      % 4x too slow
ekf_misspec.q = 1e-2;         % 100x too large
ekf_misspec.floor = 1.0;      % 10x too high

fprintf('=== MISSPECIFICATION ANALYSIS ===\n');
fprintf('Patients: %d, Mortality runs: %d\n\n', Npat, Nruns_mort);

%% Generate patient populations
fprintf('Generating %d CICADAS SDE patients...\n', Npat);
patients_sde = cell(Npat, 1);
for p = 1:Npat
    patients_sde{p} = fcn_generate_patient_cicadas(T, patient_params);
end

fprintf('Generating %d rise-decay patients (well-specified)...\n', Npat);
patients_rd = cell(Npat, 1);
for p = 1:Npat
    patients_rd{p} = fcn_generate_patient(T, patient_params);
end

%% =========================================================
%% EXPERIMENT 1: THREE-LAYER DECOMPOSITION
%% =========================================================
fprintf('\n############################################\n');
fprintf('# EXPERIMENT 1: THREE-LAYER DECOMPOSITION  #\n');
fprintf('############################################\n\n');

% --- Reference: Well-specified (rise-decay DGP + rise-decay estimation) ---
fprintf('--- Reference: Well-specified g-formula ---\n');
tic;
[ref_rct_ate, ref_naive_ate, ref_gform_ate] = run_gformula(...
    patients_rd, drug_params, ctrl_params, T, dt, theta_mort, ...
    Npat, Nruns_mort, patient_params, []);
fprintf('Well-specified: RCT=%.1f%%, Naive=%.1f%%, G-form=%.1f%% (%.1fs)\n', ...
    ref_rct_ate, ref_naive_ate, ref_gform_ate, toc);
ref_gform_bias = abs(ref_gform_ate - ref_rct_ate);
fprintf('  Residual bias: %.1f pp\n\n', ref_gform_bias);

% --- Layer A: EKF misspecification only ---
fprintf('--- Layer A: EKF misspecification only ---\n');
fprintf('  (rise-decay DGP, rise-decay estimation, WRONG EKF)\n');
tic;
[layerA_rct_ate, layerA_naive_ate, layerA_gform_ate] = run_gformula(...
    patients_rd, drug_params, ctrl_params, T, dt, theta_mort, ...
    Npat, Nruns_mort, patient_params, ekf_misspec);
fprintf('Layer A: RCT=%.1f%%, Naive=%.1f%%, G-form=%.1f%% (%.1fs)\n', ...
    layerA_rct_ate, layerA_naive_ate, layerA_gform_ate, toc);
layerA_gform_bias = abs(layerA_gform_ate - layerA_rct_ate);
fprintf('  Residual bias: %.1f pp\n\n', layerA_gform_bias);

% --- Layer B: Trajectory estimation misspecification only ---
fprintf('--- Layer B: Trajectory estimation misspecification only ---\n');
fprintf('  (SDE DGP, rise-decay estimation, default EKF)\n');
tic;
[layerB_rct_ate, layerB_naive_ate, layerB_gform_ate] = run_gformula(...
    patients_sde, drug_params, ctrl_params, T, dt, theta_mort, ...
    Npat, Nruns_mort, patient_params, []);
fprintf('Layer B: RCT=%.1f%%, Naive=%.1f%%, G-form=%.1f%% (%.1fs)\n', ...
    layerB_rct_ate, layerB_naive_ate, layerB_gform_ate, toc);
layerB_gform_bias = abs(layerB_gform_ate - layerB_rct_ate);
fprintf('  Residual bias: %.1f pp\n\n', layerB_gform_bias);

% --- Layer C: Joint misspecification ---
fprintf('--- Layer C: Joint misspecification ---\n');
fprintf('  (SDE DGP, rise-decay estimation, WRONG EKF)\n');
tic;
[layerC_rct_ate, layerC_naive_ate, layerC_gform_ate] = run_gformula(...
    patients_sde, drug_params, ctrl_params, T, dt, theta_mort, ...
    Npat, Nruns_mort, patient_params, ekf_misspec);
fprintf('Layer C: RCT=%.1f%%, Naive=%.1f%%, G-form=%.1f%% (%.1fs)\n', ...
    layerC_rct_ate, layerC_naive_ate, layerC_gform_ate, toc);
layerC_gform_bias = abs(layerC_gform_ate - layerC_rct_ate);
fprintf('  Residual bias: %.1f pp\n\n', layerC_gform_bias);

% Decomposition summary
fprintf('=====================================================\n');
fprintf('THREE-LAYER DECOMPOSITION SUMMARY\n');
fprintf('=====================================================\n');
fprintf('%-35s %10s %10s %10s\n', 'Condition', 'RCT ATE', 'G-form ATE', 'Bias');
fprintf('%s\n', repmat('-', 1, 70));
fprintf('%-35s %+9.1f%% %+9.1f%% %8.1f pp\n', ...
    'Reference (well-specified)', ref_rct_ate, ref_gform_ate, ref_gform_bias);
fprintf('%-35s %+9.1f%% %+9.1f%% %8.1f pp\n', ...
    'Layer A (EKF misspec only)', layerA_rct_ate, layerA_gform_ate, layerA_gform_bias);
fprintf('%-35s %+9.1f%% %+9.1f%% %8.1f pp\n', ...
    'Layer B (trajectory misspec only)', layerB_rct_ate, layerB_gform_ate, layerB_gform_bias);
fprintf('%-35s %+9.1f%% %+9.1f%% %8.1f pp\n', ...
    'Layer C (joint misspec)', layerC_rct_ate, layerC_gform_ate, layerC_gform_bias);

fprintf('\nError attribution:\n');
fprintf('  EKF alone adds:        %.1f pp above reference\n', layerA_gform_bias - ref_gform_bias);
fprintf('  Trajectory alone adds: %.1f pp above reference\n', layerB_gform_bias - ref_gform_bias);
fprintf('  Joint adds:            %.1f pp above reference\n', layerC_gform_bias - ref_gform_bias);
fprintf('  Interaction:           %.1f pp\n', ...
    layerC_gform_bias - (layerA_gform_bias + layerB_gform_bias - ref_gform_bias));

%% =========================================================
%% EXPERIMENT 2: GRADED TRAJECTORY MISSPECIFICATION
%% =========================================================
fprintf('\n\n############################################\n');
fprintf('# EXPERIMENT 2: GRADED MISSPECIFICATION     #\n');
fprintf('############################################\n\n');

mix_fractions = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
n_mix = numel(mix_fractions);
graded_rct = zeros(n_mix, 1);
graded_gform = zeros(n_mix, 1);
graded_bias = zeros(n_mix, 1);

for m = 1:n_mix
    frac = mix_fractions(m);
    n_sde = round(frac * Npat);
    n_rd = Npat - n_sde;

    fprintf('Mixture: %.0f%% SDE, %.0f%% rise-decay...\n', frac*100, (1-frac)*100);
    tic;

    % Build mixed population
    patients_mix = cell(Npat, 1);
    rng(42 + m);
    for p = 1:n_rd
        patients_mix{p} = fcn_generate_patient(T, patient_params);
    end
    for p = (n_rd+1):Npat
        patients_mix{p} = fcn_generate_patient_cicadas(T, patient_params);
    end
    % Shuffle
    shuffle_idx = randperm(Npat);
    patients_mix = patients_mix(shuffle_idx);

    [rct_ate, ~, gform_ate] = run_gformula(...
        patients_mix, drug_params, ctrl_params, T, dt, theta_mort, ...
        Npat, Nruns_mort, patient_params, []);

    graded_rct(m) = rct_ate;
    graded_gform(m) = gform_ate;
    graded_bias(m) = abs(gform_ate - rct_ate);

    fprintf('  RCT=%.1f%%, G-form=%.1f%%, Bias=%.1f pp (%.1fs)\n', ...
        rct_ate, gform_ate, graded_bias(m), toc);
end

fprintf('\nGraded misspecification results:\n');
fprintf('%-15s %10s %10s %10s\n', 'SDE fraction', 'RCT ATE', 'G-form ATE', 'Bias');
fprintf('%s\n', repmat('-', 1, 50));
for m = 1:n_mix
    fprintf('%-15.0f%% %+9.1f%% %+9.1f%% %8.1f pp\n', ...
        mix_fractions(m)*100, graded_rct(m), graded_gform(m), graded_bias(m));
end

%% =========================================================
%% EXPERIMENT 3: MORTALITY MODEL MISSPECIFICATION
%% =========================================================
fprintf('\n\n############################################\n');
fprintf('# EXPERIMENT 3: MORTALITY MISSPECIFICATION  #\n');
fprintf('############################################\n\n');

% Generate with squared-burden mortality, estimate with linear.
theta_squared = [-4, 0.5, 1.0, 0.7, 0.10, 0.05];

fprintf('--- Squared-burden DGP, linear estimation ---\n');
tic;

[mort_rct_ate, mort_naive_ate, mort_gform_ate] = run_gformula_mort_misspec(...
    patients_rd, drug_params, ctrl_params, T, dt, ...
    theta_squared, theta_mort, Npat, Nruns_mort, patient_params);

fprintf('Mortality misspec: RCT=%.1f%%, Naive=%.1f%%, G-form=%.1f%% (%.1fs)\n', ...
    mort_rct_ate, mort_naive_ate, mort_gform_ate, toc);
mort_gform_bias = abs(mort_gform_ate - mort_rct_ate);
fprintf('  Residual bias: %.1f pp (vs well-specified %.1f pp)\n\n', ...
    mort_gform_bias, ref_gform_bias);

%% Save all results
save('misspecification_results.mat', ...
    'ref_rct_ate', 'ref_gform_ate', 'ref_gform_bias', ...
    'layerA_rct_ate', 'layerA_gform_ate', 'layerA_gform_bias', ...
    'layerB_rct_ate', 'layerB_gform_ate', 'layerB_gform_bias', ...
    'layerC_rct_ate', 'layerC_gform_ate', 'layerC_gform_bias', ...
    'mix_fractions', 'graded_rct', 'graded_gform', 'graded_bias', ...
    'mort_rct_ate', 'mort_gform_ate', 'mort_gform_bias', ...
    'Npat', 'Nruns_mort', 'ekf_misspec', 'theta_squared');
fprintf('Results saved to misspecification_results.mat\n');

%% =========================================================
%% FIGURES
%% =========================================================

%% Figure 1: Three-layer decomposition
figure('Position', [50 50 700 500]); clf;
bias_vals = [ref_gform_bias, layerA_gform_bias, layerB_gform_bias, layerC_gform_bias];
b = bar(bias_vals, 0.6);
b.FaceColor = 'flat';
b.CData(1,:) = [0.3 0.7 0.3];
b.CData(2,:) = [0.3 0.3 0.8];
b.CData(3,:) = [0.8 0.5 0.2];
b.CData(4,:) = [0.8 0.2 0.2];
set(gca, 'XTickLabel', {'Well-specified', 'EKF misspec', 'Traj misspec', 'Joint'}, 'FontSize', 11);
ylabel('G-formula Residual Bias (pp)', 'FontSize', 12);
title('Misspecification Decomposition: Error Attribution', 'FontSize', 13);
grid on;
for i = 1:4
    text(i, bias_vals(i) + 0.3, sprintf('%.1f pp', bias_vals(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 11, 'FontWeight', 'bold');
end

%% Figure 2: Graded misspecification robustness curve
figure('Position', [50 50 700 500]); clf;
yyaxis left
plot(mix_fractions * 100, graded_bias, 'bo-', 'LineWidth', 2, 'MarkerSize', 8);
ylabel('G-formula Residual Bias (pp)', 'FontSize', 12);
ylim([0, max(graded_bias) * 1.3 + 1]);

yyaxis right
plot(mix_fractions * 100, graded_gform, 'rs--', 'LineWidth', 1.5, 'MarkerSize', 6);
hold on;
plot(mix_fractions * 100, graded_rct, 'k:+', 'LineWidth', 1.5, 'MarkerSize', 6);
ylabel('ATE (%)', 'FontSize', 12);
legend('Bias (left axis)', 'G-formula ATE', 'RCT ATE (truth)', 'Location', 'best');

xlabel('Fraction of SDE patients (%)', 'FontSize', 12);
title('Graded Trajectory Misspecification: Robustness Curve', 'FontSize', 13);
grid on;

%% Figure 3: All experiments summary
figure('Position', [50 50 800 400]); clf;
all_labels = {'Well-specified', 'EKF only', 'Trajectory only', 'Joint', 'Mortality'};
all_biases = [ref_gform_bias, layerA_gform_bias, layerB_gform_bias, ...
              layerC_gform_bias, mort_gform_bias];
b = barh(all_biases);
b.FaceColor = 'flat';
b.CData(1,:) = [0.3 0.7 0.3];
b.CData(2,:) = [0.3 0.3 0.8];
b.CData(3,:) = [0.8 0.5 0.2];
b.CData(4,:) = [0.8 0.2 0.2];
b.CData(5,:) = [0.6 0.2 0.6];
set(gca, 'YTickLabel', all_labels, 'FontSize', 11);
xlabel('G-formula Residual Bias (pp)', 'FontSize', 12);
title('All Misspecification Experiments', 'FontSize', 13);
grid on;
for i = 1:5
    text(all_biases(i) + 0.2, i, sprintf('%.1f pp', all_biases(i)), ...
        'FontSize', 10, 'FontWeight', 'bold', 'VerticalAlignment', 'middle');
end

sgtitle('Model Misspecification Analysis', 'FontSize', 14, 'FontWeight', 'bold');


%% =========================================================
%% HELPER FUNCTIONS (local functions at end of script, R2016b+)
%% =========================================================

function [rct_ate, naive_ate, gform_ate] = run_gformula(...
        patients, drug_params, ctrl_params, T, dt, theta_mort, ...
        Npat, Nruns_mort, patient_params, ekf_override)
    % Full g-formula pipeline: RCT + naive observational + g-formula.
    % ekf_override: if non-empty struct, use misspecified EKF for adaptive.

    Nt = numel(T);
    use_ekf = true;
    use_custom_ekf = ~isempty(ekf_override) && isstruct(ekf_override);

    %% RCT
    rng(100);
    rct_assignment = rand(Npat, 1) > 0.5;
    surv_rct_t = zeros(Npat, 1);
    surv_rct_u = zeros(Npat, 1);

    for p = 1:Npat
        [dp] = setup_patient_dp(patients{p}, drug_params);

        if rct_assignment(p)
            if use_custom_ekf
                r = fcn_run_adaptive_with_ekf(patients{p}, dp, ctrl_params, T, dt, ekf_override);
            else
                r = fcn_run_strategy('adaptive', patients{p}.lambda_base, dp, ctrl_params, T, dt, use_ekf);
            end
        else
            r = fcn_run_strategy('none', patients{p}.lambda_base, dp, ctrl_params, T, dt, use_ekf);
        end

        surv_rct_t(p) = mc_survival(r, dt, theta_mort, patients{p}, Nruns_mort);
        if ~rct_assignment(p), surv_rct_t(p) = 0; end

        surv_rct_u(p) = surv_rct_t(p);
        if rct_assignment(p), surv_rct_u(p) = 0; end

        % Fix: only compute for assigned group
        alive_sum = 0;
        for run = 1:Nruns_mort
            [a, ~] = fcn_mortality(r.lambda_obs, r.x, dt, theta_mort, patients{p}.age, patients{p}.sofa);
            alive_sum = alive_sum + a;
        end

        if rct_assignment(p)
            surv_rct_t(p) = alive_sum / Nruns_mort;
        else
            surv_rct_u(p) = alive_sum / Nruns_mort;
        end
    end

    rct_ate = mean(surv_rct_t(rct_assignment))*100 - mean(surv_rct_u(~rct_assignment))*100;

    %% Observational with confounding
    rng(200);
    early_steps = min(round(1.0 / dt), Nt);
    half_steps = floor(early_steps / 2);
    obs_assignment = false(Npat, 1);

    for p = 1:Npat
        obs_assignment(p) = assign_treatment(patients{p}, early_steps, half_steps, ...
            patient_params.peak_rate, dt);
    end

    surv_obs_t = zeros(Npat, 1);
    surv_obs_u = zeros(Npat, 1);

    for p = 1:Npat
        [dp] = setup_patient_dp(patients{p}, drug_params);

        if obs_assignment(p)
            if use_custom_ekf
                r = fcn_run_adaptive_with_ekf(patients{p}, dp, ctrl_params, T, dt, ekf_override);
            else
                r = fcn_run_strategy('adaptive', patients{p}.lambda_base, dp, ctrl_params, T, dt, use_ekf);
            end
        else
            r = fcn_run_strategy('none', patients{p}.lambda_base, dp, ctrl_params, T, dt, use_ekf);
        end

        alive_sum = 0;
        for run = 1:Nruns_mort
            [a, ~] = fcn_mortality(r.lambda_obs, r.x, dt, theta_mort, patients{p}.age, patients{p}.sofa);
            alive_sum = alive_sum + a;
        end

        if obs_assignment(p)
            surv_obs_t(p) = alive_sum / Nruns_mort;
        else
            surv_obs_u(p) = alive_sum / Nruns_mort;
        end
    end

    naive_ate = mean(surv_obs_t(obs_assignment))*100 - mean(surv_obs_u(~obs_assignment))*100;

    %% G-formula: estimate + counterfactual
    est_lambda_base = zeros(Npat, Nt);

    for p = 1:Npat
        [dp] = setup_patient_dp(patients{p}, drug_params);

        if obs_assignment(p)
            if use_custom_ekf
                r = fcn_run_adaptive_with_ekf(patients{p}, dp, ctrl_params, T, dt, ekf_override);
            else
                r = fcn_run_strategy('adaptive', patients{p}.lambda_base, dp, ctrl_params, T, dt, use_ekf);
            end

            if strcmpi(drug_params.pd_mode, 'hill')
                [~, fi] = fcn_estimate_natural_history(r.seizure_counts, T, dt, ...
                    'peak_rate', patient_params.peak_rate, ...
                    'k_e', dp.k_e, 'doses_given', r.doses_given, ...
                    'pd_mode', 'hill', 'C50', dp.patient_C50, 'gamma_hill', dp.patient_gamma);
            else
                [~, fi] = fcn_estimate_natural_history(r.seizure_counts, T, dt, ...
                    'peak_rate', patient_params.peak_rate, ...
                    'k_e', dp.k_e, 'doses_given', r.doses_given, ...
                    'pd_mode', 'linear', 'alpha', dp.alpha);
            end
        else
            counts = poissrnd(patients{p}.lambda_base * dt);
            [~, fi] = fcn_estimate_natural_history(counts, T, dt, ...
                'peak_rate', patient_params.peak_rate);
        end
        est_lambda_base(p,:) = fi.lambda_base_hat;
    end

    % Counterfactual simulation (always uses default EKF for g-formula)
    gform_surv_t = zeros(Npat, 1);
    gform_surv_u = zeros(Npat, 1);

    for p = 1:Npat
        [dp] = setup_patient_dp(patients{p}, drug_params);
        lam_est = est_lambda_base(p,:);

        r_treat = fcn_run_strategy('adaptive', lam_est, dp, ctrl_params, T, dt, use_ekf);
        r_none = fcn_run_strategy('none', lam_est, dp, ctrl_params, T, dt, use_ekf);

        at = 0; an = 0;
        for run = 1:Nruns_mort
            [a, ~] = fcn_mortality(r_treat.lambda_obs, r_treat.x, dt, theta_mort, patients{p}.age, patients{p}.sofa);
            at = at + a;
            [a, ~] = fcn_mortality(r_none.lambda_obs, r_none.x, dt, theta_mort, patients{p}.age, patients{p}.sofa);
            an = an + a;
        end
        gform_surv_t(p) = at / Nruns_mort;
        gform_surv_u(p) = an / Nruns_mort;
    end

    gform_ate = mean(gform_surv_t)*100 - mean(gform_surv_u)*100;
end


function [rct_ate, naive_ate, gform_ate] = run_gformula_mort_misspec(...
        patients, drug_params, ctrl_params, T, dt, ...
        theta_gen, theta_est, Npat, Nruns_mort, patient_params)
    % G-formula with mortality misspecification.
    % Data GENERATED with theta_gen (squared-burden via fcn_mortality_squared).
    % G-formula ESTIMATES counterfactuals with theta_est (linear via fcn_mortality).

    Nt = numel(T);
    use_ekf = true;

    %% RCT with squared mortality
    rng(100);
    rct_assignment = rand(Npat, 1) > 0.5;
    surv_rct_t = zeros(Npat, 1);
    surv_rct_u = zeros(Npat, 1);

    for p = 1:Npat
        [dp] = setup_patient_dp(patients{p}, drug_params);
        if rct_assignment(p)
            r = fcn_run_strategy('adaptive', patients{p}.lambda_base, dp, ctrl_params, T, dt, use_ekf);
        else
            r = fcn_run_strategy('none', patients{p}.lambda_base, dp, ctrl_params, T, dt, use_ekf);
        end

        alive_sum = 0;
        for run = 1:Nruns_mort
            [a, ~] = fcn_mortality_squared(r.lambda_obs, r.x, dt, theta_gen, patients{p}.age, patients{p}.sofa);
            alive_sum = alive_sum + a;
        end

        if rct_assignment(p)
            surv_rct_t(p) = alive_sum / Nruns_mort;
        else
            surv_rct_u(p) = alive_sum / Nruns_mort;
        end
    end
    rct_ate = mean(surv_rct_t(rct_assignment))*100 - mean(surv_rct_u(~rct_assignment))*100;

    %% Observational with squared mortality
    rng(200);
    early_steps = min(round(1.0 / dt), Nt);
    half_steps = floor(early_steps / 2);
    obs_assignment = false(Npat, 1);
    for p = 1:Npat
        obs_assignment(p) = assign_treatment(patients{p}, early_steps, half_steps, ...
            patient_params.peak_rate, dt);
    end

    surv_obs_t = zeros(Npat, 1);
    surv_obs_u = zeros(Npat, 1);
    for p = 1:Npat
        [dp] = setup_patient_dp(patients{p}, drug_params);
        if obs_assignment(p)
            r = fcn_run_strategy('adaptive', patients{p}.lambda_base, dp, ctrl_params, T, dt, use_ekf);
        else
            r = fcn_run_strategy('none', patients{p}.lambda_base, dp, ctrl_params, T, dt, use_ekf);
        end
        alive_sum = 0;
        for run = 1:Nruns_mort
            [a, ~] = fcn_mortality_squared(r.lambda_obs, r.x, dt, theta_gen, patients{p}.age, patients{p}.sofa);
            alive_sum = alive_sum + a;
        end
        if obs_assignment(p)
            surv_obs_t(p) = alive_sum / Nruns_mort;
        else
            surv_obs_u(p) = alive_sum / Nruns_mort;
        end
    end
    naive_ate = mean(surv_obs_t(obs_assignment))*100 - mean(surv_obs_u(~obs_assignment))*100;

    %% G-formula with LINEAR mortality (misspecified estimation)
    est_lambda_base = zeros(Npat, Nt);
    for p = 1:Npat
        [dp] = setup_patient_dp(patients{p}, drug_params);
        if obs_assignment(p)
            r = fcn_run_strategy('adaptive', patients{p}.lambda_base, dp, ctrl_params, T, dt, use_ekf);
            if strcmpi(drug_params.pd_mode, 'hill')
                [~, fi] = fcn_estimate_natural_history(r.seizure_counts, T, dt, ...
                    'peak_rate', patient_params.peak_rate, ...
                    'k_e', dp.k_e, 'doses_given', r.doses_given, ...
                    'pd_mode', 'hill', 'C50', dp.patient_C50, 'gamma_hill', dp.patient_gamma);
            else
                [~, fi] = fcn_estimate_natural_history(r.seizure_counts, T, dt, ...
                    'peak_rate', patient_params.peak_rate, ...
                    'k_e', dp.k_e, 'doses_given', r.doses_given, ...
                    'pd_mode', 'linear', 'alpha', dp.alpha);
            end
        else
            counts = poissrnd(patients{p}.lambda_base * dt);
            [~, fi] = fcn_estimate_natural_history(counts, T, dt, ...
                'peak_rate', patient_params.peak_rate);
        end
        est_lambda_base(p,:) = fi.lambda_base_hat;
    end

    gform_surv_t = zeros(Npat, 1);
    gform_surv_u = zeros(Npat, 1);
    for p = 1:Npat
        [dp] = setup_patient_dp(patients{p}, drug_params);
        lam_est = est_lambda_base(p,:);
        r_treat = fcn_run_strategy('adaptive', lam_est, dp, ctrl_params, T, dt, use_ekf);
        r_none = fcn_run_strategy('none', lam_est, dp, ctrl_params, T, dt, use_ekf);

        % Use LINEAR (misspecified) mortality for g-formula counterfactuals
        at = 0; an = 0;
        for run = 1:Nruns_mort
            [a, ~] = fcn_mortality(r_treat.lambda_obs, r_treat.x, dt, theta_est, patients{p}.age, patients{p}.sofa);
            at = at + a;
            [a, ~] = fcn_mortality(r_none.lambda_obs, r_none.x, dt, theta_est, patients{p}.age, patients{p}.sofa);
            an = an + a;
        end
        gform_surv_t(p) = at / Nruns_mort;
        gform_surv_u(p) = an / Nruns_mort;
    end
    gform_ate = mean(gform_surv_t)*100 - mean(gform_surv_u)*100;
end


function dp = setup_patient_dp(patient, drug_params)
    % Set up patient-specific drug params
    dp = drug_params;
    dp.alpha = patient.alpha;
    if isfield(patient, 'C50')
        dp.patient_C50 = patient.C50;
        dp.patient_gamma = patient.gamma_hill;
    end
end


function assigned = assign_treatment(patient, early_steps, half_steps, peak_rate, dt)
    % Confounding-by-indication treatment assignment (8-feature logistic)
    L_vec = patient.lambda_base(1:early_steps);

    feat_mean = mean(L_vec) / peak_rate;
    feat_max = max(L_vec) / (2 * peak_rate);
    raw_growth = (L_vec(end) - L_vec(1)) / (L_vec(1) + 0.1);
    feat_growth = log1p(max(0, raw_growth));

    p_fit = polyfit((1:early_steps)', L_vec(:), 1);
    L_max = max(L_vec);
    feat_slope = p_fit(1) * early_steps / max(L_max, 0.1);

    if half_steps > 1
        p1 = polyfit((1:half_steps)', L_vec(1:half_steps)', 1);
        p2 = polyfit((half_steps+1:early_steps)', L_vec(half_steps+1:end)', 1);
        feat_accel = (p2(1) - p1(1)) * early_steps / max(L_max, 0.1);
    else
        feat_accel = 0;
    end

    z = -0.5 + 0.8*feat_mean + 0.6*feat_max + 0.4*feat_growth ...
        + 0.3*feat_slope + 0.3*feat_accel ...
        + 0.5*(patient.sofa/10) - 0.3*(patient.age/65) + randn()*0.3;

    assigned = rand() < 1/(1+exp(-z));
end


function surv = mc_survival(r, dt, theta_mort, patient, Nruns_mort)
    % Monte Carlo survival estimate
    alive_sum = 0;
    for run = 1:Nruns_mort
        [a, ~] = fcn_mortality(r.lambda_obs, r.x, dt, theta_mort, patient.age, patient.sofa);
        alive_sum = alive_sum + a;
    end
    surv = alive_sum / Nruns_mort;
end
