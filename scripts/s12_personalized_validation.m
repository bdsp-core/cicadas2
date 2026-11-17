% S12_PERSONALIZED_OMEGA_VALIDATION
% End-to-end test: can we recover personalized omega* from estimated data?
%
% Combines s10 (smart adaptive) with s8 (estimation pipeline):
%   1. Generate patients with known disease parameters
%   2. Observe them (50% treated, 50% untreated)
%   3. Estimate {A_p, r, d} from observations (pop-mean PKPD)
%   4. Sweep omega on BOTH true and estimated trajectories
%   5. Compare: does personalized omega* agree? Does survival hold?
%
% This validates Path A: "at admission, compute omega* from data"

clear; clc;
addpath('../functions');

%% Load parameters
[time_params, drug_params, ctrl_params, mort_params] = default_params();
T = time_params.T;
Nt = time_params.Nt;
dt = time_params.dt;

%% Settings
Npat = 200;
Nruns = 200;
frac_treated = 0.5;
N_treated = round(Npat * frac_treated);
N_untreated = Npat - N_treated;
use_ekf = true;

omega_vals = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, Inf];
n_omega = numel(omega_vals);

rng(42);
patient_params.peak_rate = 10;
patient_params.drug_params = drug_params;

%% Regimes (must match s5/s8/s10)
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

fprintf('=== PERSONALIZED OMEGA VALIDATION ===\n');
fprintf('Patients: %d | Omega values: %d | Mortality runs: %d\n\n', ...
    Npat, n_omega, Nruns);

%% Step 1: Generate patients
patients = cell(Npat, 1);
for p = 1:Npat
    patients{p} = fcn_generate_patient(T, patient_params);
end
fprintf('Generated %d patients.\n', Npat);

%% Step 2: Estimate disease parameters (same as s8, pop-mean PKPD)
fprintf('Estimating disease parameters...\n');
tic;

true_lambda_base = zeros(Npat, Nt);
est_lambda_base = zeros(Npat, Nt);

for p = 1:Npat
    patient = patients{p};
    dp = drug_params;
    dp.alpha = patient.alpha;
    if isfield(patient, 'C50')
        dp.patient_C50 = patient.C50;
        dp.patient_gamma = patient.gamma_hill;
    end
    
    true_lambda_base(p,:) = patient.lambda_base;
    
    if p <= N_untreated
        counts = poissrnd(patient.lambda_base * dt);
        [~, fit_info] = fcn_estimate_natural_history(counts, T, dt, ...
            'peak_rate', patient_params.peak_rate);
    else
        r = fcn_run_strategy('fixed', patient.lambda_base, dp, ctrl_params, T, dt, false);
        [~, fit_info] = fcn_estimate_natural_history(r.seizure_counts, T, dt, ...
            'peak_rate', patient_params.peak_rate, ...
            'k_e', dp.k_e, ...
            'doses_given', r.doses_given, ...
            'pd_mode', 'hill', ...
            'C50', drug_params.C50, ...
            'gamma_hill', drug_params.gamma_hill);
    end
    est_lambda_base(p,:) = fit_info.lambda_base_hat;
end

corr_Lcum = corr(sum(true_lambda_base,2)*dt, sum(est_lambda_base,2)*dt);
fprintf('Estimation done. L_cum corr=%.3f (%.1fs)\n\n', corr_Lcum, toc);

%% Step 3: Sweep omega on BOTH true and estimated trajectories
fprintf('Running omega sweep on true and estimated trajectories...\n');

% strat_true{p, oi} and strat_est{p, oi}
strat_true = cell(Npat, n_omega);
strat_est = cell(Npat, n_omega);

for p = 1:Npat
    patient = patients{p};
    dp = drug_params;
    dp.alpha = patient.alpha;
    if isfield(patient, 'C50')
        dp.patient_C50 = patient.C50;
        dp.patient_gamma = patient.gamma_hill;
    end
    
    for oi = 1:n_omega
        cp = ctrl_params;
        cp.omega = omega_vals(oi);
        
        if isinf(omega_vals(oi))
            strat_true{p,oi} = fcn_run_strategy('none', ...
                true_lambda_base(p,:), dp, cp, T, dt, use_ekf);
            strat_est{p,oi} = fcn_run_strategy('none', ...
                est_lambda_base(p,:), dp, cp, T, dt, use_ekf);
        else
            strat_true{p,oi} = fcn_run_strategy('adaptive', ...
                true_lambda_base(p,:), dp, cp, T, dt, use_ekf);
            strat_est{p,oi} = fcn_run_strategy('adaptive', ...
                est_lambda_base(p,:), dp, cp, T, dt, use_ekf);
        end
    end
    if mod(p, 50) == 0
        fprintf('  %d/%d patients (%.1fs)\n', p, Npat, toc);
    end
end
fprintf('Omega sweep done (%.1fs).\n\n', toc);

%% Step 4: Evaluate mortality at each omega under each regime
% BIAS CORRECTION (split-sample / honest argmax):
%   Picking omega* as argmax over noisy Monte Carlo survival estimates
%   and then reporting that same estimate biases the reported survival
%   upward (winner's curse).  We split the MC draws into independent
%   SELECTION and EVALUATION halves for BOTH true and est:
%
%     surv_*_select  -> argmax over omega -> omega_true / omega_est
%     surv_*_eval    -> survival reported at omega_true / omega_est  (unbiased)
%
%   The uniform-omega baseline is also read from the eval set so that
%   the personalized-vs-uniform comparison is apples-to-apples.  Total
%   mortality calls per (p, oi, reg) on each of true/est: 2*Nruns.
fprintf('Evaluating mortality (split-sample)...\n');

surv_true_select = zeros(Npat, n_omega, n_regimes);
surv_true_eval   = zeros(Npat, n_omega, n_regimes);
surv_est_select  = zeros(Npat, n_omega, n_regimes);
surv_est_eval    = zeros(Npat, n_omega, n_regimes);

for reg = 1:n_regimes
    fprintf('  Regime %d: %s\n', reg, regimes(reg).name);
    theta = regimes(reg).theta;

    for p = 1:Npat
        for oi = 1:n_omega
            % --- Selection draws (true & est) ---
            at = 0; ae = 0;
            for run = 1:Nruns
                [a,~] = fcn_mortality(strat_true{p,oi}.lambda_obs, ...
                    strat_true{p,oi}.x, dt, theta, ...
                    patients{p}.age, patients{p}.sofa);
                at = at + a;
                [a,~] = fcn_mortality(strat_est{p,oi}.lambda_obs, ...
                    strat_est{p,oi}.x, dt, theta, ...
                    patients{p}.age, patients{p}.sofa);
                ae = ae + a;
            end
            surv_true_select(p, oi, reg) = at / Nruns;
            surv_est_select(p,  oi, reg) = ae / Nruns;

            % --- Evaluation draws (independent: fresh RNG state) ---
            at = 0; ae = 0;
            for run = 1:Nruns
                [a,~] = fcn_mortality(strat_true{p,oi}.lambda_obs, ...
                    strat_true{p,oi}.x, dt, theta, ...
                    patients{p}.age, patients{p}.sofa);
                at = at + a;
                [a,~] = fcn_mortality(strat_est{p,oi}.lambda_obs, ...
                    strat_est{p,oi}.x, dt, theta, ...
                    patients{p}.age, patients{p}.sofa);
                ae = ae + a;
            end
            surv_true_eval(p, oi, reg) = at / Nruns;
            surv_est_eval(p,  oi, reg) = ae / Nruns;
        end
    end
    fprintf('    Done (%.1fs)\n', toc);
end
fprintf('Mortality done (%.1fs).\n\n', toc);

% Aliases for downstream figure / save code: surv_*_grid := eval set.
surv_true_grid = surv_true_eval;
surv_est_grid  = surv_est_eval;

%% Step 5: Find optimal omega from true vs estimated (split-sample)
omega_true = zeros(Npat, n_regimes);
omega_est = zeros(Npat, n_regimes);
surv_at_true_opt = zeros(Npat, n_regimes);
surv_at_est_opt = zeros(Npat, n_regimes);

for reg = 1:n_regimes
    for p = 1:Npat
        % Pick omega via selection set; report survival via eval set
        [~, idx_t] = max(surv_true_select(p, :, reg));
        omega_true(p, reg) = omega_vals(idx_t);
        surv_at_true_opt(p, reg) = surv_true_eval(p, idx_t, reg);

        [~, idx_e] = max(surv_est_select(p, :, reg));
        omega_est(p, reg) = omega_vals(idx_e);
        surv_at_est_opt(p, reg) = surv_est_eval(p, idx_e, reg);
    end
end

%% Print comparison
fprintf('=====================================================\n');
fprintf('PERSONALIZED OMEGA VALIDATION: True vs Estimated\n');
fprintf('=====================================================\n\n');

for reg = 1:n_regimes
    fprintf('--- %s ---\n', regimes(reg).name);
    
    % omega* agreement
    exact_match = sum(omega_true(:,reg) == omega_est(:,reg));
    within_one = sum(abs(omega_true(:,reg) - omega_est(:,reg)) <= 1.0 | ...
        (isinf(omega_true(:,reg)) & isinf(omega_est(:,reg))));
    
    fprintf('  omega* exact match:    %d/%d (%.0f%%)\n', ...
        exact_match, Npat, exact_match/Npat*100);
    fprintf('  omega* within 1 step:  %d/%d (%.0f%%)\n', ...
        within_one, Npat, within_one/Npat*100);
    fprintf('  True median omega*:    %.1f\n', median(omega_true(:,reg)));
    fprintf('  Est median omega*:     %.1f\n', median(omega_est(:,reg)));
    
    % Survival comparison: personalized from true vs estimated vs uniform
    surv_personal_true = mean(surv_at_true_opt(:,reg)) * 100;
    surv_personal_est = mean(surv_at_est_opt(:,reg)) * 100;
    
    [~, idx_uniform] = min(abs(omega_vals - 2.0));
    surv_uniform = mean(surv_true_grid(:, idx_uniform, reg)) * 100;
    
    fprintf('  Survival (uniform w=2):          %.1f%%\n', surv_uniform);
    fprintf('  Survival (personalized, true):    %.1f%%\n', surv_personal_true);
    fprintf('  Survival (personalized, est):     %.1f%%\n', surv_personal_est);
    fprintf('  Improvement retained:             %.1f%% of %.1f%% possible\n', ...
        surv_personal_est - surv_uniform, surv_personal_true - surv_uniform);
    fprintf('\n');
end

%% FIGURE 1: omega* true vs estimated scatter
figure('Position', [50 50 1200 500]); clf;

for reg = 1:n_regimes
    subplot(1, n_regimes, reg);
    
    ot = omega_true(:,reg); oe = omega_est(:,reg);
    ot(isinf(ot)) = 12; oe(isinf(oe)) = 12;
    
    scatter(ot, oe, 40, 'filled', 'MarkerFaceAlpha', 0.4);
    hold on;
    plot([0 13], [0 13], 'k--', 'LineWidth', 1);
    xlabel('\omega^* (true)'); ylabel('\omega^* (estimated)');
    title(regimes(reg).name);
    xlim([0 13]); ylim([0 13]); grid on;
end
sgtitle('Personalized \omega^*: True vs Estimated Parameters');

%% FIGURE 2: Survival comparison — uniform vs personalized (true) vs personalized (est)
figure('Position', [50 50 900 500]); clf;

bar_data = zeros(n_regimes, 3);
[~, idx_uniform] = min(abs(omega_vals - 2.0));
for reg = 1:n_regimes
    bar_data(reg, 1) = mean(surv_true_grid(:, idx_uniform, reg)) * 100;
    bar_data(reg, 2) = mean(surv_at_true_opt(:, reg)) * 100;
    bar_data(reg, 3) = mean(surv_at_est_opt(:, reg)) * 100;
end

b = bar(bar_data);
b(1).FaceColor = [0.5 0.5 0.5];
b(2).FaceColor = [0.2 0.7 0.2];
b(3).FaceColor = [0.2 0.4 0.8];
set(gca, 'XTickLabel', {regimes.name});
ylabel('Survival (%)');
legend('Uniform (\omega=2)', 'Personalized (true \lambda)', ...
    'Personalized (estimated \lambda)', 'Location', 'best');
title('Can Estimated Parameters Recover Personalized Benefit?');
ylim([0 100]); grid on;

% Add improvement labels
for reg = 1:n_regimes
    for col = 2:3
        improvement = bar_data(reg, col) - bar_data(reg, 1);
        x_pos = reg + (col - 2.5) * 0.22;
        text(x_pos, bar_data(reg, col) + 1, sprintf('%+.1f%%', improvement), ...
            'HorizontalAlignment', 'center', 'FontSize', 8, 'FontWeight', 'bold');
    end
end

%% Save
% Persist both the selection and evaluation grids so downstream
% analyses can reconstruct the bias correction independently.
% surv_*_grid are aliases of surv_*_eval kept for backward compat.
save('personalized_validation_results.mat', 'omega_vals', ...
    'omega_true', 'omega_est', ...
    'surv_true_grid', 'surv_est_grid', ...
    'surv_true_select', 'surv_true_eval', ...
    'surv_est_select',  'surv_est_eval', ...
    'surv_at_true_opt', 'surv_at_est_opt', ...
    'regimes', 'Npat', 'Nruns', 'corr_Lcum');
fprintf('\nResults saved to personalized_validation_results.mat\n');

%% Conclusion
fprintf('\n=== KEY QUESTION ===\n');
fprintf('Can we recover the personalized omega benefit from estimated data?\n\n');

for reg = 1:n_regimes
    [~, idx_uniform] = min(abs(omega_vals - 2.0));
    surv_u = mean(surv_true_grid(:, idx_uniform, reg)) * 100;
    surv_pt = mean(surv_at_true_opt(:, reg)) * 100;
    surv_pe = mean(surv_at_est_opt(:, reg)) * 100;
    
    possible = surv_pt - surv_u;
    recovered = surv_pe - surv_u;
    if possible > 0.5
        pct_recovered = recovered / possible * 100;
        fprintf('  %s: %.0f%% of benefit recovered (%.1f / %.1f pp)\n', ...
            regimes(reg).name, pct_recovered, recovered, possible);
    else
        fprintf('  %s: benefit too small to measure (%.1f pp)\n', ...
            regimes(reg).name, possible);
    end
end
