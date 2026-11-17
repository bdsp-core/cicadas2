% S8_CLOSED_LOOP_VALIDATION
% End-to-end pipeline test: estimate parameters from observations, then run
% strategy comparison on estimated trajectories. Compare conclusions (which
% strategy wins each quadrant) to ground truth.
%
% Flow:
%   1. Generate patients with known {A_p, r, d}
%   2. Estimate {A_p, r, d} from Poisson counts (50% treated, 50% untreated)
%   3. Reconstruct lambda_base from estimated parameters
%   4. Run all strategies on BOTH true and estimated trajectories
%   5. Evaluate mortality under the s5 quadrant regimes
%   6. Compare survival rankings: true vs estimated

clear; clc;
addpath('../functions');

%% Load parameters
[time_params, drug_params, ctrl_params, mort_params] = default_params();
T = time_params.T;
Nt = time_params.Nt;
dt = time_params.dt;

%% Settings
Npat = 300;
Nruns = 100;
frac_treated = 0.5;
N_treated = round(Npat * frac_treated);
N_untreated = Npat - N_treated;
use_ekf = true;

% Personalized adaptive (omega*) — 5th strategy.
% Sweep these omega values per patient per regime on BOTH true and
% estimated trajectories; pick the argmax-survival omega as omega_i*
% (separately for true and est). Set ENABLE_OMEGA_STAR=false to keep
% the original 4-strategy comparison.
ENABLE_OMEGA_STAR = true;
omega_vals = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, Inf];

% PKPD knowledge mode:
%   'oracle'   = use patient-specific C50, gamma (unrealistically good)
%   'pop_mean' = use population-mean C50, gamma (realistic clinical scenario)
pkpd_mode = 'pop_mean';

rng(42);

patient_params.peak_rate = 10;
patient_params.drug_params = drug_params;

%% Define quadrant regimes (must match s5)
% See s5 for sweep logic. theta4_high controls the "High Drug" axis.
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
% Base strategies (always run). Personalized adaptive (omega*) is
% appended as a 5th strategy below if ENABLE_OMEGA_STAR=true.
strategies = {'none', 'fixed', 'prn', 'adaptive'};
strat_labels = {'None', 'Fixed', 'PRN', 'Adaptive'};
n_base = numel(strategies);
if ENABLE_OMEGA_STAR
    strategies = [strategies, {'adaptive_omega_star'}];
    strat_labels = [strat_labels, {'Adapt(w*)'}];
end
n_strats = numel(strategies);

%% Step 1: Generate patients
fprintf('=== CLOSED-LOOP VALIDATION ===\n');
fprintf('Patients: %d | Mortality runs: %d | Treated: %.0f%%\n', Npat, Nruns, frac_treated*100);
fprintf('PKPD mode: %s\n\n', pkpd_mode);

patients = cell(Npat, 1);
for p = 1:Npat
    patients{p} = fcn_generate_patient(T, patient_params);
end
fprintf('Generated %d patients.\n', Npat);

%% Step 2: Estimate parameters from observations
fprintf('Estimating natural history parameters...\n');
tic;

est_lambda_base = zeros(Npat, Nt);
true_lambda_base = zeros(Npat, Nt);
est_Lcum = zeros(Npat, 1);
true_Lcum = zeros(Npat, 1);

for p = 1:Npat
    patient = patients{p};
    dp = drug_params;
    dp.alpha = patient.alpha;
    if isfield(patient, 'C50')
        dp.patient_C50 = patient.C50;
        dp.patient_gamma = patient.gamma_hill;
    end
    
    true_lambda_base(p,:) = patient.lambda_base;
    true_Lcum(p) = sum(patient.lambda_base) * dt;

    if p <= N_untreated
        counts = poissrnd(patient.lambda_base * dt);
        [params_est, fit_info] = fcn_estimate_natural_history(counts, T, dt, ...
            'peak_rate', patient_params.peak_rate);
    else
        r = fcn_run_strategy('fixed', patient.lambda_base, dp, ctrl_params, T, dt, false);
        
        % Select PKPD params for estimation
        if strcmpi(pkpd_mode, 'oracle')
            est_C50 = dp.patient_C50;
            est_gamma = dp.patient_gamma;
        else  % pop_mean
            est_C50 = drug_params.C50;          % population mean (500)
            est_gamma = drug_params.gamma_hill;  % population mean (2)
        end
        
        if strcmpi(drug_params.pd_mode, 'hill')
            [params_est, fit_info] = fcn_estimate_natural_history(r.seizure_counts, T, dt, ...
                'peak_rate', patient_params.peak_rate, ...
                'k_e', dp.k_e, ...
                'doses_given', r.doses_given, ...
                'pd_mode', 'hill', ...
                'C50', est_C50, ...
                'gamma_hill', est_gamma);
        else
            [params_est, fit_info] = fcn_estimate_natural_history(r.seizure_counts, T, dt, ...
                'peak_rate', patient_params.peak_rate, ...
                'k_e', dp.k_e, ...
                'doses_given', r.doses_given, ...
                'pd_mode', 'linear', ...
                'alpha', dp.alpha);
        end
    end
    
    est_lambda_base(p,:) = fit_info.lambda_base_hat;
    est_Lcum(p) = sum(fit_info.lambda_base_hat) * dt;
    
    if mod(p, 50) == 0
        fprintf('  %d/%d estimated (%.1fs)\n', p, Npat, toc);
    end
end

corr_Lcum = corr(true_Lcum, est_Lcum);
fprintf('Estimation done. L_cum corr=%.3f (%.1fs)\n\n', corr_Lcum, toc);

%% Step 3: Run base strategies on TRUE and ESTIMATED trajectories
fprintf('Running base strategies on true and estimated trajectories...\n');
tic;

% Store base-strategy results for both
true_strat = cell(Npat, n_base);   % results from true lambda_base
est_strat = cell(Npat, n_base);    % results from estimated lambda_base

for p = 1:Npat
    patient = patients{p};
    dp = drug_params;
    dp.alpha = patient.alpha;
    if isfield(patient, 'C50')
        dp.patient_C50 = patient.C50;
        dp.patient_gamma = patient.gamma_hill;
    end

    for s = 1:n_base
        true_strat{p,s} = fcn_run_strategy(strategies{s}, true_lambda_base(p,:), ...
            dp, ctrl_params, T, dt, use_ekf);
        est_strat{p,s} = fcn_run_strategy(strategies{s}, est_lambda_base(p,:), ...
            dp, ctrl_params, T, dt, use_ekf);
    end

    if mod(p, 100) == 0
        fprintf('  %d/%d strategies done (%.1fs)\n', p, Npat, toc);
    end
end
fprintf('Base strategies done (%.1fs).\n\n', toc);

%% Step 3b: Adaptive omega sweep on true and estimated trajectories
% Pre-run adaptive at every omega in omega_vals on both true and est
% trajectories; later, per regime, we pick the argmax-survival omega
% as omega_i*. omega = Inf is treated as 'none' (never trigger).
n_omega = numel(omega_vals);
true_omega = cell(Npat, n_omega);
est_omega  = cell(Npat, n_omega);

if ENABLE_OMEGA_STAR
    fprintf('Sweeping adaptive omega (%d values) on true and est trajectories...\n', n_omega);
    tic;
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
                true_omega{p, oi} = fcn_run_strategy('none', true_lambda_base(p,:), ...
                    dp, cp, T, dt, use_ekf);
                est_omega{p, oi}  = fcn_run_strategy('none', est_lambda_base(p,:), ...
                    dp, cp, T, dt, use_ekf);
            else
                true_omega{p, oi} = fcn_run_strategy('adaptive', true_lambda_base(p,:), ...
                    dp, cp, T, dt, use_ekf);
                est_omega{p, oi}  = fcn_run_strategy('adaptive', est_lambda_base(p,:), ...
                    dp, cp, T, dt, use_ekf);
            end
        end

        if mod(p, 50) == 0
            fprintf('  %d/%d swept (%.1fs)\n', p, Npat, toc);
        end
    end
    fprintf('Omega sweep done (%.1fs).\n\n', toc);
end

%% Step 4: Evaluate mortality under each regime
fprintf('Evaluating mortality (%d regimes x %d runs)...\n', n_regimes, Nruns);
tic;

surv_true = zeros(n_regimes, n_strats);   % population survival rates (true)
surv_est = zeros(n_regimes, n_strats);    % population survival rates (estimated)

% Per-patient omega* tracking (only used if ENABLE_OMEGA_STAR)
omega_star_true = nan(Npat, n_regimes);
omega_star_est  = nan(Npat, n_regimes);

% Per-patient survival matrices for the final regime (kept for compat).
% We allocate fresh per regime inside the loop.
for reg = 1:n_regimes
    fprintf('  Regime %d: %s\n', reg, regimes(reg).name);

    surv_pat_true = zeros(Npat, n_strats);
    surv_pat_est  = zeros(Npat, n_strats);

    for p = 1:Npat
        patient = patients{p};

        % --- Base strategies (columns 1..n_base) ---
        for s = 1:n_base
            rt = true_strat{p,s};
            re = est_strat{p,s};

            alive_true = 0;
            alive_est  = 0;
            for run = 1:Nruns
                [a, ~] = fcn_mortality(rt.lambda_obs, rt.x, dt, ...
                    regimes(reg).theta, patient.age, patient.sofa);
                alive_true = alive_true + a;

                [a, ~] = fcn_mortality(re.lambda_obs, re.x, dt, ...
                    regimes(reg).theta, patient.age, patient.sofa);
                alive_est = alive_est + a;
            end
            surv_pat_true(p,s) = alive_true / Nruns;
            surv_pat_est(p,s)  = alive_est  / Nruns;
        end

        % --- Personalized adaptive omega* (column n_base+1, if enabled) ---
        %
        % BIAS CORRECTION (split-sample / honest argmax):
        %   Picking omega_i* as argmax over noisy Monte Carlo survival
        %   estimates and then reporting that same estimate biases the
        %   reported survival upward (winner's curse). We split the MC
        %   draws into independent SELECTION and EVALUATION halves:
        %
        %     selection set  -> argmax over omega -> omega_i*
        %     evaluation set -> survival reported at omega_i*  (unbiased)
        %
        %   Both halves are taken over the SAME true_omega / est_omega
        %   trajectories (only mortality draws differ), so this corrects
        %   the selection bias without changing the experiment. Total
        %   mortality calls per (p, oi) on each of true/est: 2*Nruns.
        %   The four base strategies are unaffected (no argmax over noise).
        if ENABLE_OMEGA_STAR
            sel_t  = zeros(1, n_omega);   eval_t = zeros(1, n_omega);
            sel_e  = zeros(1, n_omega);   eval_e = zeros(1, n_omega);
            for oi = 1:n_omega
                rt = true_omega{p, oi};
                re = est_omega{p, oi};

                % --- Selection draws (true & est) ---
                at = 0; ae = 0;
                for run = 1:Nruns
                    [a, ~] = fcn_mortality(rt.lambda_obs, rt.x, dt, ...
                        regimes(reg).theta, patient.age, patient.sofa);
                    at = at + a;
                    [a, ~] = fcn_mortality(re.lambda_obs, re.x, dt, ...
                        regimes(reg).theta, patient.age, patient.sofa);
                    ae = ae + a;
                end
                sel_t(oi) = at / Nruns;
                sel_e(oi) = ae / Nruns;

                % --- Evaluation draws (independent: fresh RNG state) ---
                at = 0; ae = 0;
                for run = 1:Nruns
                    [a, ~] = fcn_mortality(rt.lambda_obs, rt.x, dt, ...
                        regimes(reg).theta, patient.age, patient.sofa);
                    at = at + a;
                    [a, ~] = fcn_mortality(re.lambda_obs, re.x, dt, ...
                        regimes(reg).theta, patient.age, patient.sofa);
                    ae = ae + a;
                end
                eval_t(oi) = at / Nruns;
                eval_e(oi) = ae / Nruns;
            end
            % Pick omega from selection set; report survival from eval set
            [~, best_t] = max(sel_t);
            [~, best_e] = max(sel_e);
            surv_pat_true(p, n_base+1) = eval_t(best_t);
            surv_pat_est(p,  n_base+1) = eval_e(best_e);
            omega_star_true(p, reg) = omega_vals(best_t);
            omega_star_est(p, reg)  = omega_vals(best_e);
        end
    end

    surv_true(reg,:) = mean(surv_pat_true) * 100;
    surv_est(reg,:)  = mean(surv_pat_est)  * 100;
    fprintf('    Done (%.1fs)\n', toc);
end
fprintf('Mortality evaluation done (%.1fs).\n\n', toc);

%% Step 5: Compare results
fprintf('=====================================================\n');
fprintf('CLOSED-LOOP VALIDATION: True vs Estimated Parameters\n');
fprintf('=====================================================\n');
fprintf('L_cum estimation correlation: %.3f\n\n', corr_Lcum);

fprintf('%-22s |', 'Regime');
for s = 1:n_strats
    fprintf(' %8s', strat_labels{s});
end
fprintf(' | %8s\n', 'Winner');
fprintf('%s\n', repmat('-', 1, 85));

winner_match = 0;
for reg = 1:n_regimes
    % True
    [~, best_true] = max(surv_true(reg,:));
    fprintf('%-22s |', [regimes(reg).name ' (true)']);
    for s = 1:n_strats
        if s == best_true
            fprintf(' %6.1f%%*', surv_true(reg,s));
        else
            fprintf(' %7.1f%%', surv_true(reg,s));
        end
    end
    fprintf(' | %s\n', strat_labels{best_true});
    
    % Estimated
    [~, best_est] = max(surv_est(reg,:));
    fprintf('%-22s |', [regimes(reg).name ' (est)']);
    for s = 1:n_strats
        if s == best_est
            fprintf(' %6.1f%%*', surv_est(reg,s));
        else
            fprintf(' %7.1f%%', surv_est(reg,s));
        end
    end
    fprintf(' | %s', strat_labels{best_est});

    if best_true == best_est
        fprintf('  [MATCH]\n');
        winner_match = winner_match + 1;
    else
        fprintf('  [MISMATCH]\n');
    end
    fprintf('%s\n', repmat('-', 1, 85));
end

fprintf('\nWinner agreement: %d / %d regimes (%.0f%%)\n', ...
    winner_match, n_regimes, winner_match/n_regimes*100);

% Compute per-regime absolute differences
fprintf('\nSurvival difference (estimated - true):\n');
fprintf('%-22s |', 'Regime');
for s = 1:n_strats
    fprintf(' %9s', strat_labels{s});
end
fprintf('\n');
fprintf('%s\n', repmat('-', 1, 75));
for reg = 1:n_regimes
    fprintf('%-22s |', regimes(reg).name);
    for s = 1:n_strats
        diff_val = surv_est(reg,s) - surv_true(reg,s);
        fprintf(' %+8.1f%%', diff_val);
    end
    fprintf('\n');
end

% Personalized adaptive (omega*) per-regime summary
if ENABLE_OMEGA_STAR
    fprintf('\n=====================================================\n');
    fprintf('PERSONALIZED ADAPTIVE (omega*) — true vs estimated\n');
    fprintf('=====================================================\n');
    fprintf('%-22s %12s %12s\n', 'Regime', 'med w*_true', 'med w*_est');
    fprintf('%s\n', repmat('-', 1, 50));
    for reg = 1:n_regimes
        mt = median(omega_star_true(:, reg));
        me = median(omega_star_est(:, reg));
        if isinf(mt), st = '  Inf'; else, st = sprintf('%6.2f', mt); end
        if isinf(me), se = '  Inf'; else, se = sprintf('%6.2f', me); end
        fprintf('%-22s %12s %12s\n', regimes(reg).name, st, se);
    end
end

%% FIGURE 1: Side-by-side quadrant comparison
figure('Position', [50 50 1400 600]); clf;

% 5-strategy color palette (5th is purple for personalized adaptive)
colors = [0.7 0.7 0.7;   % None
          0.3 0.3 0.8;   % Fixed
          0.8 0.3 0.3;   % PRN
          0.3 0.8 0.3;   % Adaptive
          0.5 0.2 0.7];  % Adapt(w*)

for reg = 1:n_regimes
    subplot(2, n_regimes, reg);
    b = bar(surv_true(reg,:), 'FaceColor', 'flat');
    for s = 1:n_strats
        b.CData(s,:) = colors(s,:);
    end
    set(gca, 'XTickLabel', strat_labels);
    xtickangle(20);
    ylabel('Survival (%)');
    ylim([0 100]);
    [~, best] = max(surv_true(reg,:));
    title(sprintf('%s\nTRUE: %s (%.0f%%)', regimes(reg).name, strat_labels{best}, max(surv_true(reg,:))));
    grid on;

    subplot(2, n_regimes, n_regimes + reg);
    b = bar(surv_est(reg,:), 'FaceColor', 'flat');
    for s = 1:n_strats
        b.CData(s,:) = colors(s,:);
    end
    set(gca, 'XTickLabel', strat_labels);
    xtickangle(20);
    ylabel('Survival (%)');
    ylim([0 100]);
    [~, best] = max(surv_est(reg,:));
    title(sprintf('EST: %s (%.0f%%)', strat_labels{best}, max(surv_est(reg,:))));
    grid on;
end

sgtitle(sprintf('Closed-Loop Validation: True (top) vs Estimated (bottom), N=%d', Npat));

%% FIGURE 2: Survival correlation (true vs estimated) per regime
figure('Position', [50 50 1000 400]); clf;

for reg = 1:n_regimes
    subplot(1, n_regimes, reg);
    hold on;
    for s = 1:n_strats
        scatter(surv_true(reg,s), surv_est(reg,s), 150, colors(s,:), 'filled', ...
            'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
    end
    % Identity line
    lims = [0 100];
    plot(lims, lims, 'k--', 'LineWidth', 1);
    xlabel('True Survival (%)');
    ylabel('Estimated Survival (%)');
    title(regimes(reg).name);
    xlim(lims); ylim(lims);
    grid on;
    if reg == n_regimes
        legend(strat_labels, 'Location', 'best');
    end
end

sgtitle('Per-Strategy Survival: True vs Estimated');

%% FIGURE 3: Difference heatmap
figure('Position', [50 50 700 400]); clf;

diff_matrix = surv_est - surv_true;
imagesc(diff_matrix);
colormap(redblue(64));
colorbar;
clim([-10 10]);

set(gca, 'XTick', 1:n_strats, 'XTickLabel', strat_labels);
set(gca, 'YTick', 1:n_regimes, 'YTickLabel', {regimes.name});
xlabel('Strategy');
ylabel('Regime');
title('Survival Difference: Estimated - True (percentage points)');

for reg = 1:n_regimes
    for s = 1:n_strats
        text(s, reg, sprintf('%+.1f', diff_matrix(reg,s)), ...
            'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 11);
    end
end

%% Save
save('closed_loop_results.mat', 'surv_true', 'surv_est', 'regimes', ...
    'strategies', 'strat_labels', 'corr_Lcum', 'Npat', 'Nruns', 'pkpd_mode', ...
    'ENABLE_OMEGA_STAR', 'omega_vals', 'omega_star_true', 'omega_star_est');
fprintf('\nResults saved to closed_loop_results.mat\n');

fprintf('\n=== CONCLUSION ===\n');
if winner_match == n_regimes
    fprintf('All %d regimes have matching winners.\n', n_regimes);
    fprintf('The pipeline is validated: estimate -> optimize -> same conclusions as ground truth.\n');
else
    fprintf('%d / %d regimes have matching winners.\n', winner_match, n_regimes);
    fprintf('Mismatches may indicate estimation error propagation in tight races.\n');
end

%% Helper: red-blue diverging colormap
function cmap = redblue(n)
    if nargin < 1, n = 64; end
    half = floor(n/2);
    r = [linspace(0, 1, half), ones(1, half)];
    g = [linspace(0, 1, half), linspace(1, 0, half)];
    b = [ones(1, half), linspace(1, 0, half)];
    cmap = [r', g', b'];
end
