% S7_ESTIMATION_ROBUSTNESS
% Tests how parameter estimation degrades as treatment fraction increases.
% Sweeps fraction of treated patients from 0% to 100%.
% Reports L_cum correlation, peak height correlation, and FIM condition number.

clear; clc;
addpath('../functions');

%% Load parameters
[time_params, drug_params, ctrl_params, mort_params] = default_params();
T = time_params.T;
Nt = time_params.Nt;
dt = time_params.dt;

%% Settings
Npat = 200;
treatment_fracs = [0.0, 0.25, 0.50, 0.75, 1.0];
n_fracs = numel(treatment_fracs);

rng(42);

patient_params.peak_rate = 10;
patient_params.drug_params = drug_params;

%% Generate patient population once
fprintf('=== ESTIMATION ROBUSTNESS ===\n');
fprintf('Patients: %d | Treatment fractions: %d\n\n', Npat, n_fracs);

patients = cell(Npat, 1);
for p = 1:Npat
    patients{p} = fcn_generate_patient(T, patient_params);
end
fprintf('Generated %d patients.\n\n', Npat);

%% Storage
results = struct();
results.frac = treatment_fracs;
results.corr_Lcum = zeros(n_fracs, 1);
results.corr_peak = zeros(n_fracs, 1);
results.corr_Lcum_ut = zeros(n_fracs, 1);
results.corr_Lcum_tx = zeros(n_fracs, 1);
results.corr_Ap = zeros(n_fracs, 1);
results.corr_r = zeros(n_fracs, 1);
results.corr_kappa = zeros(n_fracs, 1);
results.mape_Lcum = zeros(n_fracs, 1);
results.fim_cond = zeros(n_fracs, 1);
results.convergence = zeros(n_fracs, 1);

%% Main sweep
tic;
for fi = 1:n_fracs
    frac = treatment_fracs(fi);
    N_treated = round(Npat * frac);
    N_untreated = Npat - N_treated;
    
    fprintf('--- Fraction treated: %.0f%% (untreated=%d, treated=%d) ---\n', ...
        frac*100, N_untreated, N_treated);
    
    % Shuffle patient order so treatment assignment is random
    rng(42 + fi);
    order = randperm(Npat);
    
    true_Ap = zeros(Npat, 1);
    true_r = zeros(Npat, 1);
    true_kappa = zeros(Npat, 1);
    est_Ap = zeros(Npat, 1);
    est_r = zeros(Npat, 1);
    est_kappa = zeros(Npat, 1);
    true_Lcum = zeros(Npat, 1);
    est_Lcum = zeros(Npat, 1);
    true_peak = zeros(Npat, 1);
    est_peak = zeros(Npat, 1);
    is_treated = false(Npat, 1);
    exitflags = zeros(Npat, 1);
    fim_conds = zeros(Npat, 1);

    for p = 1:Npat
        pidx = order(p);
        patient = patients{pidx};
        
        dp = drug_params;
        dp.alpha = patient.alpha;
        if isfield(patient, 'C50')
            dp.patient_C50 = patient.C50;
            dp.patient_gamma = patient.gamma_hill;
        end
        
        true_Ap(p) = patient.A_patient;
        true_r(p) = patient.r;
        true_kappa(p) = patient.kappa;
        true_Lcum(p) = sum(patient.lambda_base) * dt;
        [pk, pk_idx] = max(patient.lambda_base);
        true_peak(p) = pk;
        
        if p <= N_untreated
            is_treated(p) = false;
            counts = poissrnd(patient.lambda_base * dt);
            
            [params_est, fit_info] = fcn_estimate_natural_history(counts, T, dt, ...
                'peak_rate', patient_params.peak_rate);
        else
            is_treated(p) = true;
            r = fcn_run_strategy('fixed', patient.lambda_base, dp, ctrl_params, T, dt, false);

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
        end
        
        est_Ap(p) = params_est.A_p;
        est_r(p) = params_est.r;
        est_kappa(p) = params_est.kappa;
        exitflags(p) = fit_info.exitflag;
        
        % Estimated L_cum and peak from fitted trajectory
        est_Lcum(p) = sum(fit_info.lambda_base_hat) * dt;
        est_peak(p) = max(fit_info.lambda_base_hat);

        % FIM condition number for this patient
        % Compute sensitivity of lambda_base to each parameter at the MLE
        log_Ap = log(params_est.A_p);
        log_r = log(params_est.r);
        log_d = log(params_est.kappa);
        
        shape_est = exp(-params_est.kappa*T) .* (1 - exp(-params_est.r*T));
        max_s = max(shape_est);
        if max_s > 1e-10
            lam_est = params_est.A_p * patient_params.peak_rate * shape_est / max_s;
            
            % Numerical partial derivatives (central difference, step=0.01)
            h_step = 0.01;
            J = zeros(3, Nt);
            for dim = 1:3
                psi_plus = [log_Ap, log_r, log_d];
                psi_minus = [log_Ap, log_r, log_d];
                psi_plus(dim) = psi_plus(dim) + h_step;
                psi_minus(dim) = psi_minus(dim) - h_step;
                
                lam_plus = build_lambda(psi_plus, T, patient_params.peak_rate);
                lam_minus = build_lambda(psi_minus, T, patient_params.peak_rate);
                J(dim, :) = (lam_plus - lam_minus) / (2 * h_step);
            end

            % FIM = sum over k of (1/lambda_k) * J(:,k) * J(:,k)' * dt
            FIM = zeros(3, 3);
            for k = 1:Nt
                if lam_est(k) > 1e-6
                    FIM = FIM + (1/lam_est(k)) * J(:,k) * J(:,k)' * dt;
                end
            end
            eigs_fim = eig(FIM);
            eigs_fim = sort(eigs_fim, 'descend');
            if eigs_fim(end) > 1e-15
                fim_conds(p) = eigs_fim(1) / eigs_fim(end);
            else
                fim_conds(p) = Inf;
            end
        else
            fim_conds(p) = Inf;
        end
    end
    
    % Store summary metrics for this fraction
    results.corr_Lcum(fi) = corr(true_Lcum, est_Lcum);
    results.corr_peak(fi) = corr(true_peak, est_peak);
    results.corr_Ap(fi) = corr(true_Ap, est_Ap);
    results.corr_r(fi) = corr(true_r, est_r);
    results.corr_kappa(fi) = corr(true_kappa, est_kappa);
    results.mape_Lcum(fi) = mean(abs(true_Lcum - est_Lcum) ./ max(true_Lcum, 0.1)) * 100;
    results.fim_cond(fi) = median(fim_conds(isfinite(fim_conds)));
    results.convergence(fi) = sum(exitflags == 1) / Npat * 100;

    % Per-group correlations
    idx_ut = ~is_treated;
    idx_tx = is_treated;
    if sum(idx_ut) > 2
        results.corr_Lcum_ut(fi) = corr(true_Lcum(idx_ut), est_Lcum(idx_ut));
    else
        results.corr_Lcum_ut(fi) = NaN;
    end
    if sum(idx_tx) > 2
        results.corr_Lcum_tx(fi) = corr(true_Lcum(idx_tx), est_Lcum(idx_tx));
    else
        results.corr_Lcum_tx(fi) = NaN;
    end
    
    fprintf('  L_cum corr=%.3f | Peak corr=%.3f | FIM cond=%.0f | Conv=%.0f%% (%.1fs)\n', ...
        results.corr_Lcum(fi), results.corr_peak(fi), results.fim_cond(fi), ...
        results.convergence(fi), toc);
end

fprintf('\nTotal time: %.1f seconds\n\n', toc);

%% Summary table
fprintf('=====================================================\n');
fprintf('ESTIMATION ROBUSTNESS: Effect of Treatment Fraction\n');
fprintf('=====================================================\n');
fprintf('%-8s %8s %8s %8s %8s %10s\n', ...
    'Frac', 'L_cum', 'Peak', 'A_p', 'MAPE%', 'FIM cond');
fprintf('%s\n', repmat('-', 1, 55));
for fi = 1:n_fracs
    fprintf('%-8.0f%% %7.3f %8.3f %8.3f %8.1f %10.0f\n', ...
        results.frac(fi)*100, results.corr_Lcum(fi), results.corr_peak(fi), ...
        results.corr_Ap(fi), results.mape_Lcum(fi), results.fim_cond(fi));
end

fprintf('\nPer-group L_cum correlation:\n');
fprintf('%-8s %10s %10s\n', 'Frac', 'Untreated', 'Treated');
fprintf('%s\n', repmat('-', 1, 30));
for fi = 1:n_fracs
    fprintf('%-8.0f%% %10.3f %10.3f\n', ...
        results.frac(fi)*100, results.corr_Lcum_ut(fi), results.corr_Lcum_tx(fi));
end

%% FIGURE 1: Correlation vs treatment fraction
figure('Position', [50 50 1200 400]); clf;

subplot(1,3,1);
plot(results.frac*100, results.corr_Lcum, 'bo-', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'b');
hold on;
plot(results.frac*100, results.corr_peak, 'rs-', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'r');
xlabel('Treatment Fraction (%)');
ylabel('Correlation (true vs estimated)');
title('Derived Quantity Recovery');
legend('L_{cum}', 'Peak Height', 'Location', 'best');
ylim([0 1]); grid on;

subplot(1,3,2);
plot(results.frac*100, results.corr_Ap, 'ko-', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'k');
hold on;
plot(results.frac*100, results.corr_r, 'g^-', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'g');
plot(results.frac*100, results.corr_kappa, 'md-', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'm');
xlabel('Treatment Fraction (%)');
ylabel('Correlation (true vs estimated)');
title('Raw Parameter Recovery');
legend('A_p', 'r', '\kappa', 'Location', 'best');
ylim([-0.2 1]); grid on;

subplot(1,3,3);
semilogy(results.frac*100, results.fim_cond, 'ko-', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'k');
xlabel('Treatment Fraction (%)');
ylabel('FIM Condition Number (log scale)');
title('Estimation Conditioning');
yline(1e4, 'r--', '\kappa = 10^4', 'LineWidth', 1.5);
grid on;

sgtitle(sprintf('Estimation Robustness (N=%d patients)', Npat));

%% FIGURE 2: L_cum correlation split by group
figure('Position', [50 50 600 400]); clf;

plot(results.frac*100, results.corr_Lcum, 'k-', 'LineWidth', 2.5, 'DisplayName', 'Overall');
hold on;
plot(results.frac*100, results.corr_Lcum_ut, 'b--', 'LineWidth', 2, 'DisplayName', 'Untreated');
plot(results.frac*100, results.corr_Lcum_tx, 'r--', 'LineWidth', 2, 'DisplayName', 'Treated');
xlabel('Treatment Fraction (%)');
ylabel('L_{cum} Correlation');
title('L_{cum} Recovery by Patient Group');
legend('Location', 'best');
ylim([0.5 1]); grid on;

%% Save
save('estimation_robustness_results.mat', 'results', 'Npat', 'treatment_fracs');
fprintf('\nResults saved to estimation_robustness_results.mat\n');

%% Helper function: build lambda from log-parameters
function lam = build_lambda(psi, T, peak_rate)
    Ap = exp(psi(1));
    r = exp(psi(2));
    d = exp(psi(3));
    shape = exp(-d*T) .* (1 - exp(-r*T));
    ms = max(shape);
    if ms < 1e-10
        lam = zeros(size(T));
    else
        lam = Ap * peak_rate * shape / ms;
    end
end
