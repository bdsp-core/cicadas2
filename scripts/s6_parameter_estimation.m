% S6_PARAMETER_ESTIMATION
% Recovers {A_p, r, d} from simulated observations.
%   Untreated: direct Poisson MLE on lambda_base.
%   Treated: invert known PKPD to recover lambda_base, then MLE.
% Corresponds to CICADAS Stage 1 (natural history estimation).

clear; clc;
addpath('functions');

%% Load parameters
[time_params, drug_params, ctrl_params, mort_params] = default_params();
T = time_params.T;
Nt = time_params.Nt;
dt = time_params.dt;

%% Simulation settings
Npat = 200;         % Patients total
frac_treated = 0.5; % Fraction receiving treatment
N_treated = round(Npat * frac_treated);
N_untreated = Npat - N_treated;

rng(42);  % Reproducibility

patient_params.peak_rate = 10;
patient_params.drug_params = drug_params;

%% Storage
true_Ap = zeros(Npat, 1);
true_r = zeros(Npat, 1);
true_kappa = zeros(Npat, 1);
est_Ap = zeros(Npat, 1);
est_r = zeros(Npat, 1);
est_kappa = zeros(Npat, 1);
is_treated = false(Npat, 1);
fit_exitflag = zeros(Npat, 1);
patient_lambdas = zeros(Npat, Nt);  % store true trajectories for plotting
true_Lcum = zeros(Npat, 1);         % true integrated burden
est_Lcum = zeros(Npat, 1);          % estimated integrated burden

%% Main loop
fprintf('=== PARAMETER ESTIMATION ===\n');
fprintf('Patients: %d (untreated: %d, treated: %d)\n', Npat, N_untreated, N_treated);
fprintf('\n');

tic;
for p = 1:Npat
    % Generate patient
    patient = fcn_generate_patient(T, patient_params);
    dp = drug_params;
    dp.alpha = patient.alpha;
    if isfield(patient, 'C50')
        dp.patient_C50 = patient.C50;
        dp.patient_gamma = patient.gamma_hill;
    end
    
    % Store ground truth
    true_Ap(p) = patient.A_patient;
    true_r(p) = patient.r;
    true_kappa(p) = patient.kappa;
    patient_lambdas(p,:) = patient.lambda_base;
    
    % Decide if treated
    if p <= N_untreated
        % --- UNTREATED ---
        is_treated(p) = false;
        
        % Generate Poisson counts from lambda_base
        counts = poissrnd(patient.lambda_base * dt);
        
        % Estimate parameters
        [params_est, fit_info] = fcn_estimate_natural_history(counts, T, dt, ...
            'peak_rate', patient_params.peak_rate);
        
    else
        % --- TREATED (fixed dose q8h) ---
        is_treated(p) = true;
        
        % Run fixed strategy to get dosing and observed rate
        r = fcn_run_strategy('fixed', patient.lambda_base, dp, ctrl_params, T, dt, false);
        
        % Estimate parameters using known PKPD
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
    
    % Store estimates
    est_Ap(p) = params_est.A_p;
    est_r(p) = params_est.r;
    est_kappa(p) = params_est.kappa;
    fit_exitflag(p) = fit_info.exitflag;
    
    % Compute true and estimated L_cum (integrated burden)
    true_Lcum(p) = sum(patient.lambda_base) * dt;
    est_Lcum(p) = sum(fit_info.lambda_base_hat) * dt;
    
    if mod(p, 50) == 0
        fprintf('  %d/%d patients (%.1f sec)\n', p, Npat, toc);
    end
end

fprintf('\nEstimation complete. Total time: %.1f seconds\n\n', toc);

%% Compute error metrics
% Separate by treated/untreated
idx_ut = ~is_treated;
idx_tx = is_treated;

fprintf('=============================================\n');
fprintf('PARAMETER RECOVERY: UNTREATED (N=%d)\n', sum(idx_ut));
fprintf('=============================================\n');

% A_p
corr_Ap_ut = corr(true_Ap(idx_ut), est_Ap(idx_ut));
mape_Ap_ut = mean(abs(true_Ap(idx_ut) - est_Ap(idx_ut)) ./ true_Ap(idx_ut)) * 100;
fprintf('A_p:  corr=%.3f  MAPE=%.1f%%\n', corr_Ap_ut, mape_Ap_ut);

% r
corr_r_ut = corr(true_r(idx_ut), est_r(idx_ut));
mape_r_ut = mean(abs(true_r(idx_ut) - est_r(idx_ut)) ./ true_r(idx_ut)) * 100;
fprintf('r:    corr=%.3f  MAPE=%.1f%%\n', corr_r_ut, mape_r_ut);

% d
corr_kappa_ut = corr(true_kappa(idx_ut), est_kappa(idx_ut));
mape_kappa_ut = mean(abs(true_kappa(idx_ut) - est_kappa(idx_ut)) ./ true_kappa(idx_ut)) * 100;
fprintf('kappa: corr=%.3f  MAPE=%.1f%%\n', corr_kappa_ut, mape_kappa_ut);

fprintf('\n=============================================\n');
fprintf('PARAMETER RECOVERY: TREATED (N=%d)\n', sum(idx_tx));
fprintf('=============================================\n');

corr_Ap_tx = corr(true_Ap(idx_tx), est_Ap(idx_tx));
mape_Ap_tx = mean(abs(true_Ap(idx_tx) - est_Ap(idx_tx)) ./ true_Ap(idx_tx)) * 100;
fprintf('A_p:  corr=%.3f  MAPE=%.1f%%\n', corr_Ap_tx, mape_Ap_tx);

corr_r_tx = corr(true_r(idx_tx), est_r(idx_tx));
mape_r_tx = mean(abs(true_r(idx_tx) - est_r(idx_tx)) ./ true_r(idx_tx)) * 100;
fprintf('r:    corr=%.3f  MAPE=%.1f%%\n', corr_r_tx, mape_r_tx);

corr_kappa_tx = corr(true_kappa(idx_tx), est_kappa(idx_tx));
mape_kappa_tx = mean(abs(true_kappa(idx_tx) - est_kappa(idx_tx)) ./ true_kappa(idx_tx)) * 100;
fprintf('d:    corr=%.3f  MAPE=%.1f%%\n', corr_kappa_tx, mape_kappa_tx);

% Convergence
fprintf('\nConvergence: %d/%d succeeded (exitflag=1)\n', sum(fit_exitflag == 1), Npat);

%% Derived quantity recovery: L_cum, peak height, peak timing
% These matter more than individual (r,d) for downstream mortality prediction
fprintf('\n=============================================\n');
fprintf('DERIVED QUANTITY RECOVERY (what actually matters)\n');
fprintf('=============================================\n');

% Compute peak height and peak timing from estimated trajectories
true_peak_height = zeros(Npat, 1);
est_peak_height = zeros(Npat, 1);
true_peak_time = zeros(Npat, 1);
est_peak_time = zeros(Npat, 1);

for p = 1:Npat
    % True
    [true_peak_height(p), idx_pk] = max(patient_lambdas(p,:));
    true_peak_time(p) = T(idx_pk);
    
    % Estimated
    shape_est = exp(-est_kappa(p)*T) .* (1 - exp(-est_r(p)*T));
    max_shape_est = max(shape_est);
    if max_shape_est < 1e-10
        lambda_est = zeros(size(T));
    else
        lambda_est = est_Ap(p) * patient_params.peak_rate * shape_est / max_shape_est;
    end
    [est_peak_height(p), idx_pk_est] = max(lambda_est);
    est_peak_time(p) = T(idx_pk_est);
end

% L_cum
corr_Lcum = corr(true_Lcum, est_Lcum);
mape_Lcum = mean(abs(true_Lcum - est_Lcum) ./ max(true_Lcum, 0.1)) * 100;
corr_Lcum_ut = corr(true_Lcum(idx_ut), est_Lcum(idx_ut));
corr_Lcum_tx = corr(true_Lcum(idx_tx), est_Lcum(idx_tx));

fprintf('\nL_cum (integrated burden — feeds directly into mortality model):\n');
fprintf('  Overall:   corr=%.3f  MAPE=%.1f%%\n', corr_Lcum, mape_Lcum);
fprintf('  Untreated: corr=%.3f\n', corr_Lcum_ut);
fprintf('  Treated:   corr=%.3f\n', corr_Lcum_tx);

% Peak height
corr_pkh = corr(true_peak_height, est_peak_height);
corr_pkh_ut = corr(true_peak_height(idx_ut), est_peak_height(idx_ut));
corr_pkh_tx = corr(true_peak_height(idx_tx), est_peak_height(idx_tx));
fprintf('\nPeak height:\n');
fprintf('  Overall:   corr=%.3f\n', corr_pkh);
fprintf('  Untreated: corr=%.3f\n', corr_pkh_ut);
fprintf('  Treated:   corr=%.3f\n', corr_pkh_tx);

% Peak timing
corr_pkt = corr(true_peak_time, est_peak_time);
corr_pkt_ut = corr(true_peak_time(idx_ut), est_peak_time(idx_ut));
corr_pkt_tx = corr(true_peak_time(idx_tx), est_peak_time(idx_tx));
mae_pkt = mean(abs(true_peak_time - est_peak_time));
fprintf('\nPeak timing:\n');
fprintf('  Overall:   corr=%.3f  MAE=%.2f days\n', corr_pkt, mae_pkt);
fprintf('  Untreated: corr=%.3f\n', corr_pkt_ut);
fprintf('  Treated:   corr=%.3f\n', corr_pkt_tx);

%% Population-level distribution recovery
fprintf('\n=============================================\n');
fprintf('POPULATION DISTRIBUTION RECOVERY\n');
fprintf('=============================================\n');

fprintf('\n  A_p (true log-mean=%.2f, log-std=%.2f):\n', mean(log(true_Ap)), std(log(true_Ap)));
fprintf('       est  log-mean=%.2f, log-std=%.2f\n', mean(log(est_Ap)), std(log(est_Ap)));

fprintf('\n  r   (true log-mean=%.2f, log-std=%.2f):\n', mean(log(true_r)), std(log(true_r)));
fprintf('       est  log-mean=%.2f, log-std=%.2f\n', mean(log(est_r)), std(log(est_r)));

fprintf('\n  d   (true log-mean=%.2f, log-std=%.2f):\n', mean(log(true_kappa)), std(log(true_kappa)));
fprintf('       est  log-mean=%.2f, log-std=%.2f\n', mean(log(est_kappa)), std(log(est_kappa)));

%% FIGURE 1: Scatter plots — true vs estimated (UNIFORM AXES)
figure('Position', [50 50 1200 800]); clf;

% Compute shared axis limits across untreated and treated
lim_Ap = [0, max([true_Ap; est_Ap]) * 1.05];
lim_r = [0, min(max([true_r; est_r]) * 1.05, 5)];  % cap at 5 to exclude outliers
lim_kappa = [0, min(max([true_kappa; est_kappa]) * 1.05, 2)];   % cap at 2

% --- Row 1: Untreated ---
subplot(2,3,1)
scatter(true_Ap(idx_ut), est_Ap(idx_ut), 30, 'b', 'filled', 'MarkerFaceAlpha', 0.5);
hold on; plot(lim_Ap, lim_Ap, 'k--', 'LineWidth', 1);
xlabel('True A_p'); ylabel('Estimated A_p');
title(sprintf('A_p (untreated) r=%.2f', corr_Ap_ut));
grid on; xlim(lim_Ap); ylim(lim_Ap);

subplot(2,3,2)
scatter(true_r(idx_ut), est_r(idx_ut), 30, 'b', 'filled', 'MarkerFaceAlpha', 0.5);
hold on; plot(lim_r, lim_r, 'k--', 'LineWidth', 1);
xlabel('True r'); ylabel('Estimated r');
title(sprintf('r (untreated) r=%.2f', corr_r_ut));
grid on; xlim(lim_r); ylim(lim_r);

subplot(2,3,3)
scatter(true_kappa(idx_ut), est_kappa(idx_ut), 30, 'b', 'filled', 'MarkerFaceAlpha', 0.5);
hold on; plot(lim_kappa, lim_kappa, 'k--', 'LineWidth', 1);
xlabel('True d'); ylabel('Estimated d');
title(sprintf('d (untreated) r=%.2f', corr_kappa_ut));
grid on; xlim(lim_kappa); ylim(lim_kappa);

% --- Row 2: Treated ---
subplot(2,3,4)
scatter(true_Ap(idx_tx), est_Ap(idx_tx), 30, 'r', 'filled', 'MarkerFaceAlpha', 0.5);
hold on; plot(lim_Ap, lim_Ap, 'k--', 'LineWidth', 1);
xlabel('True A_p'); ylabel('Estimated A_p');
title(sprintf('A_p (treated) r=%.2f', corr_Ap_tx));
grid on; xlim(lim_Ap); ylim(lim_Ap);

subplot(2,3,5)
scatter(true_r(idx_tx), est_r(idx_tx), 30, 'r', 'filled', 'MarkerFaceAlpha', 0.5);
hold on; plot(lim_r, lim_r, 'k--', 'LineWidth', 1);
xlabel('True r'); ylabel('Estimated r');
title(sprintf('r (treated) r=%.2f', corr_r_tx));
grid on; xlim(lim_r); ylim(lim_r);

subplot(2,3,6)
scatter(true_kappa(idx_tx), est_kappa(idx_tx), 30, 'r', 'filled', 'MarkerFaceAlpha', 0.5);
hold on; plot(lim_kappa, lim_kappa, 'k--', 'LineWidth', 1);
xlabel('True d'); ylabel('Estimated d');
title(sprintf('d (treated) r=%.2f', corr_kappa_tx));
grid on; xlim(lim_kappa); ylim(lim_kappa);

sgtitle('Parameter Recovery: True vs Estimated');

%% FIGURE 2: Example trajectory fits
figure('Position', [50 50 1200 600]); clf;

% Pick 4 example patients (2 untreated, 2 treated)
examples = [1, round(N_untreated/2), N_untreated+1, N_untreated+round(N_treated/2)];

for ei = 1:4
    pidx = examples(ei);
    
    % Fitted trajectory
    shape_est = exp(-est_kappa(pidx)*T) .* (1 - exp(-est_r(pidx)*T));
    max_shape_est = max(shape_est);
    if max_shape_est < 1e-10
        lambda_est = zeros(size(T));
    else
        lambda_est = est_Ap(pidx) * patient_params.peak_rate * shape_est / max_shape_est;
    end
    
    subplot(2,2,ei)
    plot(T, patient_lambdas(pidx,:), 'k-', 'LineWidth', 2); hold on;
    plot(T, lambda_est, 'r--', 'LineWidth', 2);
    if is_treated(pidx)
        title_str = sprintf('Patient %d (TREATED)', pidx);
    else
        title_str = sprintf('Patient %d (untreated)', pidx);
    end
    xlabel('Days'); ylabel('Seizures/day');
    title(title_str);
    legend('True \lambda_{base}', 'Estimated', 'Location', 'best');
    xlim([0 T(end)]); grid on;
end

sgtitle('Example Trajectory Fits');

%% FIGURE 3: Population distribution recovery
figure('Position', [50 50 1200 350]); clf;

subplot(1,3,1)
histogram(log(true_Ap), 20, 'FaceColor', [0.3 0.3 0.8], 'FaceAlpha', 0.5, 'DisplayName', 'True');
hold on;
histogram(log(est_Ap), 20, 'FaceColor', [0.8 0.3 0.3], 'FaceAlpha', 0.5, 'DisplayName', 'Estimated');
xlabel('log(A_p)'); ylabel('Count');
title('Severity Distribution');
legend('Location', 'best'); grid on;

subplot(1,3,2)
histogram(log(true_r), 20, 'FaceColor', [0.3 0.3 0.8], 'FaceAlpha', 0.5, 'DisplayName', 'True');
hold on;
histogram(log(est_r), 20, 'FaceColor', [0.8 0.3 0.3], 'FaceAlpha', 0.5, 'DisplayName', 'Estimated');
xlabel('log(r)'); ylabel('Count');
title('Rise Rate Distribution');
legend('Location', 'best'); grid on;

subplot(1,3,3)
histogram(log(true_kappa), 20, 'FaceColor', [0.3 0.3 0.8], 'FaceAlpha', 0.5, 'DisplayName', 'True');
hold on;
histogram(log(est_kappa), 20, 'FaceColor', [0.8 0.3 0.3], 'FaceAlpha', 0.5, 'DisplayName', 'Estimated');
xlabel('log(d)'); ylabel('Count');
title('Decay Rate Distribution');
legend('Location', 'best'); grid on;

sgtitle('Population-Level Distribution Recovery');

%% FIGURE 4: Derived quantity recovery (UNIFORM AXES)
figure('Position', [50 50 1200 400]); clf;

% L_cum: true vs estimated
lim_lc = [0, max([true_Lcum; est_Lcum]) * 1.05];
subplot(1,3,1)
scatter(true_Lcum(idx_ut), est_Lcum(idx_ut), 30, 'b', 'filled', 'MarkerFaceAlpha', 0.5); hold on;
scatter(true_Lcum(idx_tx), est_Lcum(idx_tx), 30, 'r', 'filled', 'MarkerFaceAlpha', 0.5);
plot(lim_lc, lim_lc, 'k--', 'LineWidth', 1);
xlabel('True L_{cum}'); ylabel('Estimated L_{cum}');
title(sprintf('L_{cum} (r=%.3f)', corr_Lcum));
legend('Untreated', 'Treated', 'Location', 'best');
grid on; xlim(lim_lc); ylim(lim_lc);

% Peak height: true vs estimated
lim_ph = [0, max([true_peak_height; est_peak_height]) * 1.05];
subplot(1,3,2)
scatter(true_peak_height(idx_ut), est_peak_height(idx_ut), 30, 'b', 'filled', 'MarkerFaceAlpha', 0.5); hold on;
scatter(true_peak_height(idx_tx), est_peak_height(idx_tx), 30, 'r', 'filled', 'MarkerFaceAlpha', 0.5);
plot(lim_ph, lim_ph, 'k--', 'LineWidth', 1);
xlabel('True Peak Height'); ylabel('Estimated Peak Height');
title(sprintf('Peak Height (r=%.3f)', corr_pkh));
legend('Untreated', 'Treated', 'Location', 'best');
grid on; xlim(lim_ph); ylim(lim_ph);

% Peak timing: true vs estimated
lim_pt = [0, max([true_peak_time; est_peak_time]) * 1.1];
subplot(1,3,3)
scatter(true_peak_time(idx_ut), est_peak_time(idx_ut), 30, 'b', 'filled', 'MarkerFaceAlpha', 0.5); hold on;
scatter(true_peak_time(idx_tx), est_peak_time(idx_tx), 30, 'r', 'filled', 'MarkerFaceAlpha', 0.5);
plot(lim_pt, lim_pt, 'k--', 'LineWidth', 1);
xlabel('True Peak Time (days)'); ylabel('Estimated Peak Time (days)');
title(sprintf('Peak Timing (r=%.3f, MAE=%.2f d)', corr_pkt, mae_pkt));
legend('Untreated', 'Treated', 'Location', 'best');
grid on; xlim(lim_pt); ylim(lim_pt);

sgtitle('Derived Quantity Recovery: What Matters for Mortality');

%% Save
save('parameter_estimation_results.mat', ...
    'true_Ap', 'true_r', 'true_kappa', ...
    'est_Ap', 'est_r', 'est_kappa', ...
    'true_Lcum', 'est_Lcum', ...
    'true_peak_height', 'est_peak_height', ...
    'true_peak_time', 'est_peak_time', ...
    'is_treated', 'fit_exitflag', ...
    'patient_lambdas', ...
    'Npat', 'frac_treated');
fprintf('\nResults saved to parameter_estimation_results.mat\n');
