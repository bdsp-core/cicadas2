% MATLAB_HARNESS  Regenerate Python port reference fixtures.
%
% Run this once on a machine with MATLAB to populate
%     python/tests/reference_data/*.mat
% with input/output pairs that the Python pytest suite consumes.
%
% Usage (from repo root):
%     addpath(genpath('.'));
%     run('tools/matlab_harness.m');
%
% Each subsection below corresponds to one Python module under test.
% Add new fixtures here as the port progresses; keep them small
% (<5 MB each, the suite has <50 MB total budget).

clear; clc;

repo_root = fileparts(fileparts(mfilename('fullpath')));
out_dir   = fullfile(repo_root, 'python', 'tests', 'reference_data');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

addpath(fullfile(repo_root, 'functions'));

fprintf('Writing reference fixtures to %s\n', out_dir);

% =====================================================================
% fcn_mortality — Phase 1
% =====================================================================
fprintf('  mortality_seed0.mat ... ');

rng(0, 'twister');

[time, ~, ~, mort] = default_params();

% Construct a non-trivial input: a moderately seizing, mid-treated
% patient over the full 7-day horizon.
Nt = time.Nt;
T  = time.T;
lambda_obs = 2 * exp(-0.3 * T) .* (1 - exp(-2 * T));  % rise-decay
lambda_obs = max(0, lambda_obs);
x = zeros(1, Nt);
% Two boluses that decay exponentially
for tdose = [0.5, 3.0]
    k0 = round(tdose / time.dt) + 1;
    x(k0:end) = x(k0:end) + 500 * exp(-log(2) * (T(k0:end) - tdose) * 24 / 8);
end

dt    = time.dt;
theta = mort.theta;
age   = 65;
sofa  = 3;

n_days  = ceil(Nt * dt);    % NB: equals 8 for default Nt/dt because Nt*dt = 7.003
u_draws = rand(1, n_days);

mort_params = mort;  % alias so the saved variable name matches the Python loader

[alive, death_time] = fcn_mortality(lambda_obs, x, dt, theta, age, sofa, ...
                                    mort_params, u_draws);

save(fullfile(out_dir, 'mortality_seed0.mat'), ...
     'lambda_obs', 'x', 'dt', 'theta', 'age', 'sofa', ...
     'mort_params', 'u_draws', 'alive', 'death_time', '-v7');

fprintf('done (alive=%d, death=%g).\n', alive, death_time);

% --- second mortality fixture: high-burden patient who dies ---------
fprintf('  mortality_high_burden.mat ... ');
rng(1, 'twister');
lambda_obs_hi = 80 * exp(-0.3 * T) .* (1 - exp(-2 * T));   % much higher rate
lambda_obs_hi = max(0, lambda_obs_hi);
x_hi = zeros(1, Nt);                                        % no drug
u_draws_hi = rand(1, n_days);
[alive_hi, death_hi] = fcn_mortality(lambda_obs_hi, x_hi, dt, theta, ...
                                     age, sofa, mort_params, u_draws_hi);
lambda_obs = lambda_obs_hi;  x = x_hi;  u_draws = u_draws_hi;
alive = alive_hi;  death_time = death_hi;
save(fullfile(out_dir, 'mortality_high_burden.mat'), ...
     'lambda_obs', 'x', 'dt', 'theta', 'age', 'sofa', ...
     'mort_params', 'u_draws', 'alive', 'death_time', '-v7');
fprintf('done (alive=%d, death=%g).\n', alive, death_time);

% =====================================================================
% fcn_drug_pk — Phase 1
% =====================================================================
fprintf('  pk_continuous.mat ... ');
rng(2, 'twister');
[time_p, drug_p, ~, ~] = default_params();
Nt_p = time_p.Nt;
% A mix of infusion rates: zero at first, ramp up, taper down
A = zeros(1, Nt_p);
A(round(Nt_p/4):round(Nt_p/2)) = 5;
A(round(Nt_p/2):round(3*Nt_p/4)) = 2;
x_out = fcn_drug_pk(A, drug_p.k_e, Nt_p);
k_e = drug_p.k_e;
Nt = Nt_p;
save(fullfile(out_dir, 'pk_continuous.mat'), 'A', 'k_e', 'Nt', 'x_out', '-v7');
fprintf('done (max x = %.2f).\n', max(x_out));

% =====================================================================
% fcn_drug_pk_pill — Phase 1
% =====================================================================
fprintf('  pk_pill.mat ... ');
dose            = 500;
dosing_interval = 8;        % hours
k_e             = drug_p.k_e;
T_in            = time_p.T;
dt_in           = time_p.dt;
x_out = fcn_drug_pk_pill(dose, dosing_interval, k_e, T_in, dt_in);
T  = T_in;
dt = dt_in;
save(fullfile(out_dir, 'pk_pill.mat'), ...
     'dose', 'dosing_interval', 'k_e', 'T', 'dt', 'x_out', '-v7');
fprintf('done (peak x = %.2f, n_doses = %d).\n', max(x_out), ...
        sum(abs(diff(x_out)) > 100));

% =====================================================================
% fcn_observed_rate — Phase 1 (both modes)
% =====================================================================
fprintf('  pd_linear.mat ... ');
lambda_base = 2 * exp(-0.3 * time_p.T) .* (1 - exp(-2 * time_p.T));
lambda_base = max(0, lambda_base);
x = x_out;   % reuse the pill PK profile from above
pd_params.alpha = 0.02;
pd_params.mode  = 'linear';
lambda_obs = fcn_observed_rate(lambda_base, x, pd_params);
save(fullfile(out_dir, 'pd_linear.mat'), ...
     'lambda_base', 'x', 'pd_params', 'lambda_obs', '-v7');
fprintf('done.\n');

fprintf('  pd_hill.mat ... ');
clear pd_params
pd_params.mode  = 'hill';
pd_params.C50   = 500;
pd_params.gamma = 2;
lambda_obs = fcn_observed_rate(lambda_base, x, pd_params);
save(fullfile(out_dir, 'pd_hill.mat'), ...
     'lambda_base', 'x', 'pd_params', 'lambda_obs', '-v7');
fprintf('done.\n');

% =====================================================================
% fcn_compute_burden — Phase 1
% =====================================================================
fprintf('  burden.mat ... ');
rng(3, 'twister');
seizure_counts = poissrnd(0.05, 1, 2017);
burden = fcn_compute_burden(seizure_counts);
save(fullfile(out_dir, 'burden.mat'), 'seizure_counts', 'burden', '-v7');
fprintf('done (burden = %d).\n', burden);

% =====================================================================
% fcn_mortality_squared — Phase 1
% =====================================================================
fprintf('  mortality_squared.mat ... ');
rng(4, 'twister');
% Reuse the high-burden patient setup that was killing patients
% under the linear model; under the squared model these should die
% earlier still.
lambda_obs = 80 * exp(-0.3 * time_p.T) .* (1 - exp(-2 * time_p.T));
lambda_obs = max(0, lambda_obs);
x = zeros(1, Nt_p);
dt = time_p.dt;
n_days  = ceil(Nt_p * dt);
u_draws = rand(1, n_days);
[~, ~, ~, mort_params] = default_params();
theta = mort_params.theta;
age   = 65;
sofa  = 3;
[alive, death_time] = fcn_mortality_squared(lambda_obs, x, dt, theta, ...
                                            age, sofa, mort_params, u_draws);
save(fullfile(out_dir, 'mortality_squared.mat'), ...
     'lambda_obs', 'x', 'dt', 'theta', 'age', 'sofa', ...
     'mort_params', 'u_draws', 'alive', 'death_time', '-v7');
fprintf('done (alive=%d, death=%g).\n', alive, death_time);

% =====================================================================
% fcn_generate_patient — Phase 2 (stochastic; RNG-injected for bit-identical)
% =====================================================================
fprintf('  patient_basic.mat ... ');
% Fresh tape of 5 uniforms (age, sofa, A_base, r, kappa); no drug_params.
rng(5, 'twister');
u_draws = struct();
u_draws.age    = rand();
u_draws.sofa   = rand();
u_draws.A_base = rand();
u_draws.r      = rand();
u_draws.kappa  = rand();

T_in = time_p.T;
params = struct('peak_rate', 10);
patient = fcn_generate_patient(T_in, params, u_draws);

T = T_in;
save(fullfile(out_dir, 'patient_basic.mat'), ...
     'T', 'params', 'u_draws', 'patient', '-v7');
fprintf('done (age=%.1f, sofa=%d, A_p=%.2f, r=%.2f, kappa=%.2f).\n', ...
        patient.age, patient.sofa, patient.A_patient, patient.r, patient.kappa);

% --- with linear drug_params (adds alpha draw) -----------------------
fprintf('  patient_with_linear_pd.mat ... ');
rng(6, 'twister');
u_draws = struct('age', rand(), 'sofa', rand(), ...
                 'A_base', rand(), 'r', rand(), 'kappa', rand(), ...
                 'alpha', rand());
dp = struct('alpha_base', 0.02, 'alpha_sofa', -0.3, 'alpha_age', 0.2);
params = struct('peak_rate', 10, 'drug_params', dp);
patient = fcn_generate_patient(T_in, params, u_draws);
save(fullfile(out_dir, 'patient_with_linear_pd.mat'), ...
     'T', 'params', 'u_draws', 'patient', '-v7');
fprintf('done (alpha=%.4f).\n', patient.alpha);

% --- with Hill drug_params (adds alpha + C50 + gamma draws) ----------
fprintf('  patient_with_hill_pd.mat ... ');
rng(7, 'twister');
u_draws = struct('age', rand(), 'sofa', rand(), ...
                 'A_base', rand(), 'r', rand(), 'kappa', rand(), ...
                 'alpha', rand(), 'C50', rand(), 'gamma_hill', rand());
dp = struct('alpha_base', 0.02, 'alpha_sofa', -0.3, 'alpha_age', 0.2, ...
            'pd_mode', 'hill', 'C50', 500, 'gamma_hill', 2);
params = struct('peak_rate', 10, 'drug_params', dp);
patient = fcn_generate_patient(T_in, params, u_draws);
save(fullfile(out_dir, 'patient_with_hill_pd.mat'), ...
     'T', 'params', 'u_draws', 'patient', '-v7');
fprintf('done (alpha=%.4f, C50=%.1f, gamma=%.2f).\n', ...
        patient.alpha, patient.C50, patient.gamma_hill);

% =====================================================================
% fcn_generate_patient_cicadas — Phase 2 (SDE generator)
% =====================================================================
fprintf('  patient_cicadas.mat ... ');
rng(8, 'twister');
Nt_p = time_p.Nt;
u_draws = struct();
u_draws.age      = rand();
u_draws.sofa     = rand();
u_draws.A_base   = rand();
u_draws.t_switch = rand();
u_draws.sde      = rand(1, Nt_p - 1);     % Euler-Maruyama noise tape
T_in = time_p.T;
params = struct('peak_rate', 10);
patient = fcn_generate_patient_cicadas(T_in, params, u_draws);
T = T_in;
save(fullfile(out_dir, 'patient_cicadas.mat'), ...
     'T', 'params', 'u_draws', 'patient', '-v7');
fprintf('done (age=%.1f, sofa=%d, A_p=%.2f, peak L=%.1f).\n', ...
        patient.age, patient.sofa, patient.A_patient, max(patient.lambda_base));

% =====================================================================
% fcn_ekf_lograte_ou — Phase 3 (EKF)
% =====================================================================
fprintf('  ekf_lograte_ou.mat ... ');
rng(9, 'twister');
% Use the basic patient's trajectory as the population prior,
% generate Poisson observations against a *scaled* trajectory,
% then verify the EKF recovers the latent rate.
T_in = time_p.T;
Nt_p = time_p.Nt;
[~,~,~,~] = default_params();  %#ok<ASGLU>  (warm path)
u_draws = struct('age', rand(), 'sofa', rand(), ...
                 'A_base', rand(), 'r', rand(), 'kappa', rand());
params  = struct('peak_rate', 10);
patient = fcn_generate_patient(T_in, params, u_draws);
mu      = patient.lambda_base;
% True rate = 1.7 * prior, plus low-level Poisson noise
lambda_true = 1.7 * mu;
y = poissrnd(lambda_true * time_p.dt);
dt    = time_p.dt;
tau_x = 0.5;
q     = 1e-4;
[lambda_hat, xh, P_hist] = fcn_ekf_lograte_ou(y, mu, dt, tau_x, q);
save(fullfile(out_dir, 'ekf_lograte_ou.mat'), ...
     'y', 'mu', 'dt', 'tau_x', 'q', ...
     'lambda_hat', 'xh', 'P_hist', '-v7');
fprintf('done (final lambda_hat=%.3f vs true=%.3f).\n', ...
        lambda_hat(end), lambda_true(end));

% =====================================================================
% fcn_estimate_natural_history — Phase 4 (Poisson MLE; uses fminsearch)
% =====================================================================
fprintf('  estimation_untreated.mat ... ');
rng(10, 'twister');
u_draws = struct('age', rand(), 'sofa', rand(), ...
                 'A_base', rand(), 'r', rand(), 'kappa', rand());
params  = struct('peak_rate', 10);
patient = fcn_generate_patient(time_p.T, params, u_draws);
% Untreated: counts straight from the natural-history rate
y = poissrnd(patient.lambda_base * time_p.dt);
seizure_counts = y;
T  = time_p.T;
dt = time_p.dt;
[params_est, fit_info] = fcn_estimate_natural_history(seizure_counts, T, dt, ...
    'peak_rate', 10);
% Probe the NLL at a known theta for the bit-identical NLL test
theta_probe = [log(patient.A_patient), log(patient.r), log(patient.kappa)];
% Recompute NLL by re-running with theta fixed (cheap: re-eval the inner fn)
% We snapshot the input to the NLL exactly, so the Python port can
% evaluate at the same theta_probe.
nll_probe = NaN;  %#ok<NASGU>  filled in by a helper below if available
save(fullfile(out_dir, 'estimation_untreated.mat'), ...
     'seizure_counts', 'T', 'dt', 'theta_probe', ...
     'params_est', 'fit_info', 'patient', '-v7');
fprintf('done (true A_p=%.2f r=%.2f kappa=%.2f -> est A_p=%.2f r=%.2f kappa=%.2f).\n', ...
        patient.A_patient, patient.r, patient.kappa, ...
        params_est.A_p, params_est.r, params_est.kappa);

% =====================================================================
% fcn_run_strategy — Phase 5 (full per-patient simulator)
% =====================================================================
% Generate one patient (using RNG injection so the Python port can
% reproduce the same patient) and run all four pill-based strategies
% against it. Each strategy needs its own Poisson uniform tape.
[time_p, drug_p, ctrl_p, ~] = default_params();
T_in = time_p.T;
dt_in = time_p.dt;
Nt_p  = time_p.Nt;

rng(20, 'twister');
u_pat = struct('age', rand(), 'sofa', rand(), ...
               'A_base', rand(), 'r', rand(), 'kappa', rand());
patient = fcn_generate_patient(T_in, struct('peak_rate', 10), u_pat);
lambda_base = patient.lambda_base;

for strat_idx = 1:4
    strat_names = {'none', 'fixed', 'prn', 'adaptive'};
    strat = strat_names{strat_idx};
    fprintf('  strategy_%s.mat ... ', strat);

    rng(30 + strat_idx, 'twister');
    u_draws = struct('poisson', rand(1, Nt_p));

    results = fcn_run_strategy(strat, lambda_base, drug_p, ctrl_p, ...
                               T_in, dt_in, true, u_draws);

    drug_params = drug_p;  ctrl_params = ctrl_p;
    T = T_in;  dt = dt_in;  use_ekf = true;
    strategy = strat;  %#ok<NASGU>
    save(fullfile(out_dir, sprintf('strategy_%s.mat', strat)), ...
         'strategy', 'lambda_base', 'drug_params', 'ctrl_params', ...
         'T', 'dt', 'use_ekf', 'u_draws', 'results', '-v7');
    fprintf('done (burden=%d, total_dose=%g, n_doses=%d).\n', ...
            results.burden, results.total_dose, results.n_doses);
end

% =====================================================================
% fcn_run_adaptive_with_ekf — Phase 5 (EKF-misspec variant)
% =====================================================================
fprintf('  adaptive_with_ekf.mat ... ');
% Reuse the patient generated above for strategy fixtures.
% Need patient.C50 and patient.gamma_hill — generate with Hill drug_params.
rng(40, 'twister');
u_pat = struct('age', rand(), 'sofa', rand(), 'A_base', rand(), ...
               'r', rand(), 'kappa', rand(), 'alpha', rand(), ...
               'C50', rand(), 'gamma_hill', rand());
dp_hill = struct('alpha_base', 0.02, 'alpha_sofa', -0.3, 'alpha_age', 0.2, ...
                 'pd_mode', 'hill', 'C50', 500, 'gamma_hill', 2);
params = struct('peak_rate', 10, 'drug_params', dp_hill);
patient = fcn_generate_patient(T_in, params, u_pat);

% drug_params needs k_e + patient_C50 / patient_gamma + the linear alpha
dp_run = struct('k_e', drug_p.k_e, 'alpha', patient.alpha, ...
                'dosing_interval', drug_p.dosing_interval, ...
                'patient_C50', patient.C50, ...
                'patient_gamma', patient.gamma_hill);

ekf_ov = struct('P_pop', 5, 'tau_x', 0.5, 'q', 1e-4, 'floor', 0.1);

rng(41, 'twister');
u_draws = struct('poisson', rand(1, Nt_p));
r = fcn_run_adaptive_with_ekf(patient, dp_run, ctrl_p, T_in, dt_in, ekf_ov, u_draws);

drug_params = dp_run;  ctrl_params = ctrl_p;
T = T_in;  dt = dt_in;
save(fullfile(out_dir, 'adaptive_with_ekf.mat'), ...
     'patient', 'drug_params', 'ctrl_params', 'T', 'dt', ...
     'ekf_ov', 'u_draws', 'r', '-v7');
fprintf('done.\n');

% =====================================================================
% fcn_run_adaptive_custom — Phase 5 (sensitivity-sweep variant w/ mortality)
% =====================================================================
fprintf('  adaptive_custom.mat ... ');
theta_mort  = mort_params.theta;
Nruns_mort  = 10;
q_val       = 1e-4;
tau_x_val   = 0.5;
P_pop_val   = 5;
floor_val   = 0.1;
omega_val   = ctrl_p.omega;

rng(42, 'twister');
n_days = ceil(Nt_p * dt_in);
u_draws = struct( ...
    'poisson',   rand(1, Nt_p), ...
    'mortality', rand(Nruns_mort, n_days));

surv = fcn_run_adaptive_custom(patient, dp_run, ctrl_p, T_in, dt_in, ...
                               theta_mort, Nruns_mort, q_val, tau_x_val, ...
                               P_pop_val, floor_val, omega_val, u_draws);

drug_params = dp_run;  ctrl_params = ctrl_p;
T = T_in;  dt = dt_in;
save(fullfile(out_dir, 'adaptive_custom.mat'), ...
     'patient', 'drug_params', 'ctrl_params', 'T', 'dt', ...
     'theta_mort', 'Nruns_mort', 'q_val', 'tau_x_val', 'P_pop_val', ...
     'floor_val', 'omega_val', 'u_draws', 'surv', '-v7');
fprintf('done (surv=%.3f).\n', surv);

fprintf('\nAll fixtures written. Run `pytest python/tests` from the repo root.\n');
