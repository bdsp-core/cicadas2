function surv = fcn_run_adaptive_custom(patient, dp, cp, T, dt, theta_mort, ...
        Nruns_mort, q_val, tau_x_val, P_pop_val, floor_val, omega_val, u_draws)
% FCN_RUN_ADAPTIVE_CUSTOM Run adaptive strategy with custom EKF parameters.
%
% Like fcn_run_strategy('adaptive', ...) but accepts EKF tuning parameters
% directly, for sensitivity analysis.
%
% Inputs:
%   patient     - patient struct from fcn_generate_patient
%   dp          - drug_params struct (with patient_C50, patient_gamma set)
%   cp          - ctrl_params struct
%   T           - time vector (days)
%   dt          - time step (days)
%   theta_mort  - mortality parameters [1x6]
%   Nruns_mort  - number of mortality MC runs
%   q_val       - EKF process noise
%   tau_x_val   - EKF mean-reversion time constant (days)
%   P_pop_val   - EKF prior peak rate (sz/day)
%   floor_val   - EKF prior floor
%   omega_val   - adaptive dosing threshold (sz/day)
%
% Output:
%   surv        - average survival probability (0-1)

Nt = numel(T);
lambda_base = patient.lambda_base;

% PD struct
pd_struct.mode = 'hill';
pd_struct.C50 = dp.patient_C50;
pd_struct.gamma = dp.patient_gamma;

% EKF setup
phi = exp(-dt / tau_x_val);
ekf_x = 0;
ekf_P = 1;

% Prior mean rate (population trajectory)
r_pop = 1.5; kappa_pop = 0.3;
shape_pop = exp(-kappa_pop * T) .* (1 - exp(-r_pop * T));
mu_prior = max(floor_val, P_pop_val * shape_pop / max(shape_pop));

% Drug PK
k_e = dp.k_e;
dosing_interval_days = dp.dosing_interval / 24;

% Find dose times
dose_indices = [];
for i = 1:Nt
    time_since_dose = mod(T(i), dosing_interval_days);
    if time_since_dose < dt/2 || (dosing_interval_days - time_since_dose) < dt/2
        dose_indices = [dose_indices, i];
    end
end

% Simulate
x = zeros(1, Nt);
doses_given = zeros(1, Nt);
lambda_obs = zeros(1, Nt);
seizure_counts = zeros(1, Nt);
lambda_hat = zeros(1, Nt);

if nargin < 13, u_draws = []; end
inject_p = ~isempty(u_draws) && isfield(u_draws, 'poisson');
inject_m = ~isempty(u_draws) && isfield(u_draws, 'mortality');

for k = 1:Nt
    % Observed rate
    lambda_obs(k) = fcn_observed_rate(lambda_base(k), x(k), pd_struct);

    % Poisson seizures
    lam_dt = lambda_obs(k) * dt;
    if inject_p
        seizure_counts(k) = poissinv(u_draws.poisson(k), lam_dt);
    else
        seizure_counts(k) = poissrnd(lam_dt);
    end

    % EKF predict
    ekf_x = phi * ekf_x;
    ekf_P = phi^2 * ekf_P + q_val;

    % EKF update
    lam_pred = mu_prior(k) * exp(ekf_x);
    yhat = lam_pred * dt;
    H = yhat;
    R = max(yhat, 1e-3);
    K = (ekf_P * H) / (H * ekf_P * H + R);
    ekf_x = ekf_x + K * (seizure_counts(k) - yhat);
    ekf_P = (1 - K*H) * ekf_P * (1 - K*H)' + K * R * K';
    lambda_hat(k) = mu_prior(k) * exp(ekf_x);

    % 5-tier adaptive dosing with custom omega
    if ismember(k, dose_indices)
        if lambda_hat(k) < 0.5 * omega_val
            doses_given(k) = 0;
        elseif lambda_hat(k) < omega_val
            doses_given(k) = cp.dose_low;
        elseif lambda_hat(k) < 1.5 * omega_val
            doses_given(k) = cp.dose_intermediate;
        elseif lambda_hat(k) < 2 * omega_val
            doses_given(k) = cp.dose_medium;
        else
            doses_given(k) = cp.dose_high;
        end
    end

    % Update drug concentration
    if k < Nt
        x(k+1) = x(k) * k_e + doses_given(k);
    end
end

% Mortality
alive_sum = 0;
for run = 1:Nruns_mort
    if inject_m
        u_mort = u_draws.mortality(run, :);
        [a, ~] = fcn_mortality(lambda_obs, x, dt, theta_mort, ...
                               patient.age, patient.sofa, [], u_mort);
    else
        [a, ~] = fcn_mortality(lambda_obs, x, dt, theta_mort, ...
                               patient.age, patient.sofa);
    end
    alive_sum = alive_sum + a;
end
surv = alive_sum / Nruns_mort;

end
