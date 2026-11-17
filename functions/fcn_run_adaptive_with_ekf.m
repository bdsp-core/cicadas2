function r = fcn_run_adaptive_with_ekf(patient, dp, ctrl_params, T, dt, ekf_ov, u_draws)
% FCN_RUN_ADAPTIVE_WITH_EKF Run adaptive strategy with overridden EKF parameters.
%
% Returns a struct compatible with fcn_run_strategy output, but uses
% custom EKF tuning parameters (for misspecification analysis).
%
% Inputs:
%   patient    - patient struct
%   dp         - drug_params with patient_C50, patient_gamma
%   ctrl_params - control parameters
%   T          - time vector
%   dt         - time step
%   ekf_ov     - struct with fields: P_pop, tau_x, q, floor
%
% Outputs:
%   r - struct with: lambda_obs, x, doses_given, seizure_counts, lambda_hat, strategy

Nt = numel(T);
lambda_base = patient.lambda_base;

pd_struct.mode = 'hill';
pd_struct.C50 = dp.patient_C50;
pd_struct.gamma = dp.patient_gamma;

k_e = dp.k_e;
dosing_interval_days = dp.dosing_interval / 24;
omega = ctrl_params.omega;

% EKF with overridden params
tau_x = ekf_ov.tau_x;
q = ekf_ov.q;
P_pop = ekf_ov.P_pop;
floor_val = ekf_ov.floor;
phi = exp(-dt / tau_x);
ekf_x = 0; ekf_P = 1;

r_pop = 1.5; kappa_pop = 0.3;
shape_pop = exp(-kappa_pop * T) .* (1 - exp(-r_pop * T));
mu_prior = max(floor_val, P_pop * shape_pop / max(shape_pop));

dose_indices = [];
for i = 1:Nt
    tsd = mod(T(i), dosing_interval_days);
    if tsd < dt/2 || (dosing_interval_days - tsd) < dt/2
        dose_indices = [dose_indices, i];
    end
end

x = zeros(1, Nt);
doses_given = zeros(1, Nt);
lambda_obs = zeros(1, Nt);
seizure_counts = zeros(1, Nt);
lambda_hat = zeros(1, Nt);

if nargin < 7, u_draws = []; end
inject_u = ~isempty(u_draws) && isfield(u_draws, 'poisson');

for k = 1:Nt
    lambda_obs(k) = fcn_observed_rate(lambda_base(k), x(k), pd_struct);
    lam_dt = lambda_obs(k) * dt;
    if inject_u
        seizure_counts(k) = poissinv(u_draws.poisson(k), lam_dt);
    else
        seizure_counts(k) = poissrnd(lam_dt);
    end

    ekf_x = phi * ekf_x;
    ekf_P = phi^2 * ekf_P + q;
    lam_pred = mu_prior(k) * exp(ekf_x);
    yhat = lam_pred * dt;
    H = yhat; R = max(yhat, 1e-3);
    K = (ekf_P * H) / (H * ekf_P * H + R);
    ekf_x = ekf_x + K * (seizure_counts(k) - yhat);
    ekf_P = (1 - K*H) * ekf_P * (1 - K*H)' + K * R * K';
    lambda_hat(k) = mu_prior(k) * exp(ekf_x);

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

    if k < Nt
        x(k+1) = x(k) * k_e + doses_given(k);
    end
end

r.lambda_obs = lambda_obs;
r.x = x;
r.doses_given = doses_given;
r.seizure_counts = seizure_counts;
r.lambda_hat = lambda_hat;
r.strategy = 'adaptive';

end
