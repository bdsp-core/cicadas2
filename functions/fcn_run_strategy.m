function results = fcn_run_strategy(strategy, lambda_base, drug_params, ctrl_params, T, dt, use_ekf, u_draws)
% FCN_RUN_STRATEGY Simulates a treatment strategy over one patient trajectory.
%
% Pill-based (q8h): 'none', 'fixed', 'prn', 'prn_ekf', 'adaptive'
% Infusion-based: 'fixed_infusion', 'pi'
%
% Inputs:
%   strategy     - string, strategy name
%   lambda_base  - base seizure rate [1 x Nt]
%   drug_params  - struct: k_e, alpha, dosing_interval, dose_standard, pd_mode, etc.
%   ctrl_params  - struct: omega, prn_threshold, k_p, k_i, A_max, dose_low, dose_standard
%   T            - time vector (days)
%   dt           - time step (days)
%   use_ekf      - boolean, use EKF estimate for decisions (default: true)

if nargin < 7
    use_ekf = true;
end
if nargin < 8, u_draws = []; end
inject_u = ~isempty(u_draws) && isfield(u_draws, 'poisson');

Nt = numel(lambda_base);

% Unpack drug parameters
k_e = drug_params.k_e;
alpha = drug_params.alpha;
dosing_interval_days = drug_params.dosing_interval / 24;

% Build PD params struct for fcn_observed_rate
if isfield(drug_params, 'pd_mode') && strcmpi(drug_params.pd_mode, 'hill')
    pd_struct.mode = 'hill';
    pd_struct.C50 = drug_params.C50;
    pd_struct.gamma = drug_params.gamma_hill;
    % Use patient-specific Hill params if available
    if isfield(drug_params, 'patient_C50')
        pd_struct.C50 = drug_params.patient_C50;
    end
    if isfield(drug_params, 'patient_gamma')
        pd_struct.gamma = drug_params.patient_gamma;
    end
else
    pd_struct.mode = 'linear';
    pd_struct.alpha = alpha;
end

% Unpack control parameters
omega = ctrl_params.omega;           % Target rate for PI controller
prn_threshold = ctrl_params.prn_threshold;
prn_legacy = ~isfield(ctrl_params, 'prn_legacy') || ctrl_params.prn_legacy;
prn_window_hours = 8;
if isfield(ctrl_params, 'prn_window_hours'), prn_window_hours = ctrl_params.prn_window_hours; end
prn_window_steps = round(prn_window_hours / 24 / dt);
prn_min_count = 1;
if isfield(ctrl_params, 'prn_min_count'), prn_min_count = ctrl_params.prn_min_count; end

% PI controller gains (with defaults)
if isfield(ctrl_params, 'k_p')
    k_p = ctrl_params.k_p;
else
    k_p = 50;  % Proportional gain
end
if isfield(ctrl_params, 'k_i')
    k_i = ctrl_params.k_i;
else
    k_i = 10;  % Integral gain
end
if isfield(ctrl_params, 'A_max')
    A_max = ctrl_params.A_max;
else
    A_max = 100;  % Max infusion rate
end

% Infusion rate for fixed_infusion strategy
if isfield(drug_params, 'infusion_rate')
    fixed_infusion_rate = drug_params.infusion_rate;
else
    fixed_infusion_rate = 20;  % Default constant infusion
end

% EKF parameters
tau_x = 0.5;  % mean-reversion time (days)
q = 1e-4;     % process noise
phi = exp(-dt / tau_x);

% EKF state
ekf_x = 0;
ekf_P = 1;

% PI controller state
I_k = 0;  % Integral term

% Initialize storage
x = zeros(1, Nt);              % drug concentration
infusion_rate = zeros(1, Nt);  % infusion rate at each time (for infusion strategies)
doses_given = zeros(1, Nt);    % bolus doses (for pill strategies)
lambda_obs = zeros(1, Nt);
lambda_hat = zeros(1, Nt);
seizure_counts = zeros(1, Nt);

% Time-varying population-level prior for EKF.
% Uses population median trajectory shape, not patient-specific parameters.
r_pop = 1.5;
kappa_pop = 0.3;
peak_pop = 5;
shape_pop = exp(-kappa_pop * T) .* (1 - exp(-r_pop * T));
mu_prior = max(0.1, peak_pop * shape_pop / max(shape_pop));

% Find pill dosing times (q8h)
dose_indices = [];
for i = 1:Nt
    time_since_dose = mod(T(i), dosing_interval_days);
    if time_since_dose < dt/2 || (dosing_interval_days - time_since_dose) < dt/2
        dose_indices = [dose_indices, i];
    end
end

% Determine if strategy is infusion-based
is_infusion = strcmpi(strategy, 'fixed_infusion') || strcmpi(strategy, 'pi');

%% Main simulation loop
for k = 1:Nt
    
    % Observed rate given current drug concentration
    lambda_obs(k) = fcn_observed_rate(lambda_base(k), x(k), pd_struct);
    
    % Poisson seizure count (RNG-injectable for bit-identical testing)
    lam_dt = lambda_obs(k) * dt;
    if inject_u
        seizure_counts(k) = poissinv(u_draws.poisson(k), lam_dt);
    else
        seizure_counts(k) = poissrnd(lam_dt);
    end
    
    % EKF update
    if use_ekf
        % Predict
        ekf_x = phi * ekf_x;
        ekf_P = phi^2 * ekf_P + q;
        
        % Update
        lam_pred = mu_prior(k) * exp(ekf_x);
        yhat = lam_pred * dt;
        H = yhat;
        R = max(yhat, 1e-3);
        K = (ekf_P * H) / (H * ekf_P * H + R);
        ekf_x = ekf_x + K * (seizure_counts(k) - yhat);
        ekf_P = (1 - K*H) * ekf_P * (1 - K*H)' + K * R * K';
        
        lambda_hat(k) = mu_prior(k) * exp(ekf_x);
    else
        lambda_hat(k) = lambda_obs(k);
    end
    
    % Treatment decision
    switch lower(strategy)
        
        case 'none'
            % No treatment
            doses_given(k) = 0;
            infusion_rate(k) = 0;
            
        case 'fixed'
            % Pill q8h - constant dose
            if ismember(k, dose_indices)
                doses_given(k) = ctrl_params.dose_standard;
            end
            
        case 'prn'
            % R&R (task 3): prn_legacy = true (default) reproduces the submitted
            % rule, a rolling 4-hour RATE > prn_threshold sz/day. With
            % prn_legacy = false, dose if the seizure COUNT in the preceding
            % prn_window_hours (default 8 h = one dosing interval) reaches
            % prn_min_count. The R&R experiments use the count rule.
            if ismember(k, dose_indices)
                if prn_legacy
                    window_steps = round(4 / 24 / dt);
                    start_idx = max(1, k - window_steps);
                    recent_rate = sum(seizure_counts(start_idx:k)) / max((k - start_idx + 1) * dt, dt);
                    give = recent_rate > prn_threshold;
                else
                    start_idx = max(1, k - prn_window_steps + 1);
                    give = sum(seizure_counts(start_idx:k)) >= prn_min_count;
                end
                if give
                    doses_given(k) = ctrl_params.dose_standard;
                end
            end
            
        case 'fixed_infusion'
            % Constant infusion rate
            infusion_rate(k) = fixed_infusion_rate;
            
        case 'pi'
            % PI controller: adjusts infusion rate to maintain lambda_hat near omega.
            
            e_k = lambda_hat(k) - omega;
            
            if e_k > 0
                I_k = I_k + e_k * dt;
            else
                I_k = I_k * 0.99;
            end
            
            % Clamp integral
            I_k = max(0, min(I_k, A_max / k_i));
            
            % Control output
            A_k = k_p * max(0, e_k) + k_i * I_k;
            
            % Clamp to [0, A_max]
            infusion_rate(k) = min(A_max, max(0, A_k));
            
        case 'adaptive'
            % 5-tier EKF-based dosing at each q8h slot.
            %   lambda_hat < 0.5*omega           -> none (seizures negligible)
            %   0.5*omega <= lambda_hat < omega   -> low (250)
            %   omega <= lambda_hat < 1.5*omega   -> intermediate (375)
            %   1.5*omega <= lambda_hat < 2*omega -> medium (500, same as fixed)
            %   lambda_hat >= 2*omega             -> high (750, exceeds fixed)
            
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
            
        case 'prn_ekf'
            % Pill q8h if EKF estimate exceeds threshold.
            
            if ismember(k, dose_indices) && lambda_hat(k) > prn_threshold
                doses_given(k) = ctrl_params.dose_standard;
            end
            
        otherwise
            error('Unknown strategy: %s', strategy);
    end
    
    % Update drug concentration
    if k < Nt
        if is_infusion
            x(k+1) = x(k) * k_e + infusion_rate(k);
        else
            x(k+1) = x(k) * k_e + doses_given(k);
        end
    end
end

%% Compute metrics
burden = sum(seizure_counts);
expected_burden = sum(lambda_obs) * dt;

drug_exposure = sum(x) * dt;

if is_infusion
    total_dose = sum(infusion_rate);  % Total drug infused (sum of rates)
    n_doses = NaN;  % Not applicable for infusion
else
    total_dose = sum(doses_given);
    n_doses = sum(doses_given > 0);
end

%% Package results
results.strategy = strategy;
results.is_infusion = is_infusion;
results.lambda_obs = lambda_obs;
results.lambda_hat = lambda_hat;
results.x = x;
results.doses_given = doses_given;
results.infusion_rate = infusion_rate;
results.seizure_counts = seizure_counts;
results.dose_indices = dose_indices;
results.burden = burden;
results.expected_burden = expected_burden;
results.drug_exposure = drug_exposure;
results.total_dose = total_dose;
results.n_doses = n_doses;
results.use_ekf = use_ekf;

% PI-specific outputs
if strcmpi(strategy, 'pi')
    results.I_final = I_k;  % Final integral state
end

end
