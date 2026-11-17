function [time_params, drug_params, ctrl_params, mort_params] = default_params()
% DEFAULT_PARAMS Returns all default parameters for the simulation
%
% Outputs:
%   time_params - struct with: Tmax, dt, T, Nt
%   drug_params - struct with: k_e, alpha, half_life, dose, dosing_interval
%   ctrl_params - struct with: omega, dose_low, dose_standard, dose_high
%   mort_params - struct with: theta (vector)

% Time
time_params.Tmax = 7;                    % days
time_params.dt = 5/60/24;                % 5 min in days
time_params.T = 0:time_params.dt:time_params.Tmax;
time_params.Nt = numel(time_params.T);

% Drug PK
drug_params.half_life = 8;               % hours
dt_hours = time_params.dt * 24;
drug_params.k_e = exp(-log(2) * dt_hours / drug_params.half_life);

% Drug PD (linear): lambda_obs = max(0, lambda_base - alpha * x)
drug_params.alpha_base = 0.02;
drug_params.alpha_sofa = -0.3;
drug_params.alpha_age = 0.2;
drug_params.alpha = drug_params.alpha_base;  % population mean

% Drug PD (Hill): s_x = 1 - 1/((C50/x)^gamma + 1)
drug_params.pd_mode = 'hill';
drug_params.C50 = 500;
drug_params.gamma_hill = 2;

% Dosing
drug_params.dosing_interval = 8;         % hours
drug_params.dose_standard = 500;
drug_params.infusion_rate = 5;

% Control
ctrl_params.omega = 2;                   % target rate (sz/day)
ctrl_params.dose_none = 0;
ctrl_params.dose_low = 250;
ctrl_params.dose_intermediate = 375;
ctrl_params.dose_medium = 500;
ctrl_params.dose_high = 750;
ctrl_params.dose_standard = 500;         % fixed strategy dose
ctrl_params.prn_threshold = 5;          % sz/day: prn_ekf, and the legacy PRN rule
ctrl_params.prn_legacy = true;          % true = submitted PRN (4 h rate); R&R experiments use false
ctrl_params.prn_window_hours = 8;       % count-rule window = one dosing interval
ctrl_params.prn_min_count = 1;          % count-rule threshold: any seizure since the last dose
ctrl_params.k_p = 0.5;
ctrl_params.k_i = 2;
ctrl_params.A_max = 8;

% Mortality: z = theta0 + theta1*(t/Tmax)^2
%   + (theta2 + theta3*SOFA/10) * L_cum/L_norm
%   + (theta4 + theta5*Age/65) * X_cum/X_norm
% h = 1/(1+exp(-z)), evaluated daily.
mort_params.theta  = [-4, 0.5, 1.5, 1.0, 0.15, 0.07];
mort_params.L_norm = 35;            % seizure burden normalization (sz*day)
mort_params.X_norm = 5000;          % drug exposure normalization (AUC units)
mort_params.Tmax   = time_params.Tmax;  % survival horizon (days); same as time_params

end
