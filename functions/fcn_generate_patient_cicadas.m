function patient = fcn_generate_patient_cicadas(T, params, u_draws)
% FCN_GENERATE_PATIENT_CICADAS  Synthetic patient via CICADAS SDE (Sec 4.1).
%
% Generates disease trajectory using stochastic differential equation with
% logistic growth (early) and mean-reverting decay (late). Produces noisy,
% asymmetric trajectories -- fast rise, slow fall, within-patient jitter.
%
% CICADAS Eq 12-14:
%   L(t+dt) = [L(t) + mu(L,t)*dt + eps(L,t)*sqrt(dt)*Z]^+
%
%   Early (t < t_switch):
%     mu = g * L * (1 - L/H)                  [logistic growth]
%
%   Late (t >= t_switch):
%     mu = -vartheta*(L - H*exp(-w*(t-t_sw))*0.2)   [mean reversion]
%         - vartheta*max(0, L - H)                    [hard cap]
%
%   Noise:
%     eps = (eps_early*exp(-t/tau) + eps_late) * max(0.001, L)
%
% Population parameters (7, shared across patients):
%   g, H_scale, vartheta, omega_decay, eps_early, eps_late, tau_noise
%
% Patient-level variation:
%   A_p (severity), t_switch (growth duration), noise realizations
%
% Inputs:
%   T      - time vector in days [1 x Nt]
%   params - struct with fields:
%       .peak_rate    - population peak rate scale (default: 10)
%       .age          - override age (optional)
%       .sofa         - override SOFA (optional)
%       .drug_params  - for PD parameter generation (optional)
%       .pop_params   - override population SDE params (optional struct)
%       .seed         - random seed (optional)
%
% Outputs:
%   patient - struct with same interface as fcn_generate_patient:
%       .age, .sofa, .A_patient, .lambda_base, .peak_rate
%       .alpha, .C50, .gamma_hill (if drug_params provided)
%       .r = NaN, .d = NaN (no parametric shape -- SDE generated)
%       .gen_mode = 'cicadas'
%       .cicadas_pop = struct of population params used

if isfield(params, 'seed'), rng(params.seed); end
if ~isfield(params, 'peak_rate'), params.peak_rate = 10; end
if nargin < 3, u_draws = []; end
use_u = ~isempty(u_draws);

%% Population SDE parameters (CICADAS Section 4.1)
% Defaults tuned for 7-day ICU trajectory: peak at day 1-2, decay over 3-7.
pop = struct();
pop.g           = 4.0;    % logistic growth rate (1/day)
pop.vartheta    = 1.0;    % mean reversion strength (1/day)
pop.omega_decay = 0.5;    % equilibrium decay rate (1/day)
pop.eps_early   = 0.5;    % early volatility coefficient
pop.eps_late    = 0.05;   % late volatility coefficient
pop.tau_noise   = 1.5;    % noise decay time constant (days)
pop.t_switch    = 1.5;    % early->late transition (days)
pop.equil_frac  = 0.2;    % fraction of H for initial late equilibrium

% Override with user-supplied population params
if isfield(params, 'pop_params')
    fnames = fieldnames(params.pop_params);
    for i = 1:numel(fnames)
        pop.(fnames{i}) = params.pop_params.(fnames{i});
    end
end

%% Patient demographics (same distributions as fcn_generate_patient)
if isfield(params, 'age')
    age = params.age;
else
    if use_u
        beta_draw = betainv(u_draws.age, 3, 5);
    else
        beta_draw = betarnd(3, 5);
    end
    age = 40 + 55 * beta_draw;
    age = max(18, min(100, age));
end

if isfield(params, 'sofa')
    sofa = params.sofa;
else
    if use_u
        gamma_draw = gaminv(u_draws.sofa, 2, 1.5);
    else
        gamma_draw = gamrnd(2, 1.5);
    end
    sofa = min(24, max(0, round(gamma_draw)));
end

%% Patient-level severity (A_p) and carrying capacity (H)
if use_u
    A_base   = logninv(u_draws.A_base,   0, 0.8);
    t_jitter = logninv(u_draws.t_switch, 0, 0.15);
else
    A_base   = lognrnd(0, 0.8);
    t_jitter = lognrnd(0, 0.15);
end
A_patient = A_base * (1 + 0.05 * sofa);
H = A_patient * params.peak_rate;   % patient-specific ceiling

% Patient-level variation in growth-to-decay transition
t_switch = pop.t_switch * t_jitter;        % ~15% CV around population mean
t_switch = max(0.5, min(3.0, t_switch));   % clamp to [0.5, 3.0] days

%% Euler-Maruyama SDE simulation (CICADAS Eq 12-14)
dt = T(2) - T(1);
Nt = numel(T);
L = zeros(1, Nt);
L(1) = 0.01 * H;   % initial: 1% of carrying capacity

for k = 1:(Nt-1)
    t = T(k);
    Lk = L(k);
    
    % --- Drift (CICADAS Eq 13) ---
    if t < t_switch
        % Early: logistic growth toward carrying capacity H
        mu = pop.g * Lk * (1 - Lk / H);
    else
        % Late: mean reversion toward decaying equilibrium
        equil = H * exp(-pop.omega_decay * (t - t_switch)) * pop.equil_frac;
        mu = -pop.vartheta * (Lk - equil);
        % Hard cap: push back if above H (prevents runaway from noise)
        mu = mu - pop.vartheta * max(0, Lk - H);
    end
    
    % --- Diffusion (CICADAS Eq 14) ---
    % State-dependent, time-decaying noise
    vol = (pop.eps_early * exp(-t / pop.tau_noise) + pop.eps_late) ...
        * max(0.001, Lk);
    
    % --- Euler-Maruyama step ---
    if use_u
        z = norminv(u_draws.sde(k), 0, 1);
    else
        z = randn();
    end
    L(k+1) = max(0, Lk + mu * dt + vol * sqrt(dt) * z);
end

%% Package output (compatible interface with fcn_generate_patient)
patient.age = age;
patient.sofa = sofa;
patient.A_patient = A_patient;
patient.r = NaN;           % no parametric shape params for SDE model
patient.kappa = NaN;
patient.lambda_base = L;
patient.peak_rate = params.peak_rate;
patient.gen_mode = 'cicadas';
patient.cicadas_pop = pop;
patient.cicadas_H = H;
patient.cicadas_t_switch = t_switch;

%% Drug sensitivity (same as fcn_generate_patient for downstream compat)
if isfield(params, 'drug_params')
    dp = params.drug_params;
    sofa_norm = sofa / 10;
    age_norm  = age / 65;
    alpha_mean = dp.alpha_base * (1 + dp.alpha_sofa * sofa_norm + dp.alpha_age * age_norm);
    if use_u
        alpha_random = exp(norminv(u_draws.alpha, 0, 1) * 0.3);
    else
        alpha_random = exp(randn() * 0.3);
    end
    patient.alpha = max(0.001, alpha_mean * alpha_random);

    if isfield(dp, 'pd_mode') && strcmpi(dp.pd_mode, 'hill')
        age_star  = (age  - 60) / 10;
        sofa_star = (sofa -  3) /  2;
        C50_mean = dp.C50 + 50*age_star*(-0.3) + 50*sofa_star*(0.5);
        if use_u
            C50_random = exp(norminv(u_draws.C50, 0, 1) * 0.2);
        else
            C50_random = exp(randn() * 0.2);
        end
        patient.C50 = max(50, C50_mean * C50_random);
        gamma_mean = dp.gamma_hill + 0.1*sofa_star + 0.05*age_star;
        if use_u
            gamma_random = exp(norminv(u_draws.gamma_hill, 0, 1) * 0.15);
        else
            gamma_random = exp(randn() * 0.15);
        end
        patient.gamma_hill = max(0.5, gamma_mean * gamma_random);
    end
else
    patient.alpha = 0.02;
end

end
