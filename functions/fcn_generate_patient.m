function patient = fcn_generate_patient(T, params, u_draws)
% FCN_GENERATE_PATIENT Creates one synthetic patient with covariates and trajectory
%
% Inputs:
%   T       - time vector [1 x Nt]
%   params  - struct with optional fields:
%       .peak_rate   - scaling factor (default: 10)
%       .age         - override age (optional)
%       .sofa        - override SOFA (optional)
%       .seed        - random seed for reproducibility (optional)
%   u_draws - (optional) struct of pre-drawn uniforms in (0,1) for
%             RNG injection. When supplied, every native rng call is
%             replaced with the inverse CDF of u_draws.<field>, giving
%             bit-identical results to the Python port (which uses
%             scipy.stats.*.ppf on the same tape). Fields used:
%                 .age, .sofa, .A_base, .r, .kappa
%                 .alpha, .C50, .gamma_hill   (only if params.drug_params)
%
% Outputs:
%   patient - struct containing:
%       .age         - patient age (years)
%       .sofa        - SOFA score (0-24)
%       .A_patient   - severity multiplier
%       .r           - rise rate
%       .kappa       - decay rate
%       .lambda_base - base seizure rate trajectory [1 x Nt]
%       .peak_rate   - peak rate scaling factor

% Set random seed if provided
if isfield(params, 'seed')
    rng(params.seed);
end

if nargin < 3, u_draws = []; end
use_u = ~isempty(u_draws);

% Default peak rate
if ~isfield(params, 'peak_rate')
    params.peak_rate = 10;
end

% Age: Beta(3,5) scaled to [40,95]
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

% SOFA: Gamma(2,1.5), clamped to [0,24]
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

% Disease trajectory
if use_u
    A_base = logninv(u_draws.A_base, 0, 0.8);
    r      = logninv(u_draws.r,      log(1.5), 0.3);
    kappa  = logninv(u_draws.kappa,  log(0.3), 0.4);
else
    A_base = lognrnd(0, 0.8);
    r      = lognrnd(log(1.5), 0.3);
    kappa  = lognrnd(log(0.3), 0.4);
end
A_patient = A_base * (1 + 0.05 * sofa);  % SOFA increases severity

% Build trajectory
sh = exp(-kappa*T) .* (1 - exp(-r*T));
lambda_base = A_patient * params.peak_rate * sh / max(sh);

% Output
patient.age = age;
patient.sofa = sofa;
patient.A_patient = A_patient;
patient.r = r;
patient.kappa = kappa;
patient.lambda_base = lambda_base;
patient.peak_rate = params.peak_rate;

% Patient-specific drug sensitivity (linear PD)
% alpha_i = alpha_base * (1 + alpha_sofa * SOFA/10 + alpha_age * Age/65) * exp(eta)
if isfield(params, 'drug_params')
    dp = params.drug_params;
    sofa_norm = sofa / 10;
    age_norm = age / 65;
    alpha_mean = dp.alpha_base * (1 + dp.alpha_sofa * sofa_norm + dp.alpha_age * age_norm);
    if use_u
        alpha_random = exp(norminv(u_draws.alpha, 0, 1) * 0.3);
    else
        alpha_random = exp(randn() * 0.3);
    end
    patient.alpha = max(0.001, alpha_mean * alpha_random);

    % Hill PD parameters (mixed effects, cf. CICADAS Eqs 18-19)
    if isfield(dp, 'pd_mode') && strcmpi(dp.pd_mode, 'hill')
        age_star  = (age  - 60) / 10;
        sofa_star = (sofa -  3) /  2;

        % C50: SOFA increases, age decreases
        C50_mean   = dp.C50 + 50*age_star*(-0.3) + 50*sofa_star*(0.5);
        if use_u
            C50_random = exp(norminv(u_draws.C50, 0, 1) * 0.2);
        else
            C50_random = exp(randn() * 0.2);
        end
        patient.C50 = max(50, C50_mean * C50_random);

        % gamma: small covariate effects (age + SOFA)
        gamma_mean   = dp.gamma_hill + 0.1*sofa_star + 0.05*age_star;
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
