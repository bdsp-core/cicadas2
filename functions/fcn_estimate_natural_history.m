function [params_est, fit_info] = fcn_estimate_natural_history(seizure_counts, T, dt, varargin)
% FCN_ESTIMATE_NATURAL_HISTORY Estimates {A_p, r, kappa} from observed seizure counts.
%
% Model: lambda_base(t) = A_p * peak_rate * shape(t; r, kappa) / max(shape)
%        shape(t) = exp(-kappa*t) * (1 - exp(-r*t))
%
% Untreated: y_k ~ Poisson(lambda_base(k) * dt)
% Treated:   y_k ~ Poisson(lambda_obs(k) * dt), with PD applied to lambda_base.
%            Requires known PKPD parameters and dosing history.
%
% Supports both linear and Hill PD models for treated patients.

%% Parse inputs
p = inputParser;
addRequired(p, 'seizure_counts');
addRequired(p, 'T');
addRequired(p, 'dt');
addParameter(p, 'peak_rate', 10);
addParameter(p, 'k_e', []);
addParameter(p, 'doses_given', []);
addParameter(p, 'infusion_rate', []);
addParameter(p, 'is_infusion', false);
addParameter(p, 'pd_mode', 'linear');
addParameter(p, 'alpha', []);
addParameter(p, 'C50', []);
addParameter(p, 'gamma_hill', []);
parse(p, seizure_counts, T, dt, varargin{:});

peak_rate = p.Results.peak_rate;
k_e = p.Results.k_e;
doses_given = p.Results.doses_given;
infusion_rate_input = p.Results.infusion_rate;
is_infusion = p.Results.is_infusion;
pd_mode = p.Results.pd_mode;
alpha = p.Results.alpha;
C50 = p.Results.C50;
gamma_hill = p.Results.gamma_hill;

Nt = numel(T);

% Determine if treated based on whether k_e and dosing info are provided
has_dosing = ~isempty(k_e) && (~isempty(doses_given) || ~isempty(infusion_rate_input));
is_treated = has_dosing;

%% Reconstruct drug concentration if treated
x = zeros(1, Nt);
if is_treated
    for k = 1:(Nt-1)
        if is_infusion && ~isempty(infusion_rate_input)
            x(k+1) = x(k) * k_e + infusion_rate_input(k);
        elseif ~isempty(doses_given)
            x(k+1) = x(k) * k_e + doses_given(k);
        end
    end
end

%% PD forward model: lambda_base -> lambda_obs
% Used inside the likelihood and for final reconstruction.
    function lobs = apply_pd(lbase)
        if ~is_treated
            lobs = lbase;
            return;
        end
        switch lower(pd_mode)
            case 'linear'
                lobs = max(1e-10, lbase - alpha * x);
            case 'hill'
                s_x = ones(size(x));
                idx = x > 0;
                s_x(idx) = 1 - 1 ./ ((C50 ./ x(idx)).^gamma_hill + 1);
                lobs = max(1e-10, lbase .* s_x);
        end
    end

%% PD inverse: lambda_obs -> lambda_base (for initialization heuristic)
    function lbase = invert_pd(lobs)
        if ~is_treated
            lbase = lobs;
            return;
        end
        switch lower(pd_mode)
            case 'linear'
                lbase = lobs + alpha * x;
            case 'hill'
                s_x = ones(size(x));
                idx = x > 0;
                s_x(idx) = 1 - 1 ./ ((C50 ./ x(idx)).^gamma_hill + 1);
                lbase = lobs ./ max(s_x, 0.01);
        end
    end

%% Negative log-likelihood
    function nll = neg_log_likelihood(theta)
        A_p_trial = exp(theta(1));
        r_trial = exp(theta(2));
        d_trial = exp(theta(3));
        
        shape = exp(-d_trial * T) .* (1 - exp(-r_trial * T));
        max_shape = max(shape);
        if max_shape < 1e-10
            nll = 1e10;
            return;
        end
        
        lambda_base_trial = A_p_trial * peak_rate * shape / max_shape;
        lambda_obs_trial = apply_pd(lambda_base_trial);
        
        mu = lambda_obs_trial * dt;
        mu = max(mu, 1e-10);
        
        nll = -sum(seizure_counts .* log(mu) - mu);
        nll = nll + 0.01 * (theta(1)^2 + theta(2)^2 + theta(3)^2);
    end

%% Initial parameter estimates
window = min(round(0.5/dt), floor(Nt/4));
if window > 1
    kernel = ones(1, window) / window;
    smooth_rate = conv(seizure_counts / dt, kernel, 'same');
else
    smooth_rate = seizure_counts / dt;
end

% Invert PD to get approximate lambda_base
smooth_rate_base = invert_pd(smooth_rate);

[peak_val, peak_idx] = max(smooth_rate_base);
peak_time = T(peak_idx);

A_p_init = max(0.1, peak_val / peak_rate);
r_init = max(0.5, 2 / max(peak_time, dt));

post_peak = smooth_rate_base(peak_idx:end);
half_val = peak_val / 2;
half_idx = find(post_peak < half_val, 1);
if ~isempty(half_idx)
    half_time = half_idx * dt;
    kappa_init = max(0.1, log(2) / half_time);
else
    kappa_init = 0.3;
end

%% Optimize
theta0 = [log(A_p_init), log(r_init), log(kappa_init)];

options = optimset('Display', 'off', 'MaxFunEvals', 5000, 'MaxIter', 2000, ...
    'TolFun', 1e-8, 'TolX', 1e-8);

[theta_opt, nll_opt, exitflag] = fminsearch(@neg_log_likelihood, theta0, options);

%% Package results
params_est.A_p = exp(theta_opt(1));
params_est.r = exp(theta_opt(2));
params_est.kappa = exp(theta_opt(3));
params_est.peak_rate = peak_rate;

shape_est = exp(-params_est.kappa * T) .* (1 - exp(-params_est.r * T));
lambda_base_hat = params_est.A_p * peak_rate * shape_est / max(shape_est);
lambda_obs_hat = apply_pd(lambda_base_hat);

fit_info.negloglik = nll_opt;
fit_info.exitflag = exitflag;
fit_info.lambda_base_hat = lambda_base_hat;
fit_info.lambda_obs_hat = lambda_obs_hat;
fit_info.x = x;
fit_info.is_treated = is_treated;
fit_info.pd_mode = pd_mode;

end
