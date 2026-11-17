function [alive, death_time] = fcn_mortality_squared(lambda_obs, x, dt, theta, age, sofa, mort_params, u_draws)
% FCN_MORTALITY_SQUARED Mortality model with SQUARED burden terms.
%
% Like fcn_mortality but uses (L_cum/L_norm)^2 and (X_cum/X_norm)^2
% instead of linear terms. Used for mortality model misspecification
% experiments.
%
%   z(day) = theta0 + theta1*(t/Tmax)^2
%            + (theta2 + theta3*SOFA/10) * (L_cum/L_norm)^2
%            + (theta4 + theta5*Age/65) * (X_cum/X_norm)^2
%   h(day) = 1 / (1 + exp(-z))
%
% mort_params and u_draws are optional and mirror fcn_mortality.m;
% defaults from default_params() when omitted.

if nargin < 7 || isempty(mort_params)
    [~, ~, ~, mort_params] = default_params();
end
inject_u = nargin >= 8 && ~isempty(u_draws);

Nt = numel(lambda_obs);

theta0 = theta(1);
theta1 = theta(2);
theta2 = theta(3);
theta3 = theta(4);
theta4 = theta(5);
theta5 = theta(6);

Tmax   = mort_params.Tmax;
L_norm = mort_params.L_norm;
X_norm = mort_params.X_norm;

sofa_norm = sofa / 10;
age_norm = age / 65;

L_cum = cumsum(lambda_obs) * dt;
X_cum = cumsum(x) * dt;

n_days = ceil(Nt * dt);

for d = 1:n_days
    idx = min(round(d / dt), Nt);
    t_frac = (d / Tmax)^2;

    z = theta0 ...
        + theta1 * t_frac ...
        + (theta2 + theta3 * sofa_norm) * (L_cum(idx) / L_norm)^2 ...
        + (theta4 + theta5 * age_norm) * (X_cum(idx) / X_norm)^2;

    h = 1 / (1 + exp(-z));

    if inject_u
        u = u_draws(d);
    else
        u = rand();
    end
    if u < h
        alive = 0;
        death_time = d;
        return
    end
end

alive = 1;
death_time = NaN;

end
