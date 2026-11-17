function [alive, death_time] = fcn_mortality(lambda_obs, x, dt, theta, age, sofa, mort_params, u_draws)
% FCN_MORTALITY Simulates survival via discrete-time daily hazard model.
%
%   z(day) = theta0 + theta1*(t/Tmax)^2
%            + (theta2 + theta3*SOFA/10) * (L_cum/L_norm)
%            + (theta4 + theta5*Age/65) * (X_cum/X_norm)
%   h(day) = 1 / (1 + exp(-z))
%
% Inputs:
%   lambda_obs  - observed seizure rate [1 x Nt]
%   x           - drug concentration [1 x Nt]
%   dt          - time step (days)
%   theta       - [theta0, theta1, theta2, theta3, theta4, theta5]
%   age         - patient age (years)
%   sofa        - SOFA score (0-24)
%   mort_params - (optional) struct with fields L_norm, X_norm, Tmax;
%                 defaults pulled from default_params() if omitted, so
%                 the normalization constants live in a single source.
%   u_draws     - (optional) length-n_days vector of pre-drawn uniforms
%                 for RNG injection. If omitted, draws via rand() each
%                 step. Used to produce bit-identical reference outputs
%                 for the Python port (see python/PORT_STATUS.md).
%
% Outputs:
%   alive      - 1 if survived, 0 if died
%   death_time - day of death (NaN if survived)

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
        + (theta2 + theta3 * sofa_norm) * (L_cum(idx) / L_norm) ...
        + (theta4 + theta5 * age_norm) * (X_cum(idx) / X_norm);
    
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
