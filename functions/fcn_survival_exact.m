function [S, h] = fcn_survival_exact(Lcum_day, Xcum_day, age, sofa, theta, form, n_days)
% FCN_SURVIVAL_EXACT  Exact 7-day survival probability under the daily logistic hazard.
%
%   S = fcn_survival_exact(Lcum_day, Xcum_day, age, sofa, theta)
%   [S, h] = fcn_survival_exact(..., form, n_days)
%
% fcn_mortality draws death day by day with probability h_d, so the expected
% value of its 0/1 output is exactly
%       S = prod_{d=1..n_days} (1 - h_d).
% Averaging Nruns calls to fcn_mortality estimates this number with Monte
% Carlo error sqrt(S(1-S)/Nruns) and nothing else. Computing it directly gives
% the same expected answer with zero mortality Monte Carlo error, and is
% thousands of times faster.
%
% Inputs
%   Lcum_day, Xcum_day : n x D (or n x S x D) day-end cumulative seizure burden
%                        and drug AUC (from fcn_simulate_cohort)
%   age, sofa          : n x 1
%   theta              : [theta0..theta5], same meaning as fcn_mortality
%   form               : 'linear' (default, = fcn_mortality),
%                        'squared' (= fcn_mortality_squared), or a struct with
%                        function handles .gL(u) and .gX(u) applied to the
%                        normalised burdens u = L/35 and u = X/5000
%                        (hook for threshold / saturating shapes, task 10)
%   n_days             : number of daily hazards (default 7 = Tmax, as the paper
%                        defines). NOTE: fcn_mortality loops to ceil(Nt*dt) = 8
%                        because Nt*dt = 7.0035; pass 8 to reproduce it exactly.
%
% Outputs
%   S : survival probability, same leading dimensions as the inputs minus D
%   h : daily hazards, same size as the inputs (first n_days slices)

if nargin < 6 || isempty(form), form = 'linear'; end
if nargin < 7 || isempty(n_days), n_days = 7; end

Tmax = 7; L_norm = 35; X_norm = 5000;

if ischar(form)
    switch lower(form)
        case 'linear'
            gL = @(u) u;  gX = @(u) u;
        case 'squared'
            gL = @(u) u.^2;  gX = @(u) u.^2;
        otherwise
            error('Unknown mortality form: %s', form);
    end
else
    gL = form.gL;  gX = form.gX;
end

nd = ndims(Lcum_day);
D = size(Lcum_day, nd);
if n_days > D
    error('Need %d day-end values, only %d stored.', n_days, D);
end

% Reshape so the day dimension is last and covariates broadcast over strategies
if nd == 2
    L = Lcum_day(:, 1:n_days);  X = Xcum_day(:, 1:n_days);
    sofa_n = sofa(:) / 10;  age_n = age(:) / 65;
    d = 1:n_days;
else
    L = Lcum_day(:, :, 1:n_days);  X = Xcum_day(:, :, 1:n_days);
    sofa_n = sofa(:) / 10;  age_n = age(:) / 65;
    d = reshape(1:n_days, 1, 1, n_days);
end

z = theta(1) + theta(2) * (d / Tmax).^2 ...
    + (theta(3) + theta(4) * sofa_n) .* gL(L / L_norm) ...
    + (theta(5) + theta(6) * age_n) .* gX(X / X_norm);
h = 1 ./ (1 + exp(-z));
S = prod(1 - h, nd);

end
