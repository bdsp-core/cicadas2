function lambda_obs = fcn_observed_rate(lambda_base, x, pd_params)
% FCN_OBSERVED_RATE Seizure rate after drug suppression.
%
% Modes:
%   'linear': lambda_obs = max(0, lambda_base - alpha * x)
%   'hill':   lambda_obs = lambda_base * s_x,  s_x = 1 - 1/((C50/x)^gamma + 1)
%
% Inputs:
%   lambda_base - base seizure rate (scalar or vector)
%   x           - drug concentration (scalar or vector)
%   pd_params   - scalar (alpha, linear mode) or struct with .mode, .alpha/.C50/.gamma
%
% Outputs:
%   lambda_obs  - observed seizure rate [1 x Nt]

% --- Backward compatibility: scalar input = alpha for linear mode ---
if ~isstruct(pd_params)
    alpha = pd_params;
    lambda_obs = max(0, lambda_base - alpha * x);
    return;
end

% --- Struct input: check mode ---
switch lower(pd_params.mode)
    case 'linear'
        alpha = pd_params.alpha;
        lambda_obs = max(0, lambda_base - alpha * x);
        
    case 'hill'
        C50 = pd_params.C50;
        gamma = pd_params.gamma;
        
        % Suppression factor: s_x goes from 1 (no drug) to 0 (full suppression)
        % s_x = 1 - 1 / ((C50/x)^gamma + 1)
        %
        % When x = 0:   s_x = 1 (no suppression)
        % When x = C50:  s_x = 0.5 (50% suppression)
        % When x >> C50: s_x → 0 (full suppression)
        
        s_x = ones(size(x));  % default: no suppression
        idx = x > 0;          % only compute where drug is present
        s_x(idx) = 1 - 1 ./ ((C50 ./ x(idx)).^gamma + 1);
        
        lambda_obs = lambda_base .* s_x;
        
    otherwise
        error('Unknown PD mode: %s. Use ''linear'' or ''hill''.', pd_params.mode);
end

end
