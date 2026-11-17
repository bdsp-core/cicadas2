function [lambda_hat, xh, P_hist] = fcn_ekf_lograte_ou(y, mu, dt, tau_x, q)
% FCN_EKF_LOGRATE_OU Extended Kalman Filter for Poisson observations with log-rate state
%
% State model (Ornstein-Uhlenbeck):
%   x_{k+1} = phi * x_k + w_k,  w_k ~ N(0, q)
%   where x = log(lambda / mu) is log-deviation from prior mean
%   phi = exp(-dt / tau_x) is the AR(1) coefficient
%
% Observation model:
%   y_k ~ Poisson(mu_k * exp(x_k) * dt)
%
% Inputs:
%   y     - observed seizure counts per time bin [1 x Nt]
%   mu    - prior mean rate at each time point [1 x Nt]
%   dt    - time step (days)
%   tau_x - mean-reversion time constant (days)
%   q     - process noise variance
%
% Outputs:
%   lambda_hat - estimated rate at each time point [1 x Nt]
%   xh         - state estimate history [1 x Nt]
%   P_hist     - variance estimate history [1 x Nt]

Nt = numel(mu);
phi = exp(-dt/tau_x);  % AR(1) coefficient from OU discretization

% Initialize
x = 0;  % state estimate (at prior mean)
P = 1;  % initial uncertainty

% Storage
lambda_hat = zeros(1, Nt);
xh = zeros(1, Nt);
P_hist = zeros(1, Nt);

for k = 1:Nt
    % Predict
    x = phi * x;
    P = phi^2 * P + q;
    
    % Update
    lam = mu(k) * exp(x);
    yhat = lam * dt;
    
    H = yhat;               % Jacobian d(yhat)/dx
    R = max(yhat, 1e-3);    % Poisson variance
    K = (P * H) / (H * P * H + R);
    x = x + K * (y(k) - yhat);
    P = (1 - K*H) * P * (1 - K*H)' + K * R * K';  % Joseph form
    
    xh(k) = x;
    P_hist(k) = P;
    lambda_hat(k) = mu(k) * exp(x);
end

end
