function x = fcn_drug_pk_pill(dose, dosing_interval, k_e, T, dt)
% FCN_DRUG_PK_PILL Simulates drug concentration with pill dosing (discrete bolus)
%
% Model: x(k+1) = k_e * x(k)           (between doses)
%        x(k+1) = k_e * x(k) + dose    (at dosing time)
%
% Inputs:
%   dose            - pill dose amount (added instantaneously)
%   dosing_interval - hours between doses (e.g., 8 for every 8 hours)
%   k_e             - decay coefficient per time step (0 < k_e < 1)
%   T               - time vector in DAYS
%   dt              - time step in DAYS
%
% Outputs:
%   x - drug concentration vector [1 x Nt]
%
% Notes:
%   - Dosing occurs at t = 0, dosing_interval, 2*dosing_interval, etc.
%   - The decay coefficient k_e should match the time step dt
%   - For half-life t_half (hours): k_e = exp(-ln(2) * dt * 24 / t_half)
%
% Example:
%   dt = 5/60/24;                          % 5 minutes in days
%   t_half = 8;                            % 8 hour half-life
%   k_e = exp(-log(2) * dt * 24 / t_half); % ≈ 0.9993
%   x = fcn_drug_pk_pill(100, 8, k_e, T, dt);

Nt = numel(T);
x = zeros(1, Nt);

% Convert dosing interval from hours to days
dosing_interval_days = dosing_interval / 24;

% Find dosing times (indices where dose is given)
% Dose at t = 0, dosing_interval, 2*dosing_interval, ...
dose_times = 0:dosing_interval_days:T(end);

for k = 1:Nt
    current_time = T(k);
    
    % Check if this is a dosing time (within half a time step)
    is_dose_time = any(abs(current_time - dose_times) < dt/2);
    
    if k == 1
        % First time step: give first dose
        x(k) = dose;
    else
        % Decay from previous step
        x(k) = x(k-1) * k_e;
        
        % Add dose if it's dosing time
        if is_dose_time
            x(k) = x(k) + dose;
        end
    end
end

end
