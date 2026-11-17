function x = fcn_drug_pk(A, k_e, Nt)
% FCN_DRUG_PK Simulates drug concentration over time (one-compartment PK)
%
% Model: x(k+1) = k_e * x(k) + A(k)
%   - x(k): drug concentration at time k
%   - k_e:  decay coefficient (0 < k_e < 1)
%   - A(k): infusion rate at time k
%
% Inputs:
%   A   - infusion rate vector [1 x Nt]
%   k_e - decay coefficient (e.g., 0.95)
%   Nt  - number of time steps
%
% Outputs:
%   x   - drug concentration vector [1 x Nt]

x = zeros(1, Nt);

for k = 1:(Nt-1)
    x(k+1) = x(k) * k_e + A(k);
end

end
