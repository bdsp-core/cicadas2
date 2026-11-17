% AUGMENT_GFORMULA_RESULTS
% Regenerate the s9 patient cohort with the SAME rng(42) seed and patient
% generation order, then append per-patient peak_rate and sofa to the
% existing gformula_results.mat. This is fast (~30s) because we skip all
% of s9's expensive mortality / g-formula simulation.

addpath('functions');
[time_params, drug_params, ~, ~] = default_params();
T = time_params.T;

S = load('results/gformula_results.mat');
Npat = double(S.Npat);
use_cicadas_dgp = logical(S.use_cicadas_dgp);

patient_params.peak_rate = 10;
patient_params.drug_params = drug_params;

fprintf('Regenerating %d patients (cicadas DGP=%d)...\n', Npat, use_cicadas_dgp);
rng(42);
peak_rates = zeros(Npat, 1);
sofas = zeros(Npat, 1);
ages = zeros(Npat, 1);
A_p_true = zeros(Npat, 1);
for p = 1:Npat
    if use_cicadas_dgp
        patient = fcn_generate_patient_cicadas(T, patient_params);
    else
        patient = fcn_generate_patient(T, patient_params);
    end
    peak_rates(p) = max(patient.lambda_base);
    sofas(p) = patient.sofa;
    ages(p) = patient.age;
    A_p_true(p) = patient.A_patient;
end
fprintf('Done.\n');

% Append + resave
S.peak_rates = peak_rates;
S.sofas = sofas;
S.ages = ages;
S.A_p_true = A_p_true;
save('results/gformula_results.mat', '-struct', 'S');

fprintf('peak_rates: mean=%.1f median=%.1f range=[%.1f, %.1f]\n', ...
    mean(peak_rates), median(peak_rates), min(peak_rates), max(peak_rates));

% Sanity check the confounding
obs = logical(S.obs_assignment);
fprintf('Treated mean peak: %.1f  Untreated mean peak: %.1f  (ratio %.2f)\n', ...
    mean(peak_rates(obs)), mean(peak_rates(~obs)), ...
    mean(peak_rates(obs)) / mean(peak_rates(~obs)));
