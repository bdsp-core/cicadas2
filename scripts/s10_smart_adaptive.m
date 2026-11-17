% S10_SMART_ADAPTIVE
% Patient-specific omega optimization.
%
% Instead of omega=2 for everyone, find each patient's optimal omega
% based on their harm profile (age, SOFA -> drug/seizure susceptibility).
%
% For each patient, sweep omega and pick the value that maximizes survival.
% Compare uniform omega vs personalized omega under each quadrant regime.
%
% This is the discrete version of CICADAS Figure 10.

clear; clc;
addpath('../functions');

%% Load parameters
[time_params, drug_params, ctrl_params, mort_params] = default_params();
T = time_params.T;
Nt = time_params.Nt;
dt = time_params.dt;

%% Settings
Npat = 200;
Nruns = 200;       % mortality runs per patient per omega
use_ekf = true;
omega_uniform = 2; % current fixed omega

% Omega sweep range
omega_vals = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, Inf];
n_omega = numel(omega_vals);

rng(42);
patient_params.peak_rate = 10;
patient_params.drug_params = drug_params;

%% Define regimes (must match s5/s8)
theta4_high = 1.25;
regimes = struct();
regimes(1).name = 'High Sz / Low Drug';
regimes(1).theta = [-5, 0.5, 3.0, 1.0, 0.05, 0.02];
regimes(2).name = 'High Sz / High Drug';
regimes(2).theta = [-5, 0.5, 3.0, 1.0, theta4_high, 0.3];
regimes(3).name = 'Low Sz / Low Drug';
regimes(3).theta = [-5, 0.5, 0.5, 0.2, 0.05, 0.02];
regimes(4).name = 'Low Sz / High Drug';
regimes(4).theta = [-5, 0.5, 0.5, 0.2, theta4_high, 0.3];
n_regimes = numel(regimes);

%% Generate patients
fprintf('=== SMART ADAPTIVE: Patient-Specific Omega ===\n');
fprintf('Patients: %d | Omega values: %d | Mortality runs: %d\n\n', ...
    Npat, n_omega, Nruns);

patients = cell(Npat, 1);
ages = zeros(Npat, 1);
sofas = zeros(Npat, 1);
peak_rates = zeros(Npat, 1);
for p = 1:Npat
    patients{p} = fcn_generate_patient(T, patient_params);
    ages(p) = patients{p}.age;
    sofas(p) = patients{p}.sofa;
    peak_rates(p) = max(patients{p}.lambda_base);
end
fprintf('Generated %d patients.\n', Npat);

%% Step 1: Run adaptive at each omega, store full results
fprintf('Running adaptive at %d omega values per patient...\n', n_omega);
tic;

% Store results: strat_results{p, oi} = struct from fcn_run_strategy
strat_results = cell(Npat, n_omega);

for p = 1:Npat
    patient = patients{p};
    dp = drug_params;
    dp.alpha = patient.alpha;
    if isfield(patient, 'C50')
        dp.patient_C50 = patient.C50;
        dp.patient_gamma = patient.gamma_hill;
    end
    
    for oi = 1:n_omega
        cp = ctrl_params;
        cp.omega = omega_vals(oi);
        
        if isinf(omega_vals(oi))
            strat_results{p, oi} = fcn_run_strategy('none', ...
                patient.lambda_base, dp, cp, T, dt, use_ekf);
        else
            strat_results{p, oi} = fcn_run_strategy('adaptive', ...
                patient.lambda_base, dp, cp, T, dt, use_ekf);
        end
    end
    
    if mod(p, 50) == 0
        fprintf('  %d/%d patients (%.1fs)\n', p, Npat, toc);
    end
end
fprintf('Strategy sweep done (%.1fs).\n\n', toc);

%% Step 2: Evaluate mortality at each omega under each regime
% BIAS CORRECTION (split-sample / honest argmax):
%   Using the same Monte Carlo draws to pick omega_i* AND to report
%   survival at omega_i* yields a winner's-curse upward bias.  We
%   split the mortality MC per (p, oi, reg) into two independent
%   halves:
%
%     surv_select(p, oi, reg)  -> argmax over omega -> omega_i*
%     surv_eval  (p, oi, reg)  -> survival reported at omega_i*  (unbiased)
%
%   Both halves use the SAME strat_results trajectories; only the
%   random mortality draws differ.  Total mortality calls per
%   (p, oi, reg) become 2*Nruns.  surv_uniform is reported from the
%   evaluation set as well, for a like-for-like comparison.
fprintf('Evaluating mortality (split-sample)...\n');

surv_select = zeros(Npat, n_omega, n_regimes);   % drives argmax
surv_eval   = zeros(Npat, n_omega, n_regimes);   % drives reported survival

for reg = 1:n_regimes
    fprintf('  Regime %d: %s\n', reg, regimes(reg).name);
    theta = regimes(reg).theta;

    for p = 1:Npat
        for oi = 1:n_omega
            r = strat_results{p, oi};

            % --- Selection draws ---
            alive_sel = 0;
            for run = 1:Nruns
                [a, ~] = fcn_mortality(r.lambda_obs, r.x, dt, ...
                    theta, patients{p}.age, patients{p}.sofa);
                alive_sel = alive_sel + a;
            end
            surv_select(p, oi, reg) = alive_sel / Nruns;

            % --- Evaluation draws (independent: fresh RNG state) ---
            alive_evl = 0;
            for run = 1:Nruns
                [a, ~] = fcn_mortality(r.lambda_obs, r.x, dt, ...
                    theta, patients{p}.age, patients{p}.sofa);
                alive_evl = alive_evl + a;
            end
            surv_eval(p, oi, reg) = alive_evl / Nruns;
        end
    end
    fprintf('    Done (%.1fs)\n', toc);
end
fprintf('Mortality evaluation done (%.1fs).\n\n', toc);

% surv_grid is kept (set to the evaluation set) for backward compatibility
% with downstream figure code that references surv_grid directly.
surv_grid = surv_eval;

%% Step 3: Find optimal omega per patient per regime (split-sample)
omega_star   = zeros(Npat, n_regimes);    % omega_i* (chosen on selection set)
surv_star    = zeros(Npat, n_regimes);    % survival at omega_i* (from eval set)
surv_uniform = zeros(Npat, n_regimes);    % survival at omega=2 (from eval set)

% Index of omega=2 in omega_vals
[~, idx_uniform] = min(abs(omega_vals - omega_uniform));

for reg = 1:n_regimes
    for p = 1:Npat
        % Pick omega_i* via the selection set; report survival via eval set
        [~, best_oi] = max(surv_select(p, :, reg));
        omega_star(p, reg) = omega_vals(best_oi);
        surv_star(p, reg)  = surv_eval(p, best_oi, reg);   % unbiased
        surv_uniform(p, reg) = surv_eval(p, idx_uniform, reg);
    end
end

%% Print summary
fprintf('=====================================================\n');
fprintf('SMART ADAPTIVE: Uniform vs Personalized Omega\n');
fprintf('=====================================================\n\n');

for reg = 1:n_regimes
    mean_surv_uniform = mean(surv_uniform(:, reg)) * 100;
    mean_surv_star = mean(surv_star(:, reg)) * 100;
    improvement = mean_surv_star - mean_surv_uniform;
    
    fprintf('--- %s ---\n', regimes(reg).name);
    fprintf('  Uniform (omega=%.1f):     %.1f%% survival\n', omega_uniform, mean_surv_uniform);
    fprintf('  Personalized (omega_i*):  %.1f%% survival\n', mean_surv_star);
    fprintf('  Improvement:              %+.1f percentage points\n', improvement);
    fprintf('  Omega* distribution:      median=%.1f, range=[%.1f, %.1f]\n', ...
        median(omega_star(:, reg)), min(omega_star(:, reg)), max(omega_star(:, reg)));
    fprintf('\n');
end

%% FIGURE 1: Omega-survival curves per regime (population average)
figure('Position', [50 50 1200 500]); clf;

for reg = 1:n_regimes
    subplot(1, n_regimes, reg);
    
    mean_surv = squeeze(mean(surv_grid(:, :, reg), 1)) * 100;
    
    % Replace Inf with a plottable value for display
    omega_plot = omega_vals;
    omega_plot(isinf(omega_plot)) = 12;  % plot Inf as 12
    
    plot(omega_plot, mean_surv, 'b-o', 'LineWidth', 2, 'MarkerSize', 6);
    hold on;
    
    % Mark uniform omega
    xline(omega_uniform, 'r--', '\omega=2', 'LineWidth', 1.5);
    
    % Mark optimal population omega
    [~, best_pop] = max(mean_surv);
    plot(omega_plot(best_pop), mean_surv(best_pop), 'rp', ...
        'MarkerSize', 15, 'MarkerFaceColor', 'r');
    
    xlabel('\omega (target rate)');
    ylabel('Survival (%)');
    title(sprintf('%s\nBest \\omega=%.1f (%.0f%%)', ...
        regimes(reg).name, omega_vals(best_pop), mean_surv(best_pop)));
    
    xticks(omega_plot);
    xticklabels(arrayfun(@(x) sprintf('%.1f', x), omega_vals, 'UniformOutput', false));
    grid on; ylim([0 100]);
end
sgtitle('Population-Average Survival vs Treatment Target \omega');

%% FIGURE 2: Optimal omega distribution per regime
figure('Position', [50 50 1200 500]); clf;

colors = [0.3 0.3 0.8; 0.8 0.3 0.3; 0.3 0.8 0.3; 0.8 0.6 0.2];

for reg = 1:n_regimes
    subplot(1, n_regimes, reg);
    
    os = omega_star(:, reg);
    os(isinf(os)) = 12;  % for histogram
    
    histogram(os, [0 0.75 1.25 1.75 2.5 3.5 5 7 10 14], ...
        'FaceColor', colors(reg,:), 'FaceAlpha', 0.7);
    xline(omega_uniform, 'r--', 'LineWidth', 2);
    xlabel('\omega_i^*');
    ylabel('Patients');
    title(sprintf('%s\nmedian \\omega^*=%.1f', regimes(reg).name, median(omega_star(:,reg))));
    grid on;
end
sgtitle('Distribution of Optimal \omega^* per Patient');

%% FIGURE 3: Personalized vs Uniform — survival comparison
figure('Position', [50 50 800 500]); clf;

bar_data = zeros(n_regimes, 2);
for reg = 1:n_regimes
    bar_data(reg, 1) = mean(surv_uniform(:, reg)) * 100;
    bar_data(reg, 2) = mean(surv_star(:, reg)) * 100;
end

b = bar(bar_data);
b(1).FaceColor = [0.5 0.5 0.5];
b(2).FaceColor = [0.2 0.7 0.2];
set(gca, 'XTickLabel', {regimes.name});
ylabel('Survival (%)');
legend('Uniform (\omega=2)', 'Personalized (\omega_i^*)', 'Location', 'best');
title('Population Survival: Uniform vs Personalized Omega');
ylim([0 100]); grid on;

% Add improvement labels
for reg = 1:n_regimes
    improvement = bar_data(reg, 2) - bar_data(reg, 1);
    text(reg, bar_data(reg, 2) + 1, sprintf('%+.1f%%', improvement), ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end

%% FIGURE 4: Omega* vs patient severity (key insight)
figure('Position', [50 50 1200 500]); clf;

for reg = 1:n_regimes
    subplot(1, n_regimes, reg);
    
    os = omega_star(:, reg);
    os(isinf(os)) = 12;
    
    scatter(peak_rates, os, 40, sofas, 'filled', 'MarkerFaceAlpha', 0.6);
    hold on;
    yline(omega_uniform, 'r--', '\omega=2', 'LineWidth', 1.5);
    
    xlabel('Peak Seizure Rate (sz/day)');
    ylabel('\omega_i^*');
    title(regimes(reg).name);
    colormap('hot');
    if reg == n_regimes
        c = colorbar; c.Label.String = 'SOFA Score';
    end
    grid on;
    ylim([0 13]);
end
sgtitle('Optimal \omega^* by Patient Severity and SOFA');

%% Save results
% Save both surv_select and surv_eval so downstream analyses can
% reconstruct the bias-correction independently.  surv_grid is kept
% (alias of surv_eval) for backward compatibility with old figure code.
save('smart_adaptive_results.mat', 'omega_vals', 'omega_star', 'surv_grid', ...
    'surv_select', 'surv_eval', ...
    'surv_star', 'surv_uniform', 'ages', 'sofas', 'peak_rates', ...
    'regimes', 'Npat', 'Nruns', 'omega_uniform');
fprintf('\nResults saved to smart_adaptive_results.mat\n');

%% Summary
fprintf('\n=== KEY FINDINGS ===\n');
fprintf('Personalized omega assigns different treatment intensities by patient type.\n');
fprintf('This is the continuous generalization of the quadrant analysis:\n');
fprintf('  - R1 (High Sz / Low Drug): low omega (aggressive, drugs are free)\n');
fprintf('  - R4 (Low Sz / High Drug): high omega or Inf (conservative, drugs dangerous)\n');
fprintf('  - The mapping omega*(patient) depends on age, SOFA, and disease severity\n');
fprintf('\nNext step: estimate theta from real data -> compute omega* per patient\n');
fprintf('This is the CICADAS vision for precision medicine in critical care.\n');
