% S5_QUADRANT_ANALYSIS
% Tests how optimal treatment strategy changes under four harm profiles.

clear; clc;
addpath('functions');

%% Load default parameters
[time_params, drug_params, ctrl_params, ~] = default_params();
T = time_params.T;
Nt = time_params.Nt;
dt = time_params.dt;

%% Simulation settings
Npat = 500;      % Patients per regime
Nruns = 1000;     % Mortality runs per patient
use_ekf = true;

% Personalized adaptive (omega*) — 5th strategy.
% Sweep these omega values per patient per regime; pick the one that
% maximizes survival as omega_i* and report adaptive(omega_i*) survival.
% Set ENABLE_OMEGA_STAR=false to skip and keep the original 4-bar comparison.
ENABLE_OMEGA_STAR = true;
omega_vals = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, Inf];

rng(42);  % Reproducibility

%% Define the four harm regimes
% theta = [theta0, theta1, theta2, theta3, theta4, theta5]
%          base    time    sz_harm sofa*sz drug_harm age*drug
%
% Quadrant axes (orthogonal):
%   High Sz: theta2=3.0, theta3=1.0  |  Low Sz: theta2=0.5, theta3=0.2
%   High Drug: theta4=theta4_high, theta5=0.3  |  Low Drug: theta4=0.05, theta5=0.02
%
% Sweep theta4_high to find where adaptive wins Regime 2.
% Realistic range for ICU anti-seizure drugs: 0.3-1.5
%   0.3-0.5 = second-line (phenytoin, valproate)
%   0.8-1.5 = third-line anesthetics (propofol, midazolam, pentobarbital)
theta4_high = 1.25;   % <-- CHANGE THIS TO SWEEP
theta4_low = 0.05;    % Fixed for low drug harm regimes

regimes = struct();

% Regime 1: High seizure harm, Low drug harm → "Treat aggressively"
regimes(1).name = 'High Sz / Low Drug';
regimes(1).theta = [-5, 0.5, 3.0, 1.0, 0.05, 0.02];
regimes(1).expected = 'Fixed wins — drugs are free, max suppression best';

% Regime 2: High seizure harm, High drug harm
%   Adaptive can now give 750 during peak (better suppression than fixed's 500)
%   while backing off during tail (less total drug than fixed).
regimes(2).name = 'High Sz / High Drug';
regimes(2).theta = [-5, 0.5, 3.0, 1.0, theta4_high, 0.3];
regimes(2).expected = 'Adaptive wins — better peak suppression + drug-sparing tail';

% Regime 3: Low seizure harm, Low drug harm → "Doesn't matter"
regimes(3).name = 'Low Sz / Low Drug';
regimes(3).theta = [-5, 0.5, 0.5, 0.2, 0.05, 0.02];
regimes(3).expected = 'All strategies similar';

% Regime 4: Low seizure harm, High drug harm → "Don't treat"
%   Seizures aren't that dangerous, but drugs are.
%   Same theta4/theta5 as Regime 2 (High Drug), same theta2/theta3 as Regime 3 (Low Sz).
regimes(4).name = 'Low Sz / High Drug';
regimes(4).theta = [-5, 0.5, 0.5, 0.2, theta4_high, 0.3];
regimes(4).expected = 'None and Adaptive comparable — drugs dangerous, seizures mild';

n_regimes = numel(regimes);

%% Strategies to compare
strategies = {'none', 'fixed', 'prn', 'adaptive'};
n_strats = numel(strategies);
strat_labels = {'None', 'Fixed', 'PRN', 'Adaptive'};

%% Run simulation for each regime
fprintf('=== QUADRANT ANALYSIS ===\n');
fprintf('Patients per regime: %d | Mortality runs: %d\n\n', Npat, Nruns);

% Storage: survival rates [n_regimes x n_strats]
surv_rates = zeros(n_regimes, n_strats);
burden_means = zeros(n_regimes, n_strats);
drug_means = zeros(n_regimes, n_strats);

% Also store per-patient for scatter plots
all_results = struct();

patient_params.peak_rate = 10;
patient_params.drug_params = drug_params;  % For patient-specific alpha

tic;

%% STEP 1: Generate patients and run strategies ONCE
fprintf('Generating %d patients and running strategies...\n', Npat);
patients = cell(Npat, 1);
strat_results = cell(Npat, n_strats);  % store lambda_obs, x per patient/strategy
burden_pat = zeros(Npat, n_strats);
drug_pat = zeros(Npat, n_strats);
ages = zeros(Npat, 1);
sofas = zeros(Npat, 1);

for p = 1:Npat
    patient = fcn_generate_patient(T, patient_params);
    patients{p} = patient;
    ages(p) = patient.age;
    sofas(p) = patient.sofa;
    
    dp = drug_params;
    dp.alpha = patient.alpha;
    if isfield(patient, 'C50')
        dp.patient_C50 = patient.C50;
        dp.patient_gamma = patient.gamma_hill;
    end
    
    for s = 1:n_strats
        r = fcn_run_strategy(strategies{s}, patient.lambda_base, ...
            dp, ctrl_params, T, dt, use_ekf);
        strat_results{p, s} = r;
        burden_pat(p, s) = r.expected_burden;
        drug_pat(p, s) = r.drug_exposure;
    end
    
    if mod(p, 100) == 0
        fprintf('  %d/%d patients generated (%.1fs)\n', p, Npat, toc);
    end
end
fprintf('Strategies complete. Now evaluating mortality under %d regimes...\n', n_regimes);

%% STEP 2: Evaluate mortality under each regime (SAME patients, different theta)
for reg = 1:n_regimes
    fprintf('--- Regime %d: %s ---\n', reg, regimes(reg).name);
    
    surv_pat = zeros(Npat, n_strats);
    
    for p = 1:Npat
        patient = patients{p};
        
        for s = 1:n_strats
            r = strat_results{p, s};
            
            surv_temp = 0;
            for run = 1:Nruns
                [alive, ~] = fcn_mortality(r.lambda_obs, r.x, dt, ...
                    regimes(reg).theta, patient.age, patient.sofa);
                surv_temp = surv_temp + alive;
            end
            surv_pat(p, s) = surv_temp / Nruns;
        end
    end
    
    % Store
    surv_rates(reg, :) = mean(surv_pat) * 100;
    burden_means(reg, :) = mean(burden_pat);
    drug_means(reg, :) = mean(drug_pat);
    
    all_results(reg).surv_pat = surv_pat;
    all_results(reg).burden_pat = burden_pat;
    all_results(reg).ages = ages;
    all_results(reg).sofas = sofas;
    
    fprintf('  Done (%.1fs)\n', toc);
end

fprintf('\nBase-strategy time: %.1f seconds\n\n', toc);

%% STEP 3 (optional): Personalized adaptive omega* — 5th strategy
% For each patient, sweep omega across omega_vals; for each regime, pick
% the omega that maximizes survival as omega_i*(regime). The 5th-strategy
% survival/burden/drug for each patient under each regime are read from
% the omega_i* sweep slot. Adds one extra "Adaptive (omega*)" bar.

n_omega = numel(omega_vals);
omega_star          = nan(Npat, n_regimes);   % chosen omega per patient per regime
surv_star_pat       = nan(Npat, n_regimes);   % survival at omega*
omega_burden_pat    = nan(Npat, n_regimes);   % expected_burden at omega*
omega_drug_pat      = nan(Npat, n_regimes);   % drug_exposure at omega*
omega_total_dose    = nan(Npat, n_regimes);   % total_dose at omega*

if ENABLE_OMEGA_STAR
    fprintf('=== Step 3: Personalized adaptive (omega*) ===\n');
    fprintf('Sweep grid: [%s]\n', strjoin(arrayfun(@(x) sprintf('%.1f', x), omega_vals, 'UniformOutput', false), ', '));

    % Step 3a: run adaptive once at each omega for every patient
    fprintf('  Running adaptive at %d omegas x %d patients...\n', n_omega, Npat);
    omega_strat_results = cell(Npat, n_omega);
    tic;
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
                % omega = Inf means "never trigger" -> equivalent to no treatment
                omega_strat_results{p, oi} = fcn_run_strategy('none', ...
                    patient.lambda_base, dp, cp, T, dt, use_ekf);
            else
                omega_strat_results{p, oi} = fcn_run_strategy('adaptive', ...
                    patient.lambda_base, dp, cp, T, dt, use_ekf);
            end
        end

        if mod(p, 100) == 0
            fprintf('    %d/%d patients swept (%.1fs)\n', p, Npat, toc);
        end
    end
    fprintf('  Omega sweep complete (%.1fs)\n', toc);

    % Step 3b: per regime, evaluate mortality at each omega and pick the best.
    %
    % BIAS CORRECTION (split-sample / honest argmax):
    %   Using the same Monte Carlo draws to pick omega_i* AND to report
    %   survival at omega_i* yields a winner's-curse upward bias -- the
    %   omega selected as argmax is the one whose MC noise was most
    %   favorable. We split the mortality MC into two independent halves:
    %
    %     selection set  (Nruns draws): used only to choose omega_i*
    %     evaluation set (Nruns draws): used only to report survival at omega_i*
    %
    %   The reported survival is then unbiased: argmax is taken over
    %   selection-set noise, but the reported value is sampled from
    %   independent evaluation noise. Both halves use the SAME
    %   omega_strat_results trajectories (only the random mortality draws
    %   differ), so this isolates the selection bias without changing
    %   the underlying patient/strategy population.
    %
    %   Total mortality calls per (p, oi) become 2*Nruns. The four base
    %   strategies are NOT affected since they are predefined and not
    %   chosen by argmax over noise.
    for reg = 1:n_regimes
        fprintf('  Regime %d (%s): finding omega* (split-sample)...\n', reg, regimes(reg).name);
        surv_select = zeros(Npat, n_omega);   % drives argmax
        surv_eval   = zeros(Npat, n_omega);   % drives reported survival
        for p = 1:Npat
            for oi = 1:n_omega
                r = omega_strat_results{p, oi};

                % --- Selection draws ---
                alive_sel = 0;
                for run = 1:Nruns
                    [a, ~] = fcn_mortality(r.lambda_obs, r.x, dt, ...
                        regimes(reg).theta, patients{p}.age, patients{p}.sofa);
                    alive_sel = alive_sel + a;
                end
                surv_select(p, oi) = alive_sel / Nruns;

                % --- Evaluation draws (independent: fresh RNG state) ---
                alive_evl = 0;
                for run = 1:Nruns
                    [a, ~] = fcn_mortality(r.lambda_obs, r.x, dt, ...
                        regimes(reg).theta, patients{p}.age, patients{p}.sofa);
                    alive_evl = alive_evl + a;
                end
                surv_eval(p, oi) = alive_evl / Nruns;
            end
        end
        for p = 1:Npat
            % Pick omega_i* using selection set, report survival from eval set
            [~, best_oi] = max(surv_select(p, :));
            surv_star_pat(p, reg) = surv_eval(p, best_oi);
            omega_star(p, reg) = omega_vals(best_oi);
            omega_burden_pat(p, reg) = omega_strat_results{p, best_oi}.expected_burden;
            omega_drug_pat(p, reg) = omega_strat_results{p, best_oi}.drug_exposure;
            omega_total_dose(p, reg) = omega_strat_results{p, best_oi}.total_dose;
        end
        fprintf('    Done (%.1fs) | median omega* = %.2f\n', toc, median(omega_star(:, reg)));
    end

    % Extend strategy arrays to include the 5th strategy.
    % Console-friendly label here (no LaTeX); figure code uses LaTeX
    % directly via 'Adaptive (\omega^*)' where it's drawn.
    strategies = [strategies, {'adaptive_omega_star'}];
    strat_labels = [strat_labels, {'Adapt(w*)'}];
    n_strats = numel(strategies);

    % Append the 5th column to the population-level results
    surv_rates(:, 5)   = mean(surv_star_pat, 1)' * 100;
    burden_means(:, 5) = mean(omega_burden_pat, 1)';
    drug_means(:, 5)   = mean(omega_drug_pat, 1)';

    % Append per-patient survival to each regime's all_results struct
    for reg = 1:n_regimes
        all_results(reg).surv_pat(:, 5)        = surv_star_pat(:, reg);
        all_results(reg).burden_pat(:, 5)      = omega_burden_pat(:, reg);
        all_results(reg).omega_star(:, 1)      = omega_star(:, reg);
    end

    fprintf('Personalized adaptive complete. Total elapsed: %.1fs\n\n', toc);
else
    fprintf('Skipping personalized adaptive (ENABLE_OMEGA_STAR=false).\n\n');
end

%% Print summary table
fprintf('=====================================================\n');
fprintf('QUADRANT ANALYSIS: Survival Rates by Regime\n');
fprintf('=====================================================\n');
fprintf('%-22s', 'Regime');
for s = 1:n_strats
    fprintf('%10s', strat_labels{s});
end
fprintf('%12s\n', 'Best');
fprintf('%s\n', repmat('-', 1, 85));

for reg = 1:n_regimes
    fprintf('%-22s', regimes(reg).name);
    [best_val, best_idx] = max(surv_rates(reg,:));
    for s = 1:n_strats
        if s == best_idx
            fprintf('  %6.1f%% *', surv_rates(reg,s));
        else
            fprintf('  %6.1f%%  ', surv_rates(reg,s));
        end
    end
    fprintf('  %s\n', strat_labels{best_idx});
end

fprintf('\n* = best strategy for that regime\n');

% Print burden and drug exposure for the 4 base strategies (same across
% regimes since same patients/strategies).  The personalized adaptive
% strategy varies per regime, so it's summarized separately below.
fprintf('\n=====================================================\n');
fprintf('STRATEGY METRICS (averaged across %d patients)\n', Npat);
fprintf('=====================================================\n');
fprintf('%-18s %12s %12s %12s\n', 'Strategy', 'L_cum', 'X_cum', 'Total Dose');
fprintf('%s\n', repmat('-', 1, 60));
n_base = 4;  % first 4 strategies are run once per patient
for s = 1:n_base
    avg_burden = mean(burden_pat(:,s));
    avg_drug = mean(drug_pat(:,s));
    % Compute average total dose
    total_doses = zeros(Npat, 1);
    for p = 1:Npat
        total_doses(p) = strat_results{p,s}.total_dose;
    end
    fprintf('%-18s %12.1f %12.0f %12.0f\n', strat_labels{s}, avg_burden, avg_drug, mean(total_doses));
end

if ENABLE_OMEGA_STAR
    fprintf('\n=====================================================\n');
    fprintf('PERSONALIZED ADAPTIVE (omega*) SUMMARY (per regime)\n');
    fprintf('=====================================================\n');
    fprintf('%-22s %10s %10s %10s %10s\n', 'Regime', 'med w*', 'L_cum', 'X_cum', 'TotalDose');
    fprintf('%s\n', repmat('-', 1, 64));
    for reg = 1:n_regimes
        med_omega = median(omega_star(:, reg));
        if isinf(med_omega)
            omega_str = '  Inf';
        else
            omega_str = sprintf('%6.2f', med_omega);
        end
        fprintf('%-22s %10s %10.1f %10.0f %10.0f\n', ...
            regimes(reg).name, omega_str, ...
            mean(omega_burden_pat(:, reg)), ...
            mean(omega_drug_pat(:, reg)), ...
            mean(omega_total_dose(:, reg)));
    end
end

fprintf('\nExpected behavior:\n');
for reg = 1:n_regimes
    fprintf('  %s: %s\n', regimes(reg).name, regimes(reg).expected);
end

%% FIGURE 1: The Quadrant — Bar chart (2x2 layout)
figure('Position', [50 50 1200 900]); clf;

% 5-strategy color palette (5th is purple for personalized adaptive)
colors = [0.7 0.7 0.7;   % None        - gray
          0.3 0.3 0.8;   % Fixed       - blue
          0.8 0.3 0.3;   % PRN         - red
          0.3 0.8 0.3;   % Adaptive    - green
          0.5 0.2 0.7];  % Adapt(w*)   - purple

% Arrange as 2x2 matching the quadrant diagram:
%   [1] High Sz / Low Drug   [2] High Sz / High Drug
%   [3] Low Sz / Low Drug    [4] Low Sz / High Drug
quad_order = [1, 2, 3, 4];

for qi = 1:4
    reg = quad_order(qi);
    subplot(2, 2, qi);

    b = bar(surv_rates(reg, :), 'FaceColor', 'flat');
    for s = 1:n_strats
        b.CData(s,:) = colors(s,:);
    end

    set(gca, 'XTickLabel', strat_labels);
    xtickangle(15);
    ylabel('Survival (%)');
    ylim([0 100]);

    [best_val, best_idx] = max(surv_rates(reg,:));
    title(sprintf('%s\nBest: %s (%.0f%%)', ...
        regimes(reg).name, strat_labels{best_idx}, best_val), ...
        'FontSize', 11);
    grid on;
end

sgtitle('Optimal Strategy Depends on Harm Profile', 'FontSize', 14, 'FontWeight', 'bold');

%% FIGURE 2: Survival benefit distributions (treatment - no treatment)
figure('Position', [50 50 1200 600]); clf;

for qi = 1:4
    reg = quad_order(qi);
    subplot(2, 2, qi);

    % Benefit = strategy survival - no treatment survival
    benefit_fixed = (all_results(reg).surv_pat(:,2) - all_results(reg).surv_pat(:,1)) * 100;
    benefit_adapt = (all_results(reg).surv_pat(:,4) - all_results(reg).surv_pat(:,1)) * 100;

    hold on;
    histogram(benefit_fixed, -50:5:100, 'FaceColor', colors(2,:), 'FaceAlpha', 0.5, 'DisplayName', 'Fixed');
    histogram(benefit_adapt, -50:5:100, 'FaceColor', colors(4,:), 'FaceAlpha', 0.5, 'DisplayName', 'Adaptive');
    if ENABLE_OMEGA_STAR && size(all_results(reg).surv_pat, 2) >= 5
        benefit_omega = (all_results(reg).surv_pat(:,5) - all_results(reg).surv_pat(:,1)) * 100;
        histogram(benefit_omega, -50:5:100, 'FaceColor', colors(5,:), 'FaceAlpha', 0.55, 'DisplayName', 'Adaptive (\omega^*)');
    end
    xline(0, 'k--', 'LineWidth', 1.5, 'HandleVisibility', 'off');
    xlabel('Survival Benefit vs No Treatment (%)');
    ylabel('Patients');
    title(regimes(reg).name);
    legend('Location', 'best');
    grid on;
end

sgtitle('Distribution of Treatment Benefit by Harm Regime', 'FontSize', 14);

%% FIGURE 3: Summary heatmap
figure('Position', [100 100 700 500]); clf;

imagesc(surv_rates);
% Dark red = bad (low survival), Light yellow/white = good (high survival)
colormap(hot);
colorbar;
clim([0 100]);

set(gca, 'XTick', 1:n_strats, 'XTickLabel', strat_labels);
set(gca, 'YTick', 1:n_regimes, 'YTickLabel', {regimes.name});
xlabel('Strategy');
ylabel('Harm Regime');
title('Survival Rate (%) — Optimal Strategy Varies by Regime');

% Add text labels
for reg = 1:n_regimes
    [~, best_idx] = max(surv_rates(reg,:));
    for s = 1:n_strats
        % Dark text on light cells, white text on dark cells
        if surv_rates(reg,s) > 50
            txt_color = 'k';
        else
            txt_color = 'w';
        end
        
        label = sprintf('%.0f%%', surv_rates(reg,s));
        if s == best_idx
            label = [label ' *'];
        end
        text(s, reg, label, 'HorizontalAlignment', 'center', ...
            'Color', txt_color, 'FontWeight', 'bold', 'FontSize', 11);
    end
end

%% Save results
save('quadrant_results.mat', 'regimes', 'surv_rates', 'burden_means', ...
     'drug_means', 'all_results', 'strategies', 'strat_labels', ...
     'ENABLE_OMEGA_STAR', 'omega_vals', 'omega_star', 'surv_star_pat', ...
     'omega_burden_pat', 'omega_drug_pat', 'omega_total_dose');
fprintf('\nResults saved to quadrant_results.mat\n');

fprintf('\n=== KEY TAKEAWAY ===\n');
fprintf('The optimal strategy changes depending on the harm profile.\n');
fprintf('This is why we need REAL patient data to estimate theta:\n');
fprintf('  - Real theta2/theta3 tell us how harmful seizures are\n');
fprintf('  - Real theta4/theta5 tell us how harmful drugs are\n');
fprintf('  - The ratio determines the optimal treatment intensity\n');
fprintf('  - Different patients (age/SOFA) may be in different quadrants\n');
fprintf('\nKey results (expected with theta4_high=%.2f):\n', theta4_high);
fprintf('  - High Sz / Low Drug: Fixed wins (drugs free, max suppression)\n');
fprintf('  - High Sz / High Drug: Adaptive wins (better peak suppression, drug-sparing tail)\n');
fprintf('  - Low Sz / Low Drug: All strategies similar\n');
fprintf('  - Low Sz / High Drug: None and Adaptive comparable (drugs dangerous, seizures mild)\n');
