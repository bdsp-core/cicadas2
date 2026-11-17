% S0_VALIDATE_DISTRIBUTIONS
% Validates that patient generation produces realistic distributions
%
% Generates N patients and plots distributions for age and SOFA

clear; clc;

% Add functions to path
addpath('functions');

%% Parameters
N = 10000;  % Number of patients to generate
dt = 5/60/24;
T = 0:dt:4;  % 4 days

%% Generate many patients
ages = zeros(1, N);
sofas = zeros(1, N);

params.peak_rate = 10;

for i = 1:N
    patient = fcn_generate_patient(T, params);
    ages(i) = patient.age;
    sofas(i) = patient.sofa;
end

%% Plot Age Distribution (smooth line - continuous variable)
figure('Position', [100 100 1000 400]);

subplot(1, 2, 1);

% Kernel density estimate for continuous age data
[f_age, xi_age] = ksdensity(ages, 'Support', [18 100], 'BoundaryCorrection', 'reflection');

% Convert density to percent (density integrates to 1, multiply by 100)
plot(xi_age, f_age * 100, 'b-', 'LineWidth', 2);

xlabel('Age (years)', 'FontSize', 12);
ylabel('Percent of Patients (%)', 'FontSize', 12);
title('Age Distribution (N = 10,000)', 'FontSize', 14);
xlim([18 100]);
grid on;

% Add summary stats
text(0.95, 0.95, sprintf('Mean: %.1f\nMedian: %.1f\nSD: %.1f', ...
    mean(ages), median(ages), std(ages)), ...
    'Units', 'normalized', 'HorizontalAlignment', 'right', ...
    'VerticalAlignment', 'top', 'FontSize', 10, 'BackgroundColor', 'w');

%% Plot SOFA Distribution (line connecting discrete counts)
subplot(1, 2, 2);

% SOFA is discrete (integers 0-24), so count occurrences
sofa_values = 0:24;
sofa_counts = histcounts(sofas, -0.5:24.5);  % bin edges for integers 0-24
sofa_percent = sofa_counts / N * 100;

% Plot as line connecting points
plot(sofa_values, sofa_percent, 'r-', 'LineWidth', 2);
hold on;
plot(sofa_values, sofa_percent, 'ro', 'MarkerSize', 4, 'MarkerFaceColor', 'r');

xlabel('SOFA Score', 'FontSize', 12);
ylabel('Percent of Patients (%)', 'FontSize', 12);
title('SOFA Distribution (N = 10,000)', 'FontSize', 14);
xlim([-0.5 24.5]);
grid on;

% Add summary stats
text(0.95, 0.95, sprintf('Mean: %.1f\nMedian: %.1f\nSD: %.1f', ...
    mean(sofas), median(sofas), std(sofas)), ...
    'Units', 'normalized', 'HorizontalAlignment', 'right', ...
    'VerticalAlignment', 'top', 'FontSize', 10, 'BackgroundColor', 'w');

%% Print summary
fprintf('\n=== Age Distribution ===\n');
fprintf('Mean: %.1f years\n', mean(ages));
fprintf('Median: %.1f years\n', median(ages));
fprintf('Std: %.1f years\n', std(ages));
fprintf('Range: [%.0f, %.0f]\n', min(ages), max(ages));
fprintf('Quartiles: [%.0f, %.0f, %.0f]\n', prctile(ages, [25, 50, 75]));

fprintf('\n=== SOFA Distribution ===\n');
fprintf('Mean: %.1f\n', mean(sofas));
fprintf('Median: %.1f\n', median(sofas));
fprintf('Std: %.1f\n', std(sofas));
fprintf('Range: [%d, %d]\n', min(sofas), max(sofas));
fprintf('Quartiles: [%.0f, %.0f, %.0f]\n', prctile(sofas, [25, 50, 75]));

% SOFA category breakdown
fprintf('\nSOFA Categories:\n');
fprintf('  0-6 (low risk):    %.1f%%\n', 100*sum(sofas <= 6)/N);
fprintf('  7-9 (moderate):    %.1f%%\n', 100*sum(sofas >= 7 & sofas <= 9)/N);
fprintf('  10-12 (high):      %.1f%%\n', 100*sum(sofas >= 10 & sofas <= 12)/N);
fprintf('  13+ (very high):   %.1f%%\n', 100*sum(sofas >= 13)/N);
