% RUN_ALL_SIMULATIONS Master reproducibility script for the CICADAS-II
% manuscript.
%
% Executes every numbered simulation script in scripts/ in order, with a
% fixed master RNG seed for reproducibility.  Each script writes its cached
% outputs to results/*.mat; those files are what the manuscript's
% quantitative claims (e.g., "g-formula reduced confounding bias by 95.4%")
% are computed from.
%
% Mapping (script -> cached output -> paper section):
%
%   s0_validate_distributions     -> (prints diagnostics only)            -> Sim. I demographics
%   s1_single_patient_demo        -> (visual demo only)                   -> Sim. I figures
%   s2_compare_strategies         -> (visual demo only)                   -> Sim. II controllers
%   s3_population_sim             -> results/population_results.mat       -> Sim. II
%   s4_threshold_sweep            -> results/threshold_sweep_results.mat  -> Fig. 8 / Table
%   s5_quadrant_analysis          -> results/quadrant_results.mat         -> Fig. 7  (quadrant)
%   s6_parameter_estimation       -> results/parameter_estimation_results.mat -> Figs. 2-3, Tables
%   s7_estimation_robustness      -> results/estimation_robustness_results.mat
%   s8_closed_loop_validation     -> results/closed_loop_results.mat      -> Fig. 11 (closed-loop)
%   s9_gformula_causal            -> results/gformula_results.mat         -> Figs. 4, 5, Table 6
%   s10_smart_adaptive            -> results/smart_adaptive_results.mat   -> Sec. 5.3
%   s11_ke_sensitivity            -> results/ke_sensitivity_results.mat   -> Fig. 12 (sensitivity)
%   s12_personalized_validation   -> results/personalized_validation_results.mat -> Fig. 9
%   s13_closed_loop_personalized  -> results/closed_loop_personalized_results.mat -> Fig. 10
%   s14_c50_sensitivity           -> results/c50_sensitivity_results.mat  -> Fig. 12
%   s15_ekf_sensitivity           -> results/ekf_sensitivity_results.mat  -> Fig. 12 (EKF)
%   s16_misspecification          -> results/misspecification_results.mat -> Fig. 13, Sec. 5.5
%
% Usage:
%   run_all_simulations             % run with default seed (42)
%   run_all_simulations(seed)       % override the seed
%
% Total wall time on a modern laptop is approximately 1-3 hours, depending
% on which scripts dominate (s12, s15, s16 are the longest).
%
% After completion, regenerate the python figures with
%     python make_paper_figures.py
% in the repo root.

function run_all_simulations(seed)
    if nargin < 1
        seed = 42;
    end
    rng(seed, 'twister');
    fprintf('=== run_all_simulations (seed = %d) ===\n', seed);
    t0_total = tic;

    repo_root = fileparts(mfilename('fullpath'));
    addpath(fullfile(repo_root, 'functions'));
    addpath(fullfile(repo_root, 'scripts'));
    if ~exist(fullfile(repo_root, 'results'), 'dir')
        mkdir(fullfile(repo_root, 'results'));
    end
    cd(fullfile(repo_root, 'scripts'));

    scripts = { ...
        's0_validate_distributions', ...
        's3_population_sim', ...
        's4_threshold_sweep', ...
        's5_quadrant_analysis', ...
        's6_parameter_estimation', ...
        's7_estimation_robustness', ...
        's8_closed_loop_validation', ...
        's9_gformula_causal', ...
        's10_smart_adaptive', ...
        's11_ke_sensitivity', ...
        's12_personalized_validation', ...
        's13_closed_loop_personalized', ...
        's14_c50_sensitivity', ...
        's15_ekf_sensitivity', ...
        's16_misspecification', ...
        };

    for i = 1:numel(scripts)
        name = scripts{i};
        fprintf('\n--- [%d/%d] %s ---\n', i, numel(scripts), name);
        t_step = tic;
        try
            run([name '.m']);
        catch ME
            fprintf(2, 'FAILED: %s (%s)\n', name, ME.message);
            rethrow(ME);
        end
        fprintf('   [%s] done in %.1f s\n', name, toc(t_step));
    end

    cd(repo_root);
    fprintf('\n=== run_all_simulations complete (total %.1f min) ===\n', ...
            toc(t0_total) / 60);
end
