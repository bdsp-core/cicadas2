function burden = fcn_compute_burden(seizure_counts)
% FCN_COMPUTE_BURDEN Total seizure count from Poisson events.
% Not currently called; burden computed inline in fcn_run_strategy.

burden = sum(seizure_counts);

end
