function out = fcn_paired_bootstrap(Sa, Sb, B, seed, alpha)
% FCN_PAIRED_BOOTSTRAP  Paired comparison of two strategies on the same patients.
%
%   out = fcn_paired_bootstrap(Sa, Sb)            % B = 2000, seed = 1, alpha = 0.05
%   out = fcn_paired_bootstrap(Sa, Sb, B, seed, alpha)
%
% Sa, Sb: n x 1 per-patient survival probabilities for strategies a and b,
% computed on the SAME patients (same cohort, common random numbers). The
% estimand is the mean difference mean(Sa - Sb). Patients are resampled with
% replacement and the SAME resampled indices are used for both strategies,
% which keeps the pairing of strategy comparisons.
%
% Output (all in percentage points):
%   .diff      - mean(Sa - Sb) * 100
%   .se        - analytic paired SE = sd(Sa - Sb)/sqrt(n) * 100 (the MC-SE)
%   .ci_normal - diff +/- z * se
%   .ci_boot   - percentile bootstrap interval
%   .sd_diff   - sd of per-patient differences (used to size n)
%   .n

if nargin < 3 || isempty(B), B = 2000; end
if nargin < 4 || isempty(seed), seed = 1; end
if nargin < 5 || isempty(alpha), alpha = 0.05; end

Sa = Sa(:); Sb = Sb(:);
n = numel(Sa);
D = Sa - Sb;

out.n = n;
out.diff = 100 * mean(D);
out.sd_diff = 100 * std(D);
out.se = out.sd_diff / sqrt(n);
zc = sqrt(2) * erfinv(1 - alpha);          % 1.96 for alpha = 0.05, no toolbox needed
out.ci_normal = out.diff + [-1, 1] * zc * out.se;

% Bootstrap with a private stream so it never disturbs the caller's RNG state
st = rng;
rng(seed);
idx = randi(n, n, B);
rng(st);
bm = 100 * mean(D(idx), 1);
bm = sort(bm);
lo = bm(max(1, floor(B * alpha / 2)));
hi = bm(min(B, ceil(B * (1 - alpha / 2))));
out.ci_boot = [lo, hi];
out.B = B;

end
