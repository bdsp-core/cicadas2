function T = fcn_summarize_survival(S, labels, ref, B, seed)
% FCN_SUMMARIZE_SURVIVAL  Survival per strategy with MC-SE / 95% CI, and paired contrasts.
%
%   T = fcn_summarize_survival(S, labels)            % contrasts vs strategy 1
%   T = fcn_summarize_survival(S, labels, ref, B, seed)
%
% S      : n x K per-patient survival probabilities (columns = strategies)
% labels : 1 x K cell
% ref    : index of the reference strategy for contrasts (default 1), or 0
%          for all pairwise contrasts
%
% Output struct T:
%   .labels, .n
%   .surv, .se, .ci_lo, .ci_hi     - 1 x K, percent. se = sd/sqrt(n) is the
%                                    Monte Carlo standard error of the cohort
%                                    mean (patient sampling + within-run
%                                    seizure-count randomness; mortality is exact)
%   .contrast(j)                   - struct from fcn_paired_bootstrap with
%                                    .a, .b labels added
%   .best, .best_label             - argmax of survival
%   .p_best_boot                   - bootstrap share of resamples in which each
%                                    strategy has the highest mean survival
%                                    (how sure is the "winner" label)

if nargin < 3 || isempty(ref), ref = 1; end
if nargin < 4 || isempty(B), B = 2000; end
if nargin < 5 || isempty(seed), seed = 1; end

[n, K] = size(S);
T.labels = labels;
T.n = n;
T.surv = 100 * mean(S, 1);
T.se = 100 * std(S, 0, 1) / sqrt(n);
T.ci_lo = T.surv - 1.96 * T.se;
T.ci_hi = T.surv + 1.96 * T.se;
[~, T.best] = max(T.surv);
T.best_label = labels{T.best};

% Contrasts
pairs = [];
if ref == 0
    for a = 1:K, for b = a+1:K, pairs(end+1, :) = [a b]; end, end %#ok<AGROW>
else
    for a = 1:K, if a ~= ref, pairs(end+1, :) = [a ref]; end, end %#ok<AGROW>
end
T.contrast = struct('a', {}, 'b', {});
for j = 1:size(pairs, 1)
    c = fcn_paired_bootstrap(S(:, pairs(j, 1)), S(:, pairs(j, 2)), B, seed + j);
    c.a = labels{pairs(j, 1)};
    c.b = labels{pairs(j, 2)};
    if j == 1
        T.contrast = c;
    else
        T.contrast(j) = c;
    end
end

% How stable is the winner? Resample patients, recompute argmax.
st = rng; rng(seed + 999);
idx = randi(n, n, B);
rng(st);
wins = zeros(1, K);
for b = 1:B
    [~, w] = max(mean(S(idx(:, b), :), 1));
    wins(w) = wins(w) + 1;
end
T.p_best_boot = wins / B;

end
