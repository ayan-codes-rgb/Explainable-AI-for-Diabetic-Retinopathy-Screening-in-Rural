function [w, T] = classWeights(labels, method)
%CLASSWEIGHTS Per-class loss weights for the imbalanced DR grading problem.
%
%   [w, T] = CLASSWEIGHTS(labels)
%   [w, T] = CLASSWEIGHTS(labels, method)
%
%   labels is a categorical vector (use S.train.labels from buildDatastores).
%   w is a row vector aligned to categories(labels), for use as the
%   ClassWeights of a weighted cross-entropy loss. T is a table showing the
%   counts and weights, so you can see what you are about to do.
%
%   method
%     'inverse'      (default) w = N / (K * n_c). Full inverse frequency.
%     'inverse-sqrt' w proportional to 1/sqrt(n_c). Gentler; usually the
%                    better choice, because full inverse frequency on a
%                    class with 17 images makes each of those images
%                    enormously influential and the model overfits them.
%     'none'         all ones, for an unweighted baseline to compare against.
%
%   WHY THIS MATTERS HERE, SPECIFICALLY
%   The scarce class on IDRiD+APTOS combined is grade 3 (severe NPDR),
%   ~6.8% of images -- not grade 0 as you might assume. Grade 3 is on the
%   REFERABLE side of the screening threshold, so missing it costs
%   sensitivity, which carries the tighter >90% target. Grade 1 sits on the
%   non-referable side and costs specificity (>85%). Weight accordingly:
%   errors are not symmetric in consequence.
%
%   See also BUILDDATASTORES, LOADSPLIT.

if nargin < 2 || isempty(method)
    method = 'inverse-sqrt';
end
method = lower(char(method));

cats = categories(labels);
n    = countcats(labels);
n    = n(:)';
K    = numel(cats);
N    = sum(n);

switch method
    case 'inverse'
        w = N ./ (K * max(n, 1));
    case 'inverse-sqrt'
        w = 1 ./ sqrt(max(n, 1));
        w = w * (K / sum(w));            % normalise so mean weight is 1
    case 'none'
        w = ones(1, K);
    otherwise
        error('classWeights:badMethod', ...
            'method must be ''inverse'', ''inverse-sqrt'' or ''none''.');
end

T = table(string(cats(:)), n(:), (100*n(:)/N), w(:), ...
    'VariableNames', {'class', 'count', 'percent', 'weight'});

if nargout == 0
    disp(T); clear w
end
end
