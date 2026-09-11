function T = tier1Experiments(roots, varargin)
%TIER1EXPERIMENTS Seven classifier variants on the SAME frozen features.
%
%   T = TIER1EXPERIMENTS(roots)
%   T = TIER1EXPERIMENTS(roots, 'InputSize',[448 448], 'Enhance','none')
%
%   Requires features already cached by a matching runBaseline call. Nothing
%   here touches the CNN, so every variant costs seconds to minutes instead
%   of hours -- the point is to find out how much of the remaining gap is
%   the CLASSIFIER rather than the representation.
%
%   WHY THESE SEVEN
%   Everything tried so far changed the inputs to a frozen network and left
%   IDRiD test performance flat (six configurations, AUC 0.835-0.877, all
%   inside the +/-0.036 measurement error). What was never varied is the
%   thing on top: a plain logistic regression. These variants attack the
%   three diagnosed problems instead.
%
%     linear        baseline reproduction -- logistic regression
%     svm-rbf       nonlinear boundary. A linear separator in 512-D is an
%                   assumption, not a law; borderline grade 1 vs 2 cases may
%                   simply not be linearly separable in this feature space.
%     boosted-trees gradient boosting -- different inductive bias again,
%                   and it picks its own feature interactions
%     mlp           small fully-connected net on the frozen features
%     ordinal       regression on grade 0-4, not binary classification.
%                   Grades are ORDERED and the referral line sits exactly at
%                   the 1-2 boundary; collapsing to binary throws that
%                   ordering away. Score = predicted severity.
%     idrid-only    head trained on IDRiD rows alone. The extreme version of
%                   domain weighting: stop letting 3,112 APTOS images outvote
%                   352 IDRiD ones entirely.
%     svm-idrid     nonlinear boundary fitted on IDRiD rows alone
%     svm-weighted  nonlinear boundary, domain-balanced sample weights --
%                   keeps APTOS volume without letting it outvote IDRiD
%     svm-l2norm    RBF on unit-length features -- the kernel is a distance,
%                   so uneven embedding magnitudes distort it
%     svm-tuned     Bayesian search over BoxConstraint and KernelScale; the
%                   kernel won on defaults that were never examined
%     svm-l2-tuned  both of the above
%     ensemble      svm-rbf + boosted trees + IDRiD-only SVM, standardised
%                   on training scores and averaged
%     coral         CORAL domain alignment -- a closed-form linear transform
%                   that recolours APTOS feature covariance to match IDRiD's
%                   before training. Attacks the domain shift in feature
%                   space rather than in pixel space.
%
%   REPORTED METRIC IS AUC, on both held-out sets. AUC is threshold-free, so
%   variants are comparable without the threshold-tuning noise that a
%   61-image IDRiD validation set introduces. Sensitivity/specificity are
%   shown too, at a threshold tuned per variant on IDRiD validation rows.
%
%   Options
%     'Backbone'  'resnet18' (default) -- must match a cached run
%     'InputSize' [448 448]
%     'Enhance'   'none' | 'bengraham' | 'clahe'
%     'Plot'      true -- PCA scatter of the two domains in feature space
%
%   See also RUNBASELINE, ROCREPORT, BUILDDATASTORES.

p = inputParser;
p.FunctionName = 'tier1Experiments';
p.addParameter('Backbone',  'resnet18');
p.addParameter('InputSize', [448 448]);
p.addParameter('Enhance',   'none');
p.addParameter('Plot',      true);
p.parse(varargin{:});
opt = p.Results;

fprintf('\n=== Tier-1 experiments (frozen features, %s @ %dx%d, enhance=%s) ===\n', ...
    opt.Backbone, opt.InputSize(1), opt.InputSize(2), lower(char(opt.Enhance)));

% ---- data ------------------------------------------------------------
S = buildDatastores(roots, 'InputSize', opt.InputSize, 'Task', 'referable', ...
                           'Augment', false, 'Enhance', opt.Enhance);

D = struct();
for nm = ["train" "val" "test" "test_aptos"]
    k = char(nm);
    D.(k).X  = iLoadCache(k, opt, S);
    D.(k).y  = S.(k).labels == "referable";
    D.(k).ds = string(S.(k).tbl.dataset);
    D.(k).g  = double(S.(k).tbl.grade);
end
fprintf('features %d-dim | train %d  val %d  IDRiD-test %d  APTOS-test %d\n\n', ...
    size(D.train.X,2), numel(D.train.y), numel(D.val.y), ...
    numel(D.test.y), numel(D.test_aptos.y));

% ---- CORAL transform, fitted on training rows only -------------------
% Recolour APTOS features so their covariance matches IDRiD's. The
% classifier then lives in IDRiD's feature space, which is where the test
% set lives. Fitted on train only; applied to APTOS rows in every split.
mA = D.train.ds == "APTOS";  mI = D.train.ds == "IDRiD";
[coralA, coralMu, coralMuT] = iFitCoral(D.train.X(mA,:), D.train.X(mI,:), 1.0);

% ---- run the variants ------------------------------------------------
rows = {};
rows(end+1,:) = iRun('linear',        @() iLinear(D.train.X, D.train.y),               D); %#ok<*AGROW>
rows(end+1,:) = iRun('svm-rbf',       @() iSvmRbf(D.train.X, D.train.y),               D);
rows(end+1,:) = iRun('boosted-trees', @() iTrees(D.train.X, D.train.y),                D);
rows(end+1,:) = iRun('mlp',           @() iMlp(D.train.X, D.train.y),                  D);
rows(end+1,:) = iRun('ordinal',       @() iOrdinal(D.train.X, D.train.g),              D);
rows(end+1,:) = iRun('idrid-only',    @() iLinear(D.train.X(mI,:), D.train.y(mI)),     D);
rows(end+1,:) = iRun('coral',         @() iCoralModel(D, coralA, coralMu, coralMuT),   D, ...
                                       @(X, ds) iApplyCoral(X, ds, coralA, coralMu, coralMuT));
% The first pass showed the winner was nonlinear (svm-rbf) and the runner-up
% was domain-specialised (idrid-only). These two combine the pair: a
% nonlinear boundary fitted only on IDRiD, and a nonlinear boundary that
% keeps APTOS's volume but stops it outvoting IDRiD nine to one.
rows(end+1,:) = iRun('svm-idrid',     @() iSvmRbf(D.train.X(mI,:), D.train.y(mI)),      D);
rows(end+1,:) = iRun('svm-weighted',  @() iSvmWeighted(D.train.X, D.train.y, D.train.ds), D);
% Second pass. The RBF kernel won on completely untuned settings, so the
% obvious unexploited knobs are the kernel's own hyperparameters and the
% scale of the features it measures distances between.
rows(end+1,:) = iRun('svm-l2norm',    @() iSvmRbf(iL2(D.train.X), D.train.y),          D, @(X,~) iL2(X));
rows(end+1,:) = iRun('svm-tuned',     @() iSvmTuned(D.train.X, D.train.y),             D);
rows(end+1,:) = iRun('svm-l2-tuned',  @() iSvmTuned(iL2(D.train.X), D.train.y),        D, @(X,~) iL2(X));
rows(end+1,:) = iRun('ensemble',      @() iEnsemble(D.train.X, D.train.y, D.train.ds), D);

T = cell2table(rows, 'VariableNames', ...
    {'variant','aucIDRiD','aucAPTOS','sensIDRiD','specIDRiD','seconds'});

% ---- report ----------------------------------------------------------
fprintf('\n%-15s%11s%11s%11s%11s%9s\n', 'variant', 'AUC IDRiD', 'AUC APTOS', ...
        'sens IDRiD', 'spec IDRiD', 'secs');
fprintf('%s\n', repmat('-', 1, 68));
for i = 1:height(T)
    fprintf('%-15s%11.3f%11.3f%10.1f%%%10.1f%%%9.0f\n', T.variant{i}, ...
        T.aucIDRiD(i), T.aucAPTOS(i), 100*T.sensIDRiD(i), 100*T.specIDRiD(i), T.seconds(i));
end
fprintf('%s\n', repmat('-', 1, 68));
[~, best] = max(T.aucIDRiD);
fprintf('best on IDRiD: %s (AUC %.3f)\n', T.variant{best}, T.aucIDRiD(best));
fprintf(['\nIDRiD test n=%d -> AUC std-error ~0.036. A difference under ~0.07\n' ...
         'between two variants is NOT distinguishable on this set. APTOS test\n' ...
         'n=%d -> ~0.022; steer by that column.\n\n'], ...
         numel(D.test.y), numel(D.test_aptos.y));

% ---- domain diagnostic ----------------------------------------------
if opt.Plot
    iPcaPlot(D.train.X, D.train.ds);
end
end

% =====================================================================
function row = iRun(name, fitFn, D, xform)
%IRUN Fit one variant, evaluate on both held-out sets, return a table row.
if nargin < 4, xform = @(X, ds) X; end
fprintf('  %-15s ', name); t0 = tic;
scoreFn = fitFn();

sv   = scoreFn(xform(D.val.X, D.val.ds));
mI   = D.val.ds == "IDRiD";                 % calibrate in the target domain
thr  = iTuneThr(D.val.y(mI), sv(mI), 0.90);

st   = scoreFn(xform(D.test.X,       D.test.ds));
sa   = scoreFn(xform(D.test_aptos.X, D.test_aptos.ds));

row = {name, iAuc(D.test.y, st), iAuc(D.test_aptos.y, sa), ...
       mean(st(D.test.y) >= thr), mean(st(~D.test.y) < thr), toc(t0)};
fprintf('AUC IDRiD %.3f   APTOS %.3f   (%.0f s)\n', row{2}, row{3}, row{6});
end

% ---------- variants --------------------------------------------------
function fn = iLinear(X, y)
m  = fitclinear(X, y, 'Learner', 'logistic', 'Solver', 'lbfgs');
fn = @(Xn) iPosScore(m, Xn);
end

function fn = iSvmRbf(X, y)
m  = fitcsvm(X, y, 'KernelFunction', 'rbf', 'KernelScale', 'auto', ...
                   'Standardize', true, 'BoxConstraint', 1);
fn = @(Xn) iPosScore(m, Xn);
end

function Xn = iL2(X)
%IL2 Unit-length rows.
%   An RBF kernel measures distance between feature vectors, so a channel
%   with a large magnitude dominates the distance regardless of how much it
%   actually says about the label. CNN embeddings have wildly uneven
%   magnitudes; projecting every image onto the unit sphere makes the kernel
%   compare directions instead. Standard practice with deep features.
n  = vecnorm(X, 2, 2);
Xn = X ./ max(n, eps);
end

function fn = iSvmTuned(X, y)
%ISVMTUNED Bayesian search over BoxConstraint and KernelScale.
%   Everything so far used KernelScale='auto' and BoxConstraint=1 -- the
%   defaults, never examined. The search cross-validates inside the training
%   set only, so no test information leaks.
m = fitcsvm(X, y, 'KernelFunction', 'rbf', 'Standardize', true, ...
    'OptimizeHyperparameters', {'BoxConstraint', 'KernelScale'}, ...
    'HyperparameterOptimizationOptions', struct( ...
        'ShowPlots', false, 'Verbose', 0, 'MaxObjectiveEvaluations', 20, ...
        'Kfold', 5, 'UseParallel', true, ...
        'AcquisitionFunctionName', 'expected-improvement-plus'));
fn = @(Xn) iPosScore(m, Xn);
end

function fn = iEnsemble(X, y, ds)
%IENSEMBLE Average three heads with different inductive biases.
%   svm-rbf, boosted trees and an IDRiD-only SVM disagree in different
%   places, so averaging them cancels some of each one's idiosyncratic
%   errors. Scores live on different scales, so each is standardised using
%   statistics fitted on the TRAINING scores -- which keeps the ensemble
%   deployable on a single image, unlike rank averaging.
mI = string(ds(:)) == "IDRiD";
m1 = fitcsvm(X, y, 'KernelFunction','rbf', 'KernelScale','auto', ...
                   'Standardize',true, 'BoxConstraint',1);
m2 = fitcensemble(X, y, 'Method','LogitBoost', 'NumLearningCycles',200, ...
        'Learners', templateTree('MaxNumSplits',24), 'LearnRate',0.1);
m3 = fitcsvm(X(mI,:), y(mI), 'KernelFunction','rbf', 'KernelScale','auto', ...
                   'Standardize',true, 'BoxConstraint',1);
ms = {m1, m2, m3};
mu = zeros(1,3); sd = ones(1,3);
for i = 1:3
    t = iPosScore(ms{i}, X);
    mu(i) = mean(t); sd(i) = max(std(t), eps);
end
fn = @(Xn) iEnsScore(ms, mu, sd, Xn);
end

function s = iEnsScore(ms, mu, sd, X)
S = zeros(size(X,1), numel(ms));
for i = 1:numel(ms)
    S(:,i) = (iPosScore(ms{i}, X) - mu(i)) / sd(i);
end
s = mean(S, 2);
end

function fn = iSvmWeighted(X, y, ds)
% Nonlinear boundary with domain-balanced sample weights: each dataset gets
% equal total influence while every image is kept.
ds = string(ds(:));
u  = unique(ds);
w  = ones(numel(y), 1);
for i = 1:numel(u)
    m = ds == u(i);
    w(m) = numel(ds) / (numel(u) * sum(m));
end
m  = fitcsvm(X, y, 'KernelFunction', 'rbf', 'KernelScale', 'auto', ...
                   'Standardize', true, 'BoxConstraint', 1, 'Weights', w);
fn = @(Xn) iPosScore(m, Xn);
end

function fn = iTrees(X, y)
m  = fitcensemble(X, y, 'Method', 'LogitBoost', 'NumLearningCycles', 200, ...
        'Learners', templateTree('MaxNumSplits', 24), 'LearnRate', 0.1);
fn = @(Xn) iPosScore(m, Xn);
end

function fn = iMlp(X, y)
if isempty(which('fitcnet'))
    error('tier1Experiments:noFitcnet', ...
        'fitcnet is unavailable in this MATLAB; drop the mlp variant.');
end
m  = fitcnet(X, y, 'LayerSizes', [256 64], 'Activations', 'relu', ...
        'Standardize', true, 'Lambda', 1e-4, 'IterationLimit', 400, 'Verbose', 0);
fn = @(Xn) iPosScore(m, Xn);
end

function fn = iOrdinal(X, g)
% Regression on the 0-4 severity scale. The score IS the predicted grade, so
% it respects the ordering that a binary target discards. Referral would
% normally threshold at 1.5; for comparability the threshold is tuned like
% every other variant, and AUC is unaffected either way.
m  = fitrlinear(X, g, 'Learner', 'leastsquares', 'Solver', 'lbfgs');
fn = @(Xn) predict(m, Xn);
end

function fn = iCoralModel(D, A, mu, muT)
mA = D.train.ds == "APTOS";
X  = D.train.X;
X(mA,:) = iApplyCoral(X(mA,:), repmat("APTOS", sum(mA), 1), A, mu, muT);
m  = fitclinear(X, D.train.y, 'Learner', 'logistic', 'Solver', 'lbfgs');
fn = @(Xn) iPosScore(m, Xn);
end

% ---------- CORAL -----------------------------------------------------
function [A, muS, muT] = iFitCoral(Xs, Xt, lambda)
%IFITCORAL Closed-form covariance alignment, source -> target.
%   Whiten the source, then recolour it with the target's covariance. No
%   labels involved, no CNN touched; it is a single 512x512 matrix.
d   = size(Xs, 2);
muS = mean(Xs, 1);  muT = mean(Xt, 1);
Cs  = cov(Xs) + lambda * eye(d);
Ct  = cov(Xt) + lambda * eye(d);
A   = iMatPow(Cs, -0.5) * iMatPow(Ct, 0.5);
end

function Xo = iApplyCoral(X, ds, A, muS, muT)
Xo = X;
m  = string(ds) == "APTOS";
if any(m)
    Xo(m,:) = (X(m,:) - muS) * A + muT;
end
end

function B = iMatPow(M, p)
M = (M + M') / 2;                       % force symmetry before eig
[V, Dg] = eig(M);
dv = max(diag(Dg), eps);
B  = V * diag(dv .^ p) * V';
B  = real((B + B') / 2);
end

% ---------- plumbing --------------------------------------------------
function s = iPosScore(m, X)
[~, sc] = predict(m, X);
s = sc(:, end);                          % last column = positive class
end

function a = iAuc(y, s)
[~, ~, ~, a] = perfcurve(y, s, true);
end

function thr = iTuneThr(y, s, targetSens)
cand = unique(s(:)); cand = [min(cand)-1; cand; max(cand)+1];
thr = median(cand); best = -Inf; hit = false;
for t = cand'
    se = mean(s(y) >= t); sp = mean(s(~y) < t);
    if se >= targetSens && sp > best, best = sp; thr = t; hit = true; end
end
if ~hit                                   % target unreachable -> Youden's J
    J = arrayfun(@(t) mean(s(y) >= t) + mean(s(~y) < t) - 1, cand);
    [~, i] = max(J); thr = cand(i);
end
end

function F = iLoadCache(splitName, opt, S)
sz  = opt.InputSize;
enh = lower(char(opt.Enhance));
tag = ''; if ~strcmp(enh, 'none'), tag = ['_' enh]; end
f = fullfile(fileparts(mfilename('fullpath')), 'cache', ...
    sprintf('feat_%s_%dx%d%s_%s.mat', lower(char(opt.Backbone)), sz(1), sz(2), tag, splitName));
if ~isfile(f)
    error('tier1Experiments:noCache', ...
        ['No cached features at\n  %s\nRun runBaseline with the same ' ...
         'Backbone/InputSize/Enhance first -- these experiments reuse its cache.'], f);
end
L = load(f, 'F', 'files');
if ~isequal(L.files, cellstr(S.(splitName).tbl.file))
    error('tier1Experiments:staleCache', ...
        'Cached "%s" features were computed for a different file list. Re-run runBaseline.', splitName);
end
F = L.F;
end

function iPcaPlot(X, ds)
%IPCAPLOT Do the two datasets occupy different regions of feature space?
%   If they separate cleanly, the representation itself is domain-sensitive
%   and alignment is worth pursuing. If they overlap, the gap is difficulty,
%   not shift, and no amount of alignment will help.
[~, Z] = pca(X, 'NumComponents', 2);
figure('Name', 'Feature space by dataset', 'Color', 'w');
ax = axes(); hold(ax, 'on'); box(ax, 'off');
ds = string(ds);
scatter(ax, Z(ds=="APTOS",1), Z(ds=="APTOS",2), 7, [0.878 0.482 0.094], ...
    'filled', 'MarkerFaceAlpha', .20, 'DisplayName', 'APTOS');
scatter(ax, Z(ds=="IDRiD",1), Z(ds=="IDRiD",2), 12, [0.184 0.435 0.816], ...
    'filled', 'MarkerFaceAlpha', .55, 'DisplayName', 'IDRiD');
grid(ax, 'on'); ax.GridAlpha = .18;
xlabel(ax, 'PC 1'); ylabel(ax, 'PC 2');
title(ax, 'Frozen features, first two principal components');
legend(ax, 'Location', 'best', 'Box', 'off');
end
