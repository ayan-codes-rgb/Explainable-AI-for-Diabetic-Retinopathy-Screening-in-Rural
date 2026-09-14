function T = tier1CV(roots, varargin)
%TIER1CV Classifier selection on cross-validated scores -- the test sets are
%        never touched.
%
%   T = TIER1CV(roots)
%   T = TIER1CV(roots, 'InputSize',[448 448], 'Enhance','none')
%   T = TIER1CV(roots, 'Tuned',true)            % + the two Bayesian searches
%   T = TIER1CV(roots, 'FinalTest','svm-rbf')   % spend the one-time reveal
%
%   WHY THIS FILE EXISTS
%   tier1Experiments ranks variants by their AUC on `test` and `test_aptos`.
%   No test image ever entered a fit, so the weights are clean -- but the
%   CHOICE of variant was made by reading those columns, and that choice has
%   now been made roughly twenty times across the project. Picking the best
%   of many near-equal options by their score on a held-out set does not
%   return the best option; it returns the luckiest one, and reports its
%   inflated score as if it were unbiased. The inflation is bounded by the
%   spread of the options considered: ~0.04 on IDRiD test (13 variants
%   spanning 0.861-0.899) and ~0.012 on APTOS test.
%
%   The fix is not a bigger test set. It is to select on data that is
%   allowed to be looked at repeatedly.
%
%   WHAT IT DOES INSTEAD
%   Pools train+val (3,800 images), makes ONE stratified k-fold partition,
%   and scores every variant out-of-fold: each image is scored by a model
%   fitted on folds that excluded it. Every variant sees the identical
%   partition, so comparisons between them are paired and the fold-to-fold
%   noise largely cancels.
%
%   That yields a selection signal with n=3,800 (standard error ~0.008) and,
%   restricted to the IDRiD rows, n=413 (~0.020). Both are tighter than the
%   103-image IDRiD test set (~0.036) that selection has been running on.
%   The test sets stay sealed for a single final measurement.
%
%   Stratification is on dataset x class jointly, so each fold holds a
%   representative share of the 413 IDRiD rows rather than whatever a
%   class-only split happens to deal out.
%
%   READING THE OUTPUT
%   dIDRiD is each variant's OOF AUC on IDRiD rows minus the linear
%   baseline's, with a 95% paired bootstrap interval. An interval that
%   straddles zero means the variant is not distinguishable from plain
%   logistic regression, however large the point estimate looks. This is the
%   column that answers "was the RBF kernel a real win or a lucky draw".
%
%   sens/spec are measured at a threshold tuned on the OOF IDRiD scores to
%   hit TargetSens. The scores are out-of-sample but the threshold is fitted
%   on them, so treat these two as indicative and quote AUC for comparisons.
%
%   COST  ~10 min for the 11 default variants (5 folds each, cached
%   features, CPU). 'Tuned',true adds the two Bayesian searches and takes it
%   past 35 min -- each is a 5-fold search nested inside every outer fold,
%   which is the correct way to evaluate a tuned model and priced
%   accordingly. Both already found nothing; off by default.
%
%   Options
%     'Backbone'   'resnet18' (default) -- must match a cached run
%     'InputSize'  [448 448]
%     'Enhance'    'none' | 'bengraham' | 'clahe'
%     'KFold'      5
%     'Seed'       20260909  (same seed as the split, for reproducibility)
%     'Tuned'      false -- include svm-tuned and svm-l2-tuned
%     'NBoot'      2000 bootstrap resamples
%     'TargetSens' 0.90
%     'FinalTest'  '' -- name one variant to refit on all of train+val and
%                  measure once on test + test_aptos. Leave empty while
%                  still choosing. Every use spends a peek.
%
%   See also TIER1EXPERIMENTS, RUNBASELINE, ROCREPORT.

p = inputParser;
p.FunctionName = 'tier1CV';
p.addParameter('Backbone',   'resnet18');
p.addParameter('InputSize',  [448 448]);
p.addParameter('Enhance',    'none');
p.addParameter('KFold',      5);
p.addParameter('Seed',       20260909);
p.addParameter('Tuned',      false);
p.addParameter('NBoot',      2000);
p.addParameter('TargetSens', 0.90);
p.addParameter('FinalTest',  '');
p.parse(varargin{:});
opt = p.Results;

reveal = ~isempty(char(opt.FinalTest));

fprintf('\n=== tier1CV: selection on out-of-fold scores (%s @ %dx%d, enhance=%s) ===\n', ...
    opt.Backbone, opt.InputSize(1), opt.InputSize(2), lower(char(opt.Enhance)));

% ---- data ------------------------------------------------------------
S = buildDatastores(roots, 'InputSize', opt.InputSize, 'Task', 'referable', ...
                           'Augment', false, 'Enhance', opt.Enhance);

need = ["train" "val"];
if reveal, need = [need "test" "test_aptos"]; end

D = struct();
for nm = need
    k = char(nm);
    D.(k).X  = iLoadCache(k, opt, S);
    D.(k).y  = S.(k).labels == "referable";
    D.(k).ds = string(S.(k).tbl.dataset);
    D.(k).g  = double(S.(k).tbl.grade);
end

X  = [D.train.X;      D.val.X];
y  = [D.train.y(:);   D.val.y(:)];
ds = [D.train.ds(:);  D.val.ds(:)];
g  = [D.train.g(:);   D.val.g(:)];

n  = numel(y);
mI = ds == "IDRiD";
mA = ds == "APTOS";
fprintf(['pool = train+val: %d images (%d IDRiD, %d APTOS), %d-dim features\n' ...
         'test sets: %s\n'], n, sum(mI), sum(mA), size(X,2), ...
         iTernary(reveal, 'LOADED for the final reveal', 'not loaded'));

% ---- one shared partition, stratified on dataset x class -------------
rng(opt.Seed);
grp = categorical(ds + "|" + string(double(y)));
cvp = cvpartition(grp, 'KFold', opt.KFold);
fprintf('partition: %d folds, stratified on dataset x class, seed %d\n\n', ...
        opt.KFold, opt.Seed);

% ---- variant registry ------------------------------------------------
% Each entry is a fit function of the FOLD'S OWN training rows, returning a
% score function of (X, dataset). Subsetting (idrid-only), transforms (L2,
% CORAL) and weighting all happen inside the fold, so nothing crosses from
% the held-out part of the pool into the fit.
V = {};
V(end+1,:) = {'linear',        @(Xf,yf,df,gf) iFitLinear(Xf, yf)};                          %#ok<*AGROW>
V(end+1,:) = {'svm-rbf',       @(Xf,yf,df,gf) iFitSvm(Xf, yf, [])};
V(end+1,:) = {'boosted-trees', @(Xf,yf,df,gf) iFitTrees(Xf, yf)};
V(end+1,:) = {'mlp',           @(Xf,yf,df,gf) iFitMlp(Xf, yf)};
V(end+1,:) = {'ordinal',       @(Xf,yf,df,gf) iFitOrdinal(Xf, gf)};
V(end+1,:) = {'idrid-only',    @(Xf,yf,df,gf) iFitLinear(Xf(df=="IDRiD",:), yf(df=="IDRiD"))};
V(end+1,:) = {'svm-idrid',     @(Xf,yf,df,gf) iFitSvm(Xf(df=="IDRiD",:), yf(df=="IDRiD"), [])};
V(end+1,:) = {'svm-weighted',  @(Xf,yf,df,gf) iFitSvm(Xf, yf, iDomainW(df))};
V(end+1,:) = {'svm-l2norm',    @(Xf,yf,df,gf) iWrapL2(iFitSvm(iL2(Xf), yf, []))};
V(end+1,:) = {'coral',         @(Xf,yf,df,gf) iFitCoral(Xf, yf, df)};
V(end+1,:) = {'ensemble',      @(Xf,yf,df,gf) iFitEnsemble(Xf, yf, df)};
if opt.Tuned
    V(end+1,:) = {'svm-tuned',    @(Xf,yf,df,gf) iFitSvmTuned(Xf, yf)};
    V(end+1,:) = {'svm-l2-tuned', @(Xf,yf,df,gf) iWrapL2(iFitSvmTuned(iL2(Xf), yf))};
end
nV = size(V, 1);

% ---- cross-validate every variant on the same folds ------------------
OOF  = nan(n, nV);
secs = zeros(nV, 1);
fprintf('%-15s %-8s %s\n', 'variant', 'folds', 'OOF AUC (all / IDRiD / APTOS)');
fprintf('%s\n', repmat('-', 1, 62));
for v = 1:nV
    fprintf('  %-13s ', V{v,1}); t0 = tic;
    for f = 1:opt.KFold
        itr = training(cvp, f);
        ite = test(cvp, f);
        sf  = V{v,2}(X(itr,:), y(itr), ds(itr), g(itr));
        OOF(ite, v) = sf(X(ite,:), ds(ite));
        fprintf('.');
    end
    secs(v) = toc(t0);
    fprintf(' %6.3f  %6.3f  %6.3f   (%.0f s)\n', ...
        iAuc(y, OOF(:,v)), iAuc(y(mI), OOF(mI,v)), iAuc(y(mA), OOF(mA,v)), secs(v));
end

% ---- paired bootstrap against the linear baseline --------------------
iL = find(strcmp(V(:,1), 'linear'), 1);
if isempty(iL), iL = 1; end
[dI, loI, hiI] = iPairedBoot(y(mI), OOF(mI,:), iL, opt.NBoot, opt.Seed + 1);
[dP, loP, hiP] = iPairedBoot(y,     OOF,       iL, opt.NBoot, opt.Seed + 2);

% ---- assemble --------------------------------------------------------
rows = cell(nV, 10);
for v = 1:nV
    sI = OOF(mI, v); yI = y(mI);
    thr = iTuneThr(yI, sI, opt.TargetSens);
    rows(v,:) = {V{v,1}, iAuc(y, OOF(:,v)), iAuc(yI, sI), iAuc(y(mA), OOF(mA,v)), ...
                 dI(v), loI(v), hiI(v), mean(sI(yI) >= thr), mean(sI(~yI) < thr), secs(v)};
end
T = cell2table(rows, 'VariableNames', {'variant','aucOOF','aucIDRiD','aucAPTOS', ...
                'dIDRiD','ciLo','ciHi','sensIDRiD','specIDRiD','seconds'});

% ---- report ----------------------------------------------------------
fprintf('\n%-15s%9s%9s%9s%10s%18s%8s\n', 'variant', 'AUC all', 'IDRiD', 'APTOS', ...
        'd vs lin', '95% CI', 'secs');
fprintf('%s\n', repmat('-', 1, 78));
for v = 1:nV
    fprintf('%-15s%9.3f%9.3f%9.3f%+10.3f   [%+.3f, %+.3f]%8.0f\n', ...
        T.variant{v}, T.aucOOF(v), T.aucIDRiD(v), T.aucAPTOS(v), ...
        T.dIDRiD(v), T.ciLo(v), T.ciHi(v), T.seconds(v));
end
fprintf('%s\n', repmat('-', 1, 78));

sig = find(T.ciLo > 0 | T.ciHi < 0);
sig = sig(sig ~= iL);
fprintf('\nn = %d pooled (SE ~%.3f) | %d IDRiD (SE ~%.3f) | %d APTOS (SE ~%.3f)\n', ...
    n, iSeAuc(y), sum(mI), iSeAuc(y(mI)), sum(mA), iSeAuc(y(mA)));
if isempty(sig)
    fprintf(['\nVERDICT: no variant is distinguishable from ''linear'' on IDRiD\n' ...
             'out-of-fold scores -- every 95%% interval contains zero. The head is\n' ...
             'exhausted, and the 0.849->0.894 jump reported on IDRiD test does not\n' ...
             'survive selection on data that was allowed to be looked at.\n' ...
             'Ship the simplest variant.\n']);
else
    fprintf('\nVERDICT: distinguishable from ''linear'' on IDRiD OOF (95%% CI excludes 0):\n');
    for k = sig(:)'
        fprintf('  %-15s %+.3f  [%+.3f, %+.3f]\n', T.variant{k}, T.dIDRiD(k), T.ciLo(k), T.ciHi(k));
    end
    fprintf('These survived a comparison that could not be gamed by repeated peeking.\n');
end
fprintf('\npooled-domain deltas vs linear (sanity check, n=%d):\n', n);
for v = 1:nV
    if v == iL, continue; end
    fprintf('  %-15s %+.3f  [%+.3f, %+.3f]%s\n', T.variant{v}, dP(v), loP(v), hiP(v), ...
        iTernary(loP(v) > 0 || hiP(v) < 0, '   *', ''));
end

[~, bestI] = max(T.aucIDRiD);
fprintf('\nhighest IDRiD OOF AUC: %s (%.3f).\n', T.variant{bestI}, T.aucIDRiD(bestI));
fprintf(['A point estimate is still a point estimate -- if its interval overlaps\n' ...
         'linear''s, prefer the cheaper and simpler head.\n']);

% ---- the one-time reveal --------------------------------------------
if reveal
    name = char(opt.FinalTest);
    iv = find(strcmp(V(:,1), name), 1);
    if isempty(iv)
        error('tier1CV:unknownVariant', 'FinalTest ''%s'' is not one of: %s', ...
            name, strjoin(V(:,1)', ', '));
    end
    fprintf('\n%s\n', repmat('=', 1, 78));
    fprintf('FINAL TEST REVEAL -- variant ''%s''\n', name);
    fprintf(['This refits on all %d pooled images and measures test + test_aptos.\n' ...
             'Every run of this block is a peek. Do not use it to compare variants;\n' ...
             'that is the mistake this file exists to undo.\n'], n);
    fprintf('%s\n', repmat('=', 1, 78));

    sf  = V{iv,2}(X, y, ds, g);
    thr = iTuneThr(y(mI), OOF(mI,iv), opt.TargetSens);   % threshold from OOF, not from test

    st = sf(D.test.X,       D.test.ds);
    sa = sf(D.test_aptos.X, D.test_aptos.ds);
    yt = D.test.y(:);  ya = D.test_aptos.y(:);

    fprintf('threshold %.4f, taken from out-of-fold IDRiD scores (test not consulted)\n\n', thr);
    fprintf('%-14s%7s%9s%9s%9s\n', 'eval set', 'n', 'sens', 'spec', 'AUC');
    fprintf('%s\n', repmat('-', 1, 50));
    fprintf('%-14s%7d%8.1f%%%8.1f%%%9.3f\n', 'IDRiD test', numel(yt), ...
        100*mean(st(yt) >= thr), 100*mean(st(~yt) < thr), iAuc(yt, st));
    fprintf('%-14s%7d%8.1f%%%8.1f%%%9.3f\n', 'APTOS test', numel(ya), ...
        100*mean(sa(ya) >= thr), 100*mean(sa(~ya) < thr), iAuc(ya, sa));
    fprintf('%s\n', repmat('-', 1, 50));
    fprintf(['\nIDRiD test SE ~%.3f, APTOS test SE ~%.3f. These are the numbers to\n' ...
             'report. The OOF table above is the number to have chosen by.\n'], ...
             iSeAuc(yt), iSeAuc(ya));
end
end

% =====================================================================
% variants -- each returns a score function of (X, dataset)
% =====================================================================
function fn = iFitLinear(X, y)
m  = fitclinear(X, y, 'Learner', 'logistic', 'Solver', 'lbfgs');
fn = @(Xn, dsn) iPosScore(m, Xn);
end

function fn = iFitSvm(X, y, w)
args = {'KernelFunction','rbf', 'KernelScale','auto', 'Standardize',true, 'BoxConstraint',1};
if ~isempty(w), args = [args, {'Weights', w}]; end
m  = fitcsvm(X, y, args{:});
fn = @(Xn, dsn) iPosScore(m, Xn);
end

function fn = iFitTrees(X, y)
m  = fitcensemble(X, y, 'Method','LogitBoost', 'NumLearningCycles',200, ...
        'Learners', templateTree('MaxNumSplits',24), 'LearnRate',0.1);
fn = @(Xn, dsn) iPosScore(m, Xn);
end

function fn = iFitMlp(X, y)
if isempty(which('fitcnet'))
    error('tier1CV:noFitcnet', 'fitcnet is unavailable in this MATLAB; drop the mlp variant.');
end
m  = fitcnet(X, y, 'LayerSizes',[256 64], 'Activations','relu', 'Standardize',true, ...
        'Lambda',1e-4, 'IterationLimit',400, 'Verbose',0);
fn = @(Xn, dsn) iPosScore(m, Xn);
end

function fn = iFitOrdinal(X, g)
% Regression on the 0-4 severity scale; the predicted grade IS the score, so
% the ordering a binary target discards is kept. AUC is rank-based, so the
% different score scale costs nothing in the comparison.
m  = fitrlinear(X, g, 'Learner', 'leastsquares', 'Solver', 'lbfgs');
fn = @(Xn, dsn) predict(m, Xn);
end

function fn = iFitSvmTuned(X, y)
% Bayesian search over BoxConstraint and KernelScale, cross-validated inside
% this fold's training rows only. Nested inside the outer CV, which is what
% makes the reported number an honest estimate of "tune, then deploy".
m = fitcsvm(X, y, 'KernelFunction','rbf', 'Standardize',true, ...
    'OptimizeHyperparameters', {'BoxConstraint','KernelScale'}, ...
    'HyperparameterOptimizationOptions', struct( ...
        'ShowPlots',false, 'Verbose',0, 'MaxObjectiveEvaluations',20, ...
        'Kfold',5, 'UseParallel',true, ...
        'AcquisitionFunctionName','expected-improvement-plus'));
fn = @(Xn, dsn) iPosScore(m, Xn);
end

function fn = iFitEnsemble(X, y, ds)
mI = ds == "IDRiD";
m1 = fitcsvm(X, y, 'KernelFunction','rbf', 'KernelScale','auto', ...
                   'Standardize',true, 'BoxConstraint',1);
m2 = fitcensemble(X, y, 'Method','LogitBoost', 'NumLearningCycles',200, ...
        'Learners', templateTree('MaxNumSplits',24), 'LearnRate',0.1);
ms = {m1, m2};
if sum(mI) >= 20 && numel(unique(y(mI))) == 2
    ms{end+1} = fitcsvm(X(mI,:), y(mI), 'KernelFunction','rbf', 'KernelScale','auto', ...
                        'Standardize',true, 'BoxConstraint',1);
end
k  = numel(ms);
mu = zeros(1,k); sd = ones(1,k);
for i = 1:k
    t = iPosScore(ms{i}, X);
    mu(i) = mean(t); sd(i) = max(std(t), eps);
end
fn = @(Xn, dsn) iEnsScore(ms, mu, sd, Xn);
end

function s = iEnsScore(ms, mu, sd, X)
Z = zeros(size(X,1), numel(ms));
for i = 1:numel(ms)
    Z(:,i) = (iPosScore(ms{i}, X) - mu(i)) / sd(i);
end
s = mean(Z, 2);
end

function fn = iFitCoral(X, y, ds)
% Covariance alignment fitted on THIS fold's training rows, then applied to
% APTOS rows wherever they are scored.
mA = ds == "APTOS";  mI = ds == "IDRiD";
if ~any(mA) || sum(mI) < 2
    fn = iFitLinear(X, y); return
end
[A, muS, muT] = iCoralFit(X(mA,:), X(mI,:), 1.0);
Xa = X;  Xa(mA,:) = iCoralApply(X(mA,:), A, muS, muT);
m  = fitclinear(Xa, y, 'Learner','logistic', 'Solver','lbfgs');
fn = @(Xn, dsn) iPosScore(m, iCoralRows(Xn, dsn, A, muS, muT));
end

function Xo = iCoralRows(X, ds, A, muS, muT)
Xo = X;
m  = string(ds) == "APTOS";
if any(m), Xo(m,:) = iCoralApply(X(m,:), A, muS, muT); end
end

function Xo = iCoralApply(X, A, muS, muT)
Xo = (X - muS) * A + muT;
end

function [A, muS, muT] = iCoralFit(Xs, Xt, lambda)
d   = size(Xs, 2);
muS = mean(Xs, 1);  muT = mean(Xt, 1);
Cs  = cov(Xs) + lambda * eye(d);
Ct  = cov(Xt) + lambda * eye(d);
A   = iMatPow(Cs, -0.5) * iMatPow(Ct, 0.5);
end

function B = iMatPow(M, pw)
M = (M + M') / 2;
[Vv, Dg] = eig(M);
dv = max(diag(Dg), eps);
B  = Vv * diag(dv .^ pw) * Vv';
B  = real((B + B') / 2);
end

function Xn = iL2(X)
nrm = vecnorm(X, 2, 2);
Xn  = X ./ max(nrm, eps);
end

function fn = iWrapL2(inner)
fn = @(Xn, dsn) inner(iL2(Xn), dsn);
end

function w = iDomainW(ds)
ds = string(ds(:));
u  = unique(ds);
w  = ones(numel(ds), 1);
for i = 1:numel(u)
    m = ds == u(i);
    w(m) = numel(ds) / (numel(u) * sum(m));
end
end

% =====================================================================
% plumbing
% =====================================================================
function s = iPosScore(m, X)
[~, sc] = predict(m, X);
col = [];
if isprop(m, 'ClassNames') || isfield(m, 'ClassNames')
    cn = m.ClassNames;
    if islogical(cn),      col = find(cn, 1);
    elseif isnumeric(cn),  col = find(cn == 1, 1);
    elseif iscategorical(cn) || isstring(cn) || iscellstr(cn)
        col = find(string(cn) == "referable" | string(cn) == "true" | string(cn) == "1", 1);
    end
end
if isempty(col), col = size(sc, 2); end
s = sc(:, col);
end

function a = iAuc(y, s)
%IAUC Mann-Whitney form of the AUC: fast enough to bootstrap, and it handles
%     ties by mid-rank the same way perfcurve does.
y  = logical(y(:));  s = s(:);
ok = ~isnan(s);
y  = y(ok);  s = s(ok);
np = sum(y);  nn = sum(~y);
if np == 0 || nn == 0, a = NaN; return; end
r = tiedrank(s);
a = (sum(r(y)) - np*(np+1)/2) / (np*nn);
end

function se = iSeAuc(y)
%ISEAUC Hanley-McNeil standard error at AUC 0.9, the scale we operate at.
y  = logical(y(:));
np = sum(y);  nn = sum(~y);
if np == 0 || nn == 0, se = NaN; return; end
a  = 0.90;
q1 = a / (2 - a);
q2 = 2*a^2 / (1 + a);
se = sqrt((a*(1-a) + (np-1)*(q1 - a^2) + (nn-1)*(q2 - a^2)) / (np*nn));
end

function [d, lo, hi] = iPairedBoot(y, Sc, iRef, B, seed)
%IPAIREDBOOT Bootstrap the AUC difference of every column against a
%   reference column, resampling positives and negatives separately so no
%   draw degenerates to one class. Every variant is resampled on the SAME
%   indices, so the difference is paired and the shared fold noise cancels.
y   = logical(y(:));
nV  = size(Sc, 2);
d   = nan(nV,1);  lo = nan(nV,1);  hi = nan(nV,1);
aRef = iAuc(y, Sc(:,iRef));
for v = 1:nV
    d(v) = iAuc(y, Sc(:,v)) - aRef;
end
pos = find(y);  neg = find(~y);
if isempty(pos) || isempty(neg), return; end
rng(seed);
Db = zeros(B, nV);
for b = 1:B
    idx = [pos(randi(numel(pos), numel(pos), 1)); neg(randi(numel(neg), numel(neg), 1))];
    yb  = y(idx);
    ab  = zeros(1, nV);
    for v = 1:nV
        ab(v) = iAuc(yb, Sc(idx, v));
    end
    Db(b,:) = ab - ab(iRef);
end
lo = prctile(Db, 2.5)';
hi = prctile(Db, 97.5)';
end

function thr = iTuneThr(y, s, targetSens)
y = logical(y(:));  s = s(:);
ok = ~isnan(s);  y = y(ok);  s = s(ok);
cand = unique(s);
cand = [min(cand)-1; cand; max(cand)+1];
thr  = median(cand);  best = -Inf;  hit = false;
for t = cand'
    se = mean(s(y) >= t);  sp = mean(s(~y) < t);
    if se >= targetSens && sp > best, best = sp; thr = t; hit = true; end
end
if ~hit
    J = arrayfun(@(t) mean(s(y) >= t) + mean(s(~y) < t) - 1, cand);
    [~, i] = max(J);  thr = cand(i);
end
end

function F = iLoadCache(splitName, opt, S)
sz  = opt.InputSize;
enh = lower(char(opt.Enhance));
tag = '';  if ~strcmp(enh, 'none'), tag = ['_' enh]; end
f = fullfile(fileparts(mfilename('fullpath')), 'cache', ...
    sprintf('feat_%s_%dx%d%s_%s.mat', lower(char(opt.Backbone)), sz(1), sz(2), tag, splitName));
if ~isfile(f)
    error('tier1CV:noCache', ...
        ['No cached features at\n  %s\nRun runBaseline with the same ' ...
         'Backbone/InputSize/Enhance first -- this reuses its cache.'], f);
end
L = load(f, 'F', 'files');
if ~isequal(L.files, cellstr(S.(splitName).tbl.file))
    error('tier1CV:staleCache', ...
        'Cached "%s" features were computed for a different file list. Re-run runBaseline.', splitName);
end
F = L.F;
end

function out = iTernary(c, a, b)
if c, out = a; else, out = b; end
end
