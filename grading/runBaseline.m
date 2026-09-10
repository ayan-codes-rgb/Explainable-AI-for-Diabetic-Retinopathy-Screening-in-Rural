function R = runBaseline(roots, varargin)
%RUNBASELINE DR grading baseline: frozen backbone + linear classifier.
%
%   R = RUNBASELINE(roots)
%   R = RUNBASELINE(roots, 'Backbone','resnet50', 'Threshold','cv')
%
%   ROOTS -- where the images are:
%       roots.IDRiD = 'C:\dr-data\IDRiD\B. Disease Grading';
%       roots.APTOS = 'C:\dr-data\APTOS';          % contains train_images\
%   A plain string is treated as the IDRiD root (APTOS rows are dropped).
%
%   WHAT THIS DOES
%   Runs each image through a FROZEN pretrained network once, keeps the
%   activations as a feature vector, and trains a linear classifier on those
%   vectors. Forward passes only -- no backpropagation, no GPU needed. It is
%   also the ablation baseline a fine-tuned model has to beat.
%
%   FEATURES ARE CACHED in grading/cache/, keyed on backbone and input size
%   and validated against the exact file list, so changing the split
%   invalidates the cache rather than silently reusing stale features.
%   First run with APTOS is slow (~4,000 images); later runs take seconds.
%
%   THRESHOLD SELECTION -- 'Threshold'
%     'cv'  (default) Pool train+val, run stratified k-fold cross-validation,
%           and choose the threshold on the out-of-fold scores. Every image
%           contributes an honest prediction, so the threshold is estimated
%           from thousands of scores rather than a few dozen. The final model
%           is then refitted on all pooled data.
%     'val' Train on train only, choose the threshold on the held-out val
%           set. Simpler, but with a small val set the chosen threshold is
%           mostly noise -- on IDRiD alone it cost ~8 points of test
%           specificity versus what the same model could deliver.
%
%   Options
%     'Backbone'   'resnet18' (default) | 'resnet50' | 'mobilenetv2' | 'squeezenet'
%     'InputSize'  [224 224]
%     'Task'       'referable' (default, binary) | 'grade' (5-class)
%     'Weights'    'inverse-sqrt' (default) | 'inverse' | 'none'
%     'TargetSens' 0.90
%     'Threshold'  'cv' (default) | 'val'
%     'KFold'      5
%     'Refresh'    false -- true recomputes and re-caches features
%     'BatchSize'  32
%     'DomainWeight' 'balanced' (default) | 'none' | numeric IDRiD multiplier
%     'CalibrateOn'  'IDRiD' (default) | 'APTOS' | 'all'
%
%   DOMAIN SHIFT -- why the last two options exist
%   Training is ~3,100 APTOS images against ~350 IDRiD, so APTOS outvotes
%   IDRiD roughly 9:1 and the classifier optimises for APTOS cameras. The
%   test set is IDRiD. Measured effect of ignoring this: cross-validated AUC
%   0.967 on the (APTOS-dominated) pooled set, but 0.837 on IDRiD test, and
%   a threshold picked on pooled scores gave 79.7% sensitivity on IDRiD when
%   the same model could reach 90.6% at a different cutoff.
%
%   'DomainWeight','balanced' gives each dataset equal total influence while
%   keeping every image. 'CalibrateOn','IDRiD' picks the threshold using only
%   IDRiD out-of-fold scores -- the model's RANKING transfers across domains,
%   its score SCALE does not, so the cutoff must come from the target domain.
%
%   See also LOADSPLIT, BUILDDATASTORES, CLASSWEIGHTS, ROCREPORT.

p = inputParser;
p.FunctionName = 'runBaseline';
p.addParameter('Backbone',   'resnet18');
p.addParameter('InputSize',  [224 224]);
p.addParameter('Task',       'referable');
p.addParameter('Weights',    'inverse-sqrt');
p.addParameter('TargetSens', 0.90);
p.addParameter('Threshold',  'cv');
p.addParameter('KFold',      5);
p.addParameter('Refresh',    false);
p.addParameter('BatchSize',  32);
p.addParameter('DomainWeight', 'balanced');
p.addParameter('CalibrateOn',  'IDRiD');
p.addParameter('Enhance',      'none');
p.parse(varargin{:});
opt = p.Results;

mode = lower(char(opt.Threshold));
if ~ismember(mode, {'cv','val'})
    error('runBaseline:badThreshold', 'Threshold must be ''cv'' or ''val''.');
end

fprintf('\n=== DR grading baseline ===\n');

% ---- 1. data ---------------------------------------------------------
S = buildDatastores(roots, 'InputSize', opt.InputSize, ...
                           'Task', opt.Task, 'Augment', false, ...
                           'Enhance', opt.Enhance);

% ---- 2. features (cached) -------------------------------------------
cacheDir = fullfile(fileparts(mfilename('fullpath')), 'cache');
if ~isfolder(cacheDir), mkdir(cacheDir); end

net = [];  featLayer = '';
F = struct();  tAll = tic;
for nm = ["train" "val" "test" "test_aptos"]
    [F.(nm), hit] = iCachedFeatures(S, char(nm), opt, cacheDir);
    if ~hit
        if isempty(net)
            fprintf('loading backbone: %s\n', opt.Backbone);
            [net, featLayer] = iBackbone(opt.Backbone);
        end
        F.(nm) = iFeatures(net, S.(nm).ds, featLayer, char(nm), ...
                           numel(S.(nm).labels), opt.BatchSize);
        iSaveCache(F.(nm), S, char(nm), opt, cacheDir);
    end
end
fprintf('features: %d-dim, %.0f s total\n', size(F.train, 2), toc(tAll));

tmpl = templateLinear('Learner', 'logistic', 'Solver', 'lbfgs');

% ---- 3. fit + choose threshold --------------------------------------
switch mode
    case 'cv'
        Xp = [F.train; F.val];
        Yp = [S.train.labels; S.val.labels];
        classes = categories(Yp);
        posName = classes{end};
        posCol  = find(strcmp(classes, posName), 1);
        dsP = [S.train.tbl.dataset; S.val.tbl.dataset];
        [w, wTable] = classWeights(Yp, opt.Weights);
        fprintf('\nclass balance (train+val pooled):\n'); disp(wTable);
        [rwAll, dwTable] = iCombinedWeights(Yp, w, dsP, opt.DomainWeight);
        if ~isempty(dwTable)
            fprintf('domain balance:\n'); disp(dwTable);
        end

        k   = opt.KFold;
        cvp = cvpartition(Yp, 'KFold', k);        % stratified by class
        oof = nan(numel(Yp), 1);
        fprintf('cross-validating threshold on %d images:\n', numel(Yp));
        for i = 1:k
            itr = training(cvp, i); ite = test(cvp, i);
            m = fitcecoc(Xp(itr,:), Yp(itr), 'Learners', tmpl, ...
                         'ClassNames', classes, ...
                         'Weights', rwAll(itr), 'FitPosterior', true);
            [~, ~, ~, P] = predict(m, Xp(ite,:));
            oof(ite) = P(:, posCol);
            fprintf('\r  fold %d/%d', i, k);
        end
        fprintf('\n');

        yTune = Yp == posName;
        calib = iCalibMask(dsP, opt.CalibrateOn);
        [thr, sensTune, specTune, reached] = ...
            iTuneThreshold(yTune(calib), oof(calib), opt.TargetSens);
        tuneOn = sprintf('%d-fold CV, %d %s images', k, sum(calib), ...
                         iTernary(strcmpi(char(opt.CalibrateOn),'all'), 'pooled', char(opt.CalibrateOn)));

        mdl = fitcecoc(Xp, Yp, 'Learners', tmpl, 'ClassNames', classes, ...
                       'Weights', rwAll, 'FitPosterior', true);
        tuneScore = oof;
        tuneDs    = dsP;

    case 'val'
        Ytr = S.train.labels;
        classes = categories(Ytr);
        posName = classes{end};
        posCol  = find(strcmp(classes, posName), 1);
        [w, wTable] = classWeights(Ytr, opt.Weights);
        fprintf('\nclass balance (train):\n'); disp(wTable);
        [rwTr, dwTable] = iCombinedWeights(Ytr, w, S.train.tbl.dataset, opt.DomainWeight);
        if ~isempty(dwTable)
            fprintf('domain balance:\n'); disp(dwTable);
        end

        mdl = fitcecoc(F.train, Ytr, 'Learners', tmpl, 'ClassNames', classes, ...
                       'Weights', rwTr, 'FitPosterior', true);
        [~, ~, ~, Pv] = predict(mdl, F.val);
        yTune = S.val.labels == posName;
        calib = iCalibMask(S.val.tbl.dataset, opt.CalibrateOn);
        [thr, sensTune, specTune, reached] = ...
            iTuneThreshold(yTune(calib), Pv(calib,posCol), opt.TargetSens);
        tuneOn = sprintf('held-out val, %d %s images', sum(calib), ...
                         iTernary(strcmpi(char(opt.CalibrateOn),'all'), 'pooled', char(opt.CalibrateOn)));
        tuneScore = Pv(:, posCol);
        tuneDs    = S.val.tbl.dataset;
end

% ---- 4. evaluation on both held-out sets ----------------------------
% Two test sets on purpose. IDRiD-103 is the benchmark that matters -- Indian
% cameras, expert consensus grades -- but with 64 positives and 39 negatives
% its AUC standard error is ~0.036, so it cannot resolve the differences
% between model variants (six configurations all landed within 0.042 of each
% other, i.e. inside the noise). test_aptos has 275 images and roughly half
% that error, so it can actually tell whether a change helped. Report IDRiD;
% steer by test_aptos.
[~, ~, ~, Pt] = predict(mdl, F.test);
if size(Pt, 2) ~= numel(classes)
    error('runBaseline:posteriorShape', ...
        'Expected %d posterior columns, got %d.', numel(classes), size(Pt,2));
end
yTest = S.test.labels == posName;
sTest = Pt(:, posCol);
sensT = mean(sTest(yTest)  >= thr);
specT = mean(sTest(~yTest) <  thr);

n = numel(yTest);
fprintf('\n--- results (positive class: %s) ---\n', posName);
fprintf('threshold %.3f, chosen on %s%s\n', thr, tuneOn, ...
    iTernary(reached, '', '   [target NOT reached]'));
fprintf('TUNING  sensitivity %5.1f%%   specificity %5.1f%%   (n=%d)\n', ...
    100*sensTune, 100*specTune, numel(yTune));
fprintf('TEST    sensitivity %5.1f%%   specificity %5.1f%%   (n=%d, IDRiD only)\n', ...
    100*sensT, 100*specT, n);


% second held-out set
[~, ~, ~, Pa] = predict(mdl, F.test_aptos);
yAp = S.test_aptos.labels == posName;
sAp = Pa(:, posCol);
fprintf('APTOS   sensitivity %5.1f%%   specificity %5.1f%%   (n=%d, steering set)\n', ...
    100*mean(sAp(yAp) >= thr), 100*mean(sAp(~yAp) < thr), numel(yAp));
fprintf('targets: sensitivity >90%%, specificity >85%%\n');
fprintf('NOTE: IDRiD test n=%d -> AUC std-error ~0.036, too coarse to compare\n', n);
fprintf('      variants. test_aptos n=%d -> ~0.022. Steer by that one.\n\n', numel(yAp));

R = struct('model', mdl, 'threshold', thr, 'targetReached', reached, ...
           'thresholdMode', mode, 'tunedOn', tuneOn, ...
           'backbone', opt.Backbone, 'task', opt.Task, 'enhance', lower(char(opt.Enhance)), ...
           'classWeights', wTable, ...
           'tuning', struct('sensitivity', sensTune, 'specificity', specTune), ...
           'test',   struct('sensitivity', sensT,    'specificity', specT), ...
           'features', struct('dim', size(F.train,2), 'layer', featLayer), ...
           'scores', struct('positiveClass', posName, ...
                'val',  struct('score', tuneScore, 'truth', yTune, 'dataset', tuneDs), ...
                'test', struct('score', sTest, 'truth', yTest, 'dataset', S.test.tbl.dataset), ...
                'test_aptos', struct('score', sAp, 'truth', yAp, 'dataset', S.test_aptos.tbl.dataset)));
end

% =====================================================================
function f = iCacheFile(splitName, opt, cacheDir)
sz  = opt.InputSize;
enh = lower(char(opt.Enhance));
if strcmp(enh, 'none')
    tag = '';
else
    tag = ['_' enh];
end
f = fullfile(cacheDir, sprintf('feat_%s_%dx%d%s_%s.mat', ...
        lower(char(opt.Backbone)), sz(1), sz(2), tag, splitName));
end

function [F, hit] = iCachedFeatures(S, splitName, opt, cacheDir)
F = []; hit = false;
if opt.Refresh, return; end
f = iCacheFile(splitName, opt, cacheDir);
if ~isfile(f), return; end
try
    L = load(f, 'F', 'files');
    if isequal(L.files, cellstr(S.(splitName).tbl.file))
        F = L.F; hit = true;
        fprintf('  %-5s %5d images  [cached]\n', splitName, size(F,1));
    end
catch
end
end

function iSaveCache(F, S, splitName, opt, cacheDir) %#ok<INUSD>
files = cellstr(S.(splitName).tbl.file); %#ok<NASGU>
try
    save(iCacheFile(splitName, opt, cacheDir), 'F', 'files', '-v7.3');
catch ME
    warning('runBaseline:cacheWriteFailed', ...
        'Could not write feature cache (%s). Continuing.', ME.message);
end
end

% =====================================================================
function [net, featLayer] = iBackbone(name)
name = lower(char(name));
switch name
    case 'resnet18',    featLayer = 'pool5';
    case 'resnet50',    featLayer = 'avg_pool';
    case 'mobilenetv2', featLayer = 'global_average_pooling2d_1';
    case 'squeezenet',  featLayer = 'pool10';
    otherwise
        error('runBaseline:badBackbone', ...
            'Backbone must be resnet18, resnet50, mobilenetv2 or squeezenet.');
end

if ~isempty(which('imagePretrainedNetwork'))
    net = imagePretrainedNetwork(name);
elseif ~isempty(which(name))
    net = feval(name);
else
    error('runBaseline:missingAddOn', ...
        ['Pretrained network "%s" is not installed. Home > Add-Ons > ' ...
         'Get Add-Ons, search "Deep Learning Toolbox Model for %s".'], name, name);
end

if ~any(strcmp({net.Layers.Name}, featLayer))
    names = {net.Layers.Name};
    pools = names(contains(lower(names), 'pool'));
    error('runBaseline:badLayer', ...
        ['Layer "%s" not found in %s. Pooling layers available: %s\n' ...
         'Pick the global pooling layer just before the classifier.'], ...
        featLayer, name, strjoin(pools, ', '));
end
end

% =====================================================================
function F = iFeatures(net, ds, layerName, label, nTotal, batchSize)
try, ds.MiniBatchSize = batchSize; catch, end %#ok<CTCH>
t0 = tic; done = 0; chunks = {};
reset(ds);
while hasdata(ds)
    b = read(ds);
    if istable(b), X = cat(4, b{:,1}{:}); else, X = b; end
    if isa(net, 'dlnetwork')
        Y = predict(net, dlarray(single(X), 'SSCB'), 'Outputs', layerName);
        Y = extractdata(gather(Y));
    else
        Y = activations(net, X, layerName);
    end
    chunks{end+1} = reshape(Y, [], size(Y, ndims(Y)))'; %#ok<AGROW>
    done = done + size(X, 4);
    el = toc(t0); rate = done / max(el, eps);
    fprintf('\r  %-5s %5d/%5d   %5.0f s elapsed, ~%5.0f s left   ', ...
        label, done, nTotal, el, max(nTotal - done, 0) / max(rate, eps));
end
fprintf('\r  %-5s %5d/%5d   done in %.0f s%20s\n', label, done, nTotal, toc(t0), '');
F = double(vertcat(chunks{:}));
end

% =====================================================================
function rw = iRowWeights(Y, w)
cats = categories(Y);
rw   = ones(numel(Y), 1);
for k = 1:numel(cats)
    rw(Y == cats{k}) = w(k);
end
end

% =====================================================================
function [thr, sens, spec, reached] = iTuneThreshold(yTrue, score, targetSens)
%ITUNETHRESHOLD Highest threshold still reaching targetSens (best specificity there).
ok = ~isnan(score);
yTrue = yTrue(ok); score = score(ok);
cand = unique([0; sort(score(:)); 1]);
best = struct('thr', 0.5, 'sens', 0, 'spec', 0);
reached = false;
for t = cand'
    s = mean(score(yTrue)  >= t);
    p = mean(score(~yTrue) <  t);
    if s >= targetSens && (~reached || p > best.spec)
        best = struct('thr', t, 'sens', s, 'spec', p);
        reached = true;
    end
end
if ~reached
    J = arrayfun(@(t) mean(score(yTrue) >= t) + mean(score(~yTrue) < t) - 1, cand);
    [~, i] = max(J);
    best = struct('thr', cand(i), 'sens', mean(score(yTrue) >= cand(i)), ...
                  'spec', mean(score(~yTrue) < cand(i)));
    warning('runBaseline:targetUnreachable', ...
        'Could not reach %.0f%% sensitivity at any threshold; used Youden''s J.', ...
        100*targetSens);
end
thr = best.thr; sens = best.sens; spec = best.spec;
end

% =====================================================================
function m = iCalibMask(dsCol, spec)
%ICALIBMASK Which rows may be used to choose the threshold.
%   The model's ranking transfers across domains; its score SCALE does not.
%   Test is IDRiD only, so the cutoff should be picked on IDRiD scores --
%   a threshold calibrated on APTOS lands in the wrong place on IDRiD.
spec = string(spec);
if strcmpi(spec, "all")
    m = true(numel(dsCol), 1);
    return
end
m = string(dsCol) == spec;
if ~any(m)
    error('runBaseline:emptyCalibration', ...
        'CalibrateOn="%s" matches no rows. Datasets present: %s', ...
        spec, strjoin(unique(string(dsCol))', ', '));
end
m = m(:);
end

% =====================================================================
function [rw, T] = iCombinedWeights(Y, w, dsCol, spec)
%ICOMBINEDWEIGHTS Per-row weight = class weight x domain weight.
%   Training is ~3,100 APTOS against ~350 IDRiD, so APTOS outvotes IDRiD
%   roughly 9:1 and the classifier optimises for APTOS cameras. Test is
%   IDRiD. 'balanced' gives each dataset equal total influence while keeping
%   every image; a number is used directly as the IDRiD multiplier.
rw = iRowWeights(Y, w);
ds = string(dsCol(:));
dw = ones(numel(ds), 1);
T  = table.empty;

if isnumeric(spec)
    dw(ds == "IDRiD") = spec;
elseif strcmpi(string(spec), "none")
    return
elseif strcmpi(string(spec), "balanced")
    u = unique(ds); K = numel(u); N = numel(ds);
    for i = 1:K
        m = ds == u(i);
        dw(m) = N / (K * sum(m));
    end
else
    error('runBaseline:badDomainWeight', ...
        'DomainWeight must be ''balanced'', ''none'', or a numeric IDRiD multiplier.');
end

u = unique(ds);
T = table(u, arrayfun(@(x) sum(ds==x), u), arrayfun(@(x) dw(find(ds==x,1)), u), ...
    'VariableNames', {'dataset', 'count', 'weight'});
rw = rw .* dw;
end

% =====================================================================
function s = iTernary(c, a, b)
if c, s = a; else, s = b; end
end
