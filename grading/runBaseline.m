function R = runBaseline(dataRoot, varargin)
%RUNBASELINE CPU-friendly DR grading baseline: frozen backbone + linear classifier.
%
%   R = RUNBASELINE(dataRoot)
%   R = RUNBASELINE(dataRoot, 'Backbone', 'resnet18', 'Weights', 'inverse')
%
%   WHAT THIS DOES AND WHY
%   Fine-tuning a CNN means running every image through the network AND
%   backpropagating through every layer, every epoch. That needs a GPU.
%   This runs each image through a FROZEN pretrained network exactly ONCE,
%   keeps the activations as a feature vector, and trains a linear
%   classifier on those vectors. Forward passes only, one pass over the data.
%
%   It is also the ablation baseline the problem statement asks for: the
%   number a fine-tuned model has to beat to justify its cost.
%
%   FEATURES ARE CACHED. The first run spends most of its time decoding
%   4288x2848 JPEGs and pushing them through the network -- several minutes.
%   The result is saved to grading/cache/, keyed on backbone and input size,
%   and validated against the exact file list. Later runs load it in about a
%   second, so iterating on class weights and thresholds costs nothing.
%   Pass 'Refresh', true to force recomputation.
%
%   Options
%     'Backbone'   'resnet18' (default) | 'mobilenetv2' | 'squeezenet'
%     'InputSize'  [224 224] by default
%     'Task'       'referable' (default, binary) | 'grade' (5-class)
%     'Weights'    'inverse-sqrt' (default) | 'inverse' | 'none'
%     'TargetSens' 0.90 by default -- the sensitivity the threshold aims for
%     'Refresh'    false by default; true recomputes features and re-caches
%     'BatchSize'  32 by default
%
%   FIRST RUN MAY ASK FOR AN ADD-ON. Pretrained weights ship as separate
%   support packages ("Deep Learning Toolbox Model for ResNet-18 Network").
%
%   See also BUILDDATASTORES, CLASSWEIGHTS, LOADSPLIT.

p = inputParser;
p.FunctionName = 'runBaseline';
p.addParameter('Backbone',   'resnet18');
p.addParameter('InputSize',  [224 224]);
p.addParameter('Task',       'referable');
p.addParameter('Weights',    'inverse-sqrt');
p.addParameter('TargetSens', 0.90);
p.addParameter('Refresh',    false);
p.addParameter('BatchSize',  32);
p.parse(varargin{:});
opt = p.Results;

fprintf('\n=== DR grading baseline ===\n');

% ---- 1. data ---------------------------------------------------------
S = buildDatastores(dataRoot, 'InputSize', opt.InputSize, ...
                              'Task', opt.Task, 'Augment', false);
% Augmentation off: features are extracted once into a fixed matrix, so
% augmented copies would just be noise. Turn it on for fine-tuning later.

[w, wTable] = classWeights(S.train.labels, opt.Weights);
fprintf('\nclass balance (train):\n'); disp(wTable);

% ---- 2. features (cached) -------------------------------------------
cacheDir = fullfile(fileparts(mfilename('fullpath')), 'cache');
if ~isfolder(cacheDir), mkdir(cacheDir); end

net = [];  featLayer = '';
F = struct();
tAll = tic;
for nm = ["train" "val" "test"]
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

% ---- 3. linear classifier -------------------------------------------
Ytrain = S.train.labels;
tmpl   = templateLinear('Learner', 'logistic', 'Solver', 'lbfgs');
mdl    = fitcecoc(F.train, Ytrain, 'Learners', tmpl, ...
                  'ClassNames', categories(Ytrain), ...
                  'Weights', iRowWeights(Ytrain, w), ...
                  'FitPosterior', true);

% ---- 4. tune threshold on VAL, evaluate TEST once --------------------
classes = categories(Ytrain);
posName = classes{end};
posCol  = find(strcmp(classes, posName), 1);
[~, ~, Pval]  = predict(mdl, F.val);
[~, ~, Ptest] = predict(mdl, F.test);

yVal  = S.val.labels  == posName;
yTest = S.test.labels == posName;

[thr, sensV, specV, reached] = iTuneThreshold(yVal, Pval(:,posCol), opt.TargetSens);
sensT = mean(Ptest(yTest,  posCol) >= thr);
specT = mean(Ptest(~yTest, posCol) <  thr);

n = numel(yTest);
fprintf('\n--- results (positive class: %s) ---\n', posName);
fprintf('threshold tuned on val : %.3f%s\n', thr, ...
    iTernary(reached, '', '   [target NOT reached on val]'));
fprintf('VAL   sensitivity %5.1f%%   specificity %5.1f%%   (n=%d)\n', ...
    100*sensV, 100*specV, numel(yVal));
fprintf('TEST  sensitivity %5.1f%%   specificity %5.1f%%   (n=%d)\n', ...
    100*sensT, 100*specT, n);
fprintf('targets: sensitivity >90%%, specificity >85%%\n');
fprintf('NOTE: test n=%d, so one image moves the number by %.1f points.\n', n, 100/n);
fprintf('      Quote a confidence interval, not a bare percentage.\n\n');

R = struct('model', mdl, 'threshold', thr, 'targetReached', reached, ...
           'backbone', opt.Backbone, 'task', opt.Task, 'classWeights', wTable, ...
           'val',  struct('sensitivity', sensV, 'specificity', specV), ...
           'test', struct('sensitivity', sensT, 'specificity', specT), ...
           'features', struct('dim', size(F.train,2), 'layer', featLayer));
end

% =====================================================================
function f = iCacheFile(splitName, opt, cacheDir)
sz = opt.InputSize;
f  = fullfile(cacheDir, sprintf('feat_%s_%dx%d_%s.mat', ...
        lower(char(opt.Backbone)), sz(1), sz(2), splitName));
end

function [F, hit] = iCachedFeatures(S, splitName, opt, cacheDir)
%ICACHEDFEATURES Load cached features, but only if they match this exact file list.
F = []; hit = false;
if opt.Refresh, return; end
f = iCacheFile(splitName, opt, cacheDir);
if ~isfile(f), return; end
try
    L = load(f, 'F', 'files');
    if isequal(L.files, cellstr(S.(splitName).tbl.file))
        F = L.F; hit = true;
        fprintf('  %-5s %4d images  [cached]\n', splitName, size(F,1));
    end
catch
    % Corrupt or old-format cache -- fall through and recompute.
end
end

function iSaveCache(F, S, splitName, opt, cacheDir) %#ok<INUSD>
files = cellstr(S.(splitName).tbl.file); %#ok<NASGU>
try
    save(iCacheFile(splitName, opt, cacheDir), 'F', 'files');
catch ME
    warning('runBaseline:cacheWriteFailed', ...
        'Could not write feature cache (%s). Continuing without it.', ME.message);
end
end

% =====================================================================
function [net, featLayer] = iBackbone(name)
%IBACKBONE Load a pretrained net across MATLAB's changing APIs.
name = lower(char(name));
switch name
    case 'resnet18',    featLayer = 'pool5';
    case 'mobilenetv2', featLayer = 'global_average_pooling2d_1';
    case 'squeezenet',  featLayer = 'pool10';
    otherwise
        error('runBaseline:badBackbone', ...
            'Backbone must be resnet18, mobilenetv2 or squeezenet.');
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
%IFEATURES Activations as an N-by-D matrix, with progress, for either net class.
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
    el   = toc(t0);
    rate = done / max(el, eps);
    fprintf('\r  %-5s %4d/%4d   %4.0f s elapsed, ~%3.0f s left   ', ...
        label, done, nTotal, el, max(nTotal - done, 0) / max(rate, eps));
end
fprintf('\r  %-5s %4d/%4d   done in %.0f s%20s\n', label, done, nTotal, toc(t0), '');
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
%ITUNETHRESHOLD Lowest threshold reaching targetSens, best specificity there.
%   Argmax optimises accuracy. This project is graded on sensitivity, so the
%   threshold is chosen deliberately: accept more false positives (one extra
%   clinic visit) to avoid false negatives (someone loses their sight).
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
        ['Could not reach %.0f%% sensitivity on validation at any threshold. ' ...
         'Using Youden''s J instead. Report this honestly rather than tuning ' ...
         'against the test set.'], 100*targetSens);
end
thr = best.thr; sens = best.sens; spec = best.spec;
end

% =====================================================================
function s = iTernary(c, a, b)
if c, s = a; else, s = b; end
end
