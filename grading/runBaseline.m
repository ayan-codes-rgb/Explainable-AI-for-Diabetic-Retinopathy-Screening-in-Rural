function R = runBaseline(dataRoot, varargin)
%RUNBASELINE CPU-friendly DR grading baseline: frozen backbone + linear classifier.
%
%   R = RUNBASELINE(dataRoot)
%   R = RUNBASELINE(dataRoot, 'Backbone', 'resnet18', 'InputSize', [224 224])
%
%   WHAT THIS DOES AND WHY
%   Fine-tuning a CNN means running every image through the network AND
%   backpropagating through every layer, every epoch. That needs a GPU.
%   This does something cheaper: it runs each image through a FROZEN
%   pretrained network exactly ONCE, keeps the activations as a feature
%   vector, and trains a linear classifier on those vectors. Forward passes
%   only, one pass over the data ever.
%
%   On CPU that is minutes rather than hours, so you can iterate on class
%   weights and the decision threshold -- which is what actually determines
%   whether you hit >90% sensitivity -- instead of getting three experiments
%   done before the deadline.
%
%   It is also the ablation baseline the problem statement asks for: the
%   number a fine-tuned model has to beat to justify its cost.
%
%   Options
%     'Backbone'   'resnet18' (default) | 'mobilenetv2' | 'squeezenet'
%     'InputSize'  [224 224] by default
%     'Task'       'referable' (default, binary) | 'grade' (5-class)
%     'Weights'    'inverse-sqrt' (default) | 'inverse' | 'none'
%
%   Returns a struct with the trained model, the validation-tuned threshold,
%   and metrics on val and test.
%
%   FIRST RUN WILL LIKELY ASK FOR AN ADD-ON. Pretrained networks ship as
%   separate support packages ("Deep Learning Toolbox Model for ResNet-18
%   Network"). MATLAB will prompt; install it once.
%
%   See also BUILDDATASTORES, CLASSWEIGHTS, LOADSPLIT.

p = inputParser;
p.FunctionName = 'runBaseline';
p.addParameter('Backbone',  'resnet18');
p.addParameter('InputSize', [224 224]);
p.addParameter('Task',      'referable');
p.addParameter('Weights',   'inverse-sqrt');
p.parse(varargin{:});
opt = p.Results;

fprintf('\n=== DR grading baseline ===\n');

% ---- 1. data ---------------------------------------------------------
S = buildDatastores(dataRoot, 'InputSize', opt.InputSize, ...
                              'Task', opt.Task, 'Augment', false);
% Augmentation off: we extract features once, so augmented copies would
% just be noise in a fixed feature matrix. Turn it on for fine-tuning later.

[w, wTable] = classWeights(S.train.labels, opt.Weights);
fprintf('\nclass balance (train):\n'); disp(wTable);

% ---- 2. frozen backbone ---------------------------------------------
fprintf('loading backbone: %s\n', opt.Backbone);
[net, featLayer] = iBackbone(opt.Backbone);

% ---- 3. extract features (the only expensive step) -------------------
t0 = tic;
Ftrain = iFeatures(net, S.train.ds, featLayer);
Fval   = iFeatures(net, S.val.ds,   featLayer);
Ftest  = iFeatures(net, S.test.ds,  featLayer);
fprintf('features: %d-dim, %.0f s total\n', size(Ftrain,2), toc(t0));

% ---- 4. linear classifier -------------------------------------------
Ytrain = S.train.labels;
tmpl   = templateLinear('Learner', 'logistic', 'Solver', 'lbfgs');
mdl    = fitcecoc(Ftrain, Ytrain, 'Learners', tmpl, ...
                  'ClassNames', categories(Ytrain), ...
                  'Weights', iRowWeights(Ytrain, w), ...
                  'FitPosterior', true);

% ---- 5. threshold tuning on VAL, then a single test evaluation -------
classes = categories(Ytrain);
posName = classes{end};                     % 'referable' or grade '4'
[~, ~, Pval]  = predict(mdl, Fval);
[~, ~, Ptest] = predict(mdl, Ftest);
posCol = find(strcmp(classes, posName), 1);

yVal  = S.val.labels  == posName;
yTest = S.test.labels == posName;

[thr, sensV, specV] = iTuneThreshold(yVal, Pval(:,posCol), 0.90);
sensT = mean(Ptest(yTest,  posCol) >= thr);
specT = mean(Ptest(~yTest, posCol) <  thr);

fprintf('\n--- results (positive class: %s) ---\n', posName);
fprintf('threshold tuned on val : %.3f\n', thr);
fprintf('VAL   sensitivity %.1f%%   specificity %.1f%%   (n=%d)\n', ...
    100*sensV, 100*specV, numel(yVal));
fprintf('TEST  sensitivity %.1f%%   specificity %.1f%%   (n=%d)\n', ...
    100*sensT, 100*specT, numel(yTest));
fprintf('targets: sensitivity >90%%, specificity >85%%\n');
fprintf('NOTE: test n=%d, so each image moves the number by ~%.1f%%. Quote a\n', ...
    numel(yTest), 100/numel(yTest));
fprintf('      confidence interval, not a bare percentage.\n\n');

R = struct('model', mdl, 'threshold', thr, 'backbone', opt.Backbone, ...
           'task', opt.Task, 'classWeights', wTable, ...
           'val',  struct('sensitivity', sensV, 'specificity', specV), ...
           'test', struct('sensitivity', sensT, 'specificity', specT), ...
           'features', struct('dim', size(Ftrain,2), 'layer', featLayer));
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
    net = imagePretrainedNetwork(name);           % newer API -> dlnetwork
elseif ~isempty(which(name))
    net = feval(name);                            % older API -> DAGNetwork
else
    error('runBaseline:missingAddOn', ...
        ['Pretrained network "%s" is not installed. It ships as a separate ' ...
         'support package -- Home > Add-Ons > Get Add-Ons, search for ' ...
         '"Deep Learning Toolbox Model for %s".'], name, name);
end

if ~any(strcmp({net.Layers.Name}, featLayer))
    names = {net.Layers.Name};
    error('runBaseline:badLayer', ...
        ['Layer "%s" not found in %s. Available pooling layers: %s\n' ...
         'Pick the global pooling layer just before the classifier.'], ...
        featLayer, name, strjoin(names(contains(lower(names), 'pool')), ', '));
end
end

% =====================================================================
function F = iFeatures(net, ds, layerName)
%IFEATURES Activations as an N-by-D matrix, for either network class.
if isa(net, 'dlnetwork')
    reset(ds);
    chunks = {};
    while hasdata(ds)
        b = read(ds);
        if istable(b), X = cat(4, b{:,1}{:}); else, X = b; end
        Y = predict(net, dlarray(single(X), 'SSCB'), 'Outputs', layerName);
        Y = extractdata(gather(Y));
        chunks{end+1} = reshape(Y, [], size(Y, ndims(Y)))'; %#ok<AGROW>
    end
    F = double(vertcat(chunks{:}));
else
    F = double(activations(net, ds, layerName, 'OutputAs', 'rows'));
end
end

% =====================================================================
function rw = iRowWeights(Y, w)
%IROWWEIGHTS Expand per-class weights to a per-observation vector.
cats = categories(Y);
rw   = ones(numel(Y), 1);
for k = 1:numel(cats)
    rw(Y == cats{k}) = w(k);
end
end

% =====================================================================
function [thr, sens, spec] = iTuneThreshold(yTrue, score, targetSens)
%ITUNETHRESHOLD Lowest threshold reaching targetSens, best specificity there.
%   Argmax optimises accuracy. This project is graded on sensitivity, so the
%   threshold is chosen deliberately: accept more false positives (one extra
%   clinic visit) to avoid false negatives (someone loses their sight).
cand = unique([0; sort(score(:)); 1]);
best = struct('thr', 0.5, 'sens', 0, 'spec', 0, 'ok', false);
for t = cand'
    s = mean(score(yTrue)  >= t);
    p = mean(score(~yTrue) <  t);
    if s >= targetSens && (~best.ok || p > best.spec)
        best = struct('thr', t, 'sens', s, 'spec', p, 'ok', true);
    end
end
if ~best.ok
    % Target unreachable on this data -- fall back to Youden's J and say so.
    J = arrayfun(@(t) mean(score(yTrue) >= t) + mean(score(~yTrue) < t) - 1, cand);
    [~, i] = max(J);
    best = struct('thr', cand(i), 'sens', mean(score(yTrue) >= cand(i)), ...
                  'spec', mean(score(~yTrue) < cand(i)), 'ok', false);
    warning('runBaseline:targetUnreachable', ...
        ['Could not reach %.0f%% sensitivity on validation at any threshold. ' ...
         'Using Youden''s J instead. Report this honestly rather than ' ...
         'tuning against the test set.'], 100*targetSens);
end
thr = best.thr; sens = best.sens; spec = best.spec;
end
