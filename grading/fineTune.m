function R = fineTune(roots, varargin)
%FINETUNE End-to-end fine-tuning of the DR grading backbone on a CUDA GPU.
%
%   R = FINETUNE(roots)
%   R = FINETUNE(roots, 'Backbone','resnet50', 'MaxEpochs',25)
%
%   THIS FUNCTION NEVER LOADS THE TEST SPLITS.
%   Not "does not tune on them" -- does not read them at all. loadSplit is
%   called for 'train' and 'val' only, so there is no code path by which a
%   test image can influence the network, the epoch choice, or the referral
%   threshold. Measuring the result is a separate, deliberate act: see
%   REVEALTEST.
%
%   WHY THE SEPARATION EXISTS
%   tier1Experiments ranked 13 classifier variants by their AUC on `test`.
%   No test image ever entered a fit, so the weights were clean -- but the
%   CHOICE of variant was made by reading that column, roughly twenty times
%   across the project. Selecting the best of many near-equal options by
%   their held-out score does not return the best option; it returns the
%   luckiest one, then reports its inflated score as if it were unbiased.
%   tier1CV re-ran that selection on out-of-fold scores and the headline
%   0.849 -> 0.894 "win" disappeared (linear 0.9628, svm-rbf 0.9608,
%   difference -0.002, 95% CI [-0.0125, +0.0093]).
%
%   Everything selected here -- epoch, and the referral cutoff -- is chosen
%   on validation data, which is allowed to be looked at repeatedly.
%
%   DOMAIN BALANCE WITHOUT A WEIGHTED LOSS
%   Training is ~3,112 APTOS against ~352 IDRiD, so APTOS would outvote
%   IDRiD roughly 9:1 while IDRiD is the reported benchmark. Rather than
%   weight the loss, IDRiD rows are repeated until each dataset contributes
%   equally. Because IDRiD is 62.6% referable and APTOS 40.6%, this also
%   lands the pooled training set near 50/50 on the class -- so no class
%   weighting is needed either, and the plain built-in loss can be used.
%   Each repeat draws different augmentation, so the copies are not
%   duplicates in any way that matters.
%
%   Options
%     'Backbone'    'resnet18' (default) | 'resnet50'
%     'InputSize'   [448 448]
%     'Enhance'     'bengraham' (default) | 'none' | 'clahe'
%     'MaxEpochs'   20        -- phase 2; phase 1 is fixed at 3
%     'MiniBatch'   16        -- raise to 32 if the card has headroom
%     'BaseLR'      1e-4      -- phase 2 learning rate
%     'HeadLR'      1e-3      -- phase 1 (head warm-up) learning rate
%     'TargetSens'  0.90
%     'CalibrateOn' 'IDRiD' (default) | 'all'
%     'SelectBy'    'auc' (default) | 'loss'
%     'Seed'        20260909
%     'OutDir'      'grading/models'
%
%   See also REVEALTEST, LOADSPLIT, MAKEGRADINGREADFCN, RUNBASELINE.

p = inputParser; p.FunctionName = 'fineTune';
p.addParameter('Backbone',    'resnet18');
p.addParameter('InputSize',   [448 448]);
p.addParameter('Enhance',     'bengraham');
p.addParameter('MaxEpochs',   20);
p.addParameter('MiniBatch',   16);
p.addParameter('BaseLR',      1e-4);
p.addParameter('HeadLR',      1e-3);
p.addParameter('TargetSens',  0.90);
p.addParameter('CalibrateOn', 'IDRiD');
p.addParameter('SelectBy',    'auc');
p.addParameter('Seed',        20260909);
p.addParameter('OutDir',      fullfile(fileparts(mfilename('fullpath')), 'models'));
p.parse(varargin{:});
opt = p.Results;
rng(opt.Seed);

% ---- 0. preconditions -------------------------------------------------
fprintf('\n=== fine-tune: %s @ %dx%d, enhance=%s ===\n', opt.Backbone, ...
        opt.InputSize(1), opt.InputSize(2), lower(char(opt.Enhance)));
iRequireGPU();
if ~isfolder(opt.OutDir), mkdir(opt.OutDir); end

% ---- 1. data — TRAIN AND VAL ONLY ------------------------------------
Ttr = loadSplit('train', roots);
Tva = loadSplit('val',   roots);

% domain balance by repetition
nI = sum(Ttr.dataset == "IDRiD");  nA = sum(Ttr.dataset == "APTOS");
rep = max(1, round(nA / max(nI, 1)));
idx = [find(Ttr.dataset == "APTOS"); repmat(find(Ttr.dataset == "IDRiD"), rep, 1)];
idx = idx(randperm(numel(idx)));
Tbal = Ttr(idx, :);
fprintf(['train %d (IDRiD %d, APTOS %d) -> IDRiD repeated x%d -> %d rows\n' ...
         '  pooled class balance: %.1f%% referable\n'], ...
        height(Ttr), nI, nA, rep, height(Tbal), 100*mean(Tbal.referable));
fprintf('val   %d (IDRiD %d, APTOS %d)  <- every decision below uses only this\n', ...
        height(Tva), sum(Tva.dataset == "IDRiD"), sum(Tva.dataset == "APTOS"));

readFcn = makeGradingReadFcn('TargetSize', opt.InputSize, 'Enhance', opt.Enhance);
mkcat   = @(T) categorical(double(T.referable), [0 1], {'nonreferable','referable'});

imTr = imageDatastore(cellstr(Tbal.file), 'ReadFcn', readFcn);
imTr.Labels = mkcat(Tbal);
imVa = imageDatastore(cellstr(Tva.file),  'ReadFcn', readFcn);
imVa.Labels = mkcat(Tva);

% A fundus photo has no canonical orientation beyond which eye it is, so
% full rotation is legitimate. Keep scale/translation mild: heavy warping
% destroys the lesion shape that separates a microaneurysm from a bleed.
aug = imageDataAugmenter('RandXReflection', true, 'RandYReflection', true, ...
        'RandRotation', [0 360], 'RandScale', [0.9 1.1], ...
        'RandXTranslation', [-12 12], 'RandYTranslation', [-12 12]);
dsTr = augmentedImageDatastore(opt.InputSize, imTr, 'DataAugmentation', aug);
dsVa = augmentedImageDatastore(opt.InputSize, imVa);          % never augmented

% ---- 2. network -------------------------------------------------------
classes = categories(imTr.Labels);
net = imagePretrainedNetwork(lower(char(opt.Backbone)), 'NumClasses', numel(classes));
headLayer = iHeadLayer(net);
fprintf('backbone %s, classification head detected as "%s"\n', opt.Backbone, headLayer);

% ---- 3. phase 1: warm the new head, backbone frozen -------------------
% A randomly initialised head pushes large, meaningless gradients into a
% well-conditioned backbone. Three epochs at zero backbone learning rate
% costs little and stops that.
netP1 = iSetLearnRate(net, 0, headLayer);
o1 = trainingOptions('adam', ...
    'InitialLearnRate', opt.HeadLR, 'MaxEpochs', 3, ...
    'MiniBatchSize', opt.MiniBatch, 'Shuffle', 'every-epoch', ...
    'ValidationData', dsVa, 'ValidationFrequency', iValFreq(height(Tbal), opt.MiniBatch), ...
    'ExecutionEnvironment', 'gpu', 'Verbose', true, 'VerboseFrequency', 50, ...
    'Plots', 'none', 'PreprocessingEnvironment', 'parallel');
fprintf('\n-- phase 1: head warm-up, backbone frozen (3 epochs) --\n');
net = trainnet(dsTr, netP1, 'crossentropy', o1);

% ---- 4. phase 2: unfreeze, low LR, checkpoint every epoch -------------
ckDir = fullfile(opt.OutDir, 'ck');
if isfolder(ckDir), rmdir(ckDir, 's'); end
mkdir(ckDir);

net = iSetLearnRate(net, 1, "");               % everything trainable again
o2 = trainingOptions('adam', ...
    'InitialLearnRate', opt.BaseLR, 'MaxEpochs', opt.MaxEpochs, ...
    'LearnRateSchedule', 'piecewise', 'LearnRateDropFactor', 0.3, ...
    'LearnRateDropPeriod', max(6, floor(opt.MaxEpochs/3)), ...
    'MiniBatchSize', opt.MiniBatch, 'Shuffle', 'every-epoch', ...
    'ValidationData', dsVa, 'ValidationFrequency', iValFreq(height(Tbal), opt.MiniBatch), ...
    'ValidationPatience', 6, 'OutputNetwork', 'best-validation-loss', ...
    'CheckpointPath', ckDir, 'CheckpointFrequency', 1, ...
    'CheckpointFrequencyUnit', 'epoch', ...
    'ExecutionEnvironment', 'gpu', 'Verbose', true, 'VerboseFrequency', 50, ...
    'Plots', 'none', 'PreprocessingEnvironment', 'parallel');
fprintf('\n-- phase 2: full fine-tune (%d epochs max, patience 6) --\n', opt.MaxEpochs);
tTrain = tic;
netBest = trainnet(dsTr, net, 'crossentropy', o2);
fprintf('training wall clock: %.1f min\n', toc(tTrain)/60);

% ---- 5. epoch selection — VALIDATION ONLY ----------------------------
% 'best-validation-loss' is already a validation-driven choice. Loss is a
% proxy, though, and the target is a ranking metric, so by default every
% epoch checkpoint is scored on validation AUC and the best one wins.
posIdx = find(strcmp(classes, 'referable'), 1);
yVa    = Tva.referable;
mCal   = iCalibMask(Tva.dataset, opt.CalibrateOn);

chosen = netBest; chosenTag = 'best-validation-loss';
if strcmpi(opt.SelectBy, 'auc')
    try
        [chosen, chosenTag] = iPickByValAUC(ckDir, dsVa, yVa, posIdx, netBest);
    catch ME
        warning('fineTune:ckSweep', ...
            'Checkpoint sweep failed (%s). Falling back to best-validation-loss.', ME.message);
    end
end

sVa = iScore(chosen, dsVa, posIdx);
aucVaAll = iAuc(yVa, sVa);
aucVaCal = iAuc(yVa(mCal), sVa(mCal));
fprintf('\nselected network: %s\n', chosenTag);
fprintf('validation AUC  pooled %.4f (n=%d)   %s %.4f (n=%d)\n', ...
        aucVaAll, numel(yVa), char(opt.CalibrateOn), aucVaCal, sum(mCal));

% ---- 6. referral threshold — VALIDATION ONLY -------------------------
% The model's ranking transfers across cameras; its score scale does not,
% so the cutoff is taken from target-domain validation rows.
[thr, sens, spec, hit] = iTuneThreshold(yVa(mCal), sVa(mCal), opt.TargetSens);
fprintf(['threshold %.4f from %d %s validation rows%s\n' ...
         '  at that cutoff, on validation: sens %.1f%%  spec %.1f%%\n'], ...
        thr, sum(mCal), char(opt.CalibrateOn), ...
        iIf(hit, '', '   [target not reachable on val]'), 100*sens, 100*spec);
if sum(mCal) < 120
    fprintf(['  NOTE: %d rows is a noisy basis for a cutoff. This is the honest\n' ...
             '  cost of never touching test. If it proves too noisy, the fix is\n' ...
             '  k-fold fine-tuning over train+val (k x the GPU time), not a peek.\n'], sum(mCal));
end

% ---- 7. save ----------------------------------------------------------
stamp = datestr(now, 'yyyymmdd_HHMMSS'); %#ok<TNOW1,DATST>
R = struct('net', chosen, 'threshold', thr, 'classes', {classes}, ...
           'positiveIndex', posIdx, 'backbone', opt.Backbone, ...
           'inputSize', opt.InputSize, 'enhance', lower(char(opt.Enhance)), ...
           'selectedBy', chosenTag, 'calibrateOn', char(opt.CalibrateOn), ...
           'valAUCpooled', aucVaAll, 'valAUCcalib', aucVaCal, ...
           'valSens', sens, 'valSpec', spec, 'opt', opt, ...
           'testEvaluated', false, 'created', stamp);
f = fullfile(opt.OutDir, sprintf('ft_%s_%s.mat', lower(char(opt.Backbone)), stamp));
save(f, '-struct', 'R', '-v7.3');
fprintf('\nsaved: %s\n', f);
fprintf(['Test sets have not been read. When you are finished choosing, run\n' ...
         '  revealTest(''%s'', roots)\n' ...
         'once, and report what it prints.\n\n'], f);
end

% =====================================================================
function iRequireGPU()
ok = false;
try, ok = (gpuDeviceCount("available") > 0); catch, end %#ok<CTCH>
if ~ok
    error('fineTune:noGPU', ['No CUDA device visible to MATLAB.\n' ...
        'Check, in order:\n' ...
        '  1) gpuDevice            -- should name the card\n' ...
        '  2) !nvidia-smi          -- driver present?\n' ...
        '  3) On a hybrid-graphics laptop, force MATLAB onto the discrete\n' ...
        '     NVIDIA in the NVIDIA Control Panel, then restart MATLAB.\n' ...
        'MATLAB has no AMD/OpenCL path -- CUDA only.']);
end
g = gpuDevice;
fprintf('GPU: %s, %.1f GB, compute %s\n', g.Name, g.TotalMemory/2^30, g.ComputeCapability);
if g.TotalMemory/2^30 < 6
    warning('fineTune:smallGPU', ...
        'Under 6 GB. If training runs out of memory, drop MiniBatch to 8.');
end
end

function name = iHeadLayer(net)
u = unique(string(net.Learnables.Layer), 'stable');
name = u(end);
end

function net = iSetLearnRate(net, factor, exceptLayer)
L = net.Learnables;
for i = 1:height(L)
    ln = string(L.Layer(i));
    if ln == exceptLayer, continue; end
    net = setLearnRateFactor(net, ln, string(L.Parameter(i)), factor);
end
end

function f = iValFreq(nRows, mb)
f = max(20, floor(nRows / max(mb, 1) / 2));      % ~twice per epoch
end

function s = iScore(net, ds, posIdx)
Y = minibatchpredict(net, ds, 'ExecutionEnvironment', 'auto');
if isa(Y, 'dlarray'), Y = extractdata(Y); end
s = double(Y(:, posIdx));
end

function [best, tag] = iPickByValAUC(ckDir, dsVa, yVa, posIdx, fallback)
files = dir(fullfile(ckDir, '*.mat'));
if isempty(files), error('no checkpoints written'); end
best = fallback; tag = 'best-validation-loss'; bestA = -Inf;
fprintf('\nscoring %d epoch checkpoints on VALIDATION only:\n', numel(files));
for k = 1:numel(files)
    Lk = load(fullfile(ckDir, files(k).name));
    fn = fieldnames(Lk);
    n_ = [];
    for j = 1:numel(fn)
        if isa(Lk.(fn{j}), 'dlnetwork'), n_ = Lk.(fn{j}); break; end
    end
    if isempty(n_), continue; end
    a = iAuc(yVa, iScore(n_, dsVa, posIdx));
    fprintf('  %-42s val AUC %.4f\n', files(k).name, a);
    if a > bestA, bestA = a; best = n_; tag = sprintf('epoch ckpt %s (val AUC %.4f)', files(k).name, a); end
end
end

function a = iAuc(y, s)
y = logical(y(:)); s = s(:);
ok = ~isnan(s); y = y(ok); s = s(ok);
np = sum(y); nn = sum(~y);
if np == 0 || nn == 0, a = NaN; return; end
r = tiedrank(s);
a = (sum(r(y)) - np*(np+1)/2) / (np*nn);
end

function m = iCalibMask(dsCol, spec)
spec = string(spec);
if strcmpi(spec, "all"), m = true(numel(dsCol), 1); return; end
m = string(dsCol) == spec;
if ~any(m)
    error('fineTune:emptyCalibration', 'CalibrateOn="%s" matches no validation rows.', spec);
end
m = m(:);
end

function [thr, sens, spec, hit] = iTuneThreshold(y, s, target)
y = logical(y(:)); s = s(:);
cand = unique([min(s)-1; sort(s); max(s)+1]);
thr = median(cand); sens = 0; spec = 0; best = -Inf; hit = false;
for t = cand'
    se = mean(s(y) >= t); sp = mean(s(~y) < t);
    if se >= target && sp > best, best = sp; thr = t; sens = se; spec = sp; hit = true; end
end
if ~hit
    J = arrayfun(@(t) mean(s(y) >= t) + mean(s(~y) < t) - 1, cand);
    [~, i] = max(J); thr = cand(i);
    sens = mean(s(y) >= thr); spec = mean(s(~y) < thr);
end
end

function o = iIf(c, a, b), if c, o = a; else, o = b; end, end
