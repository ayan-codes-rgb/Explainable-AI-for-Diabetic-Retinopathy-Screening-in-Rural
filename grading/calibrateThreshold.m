function out = calibrateThreshold(modelFile, roots, varargin)
%CALIBRATETHRESHOLD Choose a defensible operating point from VALIDATION ONLY.
%
%   out = CALIBRATETHRESHOLD(modelFile, roots)
%   out = CALIBRATETHRESHOLD(modelFile, roots, 'Apply', true)   % write it back
%
%   THIS FUNCTION NEVER READS TEST. loadSplit is called for 'val' only.
%
%   -------------------------------------------------------------------
%   REVISION, 15 Sep 2026. The first version of this file compared the
%   POSITIONS of the per-domain cutoffs and declared the score scale
%   "untransferable" when they sat more than a quarter of a standard
%   deviation apart. That was the wrong quantity. Position only matters if
%   it changes what the model does, and on the run it was written for it
%   did not: all three candidate cutoffs caught the same 35 of 38 IDRiD
%   validation positives, and the pooled cutoff matched IDRiD's specificity
%   exactly while beating it on APTOS. The old rule recommended the option
%   fitted on 61 rows over an equally good one fitted on 335.
%
%   This version ranks candidates by what they DO on the target domain, and
%   only then breaks ties by how much data each was estimated from.
%   -------------------------------------------------------------------
%
%   THE MARGIN, AND WHY IT IS NOT OPTIONAL ON SMALL SETS
%   A threshold is normally chosen as the HIGHEST cutoff still reaching the
%   target sensitivity, because that maximises specificity. On a large
%   calibration set that is fine. On 61 rows it puts the cutoff flush
%   against the last positive it can afford to keep, with no headroom at
%   all -- and on fresh data a few positives inevitably fall below it. That
%   is a systematic optimism, not bad luck, and it is exactly why a cutoff
%   that measured 92.1% sensitivity on validation delivered 85.9% on test.
%
%   So the cutoff is fitted at TargetSens + SensMargin and deployed against
%   TargetSens. You give up a little specificity to buy back sensitivity
%   you would otherwise lose the moment the model sees new images.
%
%   Options
%     'TargetSens'  0.90  -- what you need in deployment
%     'SensMargin'  0.05  -- extra sensitivity demanded at calibration time
%     'Apply'       false -- true rewrites modelFile with the chosen cutoff
%
%   See also FINETUNE, REVEALTEST, COMPAREMODELS.

p = inputParser; p.FunctionName = 'calibrateThreshold';
p.addParameter('TargetSens', 0.90);
p.addParameter('SensMargin', 0.05);
p.addParameter('Apply',      false);
p.parse(varargin{:});
opt = p.Results;
calibTarget = min(0.995, opt.TargetSens + opt.SensMargin);

M = load(modelFile);
fprintf('\n=== threshold calibration (validation only) ===\n');
fprintf('model             : %s\n', modelFile);
fprintf('current cutoff    : %.4f, fitted on %s rows\n', M.threshold, M.calibrateOn);
fprintf('deployment target : %.0f%% sensitivity\n', 100*opt.TargetSens);
fprintf('calibration target: %.0f%% (margin %.0f points)\n\n', ...
        100*calibTarget, 100*opt.SensMargin);

% ---- score validation -------------------------------------------------
T = loadSplit('val', roots);
readFcn = makeGradingReadFcn('TargetSize', M.inputSize, 'Enhance', M.enhance);
imds = imageDatastore(cellstr(T.file), 'ReadFcn', readFcn);
ds   = augmentedImageDatastore(M.inputSize, imds);
Y = minibatchpredict(M.net, ds, 'ExecutionEnvironment', 'auto');
if isa(Y, 'dlarray'), Y = extractdata(Y); end
s   = double(Y(:, M.positiveIndex));
y   = logical(T.referable);
dsn = string(T.dataset);
mI  = dsn == "IDRiD";  mA = dsn == "APTOS";

nPosI = sum(y & mI);
fprintf('validation: %d rows (%d IDRiD / %d positive, %d APTOS)\n\n', ...
        numel(y), sum(mI), nPosI, sum(mA));

% ---- candidates -------------------------------------------------------
nm   = {'IDRiD', 'APTOS', 'pooled'};
msk  = {mI, mA, true(size(y))};
thr  = zeros(3,1);  nFit = zeros(3,1);
for k = 1:3
    thr(k)  = iTune(y(msk{k}), s(msk{k}), calibTarget);
    nFit(k) = sum(msk{k});
end

% ---- what each one DOES, on the domain that is the benchmark ---------
sI = zeros(3,1); pI = zeros(3,1); sA = zeros(3,1); pA = zeros(3,1);
for k = 1:3
    t = thr(k);
    sI(k) = mean(s(y & mI) >= t);   pI(k) = mean(s(~y & mI) < t);
    sA(k) = mean(s(y & mA) >= t);   pA(k) = mean(s(~y & mA) < t);
end

fprintf('%-8s%7s%8s%11s%9s%11s%9s\n', 'fitted', 'n', 'cutoff', ...
        'IDRiD sens', 'spec', 'APTOS sens', 'spec');
fprintf('%s\n', repmat('-', 1, 64));
for k = 1:3
    fprintf('%-8s%7d%8.4f%10.1f%%%8.1f%%%10.1f%%%8.1f%%\n', ...
        nm{k}, nFit(k), thr(k), 100*sI(k), 100*pI(k), 100*sA(k), 100*pA(k));
end
fprintf('%s\n', repmat('-', 1, 64));

% ---- can 61 rows even resolve this choice? ---------------------------
% How far can the cutoff move without changing how many IDRiD validation
% positives are caught? A wide plateau means the calibration set is not
% measuring the threshold at all, it is measuring a gap between two scores.
[lo, hi] = iPlateau(s(y & mI), thr(strcmp(nm,'pooled')));
sd   = std(s);
plat = (hi - lo) / max(sd, eps);
fprintf('\nIDRiD sensitivity is unchanged for any cutoff in [%.4f, %.4f]\n', lo, hi);
fprintf('  that plateau is %.2f x the score s.d. and spans %d of the %d candidates\n', ...
        plat, sum(thr >= lo & thr <= hi), numel(thr));

% ---- choose: performance first, sample size only to break ties -------
tol   = 1 / max(nPosI, 1);                    % one image's worth of sensitivity
elig  = find(sI >= max(sI) - tol/2);
[~,j] = max(nFit(elig));
best  = elig(j);

fprintf('\ncandidates within one image of the best IDRiD sensitivity: %s\n', ...
        strjoin(nm(elig), ', '));
fprintf('of those, the one fitted on the most rows is %s (n=%d).\n', ...
        nm{best}, nFit(best));

chosen   = thr(best);
chosenOn = nm{best};
if strcmp(chosenOn, 'pooled'), chosenOn = 'all'; end

fprintf(['\nVERDICT: adopt the %s cutoff %.4f.\n' ...
         'It catches %.1f%% of IDRiD validation positives at %.1f%% specificity -- ' ...
         'within\none image of every alternative -- and is estimated from %d rows ' ...
         'rather than %d.\n'], nm{best}, chosen, 100*sI(best), 100*pI(best), ...
         nFit(best), min(nFit));
if plat > 0.15
    fprintf(['\nCaveat worth writing down: the plateau above is wide, so these 61 ' ...
             'rows\ncannot really distinguish these cutoffs. The margin is doing the ' ...
             'work here,\nnot the calibration set. The rigorous fix remains k-fold ' ...
             'fine-tuning for an\nout-of-fold threshold over all 413 IDRiD rows.\n']);
end

out = struct('names', {nm}, 'thresholds', thr, 'nFitted', nFit, ...
             'sensIDRiD', sI, 'specIDRiD', pI, 'sensAPTOS', sA, 'specAPTOS', pA, ...
             'plateauLo', lo, 'plateauHi', hi, 'plateauSD', plat, ...
             'chosen', chosen, 'chosenOn', chosenOn, 'calibTarget', calibTarget, ...
             'nVal', numel(y));

% ---- optionally write it back ----------------------------------------
if opt.Apply
    old = M.threshold;
    M.threshold     = chosen;
    M.calibrateOn   = chosenOn;
    M.thresholdNote = sprintf(['recalibrated %s: %.4f on %d validation rows at ' ...
        'target %.0f%% (was %.4f on %d IDRiD rows at %.0f%%). Test never consulted.'], ...
        datestr(now,'yyyy-mm-dd'), chosen, nFit(best), 100*calibTarget, ...
        old, sum(mI), 100*opt.TargetSens); %#ok<TNOW1,DATST>
    save(modelFile, '-struct', 'M', '-v7.3');
    fprintf('\nwritten: cutoff %.4f -> %s\n', chosen, modelFile);
    fprintf(['Running revealTest again is a SECOND peek at the test set. It is\n' ...
             'defensible -- the new cutoff came from validation -- but the report must\n' ...
             'say two reveals, not one. revealTest logs it either way.\n']);
else
    fprintf('\nNothing written. Re-run with ''Apply'',true to adopt %.4f.\n', chosen);
end
fprintf('\n');
end

% =====================================================================
function thr = iTune(y, s, target)
%ITUNE Highest cutoff still reaching target sensitivity (best specificity there).
y = logical(y(:)); s = s(:);
cand = unique([min(s)-1; sort(s); max(s)+1]);
thr = median(cand); best = -Inf; hit = false;
for t = cand'
    se = mean(s(y) >= t); sp = mean(s(~y) < t);
    if se >= target && sp > best, best = sp; thr = t; hit = true; end
end
if ~hit
    J = arrayfun(@(t) mean(s(y) >= t) + mean(s(~y) < t) - 1, cand);
    [~, i] = max(J); thr = cand(i);
end
end

function [lo, hi] = iPlateau(sPos, t)
%IPLATEAU Range over which the cutoff catches the same number of positives.
sPos = sort(sPos(:));
below = sPos(sPos < t);
above = sPos(sPos >= t);
if isempty(below), lo = -Inf; else, lo = below(end); end
if isempty(above), hi =  Inf; else, hi = above(1);  end
end
