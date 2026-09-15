function T = compareModels(frozenR, ftModelFile, roots, varargin)
%COMPAREMODELS Paired bootstrap CI on the AUC gap between two models.
%
%   T = COMPAREMODELS(Rb, 'grading/models/ft_resnet18_....mat', roots)
%     Rb is the struct returned by runBaseline (the frozen backbone).
%
%   WHAT IT ANSWERS
%   The frozen baseline scored 0.887 on IDRiD test and the fine-tuned model
%   scored 0.933. Each carries a standard error near 0.030, so comparing
%   them by "is 0.046 bigger than 0.030" is crude -- those standard errors
%   describe each AUC on its own, not the difference.
%
%   Both models score the SAME 103 images, so the comparison is paired: an
%   image that is hard for one is usually hard for the other, and that
%   shared difficulty cancels. Resampling the two score vectors on identical
%   indices measures the gap directly, and its interval is much tighter than
%   the two individual errors suggest. If the 95% interval excludes zero,
%   fine-tuning beat the frozen backbone on evidence rather than on hope.
%
%   ON REUSING TEST
%   This reads test again, but it selects nothing -- both models were already
%   revealed, and this only puts an interval around a comparison that has
%   already been made. That is legitimate. Choosing BETWEEN models by this
%   output would not be; that decision belongs to validation.
%
%   Options
%     'NBoot' 2000
%     'Seed'  20260909
%
%   See also REVEALTEST, CALIBRATETHRESHOLD, TIER1CV.

p = inputParser; p.FunctionName = 'compareModels';
p.addParameter('NBoot', 2000);
p.addParameter('Seed',  20260909);
p.parse(varargin{:});
opt = p.Results;

if ~isfield(frozenR, 'scores') || ~isfield(frozenR.scores, 'test')
    error('compareModels:noScores', ...
        'frozenR has no scores.test -- pass the struct returned by runBaseline.');
end

% ---- frozen model, already scored ------------------------------------
sFroz = frozenR.scores.test.score(:);
yFroz = logical(frozenR.scores.test.truth(:));

% ---- fine-tuned model, scored on the same rows -----------------------
M = load(ftModelFile);
T_ = loadSplit('test', roots);
readFcn = makeGradingReadFcn('TargetSize', M.inputSize, 'Enhance', M.enhance);
imds = imageDatastore(cellstr(T_.file), 'ReadFcn', readFcn);
ds   = augmentedImageDatastore(M.inputSize, imds);
Y = minibatchpredict(M.net, ds, 'ExecutionEnvironment', 'auto');
if isa(Y, 'dlarray'), Y = extractdata(Y); end
sFine = double(Y(:, M.positiveIndex));
yFine = logical(T_.referable);

if numel(sFroz) ~= numel(sFine)
    error('compareModels:lengthMismatch', ...
        ['Frozen scores cover %d rows, fine-tuned %d. The two runs saw ' ...
         'different splits -- re-run runBaseline against the current ' ...
         'split before comparing.'], numel(sFroz), numel(sFine));
end
if ~isequal(yFroz, yFine)
    error('compareModels:labelMismatch', ...
        'Row order differs between the two score vectors; the pairing would be wrong.');
end
y = yFine;

% ---- point estimates --------------------------------------------------
aF = iAuc(y, sFroz);
aT = iAuc(y, sFine);

% ---- paired bootstrap -------------------------------------------------
% Positives and negatives are resampled separately so no draw degenerates
% to one class, and both models are resampled on identical indices so the
% difference stays paired.
rng(opt.Seed);
pos = find(y); neg = find(~y);
d = zeros(opt.NBoot, 1);
for b = 1:opt.NBoot
    idx = [pos(randi(numel(pos), numel(pos), 1)); neg(randi(numel(neg), numel(neg), 1))];
    yb  = y(idx);
    d(b) = iAuc(yb, sFine(idx)) - iAuc(yb, sFroz(idx));
end
lo = prctile(d, 2.5); hi = prctile(d, 97.5);
pTwo = 2 * min(mean(d <= 0), mean(d >= 0));

% ---- report -----------------------------------------------------------
fprintf('\n=== frozen vs fine-tuned, IDRiD test (n=%d, %d referable) ===\n\n', ...
        numel(y), sum(y));
fprintf('  frozen backbone + SVM   AUC %.4f\n', aF);
fprintf('  fine-tuned %-12s AUC %.4f\n', M.backbone, aT);
fprintf('  difference              %+.4f   95%% CI [%+.4f, %+.4f]\n', aT-aF, lo, hi);
fprintf('  paired bootstrap, %d resamples, two-sided p ~ %.4f\n\n', opt.NBoot, pTwo);

if lo > 0
    fprintf(['VERDICT: the interval excludes zero. Fine-tuning is a real\n' ...
             'improvement on IDRiD, not a lucky draw -- the first change in this\n' ...
             'project that survives its own error bar.\n']);
elseif hi < 0
    fprintf('VERDICT: fine-tuning is significantly WORSE. Do not ship it.\n');
else
    fprintf(['VERDICT: the interval contains zero. The gap is consistent with\n' ...
             'noise on 103 images; report it as "no measurable difference" rather\n' ...
             'than as an improvement, however large the point estimate looks.\n']);
end
fprintf('\n');

T = table(aF, aT, aT-aF, lo, hi, pTwo, numel(y), ...
    'VariableNames', {'aucFrozen','aucFineTuned','difference','ciLo','ciHi','pValue','n'});
end

% =====================================================================
function a = iAuc(y, s)
y = logical(y(:)); s = s(:);
np = sum(y); nn = sum(~y);
if np == 0 || nn == 0, a = NaN; return; end
r = tiedrank(s);
a = (sum(r(y)) - np*(np+1)/2) / (np*nn);
end
