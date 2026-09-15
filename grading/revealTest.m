function T = revealTest(modelFile, roots)
%REVEALTEST Measure a finished model on the held-out test sets. Once.
%
%   T = REVEALTEST('grading/models/ft_resnet18_20260914_193000.mat', roots)
%
%   This is the ONLY function in the grading module that reads `test` or
%   `test_aptos`. It exists as a separate file so that reading them is a
%   deliberate act you can see in your own command history, rather than
%   something that happens as a side effect of an experiment.
%
%   THE RULE
%   Do not run this to compare two models. That is precisely the mistake
%   this separation exists to prevent: choosing between near-equal options
%   by their test score returns the luckiest option and reports its
%   inflated score as unbiased. Choose on validation with fineTune, then
%   run this once on the winner and report what it prints.
%
%   Every call appends to grading/models/reveal_log.txt. The log is not a
%   safeguard -- it is a record, so that when the report is written you can
%   state honestly how many times the test set was consulted.
%
%   See also FINETUNE.

M = load(modelFile);
if ~isfield(M, 'net') || ~isfield(M, 'threshold')
    error('revealTest:badModel', 'Not a fineTune model file: %s', modelFile);
end

logf = fullfile(fileparts(modelFile), 'reveal_log.txt');
prior = 0;
if isfile(logf)
    prior = numel(strfind(fileread(logf), 'REVEAL'));
end
fprintf('\n%s\n', repmat('=', 1, 72));
fprintf('TEST REVEAL — %s\n', modelFile);
if prior > 0
    fprintf(['This test set has already been consulted %d time(s) from this\n' ...
             'folder. Each additional look inflates the number you report.\n'], prior);
end
fprintf('%s\n', repmat('=', 1, 72));

readFcn = makeGradingReadFcn('TargetSize', M.inputSize, 'Enhance', M.enhance);
rows = {};
for nm = ["test" "test_aptos"]
    T_ = loadSplit(char(nm), roots);
    imds = imageDatastore(cellstr(T_.file), 'ReadFcn', readFcn);
    ds   = augmentedImageDatastore(M.inputSize, imds);
    Y = minibatchpredict(M.net, ds, 'ExecutionEnvironment', 'auto');
    if isa(Y, 'dlarray'), Y = extractdata(Y); end
    s = double(Y(:, M.positiveIndex));
    y = logical(T_.referable);
    rows(end+1, :) = {char(nm), numel(y), sum(y), ...
        mean(s(y) >= M.threshold), mean(s(~y) < M.threshold), iAuc(y, s), iSe(y)}; %#ok<AGROW>
end

T = cell2table(rows, 'VariableNames', ...
    {'split', 'n', 'referable', 'sensitivity', 'specificity', 'auc', 'aucStdErr'});

fprintf('\nthreshold %.4f, taken from validation (test was not consulted for it)\n\n', M.threshold);
fprintf('%-12s%7s%9s%9s%9s%11s\n', 'eval set', 'n', 'sens', 'spec', 'AUC', 'AUC s.e.');
fprintf('%s\n', repmat('-', 1, 58));
for i = 1:height(T)
    fprintf('%-12s%7d%8.1f%%%8.1f%%%9.3f%11.3f\n', T.split{i}, T.n(i), ...
        100*T.sensitivity(i), 100*T.specificity(i), T.auc(i), T.aucStdErr(i));
end
fprintf('%s\n', repmat('-', 1, 58));
fprintf(['\nTargets: sensitivity > 90%%, specificity > 85%%.\n' ...
         'Frozen-backbone baseline to beat: IDRiD AUC 0.887, APTOS 0.972.\n' ...
         'A gain smaller than the s.e. above is not a gain.\n\n']);

fid = fopen(logf, 'a');
if fid > 0
    fprintf(fid, 'REVEAL %s  model=%s  IDRiD auc=%.4f sens=%.3f spec=%.3f  APTOS auc=%.4f\n', ...
        datestr(now, 'yyyy-mm-dd HH:MM:SS'), modelFile, ...
        T.auc(1), T.sensitivity(1), T.specificity(1), T.auc(2)); %#ok<TNOW1,DATST>
    fclose(fid);
end
end

% =====================================================================
function a = iAuc(y, s)
y = logical(y(:)); s = s(:);
np = sum(y); nn = sum(~y);
if np == 0 || nn == 0, a = NaN; return; end
r = tiedrank(s);
a = (sum(r(y)) - np*(np+1)/2) / (np*nn);
end

function se = iSe(y)
%ISE Hanley-McNeil standard error at AUC 0.9, the scale we operate at.
y = logical(y(:)); np = sum(y); nn = sum(~y);
if np == 0 || nn == 0, se = NaN; return; end
a = 0.90; q1 = a/(2-a); q2 = 2*a^2/(1+a);
se = sqrt((a*(1-a) + (np-1)*(q1-a^2) + (nn-1)*(q2-a^2)) / (np*nn));
end
