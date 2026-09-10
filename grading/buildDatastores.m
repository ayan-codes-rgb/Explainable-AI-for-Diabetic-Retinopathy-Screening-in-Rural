function S = buildDatastores(roots, varargin)
%BUILDDATASTORES Datastores for DR grading, wired to the fixed split.
%
%   S = BUILDDATASTORES(dataRoot)
%   S = BUILDDATASTORES(dataRoot, 'Name', Value, ...)
%
%   roots is a struct of dataset roots, e.g.
%       roots.IDRiD = 'C:\dr-data\IDRiD\B. Disease Grading';
%       roots.APTOS = 'C:\dr-data\APTOS';
%   A plain string is treated as the IDRiD root (APTOS rows dropped).
%
%   Options
%     'InputSize'  [r c], default [224 224]. 224 keeps CPU work sane; raise
%                  it once training moves to a GPU machine.
%     'Task'       'grade' (5-class, default) | 'referable' (binary)
%     'Augment'    true (default) | false. Training set only -- never
%                  augment val or test, or your metrics stop meaning anything.
%
%   Returns a struct with .train, .val, .test, each holding
%     .ds      datastore ready for training/inference
%     .labels  categorical response
%     .tbl     the split table (grade, referable, dme_risk, file, ...)
%   plus .inputSize, .classes and .task.
%
%   Every image is read through loadFundus via makeFundusReadFcn, so the
%   network is trained on exactly what the integrated pipeline will feed it
%   at inference time.
%
%   Example
%     roots.IDRiD = 'C:\dr-data\IDRiD\B. Disease Grading';
%     roots.APTOS = 'C:\dr-data\APTOS';
%     S = buildDatastores(roots);
%     numel(S.train.labels)
%
%   See also LOADSPLIT, MAKEFUNDUSREADFCN, CLASSWEIGHTS.

p = inputParser;
p.FunctionName = 'buildDatastores';
p.addParameter('InputSize', [224 224], @(x) isnumeric(x) && numel(x) == 2);
p.addParameter('Task', 'grade', @(x) ischar(x) || isstring(x));
p.addParameter('Augment', true, @(x) islogical(x) && isscalar(x));
p.addParameter('Enhance', 'none', @(x) ischar(x) || isstring(x));
p.parse(varargin{:});
opt = p.Results;

task = lower(char(opt.Task));
if ~ismember(task, {'grade', 'referable'})
    error('buildDatastores:badTask', 'Task must be ''grade'' or ''referable''.');
end

inputSize = double(opt.InputSize(:)');
readFcn   = makeGradingReadFcn('TargetSize', inputSize, 'Enhance', opt.Enhance);

if task == "grade"
    classes = categorical(0:4, 0:4, {'0','1','2','3','4'});
else
    classes = categorical([0 1], [0 1], {'nonreferable','referable'});
end

S = struct('inputSize', inputSize, 'task', task, 'classes', classes, ...
           'enhance', lower(char(opt.Enhance)));

for nm = ["train" "val" "test" "test_aptos"]
    T = loadSplit(char(nm), roots);

    if task == "grade"
        Y = categorical(T.grade, 0:4, {'0','1','2','3','4'});
    else
        Y = categorical(double(T.referable), [0 1], {'nonreferable','referable'});
    end

    imds = imageDatastore(cellstr(T.file), 'ReadFcn', readFcn);
    imds.Labels = Y;

    if nm == "train" && opt.Augment
        % No canonical orientation in a fundus photo beyond which eye it is,
        % so reflections and modest rotations produce plausible retinas.
        % Keep it mild -- heavy warping destroys lesion shape, which is the
        % feature that separates a microaneurysm from a haemorrhage.
        % A fundus photo has no canonical orientation beyond which eye it is,
        % so full 360-degree rotation is legitimate here and gives far more
        % variety than the +/-15 degrees a natural-image pipeline would use.
        % Brightness and contrast jitter simulates the camera-to-camera
        % variation that is the whole point of the deployment scenario.
        aug = imageDataAugmenter( ...
            'RandXReflection',      true, ...
            'RandYReflection',      true, ...
            'RandRotation',         [0 360], ...
            'RandScale',            [0.85 1.15], ...
            'RandXTranslation',     [-10 10], ...
            'RandYTranslation',     [-10 10]);
        ds = augmentedImageDatastore(inputSize, imds, 'DataAugmentation', aug);
    else
        ds = augmentedImageDatastore(inputSize, imds);
    end

    S.(nm) = struct('ds', ds, 'imds', imds, 'labels', Y, 'tbl', T);
end

fprintf(['buildDatastores: task=%s  input=%dx%d  enhance=%s\n' ...
         '  train=%d  val=%d  test(IDRiD)=%d  test_aptos=%d\n'], ...
    task, inputSize(1), inputSize(2), S.enhance, ...
    numel(S.train.labels), numel(S.val.labels), ...
    numel(S.test.labels), numel(S.test_aptos.labels));
end
