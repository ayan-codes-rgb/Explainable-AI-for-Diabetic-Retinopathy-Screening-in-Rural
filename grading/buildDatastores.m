function S = buildDatastores(dataRoot, varargin)
%BUILDDATASTORES Datastores for DR grading, wired to the fixed split.
%
%   S = BUILDDATASTORES(dataRoot)
%   S = BUILDDATASTORES(dataRoot, 'Name', Value, ...)
%
%   dataRoot is the unzipped "B. Disease Grading" folder.
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
%     S = buildDatastores('C:\dr-data\IDRiD\B. Disease Grading');
%     numel(S.train.labels)
%
%   See also LOADSPLIT, MAKEFUNDUSREADFCN, CLASSWEIGHTS.

p = inputParser;
p.FunctionName = 'buildDatastores';
p.addParameter('InputSize', [224 224], @(x) isnumeric(x) && numel(x) == 2);
p.addParameter('Task', 'grade', @(x) ischar(x) || isstring(x));
p.addParameter('Augment', true, @(x) islogical(x) && isscalar(x));
p.parse(varargin{:});
opt = p.Results;

task = lower(char(opt.Task));
if ~ismember(task, {'grade', 'referable'})
    error('buildDatastores:badTask', 'Task must be ''grade'' or ''referable''.');
end

inputSize = double(opt.InputSize(:)');
readFcn   = makeFundusReadFcn('TargetSize', inputSize);

if task == "grade"
    classes = categorical(0:4, 0:4, {'0','1','2','3','4'});
else
    classes = categorical([0 1], [0 1], {'nonreferable','referable'});
end

S = struct('inputSize', inputSize, 'task', task, 'classes', classes);

for nm = ["train" "val" "test"]
    T = loadSplit(char(nm), dataRoot);

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
        aug = imageDataAugmenter( ...
            'RandXReflection', true, ...
            'RandYReflection', true, ...
            'RandRotation',    [-15 15], ...
            'RandScale',       [0.9 1.1]);
        ds = augmentedImageDatastore(inputSize, imds, 'DataAugmentation', aug);
    else
        ds = augmentedImageDatastore(inputSize, imds);
    end

    S.(nm) = struct('ds', ds, 'imds', imds, 'labels', Y, 'tbl', T);
end

fprintf('buildDatastores: task=%s  input=%dx%d  train=%d  val=%d  test=%d\n', ...
    task, inputSize(1), inputSize(2), ...
    numel(S.train.labels), numel(S.val.labels), numel(S.test.labels));
end
