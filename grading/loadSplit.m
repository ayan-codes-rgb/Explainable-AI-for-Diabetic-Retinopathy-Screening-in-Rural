function T = loadSplit(splitName, dataRoot)
%LOADSPLIT Read the fixed IDRiD train/val/test split.
%
%   T = LOADSPLIT()                          every row, no file paths
%   T = LOADSPLIT('train', dataRoot)         training rows, with full paths
%   T = LOADSPLIT({'train','val'}, dataRoot) more than one split at once
%
%   dataRoot is the folder that CONTAINS "1. Original Images" -- i.e. the
%   unzipped "B. Disease Grading" folder, wherever you put it. The images
%   are not in this repo, so the path is yours to supply.
%
%   Columns
%     id          unique row id. Image names REPEAT between the official
%                 training and testing folders (both contain IDRiD_001.jpg),
%                 so never key on image_name alone.
%     dataset     'IDRiD'
%     split       'train' | 'val' | 'test'
%     image_name  e.g. IDRiD_001
%     relpath     path relative to dataRoot
%     grade       0-4, International Clinical DR severity scale
%     referable   true when grade >= 2 (this is the screening decision)
%     dme_risk    0-2, risk of macular oedema (a second, independent label)
%     file        full path -- added only when dataRoot is supplied
%
%   THIS SPLIT IS A FIXED ARTIFACT. Do not regenerate it.
%   It was produced once by a stratified holdout -- 15% of the official
%   413-image training set, seed 20260909 -- and committed so that every
%   run, every teammate and every reported number refers to the same
%   partition. Regenerate it and your results stop being comparable with
%   anyone else's, including your own from yesterday.
%
%   The official 103-image test set is untouched and must stay that way
%   until final evaluation. Tune on val, report on test, once.
%
%   Example
%     root = 'C:\dr-data\IDRiD\B. Disease Grading';
%     tr   = loadSplit('train', root);
%     imds = imageDatastore(tr.file, 'ReadFcn', makeFundusReadFcn());
%     Y    = categorical(tr.grade);
%
%   See also MAKEFUNDUSREADFCN, LOADFUNDUS.

here   = fileparts(mfilename('fullpath'));
csvPath = fullfile(here, 'idrid_split.csv');
if exist(csvPath, 'file') ~= 2
    error('loadSplit:missingCsv', 'Split file not found: %s', csvPath);
end

T = readtable(csvPath, 'TextType', 'string');
T.referable = logical(T.referable);

% ---- filter by split -------------------------------------------------
if nargin >= 1 && ~isempty(splitName)
    want  = string(splitName);
    valid = ["train" "val" "test"];
    bad   = want(~ismember(want, valid));
    if ~isempty(bad)
        error('loadSplit:badSplit', ...
            'Unknown split "%s". Valid: train, val, test.', bad(1));
    end
    T = T(ismember(T.split, want), :);
end

% ---- attach full paths -----------------------------------------------
if nargin >= 2 && ~isempty(dataRoot)
    rel    = strrep(T.relpath, "/", filesep);
    T.file = fullfile(string(dataRoot), rel);

    present = arrayfun(@(f) isfile(f), T.file);
    if ~any(present)
        error('loadSplit:wrongDataRoot', ...
            ['No images found under "%s".\nExpected to find, for example:\n  %s\n' ...
             'dataRoot should be the unzipped "B. Disease Grading" folder.'], ...
            dataRoot, T.file(1));
    elseif ~all(present)
        error('loadSplit:missingImages', ...
            '%d of %d images are missing under "%s". First missing: %s', ...
            sum(~present), height(T), dataRoot, T.file(find(~present, 1)));
    end
end
end
