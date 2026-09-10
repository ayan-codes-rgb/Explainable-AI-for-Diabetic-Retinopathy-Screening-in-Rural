function T = loadSplit(splitName, roots, varargin)
%LOADSPLIT Read the fixed train/val/test split (IDRiD + APTOS).
%
%   T = LOADSPLIT()                              every row, no file paths
%   T = LOADSPLIT('train', roots)                training rows, with paths
%   Splits: 'train' | 'val' | 'test' (IDRiD, the benchmark) | 'test_aptos'
%   T = LOADSPLIT({'train','val'}, roots)        more than one split
%   T = LOADSPLIT('val', roots, 'Dataset','IDRiD')   one dataset only
%
%   ROOTS -- where the images live on your disk
%     A struct with one field per dataset:
%         roots.IDRiD = 'C:\dr-data\IDRiD\B. Disease Grading';
%         roots.APTOS = 'C:\dr-data\APTOS';        % contains train_images\
%     A plain string is accepted for backwards compatibility and is treated
%     as the IDRiD root; APTOS rows are then dropped with a warning.
%
%   Columns
%     id          unique row id. Image names REPEAT between IDRiD's official
%                 training and testing folders, so never key on image_name.
%     dataset     'IDRiD' | 'APTOS'
%     split       'train' | 'val' | 'test'
%     grade       0-4, International Clinical DR severity scale
%     referable   true when grade >= 2 -- the screening decision
%     dme_risk    0-2, IDRiD only (empty for APTOS)
%     file        full path -- added only when roots is supplied
%
%   THE SPLIT IS A FIXED ARTIFACT. Do not regenerate it. IDRiD's assignments
%   were made first and have been preserved byte-for-byte since; APTOS was
%   appended later with the same seed and the same 15% validation fraction.
%   Regenerate it and your numbers stop being comparable with your own from
%   last week, let alone your teammates'.
%
%   TEST IS IDRiD ONLY -- 103 images, the official IDRiD test set, untouched.
%   APTOS contributes to train and val but never to test, so the reported
%   benchmark stays on Indian fundus cameras with expert consensus grades.
%
%   NOTE ON DOMAIN SHIFT. Validation is now mostly APTOS (611 rows, 42.7%
%   referable) while test is IDRiD (103 rows, 62.1% referable). Different
%   cameras, different disease prevalence. A threshold tuned on val may
%   transfer imperfectly; use 'Dataset','IDRiD' if you want to check.
%
%   See also BUILDDATASTORES, RUNBASELINE, MAKEFUNDUSREADFCN.

p = inputParser;
p.FunctionName = 'loadSplit';
p.addParameter('Dataset', '', @(x) ischar(x) || isstring(x) || iscellstr(x));
p.parse(varargin{:});

here    = fileparts(mfilename('fullpath'));
csvPath = fullfile(here, 'split.csv');
if exist(csvPath, 'file') ~= 2
    error('loadSplit:missingCsv', 'Split file not found: %s', csvPath);
end

T = readtable(csvPath, 'TextType', 'string');
T.referable = logical(T.referable);

% ---- filter by split -------------------------------------------------
if nargin >= 1 && ~isempty(splitName)
    want  = string(splitName);
    valid = ["train" "val" "test" "test_aptos"];
    bad   = want(~ismember(want, valid));
    if ~isempty(bad)
        error('loadSplit:badSplit', 'Unknown split "%s". Valid: train, val, test.', bad(1));
    end
    T = T(ismember(T.split, want), :);
end

% ---- filter by dataset -----------------------------------------------
wantDs = string(p.Results.Dataset);
if ~isempty(wantDs) && wantDs ~= ""
    T = T(ismember(T.dataset, wantDs), :);
end

if nargin < 2 || isempty(roots)
    return
end

% ---- normalise roots -------------------------------------------------
if ischar(roots) || isstring(roots)
    roots = struct('IDRiD', char(roots));
elseif ~isstruct(roots)
    error('loadSplit:badRoots', ...
        'roots must be a struct of dataset roots, or a string for IDRiD.');
end

present  = string(fieldnames(roots))';
needed   = unique(T.dataset)';
missing  = setdiff(needed, present);
if ~isempty(missing)
    warning('loadSplit:noRootFor', ...
        ['No root given for %s -- those %d rows are dropped. Supply ' ...
         'roots.%s to include them.'], missing(1), sum(T.dataset == missing(1)), missing(1));
    T = T(ismember(T.dataset, present), :);
end
if isempty(T)
    error('loadSplit:nothingLeft', 'No rows remain after filtering.');
end

% ---- attach full paths -----------------------------------------------
T.file = strings(height(T), 1);
for ds = unique(T.dataset)'
    m   = T.dataset == ds;
    rel = strrep(T.relpath(m), "/", filesep);
    T.file(m) = fullfile(string(roots.(char(ds))), rel);
end

present = arrayfun(@(f) isfile(f), T.file);
if ~any(present)
    error('loadSplit:wrongRoot', ...
        ['No images found. Expected for example:\n  %s\n' ...
         'IDRiD root = the unzipped "B. Disease Grading" folder.\n' ...
         'APTOS root = the folder CONTAINING train_images.'], T.file(1));
elseif ~all(present)
    bad = find(~present, 1);
    error('loadSplit:missingImages', ...
        '%d of %d images are missing. First: %s', ...
        sum(~present), height(T), T.file(bad));
end
end
