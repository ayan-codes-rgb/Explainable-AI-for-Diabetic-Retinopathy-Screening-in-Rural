function fcn = makeFundusReadFcn(varargin)
%MAKEFUNDUSREADFCN ReadFcn handle for imageDatastore, backed by loadFundus.
%
%   ds = imageDatastore(folder, ...
%            'IncludeSubfolders', true, ...
%            'LabelSource',       'foldernames', ...
%            'ReadFcn',           makeFundusReadFcn());
%
%   ds = imageDatastore(..., 'ReadFcn', makeFundusReadFcn('TargetSize', [224 224]));
%
%   This exists so the grading network is trained on exactly the
%   preprocessing the rest of the pipeline produces at inference time. Train
%   on raw imread() output, then feed the integrated pipeline's loadFundus
%   output into the same network, and accuracy drops for no visible reason.
%
%   Only the image is returned, because imageDatastore expects a single
%   output. If you need the FOV mask or the geometry during training, call
%   loadFundus directly inside a transform() or combine() datastore.
%
%   See also LOADFUNDUS, DRCONFIG.

opts = varargin;
fcn  = @(filename) iRead(filename, opts);
end

function img = iRead(filename, opts)
img = loadFundus(filename, opts{:});
end
