function cfg = drConfig(varargin)
%DRCONFIG Central configuration for the DR screening pipeline.
%
%   cfg = DRCONFIG() returns the shared settings every module must use.
%   cfg = DRCONFIG('Name',Value,...) returns those settings with overrides
%   applied, for one-off experiments.
%
%   Do NOT hardcode any of these values anywhere else. If your module needs
%   a different resolution, pass it explicitly at the call site so it is
%   visible in a diff -- a silently different default is exactly how six
%   modules end up disagreeing about what an image is.
%
%   Fields
%     imageSize     [rows cols] canonical working resolution for all modules
%     fovThreshold  intensity fraction (0-1) separating retina from the black
%                   border, used by fundusFOVMask
%     interpImage   interpolation used when resizing intensity images
%     interpMask    interpolation used when resizing label/binary masks.
%                   Keep this 'nearest' -- anything else averages label
%                   values into values that do not exist.
%     minFovArea    if the detected retina covers less than this fraction of
%                   the frame, treat the detection as failed (guards against
%                   very dark or corrupt files)
%     projectRoot   repo root, derived from this file's own location
%     dataRoot      where datasets live (kept out of git)
%
%   See also LOADFUNDUS, APPLYFUNDUSGEOM, MAPFUNDUSPOINTS, MAKEFUNDUSREADFCN.

cfg = struct( ...
    'imageSize',    [512 512], ...
    'fovThreshold', 0.06, ...
    'interpImage',  'bilinear', ...
    'interpMask',   'nearest', ...
    'minFovArea',   0.05);

thisDir         = fileparts(mfilename('fullpath'));
cfg.projectRoot = fileparts(thisDir);
cfg.dataRoot    = fullfile(cfg.projectRoot, 'data');

% ---- name/value overrides -------------------------------------------
if mod(numel(varargin), 2) ~= 0
    error('drConfig:badArgs', 'Overrides must be Name,Value pairs.');
end
for k = 1:2:numel(varargin)
    name = varargin{k};
    if ~(ischar(name) || isstring(name))
        error('drConfig:badArgs', 'Override names must be text.');
    end
    name = char(name);
    if ~isfield(cfg, name)
        error('drConfig:unknownOption', ...
            'Unknown setting "%s". Valid settings: %s', ...
            name, strjoin(fieldnames(cfg)', ', '));
    end
    cfg.(name) = varargin{k+1};
end
end
