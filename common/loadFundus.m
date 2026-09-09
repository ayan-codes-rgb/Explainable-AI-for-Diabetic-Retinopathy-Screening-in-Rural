function [img, fovMask, geom] = loadFundus(src, varargin)
%LOADFUNDUS Canonical fundus image loader. Every module reads images through this.
%
%   [img, fovMask, geom] = LOADFUNDUS(pathOrArray)
%   [...]                = LOADFUNDUS(pathOrArray, 'Name', Value, ...)
%
%   DO NOT CALL imread() DIRECTLY ANYWHERE ELSE IN THIS PROJECT.
%   Channel order, bit depth and resolution drifting apart between modules
%   is the classic way a six-person imaging project loses two days: by the
%   time it surfaces it looks like an algorithm bug, not a loading bug.
%
%   Guarantees
%     img       HxWx3 uint8, RGB channel order, 0-255, pixels outside the
%               retina forced to 0
%     fovMask   HxW logical, true inside the retina, same size as img
%     geom      struct recording every geometric operation applied, so the
%               matching ground truth can be put through the identical
%               transform -- see APPLYFUNDUSGEOM (masks) and
%               MAPFUNDUSPOINTS (coordinates)
%
%   What it does, in order
%     1. reads the file (or accepts an array you already have in memory)
%     2. forces 3-channel RGB uint8: grayscale is replicated, an alpha
%        channel is dropped, uint16 is scaled down, double is handled
%        whether it is stored 0-1 or 0-255
%     3. finds the retina and crops the black border away
%     4. pads the crop to a SQUARE before resizing, so aspect ratio is
%        preserved. Resizing a 4:3 fundus straight to 512x512 squashes the
%        retina into an ellipse, which corrupts vessel geometry, lesion
%        shape features and optic-disc circularity all at once
%     5. resizes to cfg.imageSize -- bilinear for the image, nearest for
%        the mask
%
%   Options
%     'TargetSize'  [rows cols]   default drConfig().imageSize
%     'Crop'        true | false  default true; false keeps the full frame
%     'Double'      true | false  default false; true returns double in [0,1]
%
%   Example
%     [im, fov, g] = loadFundus('data/IDRiD/images/IDRiD_001.jpg');
%     maskAligned  = applyFundusGeom(imread('IDRiD_001_MA.tif') > 0, g);
%     odAligned    = mapFundusPoints([odX odY], g);
%
%   See also DRCONFIG, FUNDUSFOVMASK, APPLYFUNDUSGEOM, MAPFUNDUSPOINTS,
%   MAKEFUNDUSREADFCN.

% ---- options ---------------------------------------------------------
p = inputParser;
p.FunctionName = 'loadFundus';
p.addParameter('TargetSize', [], @(x) isempty(x) || (isnumeric(x) && numel(x) == 2));
p.addParameter('Crop',   true,  @(x) islogical(x) && isscalar(x));
p.addParameter('Double', false, @(x) islogical(x) && isscalar(x));
p.parse(varargin{:});
opt = p.Results;

cfg = drConfig();
tgt = opt.TargetSize;
if isempty(tgt)
    tgt = cfg.imageSize;
end
tgt = double(tgt(:)');

% ---- 1. read ---------------------------------------------------------
if ischar(src) || isstring(src)
    srcPath = char(src);
    if exist(srcPath, 'file') ~= 2
        error('loadFundus:fileNotFound', 'Image not found: %s', srcPath);
    end
    raw = imread(srcPath);
else
    srcPath = '<array>';
    raw = src;
end

% ---- 2. canonical RGB uint8 -----------------------------------------
raw      = iToRgbUint8(raw);
origSize = [size(raw, 1) size(raw, 2)];

% ---- 3. retina mask, then crop the black border ---------------------
fov = fundusFOVMask(raw, cfg.fovThreshold);

if mean(fov(:)) < cfg.minFovArea
    warning('loadFundus:tinyFOV', ...
        ['Retina covers only %.1f%% of %s. The image is probably very dark ' ...
         'or corrupt -- falling back to the full frame. Flag this file for ' ...
         'the quality module.'], 100 * mean(fov(:)), srcPath);
    fov = true(origSize);
end

if opt.Crop
    rows = find(any(fov, 2));
    cols = find(any(fov, 1));
    bbox = [rows(1), cols(1), rows(end) - rows(1) + 1, cols(end) - cols(1) + 1];
else
    bbox = [1, 1, origSize(1), origSize(2)];
end

r0 = bbox(1); c0 = bbox(2); rh = bbox(3); cw = bbox(4);
img     = raw(r0:r0+rh-1, c0:c0+cw-1, :);
fovMask = fov(r0:r0+rh-1, c0:c0+cw-1);

% ---- 4. pad to square, so the resize cannot distort -----------------
side = max(rh, cw);
padT = floor((side - rh) / 2);   padB = side - rh - padT;
padL = floor((side - cw) / 2);   padR = side - cw - padL;

img     = padarray(img,     [padT padL], 0,     'pre');
img     = padarray(img,     [padB padR], 0,     'post');
fovMask = padarray(fovMask, [padT padL], false, 'pre');
fovMask = padarray(fovMask, [padB padR], false, 'post');

% ---- 5. resize -------------------------------------------------------
img     = imresize(img,     tgt, cfg.interpImage);
fovMask = imresize(fovMask, tgt, cfg.interpMask);

% Bilinear resizing smears retina pixels a little way past the edge; clear
% that fringe so nobody detects a lesion in it.
img = bsxfun(@times, img, uint8(fovMask));

% ---- geometry record -------------------------------------------------
geom = struct( ...
    'sourcePath',   srcPath, ...
    'originalSize', origSize, ...
    'bbox',         bbox, ...                  % [row0 col0 height width], ORIGINAL coords
    'pad',          [padT padB padL padR], ... % [top bottom left right]
    'squareSide',   side, ...
    'targetSize',   tgt, ...
    'scale',        tgt ./ [side side], ...    % [rowScale colScale]
    'cropped',      opt.Crop, ...
    'interpImage',  cfg.interpImage, ...
    'interpMask',   cfg.interpMask);

if opt.Double
    img = im2double(img);
end
end

% =====================================================================
function out = iToRgbUint8(a)
%ITORGBUINT8 Force any imread() output into HxWx3 uint8 RGB.

if ndims(a) > 3
    error('loadFundus:badDims', 'Expected a 2-D or 3-D image, got %d-D.', ndims(a));
end

if size(a, 3) == 4
    a = a(:, :, 1:3);            % drop alpha (some PNG exports carry one)
end

if ismatrix(a)
    a = repmat(a, 1, 1, 3);      % grayscale -> RGB, so the contract stays simple
end

if size(a, 3) ~= 3
    error('loadFundus:badChannels', 'Expected 1, 3 or 4 channels, got %d.', size(a, 3));
end

switch class(a)
    case 'uint8'
        out = a;
    case {'uint16', 'int16'}
        out = im2uint8(a);       % 16-bit TIFFs turn up in DRIVE / e-ophtha
    case 'logical'
        out = im2uint8(a);
    case {'single', 'double'}
        if max(a(:)) > 1
            a = a / 255;         % stored 0-255 as double
        end
        out = im2uint8(min(max(a, 0), 1));
    otherwise
        out = im2uint8(a);
end
end
