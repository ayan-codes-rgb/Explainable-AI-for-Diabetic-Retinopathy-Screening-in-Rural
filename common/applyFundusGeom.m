function out = applyFundusGeom(A, geom, varargin)
%APPLYFUNDUSGEOM Replay loadFundus's geometry onto a mask or annotation image.
%
%   out = APPLYFUNDUSGEOM(A, geom)
%   out = APPLYFUNDUSGEOM(A, geom, 'Interp', 'nearest')
%
%   loadFundus crops, pads and resizes. Ground truth that does not go
%   through the identical sequence ends up misaligned with its image by tens
%   of pixels, which silently destroys every segmentation metric you compute
%   -- the numbers still come out, they are just wrong. Use this for IDRiD
%   lesion masks, DRIVE vessel masks, e-ophtha masks and any label image.
%
%   A must be at the ORIGINAL image resolution (geom.originalSize); the
%   function checks and errors rather than guessing.
%
%   Interpolation defaults to geom.interpMask ('nearest'), which is what you
%   want for anything with label semantics. Only pass 'bilinear' for a
%   genuinely continuous map, such as a probability heatmap.
%
%   See also LOADFUNDUS, MAPFUNDUSPOINTS.

p = inputParser;
p.FunctionName = 'applyFundusGeom';
p.addParameter('Interp', '', @(x) ischar(x) || isstring(x));
p.parse(varargin{:});

interp = char(p.Results.Interp);
if isempty(interp)
    interp = geom.interpMask;
end

sz = [size(A, 1) size(A, 2)];
if ~isequal(sz, geom.originalSize)
    error('applyFundusGeom:sizeMismatch', ...
        ['Annotation is %dx%d but its image was %dx%d before loading. Pass ' ...
         'the annotation at its original resolution, not a pre-resized one.'], ...
        sz(1), sz(2), geom.originalSize(1), geom.originalSize(2));
end

wasLogical = islogical(A);

% 1. crop  (same bbox loadFundus used)
b   = geom.bbox;
out = A(b(1):b(1)+b(3)-1, b(2):b(2)+b(4)-1, :);

% 2. pad to square  (same amounts)
pd  = geom.pad;                                  % [top bottom left right]
out = padarray(out, [pd(1) pd(3)], 0, 'pre');
out = padarray(out, [pd(2) pd(4)], 0, 'post');

% 3. resize
out = imresize(out, geom.targetSize, interp);

if wasLogical
    out = logical(out);
end
end
