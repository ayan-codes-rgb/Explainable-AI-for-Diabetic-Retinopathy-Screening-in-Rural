function xyOut = mapFundusPoints(xyIn, geom)
%MAPFUNDUSPOINTS Map (x,y) coordinates from the original image into loadFundus space.
%
%   xyOut = MAPFUNDUSPOINTS(xyIn, geom)
%
%   xyIn is Nx2 as [x y], i.e. [column row], 1-based, in ORIGINAL image
%   coordinates. That is how IDRiD ships optic-disc and fovea centres and
%   how e-ophtha lists microaneurysm locations, so this is the function that
%   keeps those annotations usable after resizing.
%
%   Points outside the retina are still returned rather than dropped -- an
%   annotation can legitimately sit a pixel outside the detected mask. Test
%   them against the returned fovMask yourself if that matters.
%
%   See also LOADFUNDUS, APPLYFUNDUSGEOM.

validateattributes(xyIn, {'numeric'}, {'2d', 'ncols', 2}, mfilename, 'xyIn');

x = double(xyIn(:, 1));
y = double(xyIn(:, 2));

% 1. crop: shift so the bbox origin becomes (1,1)
x = x - geom.bbox(2) + 1;
y = y - geom.bbox(1) + 1;

% 2. pad: shift by the leading pad
x = x + geom.pad(3);      % left
y = y + geom.pad(1);      % top

% 3. resize. imresize maps pixel CENTRES, so convert to a 0-based centre
%    coordinate, scale, then convert back. Writing x*scale directly is the
%    classic half-pixel error -- small here, but it is exactly the size of
%    the microaneurysms M3 has to hit.
x = (x - 0.5) * geom.scale(2) + 0.5;
y = (y - 0.5) * geom.scale(1) + 0.5;

xyOut = [x y];
end
