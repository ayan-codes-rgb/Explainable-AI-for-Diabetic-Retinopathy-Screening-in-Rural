function test_loadFundus(imageFolder)
%TEST_LOADFUNDUS Smoke test and visual check for the shared fundus loader.
%
%   test_loadFundus()                      synthetic test only -- run this
%                                          today, before the datasets have
%                                          finished downloading
%   test_loadFundus('data/IDRiD/images')   also runs over real files
%
%   Passing means two things: the loader honours its contract (uint8, RGB,
%   fixed size), and a ground-truth mask put through applyFundusGeom still
%   lands on the same feature it marked in the original image. The second
%   one is the check that actually matters -- a misaligned mask does not
%   throw an error, it just quietly ruins every metric computed from it.

cfg = drConfig();

fprintf('\n== synthetic test ==\n');

% ---- build a fake fundus: off-centre circle on a non-square frame ----
H = 1400; W = 2000;
[X, Y] = meshgrid(1:W, 1:H);
cx = 980; cy = 700; r = 660;
retina = (X - cx).^2 + (Y - cy).^2 <= r^2;

rgb = zeros(H, W, 3, 'uint8');
rgb(:,:,1) = uint8(retina) * 200;         % red-dominant, like a real fundus
rgb(:,:,2) = uint8(retina) * 90;
rgb(:,:,3) = uint8(retina) * 60;

% a bright square "lesion" we can track through the whole transform
lx = cx + 300; ly = cy - 200; half = 12;
rgb(ly-half:ly+half, lx-half:lx+half, :) = 255;

gt = false(H, W);
gt(ly-half:ly+half, lx-half:lx+half) = true;

% ---- run the loader --------------------------------------------------
[img, fov, geom] = loadFundus(rgb);

assert(isa(img, 'uint8'), 'img must be uint8, got %s', class(img));
assert(size(img, 3) == 3, 'img must have 3 channels, got %d', size(img, 3));
assert(isequal([size(img,1) size(img,2)], cfg.imageSize), ...
    'img must be %dx%d, got %dx%d', cfg.imageSize(1), cfg.imageSize(2), ...
    size(img,1), size(img,2));
assert(islogical(fov), 'fovMask must be logical, got %s', class(fov));
assert(isequal(size(fov), cfg.imageSize), 'fovMask must match img size');
fprintf('  contract ....... ok  (%s, %dx%dx%d)\n', ...
    class(img), size(img,1), size(img,2), size(img,3));

% the crop should have thrown most of the black frame away
assert(mean(fov(:)) > 0.6, ...
    'Retina should fill most of the cropped frame, got %.2f', mean(fov(:)));
fprintf('  crop ........... ok  (retina fills %.0f%% of frame, was %.0f%%)\n', ...
    100*mean(fov(:)), 100*mean(retina(:)));

% ---- mask alignment: the part that matters --------------------------
gtOut = applyFundusGeom(gt, geom);
assert(islogical(gtOut), 'mask must stay logical');
assert(any(gtOut(:)), 'lesion mask vanished during the transform');

green   = img(:,:,2);
underMask = mean(double(green(gtOut)));
assert(underMask > 200, ...
    ['Mask and image are misaligned: mean intensity under the mask is %.0f, ' ...
     'expected >200 because the lesion is white.'], underMask);
fprintf('  mask replay .... ok  (mean intensity under mask %.0f/255)\n', underMask);

% ---- point mapping ---------------------------------------------------
ptOut = mapFundusPoints([lx ly], geom);
st    = regionprops(gtOut, 'Centroid');
ctr   = st(1).Centroid;                        % [x y]
err   = hypot(ptOut(1) - ctr(1), ptOut(2) - ctr(2));
assert(err < 2, 'Point mapping is off by %.2f px (expected < 2)', err);
fprintf('  point mapping .. ok  (%.2f px from the mask centroid)\n', err);

fprintf('  SYNTHETIC TEST PASSED\n');

% ---- optional: run over real files ----------------------------------
if nargin < 1 || isempty(imageFolder)
    fprintf('\nNo image folder given, so the real-data pass was skipped.\n');
    fprintf('Once IDRiD has downloaded:  test_loadFundus(''<path>/IDRiD/images'')\n\n');
    return
end

fprintf('\n== real data: %s ==\n', imageFolder);
ds = imageDatastore(imageFolder, 'IncludeSubfolders', true);
n  = min(12, numel(ds.Files));
if n == 0
    warning('test_loadFundus:noFiles', 'No readable images found in %s', imageFolder);
    return
end

rows  = cell(n, 1);
tiles = cell(n, 1);
for k = 1:n
    f  = ds.Files{k};
    t0 = tic;
    [im, fv, g] = loadFundus(f);
    dt = toc(t0);
    [~, nm, ex] = fileparts(f);
    rows{k}  = { string([nm ex]), ...
                 sprintf('%dx%d', g.originalSize(1), g.originalSize(2)), ...
                 sprintf('%dx%d', g.bbox(3), g.bbox(4)), ...
                 round(100*mean(fv(:))), ...
                 round(1000*dt) };
    tiles{k} = im;
end

T = cell2table(vertcat(rows{:}), 'VariableNames', ...
    {'file', 'original', 'afterCrop', 'retinaPct', 'ms'});
disp(T);

figure('Name', 'loadFundus output');
montage(cat(4, tiles{:}));
title(sprintf('loadFundus @ %dx%d', cfg.imageSize(1), cfg.imageSize(2)));

fprintf(['\nEyeball the montage before you trust it: every retina centred, ' ...
         'no black bars\nleft or right, nothing squashed into an ellipse.\n\n']);
end
