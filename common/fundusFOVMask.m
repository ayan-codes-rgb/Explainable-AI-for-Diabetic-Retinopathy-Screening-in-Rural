function mask = fundusFOVMask(rgb, thr)
%FUNDUSFOVMASK Binary mask of the circular retina, excluding the black border.
%
%   mask = FUNDUSFOVMASK(rgb)        uses the threshold from drConfig
%   mask = FUNDUSFOVMASK(rgb, thr)   overrides it (fraction of full scale, 0-1)
%
%   A fundus camera images a circle onto a rectangular sensor, so every raw
%   image has black corners. Every module needs to know where they are:
%     M1     do not measure focus/illumination over the black border, or the
%            statistics are dominated by pixels that carry no information
%     M2/M3  the border edge is a hard, high-contrast circle -- it is the
%            single biggest false positive source for vessel and lesion
%            detectors
%     M4     border pixels are constant across the dataset and waste capacity
%
%   The mask is the largest bright connected component, hole-filled, so a
%   dark fovea or a dark haemorrhage inside the retina never punches a hole
%   through it.

if nargin < 2 || isempty(thr)
    thr = drConfig().fovThreshold;
end

% Brightest channel per pixel. The red channel saturates inside the retina
% while the background stays near zero in all three, so max() over channels
% gives the cleanest separation across different cameras.
v = im2double(max(rgb, [], 3));

mask = v > thr;

if ~any(mask(:))
    % Fully black or corrupt file. Return everything and let the caller
    % detect it via the area check rather than erroring out down here.
    mask = true(size(v));
    return
end

mask = imfill(mask, 'holes');
mask = imopen(mask, strel('disk', 5));   % drop speckle, timestamps, thin bridges

if ~any(mask(:))
    mask = true(size(v));
    return
end

mask = bwareafilt(mask, 1);              % keep the retina only
mask = imfill(mask, 'holes');
end
