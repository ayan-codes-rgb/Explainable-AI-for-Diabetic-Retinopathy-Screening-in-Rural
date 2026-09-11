function vesselMask = detectVessels(img, fovMask)
% detectVessels  Binary vessel segmentation via morphological top-hat.
%   Uses green channel top-hat with percentile thresholding.
%   Qualitative validation only (no IDRiD vessel ground truth).

    g = uint8(img(:,:,2));

    % Top-hat on inverted green — vessels are darker than background
    % Disk size ~12: larger than vessel width, smaller than OD/exudates
    se = strel('disk', 12);
    tophat = double(imtophat(imcomplement(g), se));

    % Zero out outside FOV
    tophat(~fovMask) = 0;

    % Percentile threshold — vessels occupy ~10-15% of retinal area
    % Otsu fails here (distribution is not bimodal)
    thresh = prctile(tophat(fovMask), 85);  % top 15% of responses
    vesselMask = tophat > thresh;

    % Morphological cleanup
    vesselMask = bwareaopen(vesselMask, 40);       % drop tiny noise blobs
    vesselMask = logical(vesselMask);
end