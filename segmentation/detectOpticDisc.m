function [center, radius] = detectOpticDisc(img, fovMask)
    redCh = img(:,:,1);

    masked = redCh;
    masked(~fovMask) = 0;
    thresh = prctile(masked(fovMask), 97);
    bw = masked >= thresh;
    bw = bwareaopen(bw, 200);
    cc = bwconncomp(bw);
    stats = regionprops(cc, 'Area', 'Centroid', 'EquivDiameter');

    if isempty(stats)
        center = [NaN NaN];
        radius = NaN;
        return;
    end

    imgWidth = size(redCh, 1);
    plausible = [stats.EquivDiameter] > 0.12*imgWidth & [stats.EquivDiameter] < 0.21*imgWidth;
    candidates = stats(plausible);

    if isempty(candidates)
        center = [NaN NaN];
        radius = NaN;
    else
        [~, idx] = max([candidates.Area]);
        center = candidates(idx).Centroid;
        radius = candidates(idx).EquivDiameter / 2;
    end
end