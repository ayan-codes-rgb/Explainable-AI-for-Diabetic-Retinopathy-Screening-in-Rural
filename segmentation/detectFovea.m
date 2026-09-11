function foveaXY = detectFovea(img, discCenter, discRadius, fovMask)
% detectFovea  Localize fovea using darkest-in-directional-annulus.
%   Search is restricted to ±60° of the disc→image-centre direction
%   to exploit the anatomical constraint (fovea is always temporal to disc).

    g = double(img(:,:,2));
    [H, W] = size(g);

    if isnan(discCenter(1)) || isnan(discRadius) || discRadius < 5 || discRadius > 150
        foveaXY = [NaN NaN];
        return
    end

    % --- Directional prior: disc → image centre is approximately temporal ---
    imgCenter = [W/2, H/2];
    dirVec = imgCenter - discCenter;
    if norm(dirVec) < 1
        foveaXY = [NaN NaN];
        return
    end
    dirVec = dirVec / norm(dirVec);   % unit vector toward image centre

    % --- Per-pixel angle relative to that direction ---
    [X, Y] = meshgrid(1:W, 1:H);
    pixVecX = X - discCenter(1);
    pixVecY = Y - discCenter(2);
    pixNorm = sqrt(pixVecX.^2 + pixVecY.^2) + 1e-6;
    cosAngle = (pixVecX*dirVec(1) + pixVecY*dirVec(2)) ./ pixNorm;

    % --- Distance annulus ---
    dist = pixNorm;
    distFromBorder = bwdist(~fovMask);

    % Search = annulus ∩ ±60° cone ∩ interior of retina
    searchMask = dist >= 4.0*discRadius  ...
               & dist <= 6.0*discRadius  ...
               & cosAngle >= cosd(60)    ...   % within ±60° of temporal direction
               & fovMask                 ...
               & distFromBorder >= 25;

    if sum(searchMask(:)) < 50
        % Fall back: relax angle to ±90°
        searchMask = dist >= 4.0*discRadius  ...
                   & dist <= 6.0*discRadius  ...
                   & cosAngle >= cosd(90)    ...
                   & fovMask                 ...
                   & distFromBorder >= 25;
    end

    if sum(searchMask(:)) < 50
        foveaXY = [NaN NaN];
        return
    end

    % Heavy smoothing, then find minimum inside search zone
    gSmooth = imgaussfilt(g, 15);
    gSearch = gSmooth;
    gSearch(~searchMask) = Inf;

    [~, idx] = min(gSearch(:));
    [fy, fx] = ind2sub([H W], idx);
    foveaXY = [fx, fy];
end