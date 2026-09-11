function [quality, focusScore, illumMean, illumStd, fovArea] = assessQuality(img, fovMask)
% ASSESSQUALITY - checks a fundus image for gradability
% img     : uint8 HxWx3 RGB image (output of loadFundus)
% fovMask : logical HxW mask, true inside the retina

    green = img(:,:,2);

    % --- focus check (blur via Laplacian variance) ---
    lap = fspecial('laplacian');
    response = imfilter(double(green), lap, 'replicate');
    focusScore = var(response(fovMask));

    % --- illumination check ---
    greenVals = double(green(fovMask));
    illumMean = mean(greenVals);
    illumStd  = std(greenVals);

    % --- field of view check ---
    fovArea = sum(fovMask(:)) / numel(fovMask);
    % Note: touchesEdge removed - loadFundus crops+pads the retina to
    % fill the square frame by design, so nearly every well-loaded
    % image touches the edge. That check was flagging good images.

    % --- thresholds (starting guesses, we'll tune these against
    % our own labelled sample in Phase 4) ---
    FOCUS_THRESH = 20;   % placeholder, needs tuning
    isBlurry   = focusScore < FOCUS_THRESH;
    isBadLight = illumMean < 40 || illumMean > 220 || illumStd < 8;
    isBadFOV   = fovArea < 0.35;

    reasons = {};
    if isBlurry,   reasons{end+1} = 'out of focus'; end
    if isBadLight, reasons{end+1} = 'poor illumination'; end
    if isBadFOV,   reasons{end+1} = 'retina out of frame'; end

    if isempty(reasons)
        quality = struct('grade','gradable','reason','');
    else
        quality = struct('grade','ungradable','reason',strjoin(reasons,', '));
    end
end