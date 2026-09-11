function [quality, enhancedImage] = analyzeQuality(img, fovMask)
% ANALYZEQUALITY - top-level entry point for M1's module
% Called by analyzeImage() during integration.

    [quality, focusScore, illumMean, illumStd, fovArea] = assessQuality(img, fovMask);

    if strcmp(quality.grade, 'ungradable')
        % still enhance so a human reviewer can see it, but grade stays ungradable
        enhancedImage = enhanceImage(img, fovMask);
    else
        % borderline-but-gradable images get enhanced too; only skip if
        % the image is already clearly good (say, well within safe margins)
        isClearlyGood = focusScore > 60 && illumMean > 60 && illumMean < 180 && illumStd > 20;
        if isClearlyGood
            enhancedImage = img;  % don't touch it
        else
            enhancedImage = enhanceImage(img, fovMask);
        end
    end
end
