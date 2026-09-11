function enhanced = enhanceImage(img, fovMask)
% ENHANCEIMAGE - CLAHE + mild denoise, applied to borderline/poor images
    lab = rgb2lab(img);
    L = lab(:,:,1);
    L_eq = adapthisteq(L/100) * 100;   % CLAHE on lightness channel
    lab(:,:,1) = L_eq;
    enhanced = lab2rgb(lab);

    enhanced = im2uint8(enhanced);
    for c = 1:3
        enhanced(:,:,c) = wiener2(enhanced(:,:,c), [3 3]);  % mild denoise
    end
    enhanced(repmat(~fovMask,1,1,3)) = 0;  % keep border black
end