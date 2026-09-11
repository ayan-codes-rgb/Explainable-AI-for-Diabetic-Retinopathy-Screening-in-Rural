function diagnoseThresholds(csvPath, apostFolder)
    T = readtable(csvPath, 'TextType', 'string');
    n = height(T);

    focus = zeros(n,1); illumM = zeros(n,1); illumS = zeros(n,1); fovA = zeros(n,1);

    for i = 1:n
        fullPath = fullfile(apostFolder, T.filename(i));
        [im, fov] = loadFundus(fullPath);
        [~, fs, im_, is_, fa] = assessQuality(im, fov);
        focus(i) = fs; illumM(i) = im_; illumS(i) = is_; fovA(i) = fa;
    end

    T.focus = focus; T.illumMean = illumM; T.illumStd = illumS; T.fovArea = fovA;

    fprintf('\n--- Your GOOD images: score distribution ---\n');
    g = T(T.my_label=="gradable", :);
    fprintf('focus:    min=%.1f  median=%.1f  max=%.1f\n', min(g.focus), median(g.focus), max(g.focus));
    fprintf('illumMean min=%.1f  median=%.1f  max=%.1f\n', min(g.illumMean), median(g.illumMean), max(g.illumMean));
    fprintf('illumStd  min=%.1f  median=%.1f  max=%.1f\n', min(g.illumStd), median(g.illumStd), max(g.illumStd));
    fprintf('fovArea   min=%.2f  median=%.2f  max=%.2f\n', min(g.fovArea), median(g.fovArea), max(g.fovArea));

    fprintf('\n--- Your BAD images: score distribution ---\n');
    b = T(T.my_label=="ungradable", :);
    fprintf('focus:    min=%.1f  median=%.1f  max=%.1f\n', min(b.focus), median(b.focus), max(b.focus));
    fprintf('illumMean min=%.1f  median=%.1f  max=%.1f\n', min(b.illumMean), median(b.illumMean), max(b.illumMean));
    fprintf('illumStd  min=%.1f  median=%.1f  max=%.1f\n', min(b.illumStd), median(b.illumStd), max(b.illumStd));
    fprintf('fovArea   min=%.2f  median=%.2f  max=%.2f\n', min(b.fovArea), median(b.fovArea), max(b.fovArea));

    writetable(T, 'quality/diagnostic_scores.csv');
end