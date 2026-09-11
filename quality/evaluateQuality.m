function evaluateQuality(csvPath, apostFolder)
% EVALUATEQUALITY - runs analyzeQuality on the labelled sample set
% and reports agreement against your own labels.

    T = readtable(csvPath, 'TextType', 'string');
    n = height(T);

    predicted = strings(n,1);

    for i = 1:n
        fname = T.filename(i);
        fullPath = fullfile(apostFolder, fname);
        try
            [im, fov] = loadFundus(fullPath);
            [q, ~] = analyzeQuality(im, fov);
            predicted(i) = string(q.grade);
        catch ME
            fprintf('ERROR on %s: %s\n', fname, ME.message);
            predicted(i) = "error";
        end
    end

    T.predicted = predicted;

    % --- collapse to binary for a simple headline number ---
    % your labels have 3 classes, your function only outputs 2
    % (gradable / ungradable), so treat "borderline" as a separate
    % check rather than forcing it into one bucket.

    exactMatch = strcmp(T.my_label, T.predicted);
    fprintf('\nExact match (3-class vs 2-class labels): %d/%d (%.1f%%)\n', ...
        sum(exactMatch), n, 100*sum(exactMatch)/n);

    % --- more meaningful: how many "gradable" you labelled came back
    % gradable, and how many "ungradable" you labelled came back ungradable ---
    isG = T.my_label == "gradable";
    isU = T.my_label == "ungradable";
    isB = T.my_label == "borderline";

    fprintf('\nOf your GOOD images: %d/%d predicted gradable\n', ...
        sum(T.predicted(isG)=="gradable"), sum(isG));
    fprintf('Of your BAD images: %d/%d predicted ungradable\n', ...
        sum(T.predicted(isU)=="ungradable"), sum(isU));
    fprintf('Of your BORDERLINE images: %d predicted gradable, %d predicted ungradable\n', ...
        sum(T.predicted(isB)=="gradable"), sum(T.predicted(isB)=="ungradable"));

    % --- save full table for your README / report ---
    writetable(T, 'quality/evaluation_results.csv');
    fprintf('\nFull results saved to quality/evaluation_results.csv\n');
end