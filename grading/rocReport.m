function T = rocReport(R, varargin)
%ROCREPORT ROC curves, AUC, and the operating points actually achievable.
%
%   T = ROCREPORT(R)                     % R comes from runBaseline
%   T = ROCREPORT(R, 'Plot', false)      % table only, no figure
%
%   WHY THIS EXISTS
%   runBaseline reports ONE operating point -- the threshold tuned for 90%
%   sensitivity. That single point says nothing about whether a different
%   threshold could do better, or whether both project targets (>90%
%   sensitivity AND >85% specificity) are reachable at all.
%
%   Sensitivity and specificity are two ends of one dial. Moving the
%   threshold slides you along a fixed curve; it cannot lift the curve. This
%   function draws that curve, so you can see what the model can and cannot
%   do independently of where you happen to have set the cutoff.
%
%   AUC (area under the curve) summarises the whole curve in one number:
%   the probability that a randomly chosen referable eye scores higher than
%   a randomly chosen non-referable one. 0.5 is a coin flip. It does not
%   depend on the threshold, which makes it the right number to quote when
%   comparing models -- and the right number to put in the report.
%
%   Rough reading: 0.70-0.80 acceptable, 0.80-0.90 good, >0.90 excellent.
%
%   Prints, for val and test:
%     - AUC with a bootstrap 95% confidence interval
%     - the best sensitivity available at 80/85/90% specificity
%     - the best specificity available at 85/90/95% sensitivity
%     - whether ANY threshold satisfies both project targets at once
%
%   See also RUNBASELINE, PERFCURVE.

p = inputParser;
p.addParameter('Plot', true, @(x) islogical(x) && isscalar(x));
p.addParameter('NBoot', 1000, @isnumeric);
p.parse(varargin{:});
opt = p.Results;

if ~isfield(R, 'scores')
    error('rocReport:noScores', ...
        ['This R has no scores field -- it came from an older runBaseline. ' ...
         'Re-run:  R = runBaseline(root);  (features are cached, takes seconds).']);
end

splits = {'val', 'test'};
COL    = [0.184 0.435 0.816;    % val   #2f6fd0
          0.878 0.482 0.094];   % test  #e07b18
GRID   = [0.62 0.62 0.60];
INK    = [0.15 0.15 0.15];

rows = {};
curves = struct();

fprintf('\n=== ROC / AUC ===\n');
for k = 1:numel(splits)
    nm = splits{k};
    y  = logical(R.scores.(nm).truth);
    s  = R.scores.(nm).score;

    [X, Y, Thr]  = perfcurve(y, s, true);
    [~, ~, ~, A] = perfcurve(y, s, true, 'NBoot', opt.NBoot);
    curves.(nm) = struct('X', X, 'Y', Y, 'T', Thr, 'auc', A(1), ...
                         'lo', A(2), 'hi', A(3), 'n', numel(y), ...
                         'nPos', sum(y), 'nNeg', sum(~y));

    fprintf('\n%s  (n=%d: %d referable, %d not)\n', upper(nm), ...
        numel(y), sum(y), sum(~y));
    fprintf('  AUC %.3f   95%% CI %.3f - %.3f\n', A(1), A(2), A(3));

    % Per-dataset breakdown. A pooled AUC mixes populations of different
    % difficulty: APTOS is a screening population, ~half of it obviously
    % normal and easy to separate; IDRiD is clinic-collected, 62% referable
    % and full of borderline cases. A high pooled AUC can hide a much lower
    % one on the population you actually deploy to.
    if isfield(R.scores.(nm), 'dataset')
        ds = string(R.scores.(nm).dataset);
        u  = unique(ds);
        if numel(u) > 1
            fprintf('  per dataset:\n');
            for d = u'
                m = ds == d;
                if numel(unique(y(m))) < 2, continue; end
                [~,~,~,Ad] = perfcurve(y(m), s(m), true, 'NBoot', 200);
                fprintf('     %-6s AUC %.3f  (n=%d: %d referable, %d not)\n', ...
                    d, Ad(1), sum(m), sum(y & m), sum(~y & m));
            end
        end
    end

    fprintf('  sensitivity available at a given specificity:\n');
    for tgt = [0.80 0.85 0.90]
        [sens, spec, t] = iBestAt(y, s, 'spec', tgt);
        rows(end+1,:) = {string(nm), "spec>=" + string(tgt*100) + "%", sens, spec, t}; %#ok<AGROW>
        fprintf('     spec >= %2.0f%%  ->  sensitivity %5.1f%%  (spec %5.1f%%, thr %.3f)\n', ...
            tgt*100, 100*sens, 100*spec, t);
    end

    fprintf('  specificity available at a given sensitivity:\n');
    for tgt = [0.85 0.90 0.95]
        [sens, spec, t] = iBestAt(y, s, 'sens', tgt);
        rows(end+1,:) = {string(nm), "sens>=" + string(tgt*100) + "%", sens, spec, t}; %#ok<AGROW>
        fprintf('     sens >= %2.0f%%  ->  specificity %5.1f%%  (sens %5.1f%%, thr %.3f)\n', ...
            tgt*100, 100*spec, 100*sens, t);
    end

    [okBoth, tBoth, sB, pB] = iBothTargets(y, s, 0.90, 0.85);
    if okBoth
        fprintf('  BOTH TARGETS: reachable at threshold %.3f (sens %.1f%%, spec %.1f%%)\n', ...
            tBoth, 100*sB, 100*pB);
    else
        fprintf('  BOTH TARGETS: NOT reachable at any threshold on this model.\n');
        fprintf('                The curve does not pass through that region --\n');
        fprintf('                a better model is needed, not a better cutoff.\n');
    end
end

T = cell2table(rows, 'VariableNames', ...
    {'split', 'constraint', 'sensitivity', 'specificity', 'threshold'});

% ---- plot ------------------------------------------------------------
if opt.Plot
    figure('Name', 'ROC - referable DR', 'Color', 'w');
    ax = axes(); hold(ax, 'on'); box(ax, 'off');

    plot(ax, [0 1], [0 1], '--', 'Color', GRID, 'LineWidth', 1, ...
        'HandleVisibility', 'off');                       % chance line

    h = gobjects(1, numel(splits));
    for k = 1:numel(splits)
        c = curves.(splits{k});
        h(k) = plot(ax, c.X, c.Y, '-', 'Color', COL(k,:), 'LineWidth', 2, ...
            'DisplayName', sprintf('%s  AUC %.3f (n=%d)', splits{k}, c.auc, c.n));
    end

    % the operating point runBaseline actually chose
    for k = 1:numel(splits)
        c = curves.(splits{k});
        [~, i] = min(abs(c.T - R.threshold));
        plot(ax, c.X(i), c.Y(i), 'o', 'MarkerSize', 9, ...
            'MarkerFaceColor', COL(k,:), 'MarkerEdgeColor', 'w', ...
            'LineWidth', 1.5, 'HandleVisibility', 'off');
    end

    % target region: sens >= 0.90 and spec >= 0.85  (i.e. FPR <= 0.15)
    patch(ax, [0 0.15 0.15 0], [0.90 0.90 1 1], [0.55 0.75 0.55], ...
        'FaceAlpha', 0.13, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    text(ax, 0.155, 0.945, 'target region', 'Color', INK, 'FontSize', 9);

    grid(ax, 'on'); ax.GridColor = GRID; ax.GridAlpha = 0.25;
    ax.XColor = GRID; ax.YColor = GRID;
    xlabel(ax, '1 - specificity  (false alarms)', 'Color', INK);
    ylabel(ax, 'sensitivity  (cases caught)',      'Color', INK);
    title(ax, 'Referable DR - ROC, frozen ResNet-18 baseline', 'Color', INK);
    legend(ax, h, 'Location', 'southeast', 'TextColor', INK, 'Box', 'off');
    axis(ax, [0 1 0 1]); axis(ax, 'square');
    text(ax, 0.42, 0.06, 'filled dot = chosen operating point', ...
        'Color', GRID, 'FontSize', 8);
end

fprintf('\n');
end

% =====================================================================
function [sens, spec, thr] = iBestAt(y, s, mode, target)
%IBESTAT Best achievable other-metric subject to one metric meeting target.
cand = unique([0; sort(s(:)); 1]);
sens = 0; spec = 0; thr = NaN; best = -Inf;
for t = cand'
    se = mean(s(y)  >= t);
    sp = mean(s(~y) <  t);
    if strcmp(mode, 'spec')
        if sp >= target && se > best, best = se; sens = se; spec = sp; thr = t; end
    else
        if se >= target && sp > best, best = sp; sens = se; spec = sp; thr = t; end
    end
end
if isnan(thr)   % target unreachable at any threshold
    sens = NaN; spec = NaN;
end
end

% =====================================================================
function [ok, thr, sens, spec] = iBothTargets(y, s, targetSens, targetSpec)
cand = unique([0; sort(s(:)); 1]);
ok = false; thr = NaN; sens = NaN; spec = NaN; best = -Inf;
for t = cand'
    se = mean(s(y)  >= t);
    sp = mean(s(~y) <  t);
    if se >= targetSens && sp >= targetSpec && (se + sp) > best
        best = se + sp; ok = true; thr = t; sens = se; spec = sp;
    end
end
end
