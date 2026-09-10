function fcn = makeGradingReadFcn(varargin)
%MAKEGRADINGREADFCN Read fcn = loadFundus + optional DR-specific enhancement.
%
%   fcn = MAKEGRADINGREADFCN('TargetSize',[448 448], 'Enhance','bengraham')
%
%   Wraps common/loadFundus rather than modifying it -- common/ is M1's
%   module and every other module depends on its current behaviour. This
%   adds grading-specific preprocessing on top, without touching it.
%
%   'Enhance'
%     'none'      (default) plain loadFundus output
%     'bengraham' local-average subtraction, the preprocessing from the
%                 winning Kaggle Diabetic Retinopathy solution and present in
%                 most strong DR pipelines since
%     'clahe'     contrast-limited adaptive histogram equalisation in Lab
%
%   WHY BEN GRAHAM PREPROCESSING MATTERS HERE
%   Fundus images carry a large, smooth illumination gradient -- bright near
%   the optic disc, dark at the periphery -- and that gradient differs
%   between cameras. It is far stronger than the lesions themselves, so a
%   network spends capacity modelling illumination instead of pathology, and
%   features learned on one camera transfer poorly to another.
%
%   Subtracting a heavily blurred copy removes everything smooth and leaves
%   local deviations: microaneurysms, haemorrhages, exudates, vessel edges.
%   Formula is 4*I - 4*blur(I) + 128, with the blur radius scaled to the
%   retina, which is the standard form.
%
%   This also attacks the IDRiD/APTOS gap at its root -- much of what differs
%   between those two datasets IS illumination and colour balance.
%
%   See also LOADFUNDUS, MAKEFUNDUSREADFCN, BUILDDATASTORES.

p = inputParser;
p.FunctionName = 'makeGradingReadFcn';
p.addParameter('TargetSize', [], @(x) isempty(x) || (isnumeric(x) && numel(x)==2));
p.addParameter('Enhance', 'none', @(x) ischar(x) || isstring(x));
p.parse(varargin{:});
opt = p.Results;

mode = lower(char(opt.Enhance));
if ~ismember(mode, {'none','bengraham','clahe'})
    error('makeGradingReadFcn:badEnhance', ...
        'Enhance must be ''none'', ''bengraham'' or ''clahe''.');
end

sz  = opt.TargetSize;
fcn = @(filename) iRead(filename, sz, mode);
end

% =====================================================================
function img = iRead(filename, sz, mode)
if isempty(sz)
    [img, fov] = loadFundus(filename);
else
    [img, fov] = loadFundus(filename, 'TargetSize', sz);
end

switch mode
    case 'none'
        return
    case 'bengraham'
        img = iBenGraham(img, fov);
    case 'clahe'
        img = iClahe(img, fov);
end
end

% =====================================================================
function out = iBenGraham(rgb, fov)
%IBENGRAHAM  4*I - 4*blur(I) + 128, blur radius scaled to the retina.
I = im2double(rgb);

% Retina radius in pixels: the FOV mask fills the frame after loadFundus,
% so half the mean extent is a good estimate and degrades gracefully.
r = 0.5 * mean([sum(any(fov,2)), sum(any(fov,1))]);
sigma = max(r/30, 1);

B = imgaussfilt(I, sigma, 'Padding', 'replicate');
out = 4*I - 4*B + 0.5;
out = im2uint8(min(max(out, 0), 1));

% Outside the retina there is no signal -- the subtraction turns it into a
% flat grey ring that the network would otherwise treat as structure.
out = bsxfun(@times, out, uint8(fov));
end

% =====================================================================
function out = iClahe(rgb, fov)
%ICLAHE Contrast-limited adaptive histogram equalisation on lightness only.
lab = rgb2lab(rgb);
L   = lab(:,:,1) / 100;
lab(:,:,1) = adapthisteq(L, 'ClipLimit', 0.01, 'Distribution', 'rayleigh') * 100;
out = im2uint8(lab2rgb(lab));
out = bsxfun(@times, out, uint8(fov));
end
