# Explainable AI for Diabetic Retinopathy Screening in Rural India

MATLAB pipeline for automated DR screening: quality assessment, retinal
structure segmentation, severity grading (ICDR 0-4), explainability, and a
Simulink model of the telemedicine workflow.

## Repo layout

| folder | owner | contents |
|---|---|---|
| `common/`         | M1 | **shared image loader — read `common/README.md` first** |
| `quality/`        | M1 | image quality assessment + enhancement |
| `segmentation/`   | M2 | optic disc, fovea, vessels |
| `lesions/`        | M3 | microaneurysms, exudates, haemorrhages, neovascularisation |
| `grading/`        | M4 | DR severity classification (deep learning) |
| `explainability/` | M5 | Grad-CAM, lesion-level evidence, reports |
| `simulink/`       | M6 | telemedicine workflow model |
| `integration/`    | M6 | `analyzeImage()` — stitches all modules together |
| `data/`           | -  | datasets, **not tracked in git** (see below) |

## Setup

```matlab
addpath(genpath('common'))
test_loadFundus()        % should print SYNTHETIC TEST PASSED
```

Datasets live on the shared Drive, not here. Download them into `data/`:

```
data/
├── IDRiD/          images + grades + pixel-level lesion masks
├── APTOS2019/      images + grades
├── DRIVE/          vessel ground truth
└── e-ophtha/       microaneurysm + exudate ground truth
```

## Rules

1. **Never call `imread()` directly.** Use `loadFundus()` from `common/`.
   Read `common/README.md` for why.
2. **Only M1 edits `common/`.** Need a change? Ask, don't patch it locally —
   otherwise you are running different preprocessing from everyone else.
3. **Don't commit to `main` directly** after the initial setup. Work on your
   own branch (`quality`, `segmentation`, `lesions`, `grading`,
   `explainability`, `simulink`) and merge when your piece works.
4. **Never commit anything in `data/`.** `.gitignore` handles this — but run
   `git status` before you commit and check.

## Every module returns the same struct

```matlab
result = analyzeImage(rgbImage);
% result.quality        'gradable' / 'ungradable' + reason
% result.enhancedImage  post-CLAHE / denoise
% result.opticDisc      centre + radius
% result.fovea          centre coords
% result.vesselMask     binary mask
% result.maList         [x, y, confidence] per microaneurysm
% result.exudateMask, result.haemorrhageMask, result.neoFlag
% result.drGrade        0-4
% result.referable      true/false (grade >= 2)
% result.confidence     calibrated score
% result.gradcamMap     heatmap
```
