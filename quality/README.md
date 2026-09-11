# Module 1 — Image Quality Assessment & Enhancement

Owner: M1
Folder: `quality/`
Fills: `result.quality`, `result.enhancedImage`

## What this module does

Given an image already loaded through `common/loadFundus.m` (RGB uint8, retina
centred and squared, with `fovMask`), this module:

1. Assesses whether the image is gradable, using three checks:
   - **Focus** — variance of the Laplacian on the green channel, restricted
     to `fovMask`. Low variance indicates blur.
   - **Illumination** — mean and standard deviation of the green channel
     inside `fovMask`. Flags images that are too dark, too bright, or too
     flat/uneven.
   - **Field of view** — fraction of the frame occupied by retina
     (`fovMask`). Flags images where too little retina is visible.
2. If ungradable, returns a human-readable reason string (e.g. "out of
   focus, poor illumination") so a field technician knows what to recapture.
3. Applies CLAHE (on the L channel in Lab space) plus mild denoising
   (`wiener2`) to borderline/poor images. Images that are clearly good are
   passed through untouched, since over-processing a good image measurably
   degrades it rather than helping.

Entry point for integration: `analyzeQuality(img, fovMask)`, which returns
`[quality, enhancedImage]` matching the shared result struct fields.

## Files

| File | Purpose |
|---|---|
| `assessQuality.m` | Computes focus/illumination/FOV scores and the gradable/ungradable decision |
| `enhanceImage.m` | CLAHE + denoise |
| `analyzeQuality.m` | Top-level function called during integration |
| `evaluateQuality.m` | Runs the module against a labelled sample and reports agreement |
| `diagnoseThresholds.m` | Prints score distributions (good vs. bad) used to set thresholds |
| `my_labels.csv` | 50 APTOS images, hand-labelled gradable/borderline/ungradable by M1 |
| `evaluation_results.csv` | Per-image predictions vs. labels for the 50-image set |

## Thresholds

Starting values were guesses; both were tuned against the 50-image labelled
set described below.

```matlab
FOCUS_THRESH = 20;      % below this, flagged as out of focus
isBadLight = illumMean < 40 || illumMean > 220 || illumStd < 8;
isBadFOV   = fovArea < 0.35;
```

All statistics are computed only inside `fovMask` — using the full frame
(`img(:)`) dilutes both illumination and focus scores with the black border,
which makes up roughly 25–30% of the frame.

## Validation — 50-image labelled set

**Method:** 50 APTOS images were sampled by M1 across a visual spread of
obviously-good, borderline, and obviously-bad, and sorted by eye into
`gradable` / `borderline` / `ungradable`. This reference is M1's own
judgment, not a clinician's, and is reported as such.

**Result (after threshold tuning):**

| Metric | Result |
|---|---|
| Exact match, 3-class labels vs. 2-class output | 22/50 (44.0%) |
| Good images (`my_label = gradable`) correctly passed | 15/20 (75%) |
| Bad images (`my_label = ungradable`) correctly caught | 7/12 (58%) |
| Borderline images passed / rejected | 15 passed, 3 rejected |

The exact-match number is naturally capped below 100% because the
classifier only outputs two classes (gradable/ungradable) while the
reference labels use three (including borderline) — a borderline image can
never be an "exact match" against a binary output by construction.

## Known limitations

- **Threshold tuning is based on 50 images and four scalar features.**
  Further hand-tuning on a sample this size risks overfitting to this
  specific set rather than generalizing; thresholds should be revisited if
  a larger labelled set becomes available.
- **Haze and lens-flare-type artifacts are not detected.** Manual review of
  the module's false negatives (bad images predicted gradable) showed two
  clear cases of haze/glare artifacts that the current focus, illumination,
  and FOV checks do not capture, since a hazy image can still have
  reasonable sharpness and brightness statistics. A contrast/clarity metric
  (e.g. Michelson contrast within the FOV) would be a natural extension to
  target this failure mode specifically.
- **Some disagreements reflect labelling subjectivity, not classifier
  error.** Manual review of remaining false negatives found several images
  that were reasonably clear (visible vessels, visible optic disc,
  acceptable contrast) despite being labelled "ungradable" by M1 — these
  were likely borderline calls rather than clear-cut misses.
- **Focus and bad-image scores overlap at this sample size.** The blurry
  subset's maximum focus score (60.4) exceeded the good subset's maximum
  (44.3), meaning no single threshold perfectly separates the two classes
  on this data. `FOCUS_THRESH = 20` was chosen to sit between the median of
  each group as the most defensible cut point, not because it eliminates
  overlap entirely.

## Usage

```matlab
[img, fovMask] = loadFundus(imagePath);
[quality, enhancedImage] = analyzeQuality(img, fovMask);

quality.grade    % 'gradable' or 'ungradable'
quality.reason   % human-readable reason string if ungradable
```
