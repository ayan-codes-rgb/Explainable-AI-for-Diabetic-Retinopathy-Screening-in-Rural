# M2 — Segmentation: Optic Disc, Fovea, Vessel Segmentation

**Owner:** M2  
**Branch:** `segmentation`  
**Fills:** `result.opticDisc`, `result.fovea`, `result.vesselMask`  
**Blocks:** M3 (exudate detection needs optic disc location)

---

## Files

| File | Purpose |
|---|---|
| `runSegmentation.m` | Entry point — call this, returns result struct |
| `detectOpticDisc.m` | Brightest-blob detector on red channel with size filter |
| `detectFovea.m` | Darkest-region in directional annulus (disc → image centre) |
| `detectVessels.m` | Morphological top-hat with percentile threshold |

---

## How to run

```matlab
addpath(genpath('common'))
addpath(genpath('segmentation'))
result = runSegmentation('C:\dr-data\IDRiD\...\IDRiD_001.jpg');
```

Output fields:

```
result.opticDisc.center   [x, y] in resized image coords
result.opticDisc.radius   scalar, pixels
result.fovea.center       [x, y] or [NaN NaN] if detection failed
result.vesselMask         logical 512×512
result.geom               geometry struct — pass to M3 for mask realignment
```

---

## Validation results

All metrics are on the **IDRiD C. Localization training set** (413 images, 512×512 after `loadFundus`).  
Ground truth realigned via `mapFundusPoints(trueXY, geom)` before error computation.

### Optic disc detection

| Metric | Value |
|---|---|
| Images tested | 413 |
| Failed detections (NaN) | 17 (4.1%) |
| Mean error | **11.55 px** (95% CI: 7.96–15.13) |
| Within 40 px (~1 disc radius) | **92.74%** |

Method: brightest percentile blob on red channel, size-plausibility filter 60–110 px diameter.  
Failure cases: images where large bright exudates outcompete the disc (e.g. IDRiD_081).

### Fovea detection

| Metric | Value |
|---|---|
| Images tested | 396 (17 excluded — disc detection failed) |
| Median error | **16.0 px** |
| P25 / P75 | 4.8 px / 196.5 px |
| Mean error | 90.9 px (95% CI: 81.3–100.5) |
| Within 40 px | **52.5%** |

**Distribution is bimodal.** ~52% of images localise within 40 px (median 16 px); ~45% show systematic failure at 100–300 px.

**Known failure mode:** hard exudates at the macula occlude the foveal avascular zone. The brightness-minimum approach is confounded when the fovea is covered by bright lesions. These cases should be flagged by M3's exudate mask and reported as low-confidence in M5's output.

Method: temporal-direction directional annulus (4–6 disc-radii from disc centre, ±60° cone toward image centre), heavily smoothed green channel (σ = 15 px), minimum-intensity search.

### Vessel segmentation

**Qualitative validation only** — IDRiD does not include vessel ground truth. DRIVE ground truth would be needed for a quantitative number; noted as a gap.

Visual inspection across 20 images confirms:
- Major vascular arcades captured in all tested images
- Background noise suppressed (top-hat disk=12, prctile=85 threshold)
- **Known false positive:** hard exudate patches produce vessel-like top-hat response — M3's exudate mask should be used to subtract these after detection

---

## Known limitations

1. **Fovea failure rate ~45%** — brightness-minimum fails when macular exudates are present (Grade 2–4). Flag via M3 exudate mask.
2. **Vessel segmentation unquantified** — no ground truth in current dataset scope.
3. **Optic disc NaN = 4.1%** — primarily images with large bright lesions that outcompete the disc in the red channel.
4. **Small validation set** — 413 images for disc/fovea. 95% CIs reported throughout; do not treat point estimates as exact.

---

## Integration notes for M3

- Call `runSegmentation()` to get `result.opticDisc.center` before running exudate detection
- Subtract disc region from exudate candidates: mask pixels within `result.opticDisc.radius * 1.2` of `result.opticDisc.center`
- Pass `result.geom` to `applyFundusGeom()` when realigning your pixel-level masks from A. Segmentation
- If `result.opticDisc.center` is `[NaN NaN]`, skip disc subtraction and flag the image
