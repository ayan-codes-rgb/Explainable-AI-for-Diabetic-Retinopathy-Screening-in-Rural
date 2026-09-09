# `common/` — shared image I/O contract

**Owner: M1. Everyone else consumes it.**

Nobody in this project calls `imread()` directly. Every image enters the
pipeline through `loadFundus`, so that channel order, bit depth and
resolution are identical in all six modules. If you bypass it, the bug you
eventually hit will look like an algorithm bug and will cost you a day.

## Setup

Add this folder to your MATLAB path once per session (or put it in
`startup.m`):

```matlab
addpath(genpath(fileparts(mfilename('fullpath'))));   % from the repo root
```

Then verify your install before you write anything:

```matlab
test_loadFundus()          % works today, no dataset needed
```

## The contract

```matlab
[img, fovMask, geom] = loadFundus(pathOrArray);
```

| output    | type                    | guarantee |
|-----------|-------------------------|-----------|
| `img`     | `uint8` H×W×3           | RGB order, 0–255, retina centred, outside-retina pixels are 0 |
| `fovMask` | `logical` H×W           | true inside the retina, same size as `img` |
| `geom`    | struct                  | record of every geometric op applied, for realigning ground truth |

Default size is `[512 512]`, from `drConfig()`. Override per call with
`loadFundus(f, 'TargetSize', [224 224])`. To change the team-wide default,
edit `drConfig.m` and tell everyone — do not hardcode a size in your module.

## What it does to an image

1. forces 3-channel RGB `uint8` (grayscale replicated, alpha dropped,
   `uint16` scaled, `double` handled whether stored 0–1 or 0–255)
2. detects the retina and **crops the black border away**
3. **pads the crop to a square before resizing** — resizing a 4:3 fundus
   straight to 512×512 squashes the retina into an ellipse, which corrupts
   vessel geometry, lesion shape features and optic-disc circularity at once
4. resizes: bilinear for the image, nearest for the mask

## Realigning ground truth — read this if you use IDRiD/DRIVE/e-ophtha

The loader crops and resizes. Ground truth that does not go through the
**identical** transform is misaligned by tens of pixels. It will not throw an
error; it will just make every segmentation metric you compute wrong.

```matlab
[img, fov, geom] = loadFundus('data/IDRiD/images/IDRiD_001.jpg');

% pixel masks (IDRiD lesions, DRIVE vessels, e-ophtha)
maMask = applyFundusGeom(imread('IDRiD_001_MA.tif') > 0, geom);

% coordinates (IDRiD optic disc / fovea centres, e-ophtha MA locations)
odXY   = mapFundusPoints([odX odY], geom);
```

Both take the annotation at its **original** resolution and check that, so
you get an error rather than silent misalignment.

## For M4 (grading)

```matlab
ds = imageDatastore(imgFolder, ...
        'IncludeSubfolders', true, ...
        'LabelSource',       'foldernames', ...
        'ReadFcn',           makeFundusReadFcn());
```

Train through this, not through the default `ReadFcn`. If the network is
trained on raw `imread()` output and the integrated pipeline later feeds it
`loadFundus` output, accuracy drops with no visible cause.

## Files

| file | what it is |
|---|---|
| `drConfig.m`          | single source of truth for sizes, thresholds, interpolation |
| `loadFundus.m`        | **the loader** — the only entry point for reading an image |
| `fundusFOVMask.m`     | retina-vs-black-border mask; also useful standalone |
| `applyFundusGeom.m`   | replay the loader's geometry onto a mask |
| `mapFundusPoints.m`   | replay the loader's geometry onto (x,y) coordinates |
| `makeFundusReadFcn.m` | `imageDatastore` adapter |
| `test_loadFundus.m`   | smoke test — synthetic, plus an optional real-data pass |

## Known limits (honest list)

- The FOV threshold (`0.06`) is tuned on the assumption of a near-black
  border. A badly over-exposed image with a grey border will over-crop.
  `test_loadFundus` on real IDRiD files is the check for this — look at the
  `retinaPct` column, anything under ~50% deserves a look.
- `bwareafilt` keeps only the largest bright blob, so a burned-in timestamp
  or camera watermark is discarded rather than included. Good default, but
  it means a fundus split into two disconnected bright regions by a heavy
  artefact would lose half.
- Masking the image by the FOV means downstream code cannot distinguish
  "outside retina" from "genuinely black pixel inside retina". Use
  `fovMask`, not `img == 0`, to test for that.
