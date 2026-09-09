# quality

**Owner: M1**

Image quality assessment (focus, illumination, field of view) and adaptive enhancement (CLAHE, illumination normalisation, denoising). Rejects ungradable images with a recapture reason.

Must fill in the shared result struct: `result.quality`, `result.enhancedImage`

Read `../common/README.md` before writing any image code. Use `loadFundus()`, never `imread()`.
