# SIH Diabetic Retinopathy — M4 (Grading) — Full Context Handoff

**Last updated:** 15 Sep 2026
**Owner:** Joseph (M4 — ML / severity grading lead)
**Purpose:** Complete, self-contained state of the grading module. Paste this into a new chat to resume without re-deriving anything.

> **What changed since 11 Sep, in one line:** the GPU arrived, fine-tuning ran, and
> **IDRiD test AUC went 0.887 → 0.933** — the first change in this project to beat its
> own error bar. Also: the RBF-SVM "win" was tested properly and **retracted**.

---

## 1. The project

Smart India Hackathon entry: **explainable AI for diabetic retinopathy screening in rural India**, built in MATLAB.

- **Repo:** `github.com/ayan-codes-rgb/Explainable-AI-for-Diabetic-Retinopathy-Screening-in-Rural`
- **Team:** 6 people, modules M1–M6. Joseph is **M4 = DR severity grading** (the actual classifier).
- **Deadlines:** POC **15 Sep** (done), final **30 Sep**.

**Targets for referable DR (grade ≥ 2):** sensitivity **> 90%**, specificity **> 85%**.

### Module status (15 Sep)

| module | branch | state |
|---|---|---|
| `common` (M1, shared loader) | main | done |
| `quality` (M1) | `origin/quality` | **delivered** — 3 checks + CLAHE, validated on 50 hand-labelled images |
| `segmentation` (M2) | `origin/segmentation` | **delivered** — OD 92.7% within one disc radius (n=413); fovea 52.5%, bimodal; vessels unquantified |
| `lesions` (M3) | — | **not started** |
| **`grading` (M4)** | `grading` | **fine-tuned, validated** — see §7 |
| `explainability` (M5) | — | **not started** |
| `simulink` / `integration` (M6) | — | **not started** |

> M2's README documents `runSegmentation.m` as its entry point. **That file is not in the
> commit** — only the three `detect*.m` functions. Integration will hit a missing-function
> error until M2 pushes it.

---

## 2. Hardware — RESOLVED

| | |
|---|---|
| **GPU machine** | `laptop-7k56ktul` — Ayan's ASUS. **NVIDIA RTX 5050 Laptop, 8.52 GB, compute 12.0, WDDM.** This is the training machine. |
| **Joseph's Omen** | AMD Radeon, CUDA-incapable. Still fine for CPU work; **fine-tuning on it would take 40–50 hours** against 2.5 on the ASUS. |
| **Repo on ASUS** | `C:\Users\ayand\Desktop\Explainable-AI-for-Diabetic-Retinopathy-Screening-in-Rural` |
| **Data on ASUS** | `C:\dr-data\IDRiD\B. Disease Grading` and `C:\dr-data\APTOS` |
| **MATLAB** | R2026a, all needed toolboxes. SimEvents still absent (M6's problem). |

**Two live gotchas:**

1. `git clone` checks out **`main`**, which has an old `loadSplit.m` and `idrid_split.csv`. Always `git switch grading`. (`git checkout grading` fails — `grading` is both a folder and a branch name.)
2. `runBaseline`'s feature extraction is **CPU-only** — `iFeatures` builds a plain `dlarray`, never a `gpuArray`, so it ignores the card entirely. ~25 min instead of ~2. Not yet fixed.

---

## 3. Data

### The split — `grading/split.csv` (4,178 rows, committed, FROZEN)

| dataset | split | n | non-referable | referable |
|---|---|---:|---:|---:|
| IDRiD | train | 352 | 131 | 221 |
| IDRiD | val | 61 | 23 | 38 |
| IDRiD | **test** | **103** | 39 | 64 |
| APTOS | train | 3,112 | 1,848 | 1,264 |
| APTOS | val | 275 | 163 | 112 |
| APTOS | **test_aptos** | **275** | 164 | 111 |

Stratified, seed **20260909**. IDRiD's official 103-image test set carried through untouched.

### ⚠️ Five missing APTOS images — declared, not hidden

Five APTOS files did not survive the transfer to the ASUS and are absent at the source copy too.
They **exist on the Omen** (runBaseline could not have produced its cached features without them).

`split.csv` is never edited. Instead `grading/excluded_images.txt` declares them, and `loadSplit`
drops only declared-missing rows — anything missing and *undeclared* is still a hard error. A
warning prints on every load that applies it, so the deviation cannot be forgotten.

**Effect: train 3464 → 3461, val 336 → 335, test_aptos 275 → 274. IDRiD test stays 103, so the
reported benchmark is unaffected.** Restore the files and delete the lines; counts snap back.

### Grade distribution

| grade | meaning | IDRiD | APTOS | combined |
|---|---|---:|---:|---:|
| 0 | no DR | 168 | 1805 | 1973 |
| 1 | mild NPDR | 25 | 370 | 395 |
| 2 | moderate NPDR | 168 | 999 | 1167 |
| 3 | severe NPDR | 93 | 193 | 286 |
| 4 | proliferative DR | 62 | 295 | 357 |

IDRiD is 12% of the data but supplies **33% of all grade-3 images**; never downweight it to zero.
IDRiD is 62.6% referable (one clinic, Nanded); APTOS is 40.6% (multi-site screening programme).

### ⚠️ Measurement error

| eval set | n | AUC standard error |
|---|---:|---|
| IDRiD test | 103 | **±0.030 – 0.036** |
| APTOS test | 274 | **±0.021** |
| IDRiD **val** | 61 | **±0.039** — too few to place a cutoff, see §8 |

**A difference smaller than the relevant s.e. is not a difference.** This has bitten the project
twice; see §9.

---

## 4. The code

### `common/` — M1's shared loader. Nobody calls `imread()`. Only M1 edits it.

### `grading/` — M4's module

| file | role |
|---|---|
| `split.csv` | the frozen split |
| `excluded_images.txt` | **declared-missing rows** — the only sanctioned way to proceed without an image |
| `loadSplit.m` | multi-root resolver, dataset filter, exclusion handling |
| `makeGradingReadFcn.m` | wraps `loadFundus`, adds `none｜bengraham｜clahe` |
| `buildDatastores.m` | datastores + 360° rotation augmentation (train only) |
| `classWeights.m` | `inverse｜inverse-sqrt｜none` |
| `runBaseline.m` | **frozen-backbone baseline** — the ablation floor a fine-tuned model must beat |
| `rocReport.m` | ROC, bootstrap CIs, per-dataset breakdown, target verdict |
| `tier1Experiments.m` | 13 classifier variants — **superseded by `tier1CV`, see §6** |
| `tier1CV.m` | **honest classifier selection** on out-of-fold scores |
| **`fineTune.m`** | **GPU fine-tuning. Never reads test.** |
| **`revealTest.m`** | the only file that reads test. Logs every call. |
| **`calibrateThreshold.m`** | re-fits the cutoff from validation; never reads test |
| **`compareModels.m`** | paired bootstrap CI on the frozen-vs-fine-tuned gap |

### The discipline, enforced structurally

`fineTune` calls `loadSplit` for `'train'` and `'val'` only — there is no code path by which a
test image can influence the network, the epoch choice, or the threshold. Reading test requires
calling `revealTest` explicitly, and every call appends to `grading/models/reveal_log.txt`.

---

## 5. Results — the frozen baseline (the floor)

**Config:** resnet18 frozen @ 448×448, `Enhance='none'`, SVM head, domain-balanced,
threshold by 5-fold CV on train+val, calibrated on IDRiD.

| eval set | n | sens | spec | AUC | 95% CI |
|---|---:|---:|---:|---:|---|
| APTOS holdout | 274 | 93.7% | 92.1% | **0.972** | 0.948 – 0.985 |
| IDRiD test | 103 | 85.9% | 79.5% | **0.887** | 0.791 – 0.938 |

`runBaseline` is **not reproducible run to run** — `fitcsvm`'s `KernelScale='auto'` picks its
scale from a random subsample and nothing seeds the RNG. Put `rng(20260909)` in front of it
before quoting a number.

---

## 6. 🔴 RETRACTED: the RBF-SVM "win"

The 11 Sep version of this document said:

> *"The one real win in the whole project: `linear` → `svm-rbf`, IDRiD AUC 0.849 → 0.894."*

**That is false and has been withdrawn.**

`tier1Experiments` ranked 13 variants by their AUC **on the test set**. No test image entered a
fit, so the weights were clean — but the *choice* of variant was made by reading that column,
roughly twenty times across the project. Picking the best of many near-equal options by their
held-out score does not return the best option; it returns the luckiest one, and reports its
inflated score as if it were unbiased.

`tier1CV` re-ran the same selection on **out-of-fold scores** over train+val (n=3,800 pooled,
413 IDRiD), where repeated looking is legitimate:

| variant | IDRiD OOF AUC | vs linear | 95% CI |
|---|---:|---:|---|
| linear | **0.9628** | — | — |
| svm-rbf | 0.9608 | **−0.002** | **[−0.0125, +0.0093]** |
| boosted-trees | 0.9424 | −0.020 | [−0.036, −0.0066] ← significantly **worse** |
| coral | 0.9424 | −0.020 | [−0.032, −0.0100] ← significantly **worse** |

Eleven variants, **zero improvements, two regressions.** The claimed +0.045 sits five times
outside the honest interval's upper bound. **The classifier head is measured as exhausted.**
Do not re-litigate it.

---

## 7. ✅ Fine-tuning — what ships now

**Run:** 15 Sep, `fineTune(roots)` defaults — resnet18, 448×448, Ben Graham, domain-balanced by
repeating IDRiD ×9, phase 1 head warm-up (3 epochs, backbone frozen), phase 2 full fine-tune
@ 1e-4. Early-stopped at **epoch 7 of 20** on validation patience. **152 minutes** wall clock.

Nine epoch checkpoints were scored on **validation AUC** and the best selected.

### Test results — one reveal, never used for selection

| eval set | n | sens | spec | AUC | s.e. |
|---|---:|---:|---:|---:|---:|
| **IDRiD test** | 103 | 85.9% | 92.3% | **0.933** | 0.030 |
| **APTOS test** | 274 | 89.2% | 95.1% | **0.979** | 0.021 |

### Against the frozen floor

| | frozen | fine-tuned | change |
|---|---:|---:|---:|
| IDRiD test AUC | 0.887 | **0.933** | **+0.046** |
| APTOS test AUC | 0.972 | 0.979 | +0.007 (inside noise) |

**+0.046 against an s.e. of 0.030.** Every previous change died inside its own error bar —
224→448 gave +1.6 points, Ben Graham neutral, CLAHE neutral, classifier head retracted above.
This one clears it. Run `compareModels` for a paired-bootstrap interval on the gap before
quoting it as significant.

### ⚠️ The operating point is in the wrong place

Both sets **miss sensitivity while overshooting specificity by 7–10 points**. The cutoff
(0.6278) was fitted on the **61** IDRiD validation rows to hit 90% sensitivity; on 103 test
images it delivers 85.9%. That is the n=61 problem, exactly as predicted.

**This is a calibration problem, not a model problem.** At AUC 0.933 the curve almost certainly
passes through the target region — the model is standing in the wrong spot on a good curve.

**It cannot be fixed by looking at test.** `calibrateThreshold` re-derives the cutoff from
validation and reports whether the score scale now transfers between domains (it did not on the
frozen model). If it does, the cutoff can be fitted on all 335 validation rows instead of 61.

---

## 8. What's left

| priority | item | cost |
|---|---|---|
| 1 | `calibrateThreshold` — get a defensible operating point | seconds |
| 2 | `compareModels` — CI on the +0.046 | seconds |
| 3 | Pre-resize images to 448 once; training is CPU-data-bound (~10 img/s, GPU idle) | ~5× speedup |
| 4 | Variants: resnet50, IDRiD-only second stage, different LR — **select on validation, reveal once** | ~2.5 h each |
| 5 | k-fold fine-tuning for an OOF threshold on 413 IDRiD rows | ~12 h, the rigorous fix for §7 |
| 6 | GPU-enable `runBaseline`'s `iFeatures` | 3 lines |

**Not M4's job:** M3's lesion detection, M5's Grad-CAM, M6's Simulink and integration.

---

## 9. Honest record — wrong calls

Keep these. They are why the error bars are taken seriously.

1. **Predicted 448px would be a big win.** +1.6 points.
2. **Predicted Ben Graham would move test AUC.** Neutral.
3. **Used logistic regression as the head and never questioned it.** Switching to RBF-SVM looked like the project's biggest gain — **and then turned out not to be one** (§6).
4. **Selected classifier variants by their test-set score, ~20 times.** Caught by Joseph, 14 Sep. This is the error that produced #3. The fix is structural, not procedural: `fineTune` cannot read test.
5. **Predicted L2 normalisation could match the linear→SVM switch.** Did nothing; `Standardize=true` had already solved it.
6. **Estimated the tuned Bayesian search at 3–5 min.** It was 20–40.
7. **Gave a `roots.IDRiD` without the `B. Disease Grading` level.** Cost one failed run.
8. **Set the fine-tuned threshold on 61 rows.** Predicted it would be noisy; it was — 90% on validation, 85.9% on test.

### Bugs hit and fixed
- `genpath('common')` silently returns empty if cwd isn't the repo. **Always `cd` first.**
- `fitcecoc` posteriors are the **4th** output, not the 3rd. Replaced with `NegLoss`.
- `git checkout grading` is ambiguous (folder vs branch). Use `git switch grading`.
- `device_bash` is unavailable on both machines (Windows update, 8 Sep). File tools still work.

---

## 10. Reproduce from scratch (on the ASUS)

```matlab
cd 'C:\Users\ayand\Desktop\Explainable-AI-for-Diabetic-Retinopathy-Screening-in-Rural'
addpath(genpath('common'), genpath('grading'));
gpuDevice                              % confirm the RTX 5050

roots.IDRiD = 'C:\dr-data\IDRiD\B. Disease Grading';
roots.APTOS = 'C:\dr-data\APTOS';

rng(20260909);
Rb = runBaseline(roots, 'InputSize', [448 448]);   % frozen floor, ~25 min cold
R  = fineTune(roots);                              % ~2.5 h on GPU, test never read

calibrateThreshold('grading/models/ft_resnet18_<stamp>.mat', roots);
compareModels(Rb, 'grading/models/ft_resnet18_<stamp>.mat', roots);

revealTest('grading/models/ft_resnet18_<stamp>.mat', roots);   % once. it logs.
```

---

## 11. Working preferences (Joseph)

- **Blunt over reassuring.** Say when something failed and why; don't dress up noise as a result.
- **Don't build unless explicitly asked.** Recommend, then wait.
- **Explain in simple terms.** Second-year B.S. Medical Sciences and Technology — strong on the biology, still learning ML tooling. Define jargon the first time.
- **Credentials are never handled.** He runs every `git push` himself.
- **Never select on test.** Validation or out-of-fold, always. He caught this; it stands as a project rule.
- **Never touch a machine mid-run.**
- Always state error bars, and flag anything under ~0.03 as noise.
