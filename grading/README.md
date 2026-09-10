# grading

**Owner: M4**

DR severity grading on the ICDR 0-4 scale using a fine-tuned pretrained CNN.
Target: >90% sensitivity and >85% specificity for **referable DR** (grade >= 2).

Must fill in the shared result struct: `result.drGrade`, `result.referable`, `result.confidence`

Read `../common/README.md` before writing any image code. Use `loadFundus()`, never `imread()`.

---

## The split

`split.csv` is the **fixed** partition, covering both datasets. Read it with
`loadSplit.m`; do not regenerate it.

```matlab
roots.IDRiD = 'C:\Users\c7849\Downloads\dr-data\IDRiD\B. Disease Grading';
roots.APTOS = 'C:\Users\c7849\Downloads\dr-data\APTOS';   % contains train_images\

S = buildDatastores(roots);
R = runBaseline(roots);          % threshold cross-validated by default
T = rocReport(R);
```

|  | g0 | g1 | g2 | g3 | g4 | total | referable |
|---|---:|---:|---:|---:|---:|---:|---:|
| IDRiD train | 114 | 17 | 116 | 63 | 42 | 352 | 62.8% |
| APTOS train | 1534 | 314 | 849 | 164 | 251 | 3112 | 40.6% |
| **all train** | 1648 | 331 | 965 | 227 | 293 | **3464** | 42.9% |
| IDRiD val | 20 | 3 | 20 | 11 | 7 | 61 | 62.3% |
| APTOS val | 271 | 56 | 150 | 29 | 44 | 550 | 40.5% |
| **all val** | 291 | 59 | 170 | 40 | 51 | **611** | 42.7% |
| **test (IDRiD only)** | 34 | 5 | 32 | 19 | 13 | **103** | 62.1% |

IDRiD assignments are byte-identical to the original split and have never been
regenerated. APTOS was appended later with the same seed and the same 15%
validation fraction. **Test is IDRiD only** -- the official 103-image test set,
untouched -- so the reported benchmark stays on Indian fundus cameras with
expert consensus grades.

## Threshold selection

`runBaseline` defaults to `'Threshold','cv'`: pool train+val, run stratified
5-fold cross-validation, and pick the threshold on the out-of-fold scores.

This exists because the old held-out approach chose the threshold from 61
validation images with only 23 non-referable ones. Each of those 23 was worth
4.3 points of specificity, so the "optimal" cutoff was largely noise -- and it
cost about 8 points of test specificity versus what the same model could
deliver at the same sensitivity. Pass `'Threshold','val'` to reproduce the old
behaviour for comparison.

## Things that will bite you

**Domain shift between val and test.** Validation is now mostly APTOS (611
rows, 42.7% referable, many cameras, variable quality) while test is IDRiD
(103 rows, 62.1% referable, one camera). A threshold tuned on val may transfer
imperfectly. `loadSplit('val', roots, 'Dataset','IDRiD')` isolates the IDRiD
part if you want to check.

**Image names repeat.** Both IDRiD folders contain an `IDRiD_001.jpg`. Key on
`id` or the full path, never `image_name`.

**Grade 1 is no longer the scarce class.** With APTOS added it has 331 training
images. **Grade 3 (severe NPDR) is now scarcest at 227** -- and it sits on the
*referable* side, so its errors cost sensitivity (the tighter >90% target).

**Do not touch the test set until the end.** Tune on CV or val, evaluate test
once. Re-tuning against test is how a 91% becomes a number that does not
survive questioning.

**Argmax is not your classifier.** Argmax optimises accuracy; you are graded on
sensitivity. The threshold is chosen deliberately and then frozen.

## First run with APTOS is slow

~4,000 images at roughly a second each -- budget an hour, once. Features are
cached in `grading/cache/` afterwards, so later runs take seconds. The cache
stores the file list and invalidates itself if the split changes.

## Files

| file | what it does |
|---|---|
| `split.csv` | the fixed partition, IDRiD + APTOS |
| `loadSplit.m` | reads it, attaches full paths, filters by split/dataset |
| `buildDatastores.m` | datastores through `loadFundus`; augmentation on train only |
| `classWeights.m` | per-class loss weights |
| `runBaseline.m` | frozen backbone -> features -> linear classifier |
| `rocReport.m` | ROC curves, AUC with CI, achievable operating points |

## Baseline result to beat

Frozen ResNet-18, IDRiD only, threshold from held-out val:
**AUC 0.877 (95% CI 0.801-0.932)**, sensitivity 90.6%, specificity 59.0% on the
103-image test set. Quote the AUC, not the operating point -- it is
threshold-independent and it is what published DR work reports.
