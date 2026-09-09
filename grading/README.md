# grading

**Owner: M4**

DR severity grading on the ICDR 0-4 scale using a fine-tuned pretrained CNN.
Target: >90% sensitivity and >85% specificity for **referable DR** (grade >= 2).

Must fill in the shared result struct: `result.drGrade`, `result.referable`, `result.confidence`

Read `../common/README.md` before writing any image code. Use `loadFundus()`, never `imread()`.

---

## The split

`idrid_split.csv` is the **fixed** partition of IDRiD's disease-grading set.
Read it with `loadSplit.m`; do not regenerate it.

```matlab
root = 'C:\dr-data\IDRiD\B. Disease Grading';   % wherever you unzipped it
tr   = loadSplit('train', root);
va   = loadSplit('val',   root);

imds = imageDatastore(tr.file, 'ReadFcn', makeFundusReadFcn());
Y    = categorical(tr.grade);
```

Built by stratified holdout: 15% of IDRiD's official 413-image training set,
seed 20260909. The official 103-image test set is carried through untouched.

| grade | train | val | test | train % | val % | test % |
|---|---:|---:|---:|---:|---:|---:|
| 0 no DR | 114 | 20 | 34 | 32.4 | 32.8 | 33.0 |
| 1 mild NPDR | 17 | 3 | 5 | 4.8 | 4.9 | 4.9 |
| 2 moderate NPDR | 116 | 20 | 32 | 33.0 | 32.8 | 31.1 |
| 3 severe NPDR | 63 | 11 | 19 | 17.9 | 18.0 | 18.4 |
| 4 proliferative | 42 | 7 | 13 | 11.9 | 11.5 | 12.6 |
| **total** | **352** | **61** | **103** | | | |

Referable DR (grade >= 2): train 62.8%, val 62.3%, test 62.1%.

## Things that will bite you

**Image names repeat.** Both the official training and testing folders contain
an `IDRiD_001.jpg`. Key on the `id` column or the full path, never on
`image_name`.

**Grade 1 validation is 3 images.** Any per-class metric you compute for mild
NPDR on the validation set is noise. That is not a bug in the split -- IDRiD
only has 25 grade-1 images in total, and stratifying honestly means val gets
3 of them. Report grade-1 performance from the test set (5 images) with the
sample size stated, or fold it into the binary referable metric where it
belongs. APTOS is the real fix; it has far more mild cases.

**Do not touch the test set until the end.** Tune the decision threshold on
`val`, then evaluate `test` once. Re-tuning against test is how a 91% becomes
a number that does not survive questioning.

**Argmax is not your classifier.** Argmax optimises accuracy. You are graded
on sensitivity at >90%, which means deliberately moving the referable
threshold to accept more false positives. Pick the threshold on `val` with
`rocmetrics`/`perfcurve`, then freeze it.

## Suggested order of work

1. Feature-extraction baseline (frozen pretrained net + `fitcecoc`) -- runs on
   CPU in minutes and gives you something to beat
2. Fine-tune once a GPU machine is available
3. Add APTOS to training, keep IDRiD's test set as the reported benchmark
4. Threshold tuning on `val` for the >90%/85% target
5. Ablation vs the baseline in step 1 -- this is the comparison the problem
   statement asks for
