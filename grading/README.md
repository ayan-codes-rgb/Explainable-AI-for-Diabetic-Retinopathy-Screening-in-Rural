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

## Day 1 — what to run

```matlab
cd 'C:\\Users\\c7849\\OneDrive\\Desktop\\SIH Diabetic Retinopathy'
addpath(genpath('common')); addpath(genpath('grading'))
root = 'C:\\Users\\c7849\\Downloads\\dr-data\\IDRiD\\B. Disease Grading';

S = buildDatastores(root);            % datastores wired to the fixed split
classWeights(S.train.labels)          % see the imbalance and the weights
R = runBaseline(root);                % frozen backbone + linear classifier
```

`runBaseline` prints validation- and test-set sensitivity/specificity for
referable DR and returns the trained model plus the tuned threshold.

### Files

| file | what it does |
|---|---|
| `loadSplit.m` | reads the fixed train/val/test split |
| `buildDatastores.m` | datastores through `loadFundus`, augmentation on train only |
| `classWeights.m` | per-class loss weights, with the reasoning |
| `runBaseline.m` | CPU baseline: frozen backbone -> features -> linear classifier |

### Backbone choice

**ResNet-18** at 224x224, used as a frozen feature extractor.

Not ResNet-50: this machine has an AMD GPU and MATLAB is CUDA-only, so
everything runs on CPU until the GPU machine is confirmed. ResNet-18 is
~4x cheaper and, as a frozen extractor, the difference in feature quality
is small compared with the difference in how many experiments you get to run.

Feature extraction rather than fine-tuning, for the same reason: one forward
pass over the data, then seconds per experiment. That is what lets you tune
class weights and the decision threshold, which is what actually moves
sensitivity.

Fine-tuning is step 2, once there is a GPU. The baseline then becomes the
ablation comparison the problem statement asks for.

### Class balancing

Default is `inverse-sqrt`, not full inverse frequency. Full inverse
frequency on a class with 17 training images makes each of those images
enormously influential and the model memorises them.

Weight grade 3, not grade 0. On IDRiD+APTOS combined, grade 3 (severe NPDR)
is the scarcest class at 6.8% -- and it is on the *referable* side, so
errors there cost **sensitivity** (>90% target). Grade 1 costs
**specificity** (>85%). The errors are not equally expensive.

### Known unknowns

`runBaseline` has not been executed -- no MATLAB on the machine that wrote
it. The logic is sound but expect one round of fixes, most likely around
the pretrained-network API (`imagePretrainedNetwork` vs `resnet18`) or the
feature layer name. Both are handled defensively and error with a message
naming the fix.
