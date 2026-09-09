# grading

**Owner: M4**

DR severity grading on the ICDR 0-4 scale using a fine-tuned pretrained CNN. Target: >90% sensitivity, >85% specificity for referable DR (grade >= 2).

Must fill in the shared result struct: `result.drGrade`, `result.referable`, `result.confidence`

Read `../common/README.md` before writing any image code. Use `loadFundus()`, never `imread()`.
