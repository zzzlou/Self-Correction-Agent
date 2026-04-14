# Shared First-Failure Subset Results

## Question

On the exact same MATH-500 problems where both the base model and `ckpt650` fail attempt 1, is `ckpt650` better at correcting itself on attempt 2?

This isolates second-turn correction ability from the main confound that the RL model already fails less often on attempt 1.

## Inputs

- Base episodes:
  - `logs/two_attempt_math_eval/base_model/episodes.json`
- Checkpoint episodes:
  - `logs/two_attempt_math_eval/step_650/episodes.json`

Join key:

- `question`

Validation checks passed:

- `question` is unique within each file
- both files contain `500` questions
- joined set size is `500`

## Shared First-Failure Subset

Definition:

- `shared_failure_subset = {q | base.attempt_1_correct == False and ckpt650.attempt_1_correct == False}`

Observed subset size:

- `100`

Main table:

| subset | size | base correction rate | ckpt650 correction rate | delta |
|---|---:|---:|---:|---:|
| shared first-failure subset | 100 | 0.040 | 0.230 | 0.190 |

Equivalent corrected counts on the subset:

- base corrected on attempt 2: `4`
- `ckpt650` corrected on attempt 2: `23`

## 2x2 Breakdown

| outcome bucket | count |
|---|---:|
| corrected by both | 3 |
| corrected by base only | 1 |
| corrected by ckpt650 only | 20 |
| corrected by neither | 76 |

## Interpretation

- `ckpt650` already improves attempt-1 accuracy substantially on the full benchmark.
- To isolate second-turn correction ability, we restrict analysis to the same problems where both models fail attempt 1.
- On this controlled failure subset, `ckpt650` achieves a higher second-turn correction rate than the base model because the measured delta is positive.
- This is the cleanest support for the “self-correction agent” story because it removes the main confound that the RL model simply fails less often on attempt 1.

The strongest concrete evidence here is:

- `correction_rate_delta = +0.190`
- `ckpt650_only = 20`
- `base_only = 1`

So on the same first-failure problems, `ckpt650` rescues many more examples on the second attempt than the base model.

## Output Files

- Markdown summary:
  - `logs/two_attempt_math_eval/shared_failure_analysis/summary.md`
- JSON summary:
  - `logs/two_attempt_math_eval/shared_failure_analysis/summary.json`
- Case pack:
  - `logs/two_attempt_math_eval/shared_failure_analysis/case_pack.json`

Implementation:

- `projects/two_attempt_math/analyze_shared_failures.py`
