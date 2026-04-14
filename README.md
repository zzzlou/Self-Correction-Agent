# Two-Attempt Math Self-Correction

This project trains a compact-state two-attempt math agent with RL.

- Attempt 1 sees only the original problem.
- If attempt 1 is wrong, attempt 2 sees:
  - the original problem
  - the previous visible answer
  - binary feedback: `incorrect`
- The prompt state is rebuilt each turn instead of accumulating full history.

The goal is not just stronger one-shot math solving, but a policy that can revise its answer after failure.

## Data

- Train: `hendrycks_math`
- Eval: `MATH-500`

Registered dataset names:

- `two_attempt_math_train`
- `two_attempt_math_test`

Prepare data with:

```bash
python -m projects.two_attempt_math.prepare_data
```

## Training

Main launcher:

```bash
bash projects/two_attempt_math/train_two_attempt_math.sh
```

Reward mapping:

- first-attempt correct: `1.0`
- corrected on attempt 2: `0.6`
- otherwise: `0.0`

## Evaluation

Serve a merged checkpoint or base model with vLLM, then run:

```bash
python -m projects.two_attempt_math.eval
```

Useful env vars:

- `MODEL_NAME`
- `BASE_URL`
- `TWO_ATTEMPT_MATH_PARALLEL`
- `TWO_ATTEMPT_MATH_OUTPUT_DIR`

Offline metric recompute:

```bash
python -m projects.two_attempt_math.compute_metrics \
  logs/two_attempt_math/episodes.json \
  --output logs/two_attempt_math/offline_summary.json
```

Core metrics:

- `first_pass_accuracy`
- `final_accuracy`
- `correction_rate`
- `second_attempt_rate`

## Main Result

Evaluated on `MATH-500` with sampling:

- `temperature=0.6`
- `top_p=0.95`
- `max_tokens=2048`

Best checkpoint: `global_step_650`

| model      | first_pass_accuracy | final_accuracy | correction_rate |
| ---------- | ------------------: | -------------: | --------------: |
| base model |               0.466 |          0.582 |           0.217 |
| ckpt650    |               0.782 |          0.834 |           0.239 |

Interpretation:

- RL strongly improves first-pass accuracy.
- RL also improves final two-attempt accuracy.
- The trained model still shows a non-trivial second-attempt correction effect on first-pass failures.

## Controlled Self-Correction Evidence

To isolate second-turn correction ability from the confound that the RL model already fails less often on attempt 1, we compare the base model and `ckpt650` on the exact same problems where both models fail attempt 1.

Shared first-failure subset:

- size: `100`
- base second-turn correction rate: `0.040`
- `ckpt650` second-turn correction rate: `0.230`
- delta: `+0.190`

2x2 breakdown on this subset:

- corrected by both: `3`
- corrected by base only: `1`
- corrected by `ckpt650` only: `20`
- corrected by neither: `76`

This suggests that the trained policy is not only stronger on attempt 1, but is also better at feedback-conditioned self-correction on attempt 2.
