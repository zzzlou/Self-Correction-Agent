# RL Post-Training for Feedback-Conditioned Self-Correction

This project turns the single-turn `simple_math` example into a two-attempt math workflow:

- Attempt 1 sees only the original problem
- The environment returns binary feedback through correctness only
- If attempt 1 is wrong, attempt 2 sees the original problem, the previous answer, and the literal feedback `incorrect`
- Attempt 2 is the final chance

The implementation reuses the existing math grading stack, but replaces cumulative transcript growth with a project-local compact-state agent and environment. Attempt 2 is a fresh prompt rebuilt from the question, the visible prior answer, and the binary feedback `incorrect`.

## Datasets

- Train: `hendrycks_math`
- Eval: `math500`

The project registers them under:

- `two_attempt_math_train`
- `two_attempt_math_test`

Prepare them with:

```bash
python -m projects.two_attempt_math.prepare_data
```

## Training

Basic training entrypoint:

```bash
python -m projects.two_attempt_math.train \
  data.train_batch_size=32 \
  data.max_prompt_length=4096 \
  data.max_response_length=2048 \
  actor_rollout_ref.model.path=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  rllm.stepwise_advantage.enable=True \
  rllm.stepwise_advantage.mode=per_step
```

The project is trained through the standard agent+environment trainer path with a non-cumulative agent. Reward mapping is:

- first-attempt correct: `1.0`
- second-attempt correction: `0.6`
- otherwise: `0.0`

Exact correctness is still determined by the existing math reward function.

## Evaluation

Start a model server, then run:

```bash
python -m projects.two_attempt_math.eval
```

Environment variables:

- `MODEL_NAME` defaults to `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- `BASE_URL` defaults to `http://localhost:30000/v1`
- `TWO_ATTEMPT_MATH_PARALLEL` controls evaluation concurrency
- `TWO_ATTEMPT_MATH_OUTPUT_DIR` controls output location

The evaluation writes:

- `summary.json`
- `episodes.json`

If you want to recompute the true plan-defined metrics offline from a saved `episodes.json`, run:

```bash
python -m projects.two_attempt_math.compute_metrics logs/two_attempt_math/episodes.json
```

You can also write a fresh summary file:

```bash
python -m projects.two_attempt_math.compute_metrics \
  logs/two_attempt_math/episodes.json \
  --output logs/two_attempt_math/offline_summary.json
```

Core metrics:

- `first_pass_accuracy`: first-attempt correctness
- `final_accuracy`: final correctness after at most two attempts
- `correction_rate`: fraction of first-attempt failures fixed on attempt 2
- `second_attempt_rate`: fraction of problems that required attempt 2

Compatibility aliases are also written:

- `pass_at_1 = first_pass_accuracy`
- `pass_at_2 = final_accuracy`
