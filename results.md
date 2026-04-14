# Two-Attempt Math Self-Correction Results

## Setup

- Task: `projects.two_attempt_math.eval`
- Eval set: `MATH-500` (`500` examples)
- Protocol:
  - Attempt 1 sees only the original problem
  - If attempt 1 is wrong, attempt 2 sees the original problem, the previous visible answer, and binary feedback `incorrect`
  - Metrics are computed from `projects.two_attempt_math.compute_metrics`
- Sampling:
  - `temperature=0.6`
  - `top_p=0.95`
  - `max_tokens=2048`
- Parallelism: `TWO_ATTEMPT_MATH_PARALLEL=32`

## Models Evaluated

- Base model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- RL checkpoints, evaluated from copied checkpoints under:
  - `checkpoints/rllm-agent/two-attempt-math_eval_copy/global_step_550/actor/hf_merged`
  - `checkpoints/rllm-agent/two-attempt-math_eval_copy/global_step_600/actor/hf_merged`
  - `checkpoints/rllm-agent/two-attempt-math_eval_copy/global_step_650/actor/hf_merged`

## Main Results

| model | first_pass_accuracy | final_accuracy | correction_gain | correction_rate | second_attempt_rate |
|---|---:|---:|---:|---:|---:|
| base_model | 0.466 | 0.582 | 0.116 | 0.217 | 0.534 |
| 550 | 0.772 | 0.828 | 0.056 | 0.246 | 0.228 |
| 600 | 0.764 | 0.810 | 0.046 | 0.195 | 0.236 |
| 650 | 0.782 | 0.834 | 0.052 | 0.239 | 0.218 |

Definitions:

- `first_pass_accuracy`: attempt-1 accuracy
- `final_accuracy`: accuracy after at most two attempts
- `correction_gain = final_accuracy - first_pass_accuracy`
- `correction_rate`: among attempt-1 failures, fraction fixed on attempt 2
- `second_attempt_rate`: fraction of examples that entered attempt 2

## Key Takeaways

1. All three RL checkpoints show a positive self-correction effect.
   - `550`: `0.772 -> 0.828`
   - `600`: `0.764 -> 0.810`
   - `650`: `0.782 -> 0.834`

2. `global_step_650` is the best checkpoint by `final_accuracy`.
   - It also has the best `first_pass_accuracy`.
   - Its `correction_rate` is `0.239`, which is clearly above zero.

3. Compared with the base model, RL training massively improves overall performance.
   - Base model: `first_pass=0.466`, `final=0.582`
   - Best RL checkpoint (`650`): `first_pass=0.782`, `final=0.834`

4. The main improvement over the base model is not only "second attempt helps", but also "attempt 1 is much stronger".
   - Base model has a larger `correction_gain` (`0.116`) mainly because it fails much more often on attempt 1 and therefore uses attempt 2 much more (`0.534` vs `0.218`).
   - For the self-correction story, `correction_rate` is the cleaner metric than raw `correction_gain`.

5. The strongest defensible project claim from the current data is:
   - The RL-trained model substantially improves first-pass and final two-attempt accuracy over the base model.
   - The RL-trained model still exhibits a non-trivial second-attempt correction effect on first-pass failures.

## Best Checkpoint

- Best checkpoint: `650`
- Offline summary:
  - `first_pass_accuracy = 0.782`
  - `final_accuracy = 0.834`
  - `correction_gain = 0.052`
  - `correction_rate = 0.239`
  - `second_attempt_rate = 0.218`

Case-pack counts from `step_650`:

- corrected examples available: `26`
- not-corrected examples available: `83`

Example pack file:

- `logs/two_attempt_math_eval/step_650/case_pack.json`

## Output Files

- Aggregate comparisons:
  - `logs/two_attempt_math_eval/comparison.md`
  - `logs/two_attempt_math_eval/comparison.json`
  - `logs/two_attempt_math_eval/comparison_with_base.md`
  - `logs/two_attempt_math_eval/comparison_with_base.json`

- Per-run outputs:
  - `logs/two_attempt_math_eval/base_model/offline_summary.json`
  - `logs/two_attempt_math_eval/step_550/offline_summary.json`
  - `logs/two_attempt_math_eval/step_600/offline_summary.json`
  - `logs/two_attempt_math_eval/step_650/offline_summary.json`

- Episode dumps:
  - `logs/two_attempt_math_eval/base_model/episodes.json`
  - `logs/two_attempt_math_eval/step_550/episodes.json`
  - `logs/two_attempt_math_eval/step_600/episodes.json`
  - `logs/two_attempt_math_eval/step_650/episodes.json`

## Notes

- The checkpoint evaluations were run from copied checkpoints under `checkpoints/rllm-agent/two-attempt-math_eval_copy` to avoid mutating the original training outputs.
- `projects/two_attempt_math/eval.py` was patched to handle the merged tokenizer/chat-template incompatibility by falling back to OpenAI chat completions mode when needed, and to match the current `Episode` API.
- Because evaluation uses sampling (`temperature=0.6`), small differences between nearby checkpoints should not be overinterpreted.
