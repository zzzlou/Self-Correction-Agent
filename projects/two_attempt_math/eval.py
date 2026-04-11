from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path

from transformers import AutoTokenizer

from projects.two_attempt_math.agent import CompactStateSelfCorrectionAgent
from projects.two_attempt_math.env import TwoAttemptSelfCorrectionEnv
from projects.two_attempt_math.prepare_data import TEST_DATASET_NAME, prepare_two_attempt_math_data
from rllm.agents.agent import Episode
from rllm.data.dataset import DatasetRegistry
from rllm.engine import OpenAIEngine
from rllm.workflows.workflow import TerminationReason


def load_eval_tasks():
    dataset = DatasetRegistry.load_dataset(TEST_DATASET_NAME, "test")
    if dataset is None:
        _, dataset = prepare_two_attempt_math_data()
    return dataset.get_data()


def summarize_results(results):
    total = len(results)
    if total == 0:
        return {
            "num_examples": 0,
            "first_pass_accuracy": 0.0,
            "final_accuracy": 0.0,
            "correction_rate": 0.0,
            "second_attempt_rate": 0.0,
            "pass_at_1": 0.0,
            "pass_at_2": 0.0,
        }

    first_try_correct = 0
    final_correct = 0
    used_second_attempt = 0
    corrected_on_second_attempt = 0
    first_try_failures = 0

    for episode in results:
        _, trajectory = episode.trajectories[0]
        info = trajectory.steps[-1].info or {}
        attempt_1_correct = bool(info.get("attempt_1_correct", False))
        final_is_correct = bool(info.get("final_correct", False))
        used_retry = bool(info.get("used_second_attempt", False))
        corrected = bool(info.get("corrected_on_second_attempt", False))

        first_try_correct += int(attempt_1_correct)
        final_correct += int(final_is_correct)
        used_second_attempt += int(used_retry)
        corrected_on_second_attempt += int(corrected)
        first_try_failures += int(not attempt_1_correct)

    correction_rate = corrected_on_second_attempt / first_try_failures if first_try_failures else 0.0
    first_pass_accuracy = first_try_correct / total
    final_accuracy = final_correct / total
    second_attempt_rate = used_second_attempt / total
    return {
        "num_examples": total,
        "first_pass_accuracy": first_pass_accuracy,
        "final_accuracy": final_accuracy,
        "correction_rate": correction_rate,
        "second_attempt_rate": second_attempt_rate,
        "pass_at_1": first_pass_accuracy,
        "pass_at_2": final_accuracy,
    }


def serialize_results(results):
    serialized = []
    for episode in results:
        _, trajectory = episode.trajectories[0]
        attempt_1 = trajectory.steps[0]
        attempt_2 = trajectory.steps[1] if len(trajectory.steps) > 1 else None
        final_info = trajectory.steps[-1].info or {}
        serialized.append(
            {
                "id": episode.id,
                "question": episode.task["question"],
                "attempt_1_response": attempt_1.model_response,
                "attempt_2_response": attempt_2.model_response if attempt_2 is not None else None,
                "attempt_1_correct": final_info.get("attempt_1_correct", False),
                "final_correct": final_info.get("final_correct", False),
                "corrected_on_second_attempt": final_info.get("corrected_on_second_attempt", False),
            }
        )
    return serialized


async def run_episode(task: dict, rollout_engine: OpenAIEngine) -> Episode:
    agent = CompactStateSelfCorrectionAgent()
    env = TwoAttemptSelfCorrectionEnv(task=task)

    observation, info = env.reset()
    agent.reset()
    agent.trajectory.task = task
    agent.update_from_env(observation=observation, reward=0.0, done=False, info=info)

    while True:
        output = await rollout_engine.get_model_response(agent.chat_completions)
        action = agent.update_from_model(output.text)
        next_observation, reward, done, step_info = env.step(action.action)
        agent.update_from_env(observation=next_observation, reward=reward, done=done, info=step_info)
        if done:
            break

    episode = Episode(
        id=str(uuid.uuid4()),
        task=task,
        termination_reason=TerminationReason.ENV_DONE,
        trajectories=[("agent", agent.trajectory)],
    )
    episode.is_correct = bool(agent.trajectory.steps[-1].info.get("final_correct", False))
    episode.metrics = summarize_results([episode])
    return episode


async def evaluate_tasks(tasks, rollout_engine, n_parallel_tasks: int):
    semaphore = asyncio.Semaphore(n_parallel_tasks)

    async def _run(task):
        async with semaphore:
            return await run_episode(task, rollout_engine)

    return await asyncio.gather(*[_run(task) for task in tasks])


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    n_parallel_tasks = int(os.environ.get("TWO_ATTEMPT_MATH_PARALLEL", "64"))
    model_name = os.environ.get("MODEL_NAME", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    base_url = os.environ.get("BASE_URL", "http://localhost:30000/v1")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    rollout_engine = OpenAIEngine(
        model=model_name,
        tokenizer=tokenizer,
        base_url=base_url,
        api_key=os.environ.get("OPENAI_API_KEY", "None"),
        sampling_params={"temperature": 0.6, "top_p": 0.95, "max_tokens": 2048},
    )

    tasks = load_eval_tasks()
    results = asyncio.run(evaluate_tasks(tasks, rollout_engine, n_parallel_tasks=n_parallel_tasks))
    summary = summarize_results(results)

    output_dir = Path(os.environ.get("TWO_ATTEMPT_MATH_OUTPUT_DIR", "logs/two_attempt_math"))
    output_dir.mkdir(parents=True, exist_ok=True)
    with output_dir.joinpath("summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with output_dir.joinpath("episodes.json").open("w", encoding="utf-8") as f:
        json.dump(serialize_results(results), f, indent=2)

    print(json.dumps(summary, indent=2))
