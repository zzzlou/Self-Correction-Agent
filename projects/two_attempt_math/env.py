from __future__ import annotations

import copy
from typing import Any

from rllm.environments.base.multi_turn_env import MultiTurnEnvironment
from rllm.rewards.reward_fn import math_reward_fn
from rllm.rewards.reward_types import RewardOutput

ATTEMPT_1_TEMPLATE = """You are solving a math problem.

Problem:
{question}

Give a concise derivation and end with exactly one final boxed answer of the form \\boxed{{...}}."""

ATTEMPT_2_TEMPLATE = """You have one final retry.

Original problem:
{question}

Your previous attempt:
{attempt_1}

Feedback from the environment:
incorrect

Revise your answer based on the previous attempt. Do not repeat the same mistake.
End with exactly one final boxed answer of the form \\boxed{{...}}."""


def _normalize_action_for_math_reward(action: str) -> str:
    """Preserve current math grading while tolerating models that omit think tags."""
    if "</think>" in action:
        return action
    return f"<think></think>\n{action}"


def two_attempt_math_reward_fn(task_info: dict, action: Any, attempt_index: int) -> RewardOutput:
    """Use the existing math grader, then remap reward based on the attempt index."""
    normalized_action = _normalize_action_for_math_reward(str(action))
    reward_output = math_reward_fn(task_info=task_info, action=normalized_action)
    if reward_output.is_correct:
        mapped_reward = 1.0 if attempt_index == 1 else 0.6
    else:
        mapped_reward = 0.0
    return RewardOutput(reward=mapped_reward, is_correct=reward_output.is_correct, metadata=reward_output.metadata)


def extract_attempt_text(action: Any) -> str:
    if hasattr(action, "action"):
        return str(action.action)
    return str(action)


def strip_hidden_reasoning(text: str) -> str:
    _, sep, visible_text = text.partition("</think>")
    if sep:
        return visible_text.strip()
    return text.strip()


class TwoAttemptSelfCorrectionEnv(MultiTurnEnvironment):
    def __init__(self, task: dict | None = None, **kwargs):
        super().__init__(task=task, max_turns=2, **kwargs)
        self.original_task: dict | None = None
        self.attempt_1_action: str = ""
        self.attempt_1_correct: bool = False
        self.corrected_on_second_attempt: bool = False

    def reset(self, task: dict | None = None):
        observation, _ = super().reset(task=task)
        assert observation is not None, "Task is required for reset"
        self.original_task = copy.deepcopy(observation)
        self.attempt_1_action = ""
        self.attempt_1_correct = False
        self.corrected_on_second_attempt = False

        initial_obs = {
            "state_prompt": self.build_attempt_1_prompt(self.original_task["question"]),
            "attempt_index": 1,
        }
        initial_info = {
            "attempt_index": 1,
            "attempt_1_correct": False,
            "used_second_attempt": False,
            "corrected_on_second_attempt": False,
            "final_correct": False,
        }
        return initial_obs, initial_info

    def step(self, action: Any) -> tuple[Any, float, bool, dict]:
        assert self.original_task is not None, "Environment must be reset before stepping"
        self.history.append(action)
        reward, next_obs = self.get_reward_and_next_obs(self.original_task, action)
        self.current_turn += 1

        info = {
            "attempt_index": self.current_turn,
            "attempt_1_correct": self.attempt_1_correct,
            "used_second_attempt": not self.attempt_1_correct,
            "corrected_on_second_attempt": self.corrected_on_second_attempt,
            "final_correct": self.attempt_1_correct if self.current_turn == 1 else self.corrected_on_second_attempt,
        }

        self.done = self.current_turn >= self.max_turns or self.attempt_1_correct
        if self.done:
            return {}, reward, True, info
        return next_obs, reward, False, info

    def get_reward_and_next_obs(self, task: dict, action: Any) -> tuple[float, dict]:
        next_attempt_index = self.current_turn + 1
        reward_output = two_attempt_math_reward_fn(task_info=task, action=action, attempt_index=next_attempt_index)

        if next_attempt_index == 1:
            self.attempt_1_action = strip_hidden_reasoning(extract_attempt_text(action))
            self.attempt_1_correct = reward_output.is_correct
            if reward_output.is_correct:
                return reward_output.reward, {}
            next_obs = {
                "state_prompt": self.build_attempt_2_prompt(
                    question=task["question"],
                    attempt_1=self.attempt_1_action,
                ),
                "attempt_index": 2,
            }
            return reward_output.reward, next_obs

        self.corrected_on_second_attempt = reward_output.is_correct and not self.attempt_1_correct
        return reward_output.reward, {}

    @staticmethod
    def build_attempt_1_prompt(question: str) -> str:
        return ATTEMPT_1_TEMPLATE.format(question=question)

    @staticmethod
    def build_attempt_2_prompt(question: str, attempt_1: str) -> str:
        return ATTEMPT_2_TEMPLATE.format(question=question, attempt_1=attempt_1)

    @staticmethod
    def from_dict(env_args: dict) -> "TwoAttemptSelfCorrectionEnv":
        if "task" in env_args:
            task = env_args["task"]
        else:
            task = env_args
        return TwoAttemptSelfCorrectionEnv(task=task)


TwoAttemptMathEnv = TwoAttemptSelfCorrectionEnv
