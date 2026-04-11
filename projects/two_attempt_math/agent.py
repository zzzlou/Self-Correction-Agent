from __future__ import annotations

import copy
from typing import Any

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory


class CompactStateSelfCorrectionAgent(BaseAgent):
    """A math agent whose prompt state is replaced each turn instead of accumulated."""

    def __init__(self, system_prompt: str | None = None):
        self.system_prompt = system_prompt
        self.reset()

    def reset(self) -> None:
        self._trajectory = Trajectory()
        self._current_messages: list[dict[str, str]] = []
        self._current_observation: Any = None
        self._current_info: dict[str, Any] = {}

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        if observation is None or (isinstance(observation, dict) and observation == {}):
            current_step = self.get_current_state()
            if current_step is not None:
                current_step.reward = reward
                current_step.done = done
                current_step.info = copy.deepcopy(info)
            return

        if not isinstance(observation, dict) or "state_prompt" not in observation:
            raise ValueError(f"Observation must contain 'state_prompt': {observation}")

        state_prompt = observation["state_prompt"]
        self._current_observation = copy.deepcopy(observation)
        self._current_info = copy.deepcopy(info)

        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": state_prompt})
        self._current_messages = messages

    def update_from_model(self, response: str, **kwargs) -> Action:
        current_step = Step(
            chat_completions=copy.deepcopy(self.chat_completions),
            observation=copy.deepcopy(self._current_observation),
            model_response=response,
            action=Action(action=response.strip()),
            info=copy.deepcopy(self._current_info),
        )
        self._trajectory.steps.append(current_step)
        return current_step.action

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        return copy.deepcopy(self._current_messages)

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory
