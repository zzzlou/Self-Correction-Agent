from __future__ import annotations

from rllm.workflows.multi_turn_workflow import MultiTurnWorkflow
from rllm.workflows.workflow import Episode, TerminationReason

from projects.two_attempt_math.agent import CompactStateSelfCorrectionAgent
from projects.two_attempt_math.env import TwoAttemptSelfCorrectionEnv


class TwoAttemptMathWorkflow(MultiTurnWorkflow):
    def __init__(self, agent_args=None, env_args=None, **kwargs):
        super().__init__(
            agent_cls=CompactStateSelfCorrectionAgent,
            env_cls=TwoAttemptSelfCorrectionEnv,
            agent_args=agent_args or {},
            env_args=env_args or {},
            max_steps=2,
            **kwargs,
        )

    async def run(self, task: dict, uid: str, **kwargs) -> Episode | None:
        observation, info = await self.run_in_executor(self.reset, task=task, uid=uid)
        self.agent.update_from_env(observation, 0.0, False, info)

        for _ in range(self.max_steps):
            response = (await self.get_model_response(self.agent, **kwargs)).text
            action = self.agent.update_from_model(response)
            action.raw_response = response

            next_obs, reward, done, step_info = await self.run_in_executor(self.env.step, action)

            current_step = self.agent.get_current_state()
            current_step.reward = reward
            current_step.done = done
            current_step.info = step_info

            if self._termination_buffer is not None:
                return self.postprocess_episode(self.collect_trajectories(), self._termination_buffer)

            if done:
                return self.postprocess_episode(self.collect_trajectories(), TerminationReason.ENV_DONE)

            self.agent.update_from_env(next_obs, 0.0, False, step_info)

        return self.postprocess_episode(self.collect_trajectories(), TerminationReason.MAX_TURNS_EXCEEDED)

    def assign_episode_correctness(self, episode: Episode) -> None:
        if not episode.trajectories:
            episode.is_correct = False
            return
        _, trajectory = episode.trajectories[0]
        final_step = trajectory.steps[-1]
        episode.is_correct = bool(final_step.info.get("final_correct", False))

    def collect_metrics(self, episode: Episode) -> None:
        if not episode.trajectories:
            episode.metrics = {
                "first_pass_accuracy": 0.0,
                "final_accuracy": 0.0,
                "correction_rate": 0.0,
                "second_attempt_rate": 0.0,
                "pass_at_1": 0.0,
                "pass_at_2": 0.0,
            }
            return

        _, trajectory = episode.trajectories[0]
        final_step = trajectory.steps[-1]
        info = final_step.info or {}

        attempt_1_correct = bool(info.get("attempt_1_correct", False))
        used_second_attempt = bool(info.get("used_second_attempt", False))
        corrected_on_second_attempt = bool(info.get("corrected_on_second_attempt", False))
        final_correct = bool(info.get("final_correct", False))

        first_pass_accuracy = float(attempt_1_correct)
        final_accuracy = float(final_correct)
        episode.metrics = {
            "first_pass_accuracy": first_pass_accuracy,
            "final_accuracy": final_accuracy,
            "correction_rate": float(corrected_on_second_attempt),
            "second_attempt_rate": float(used_second_attempt),
            "pass_at_1": first_pass_accuracy,
            "pass_at_2": final_accuracy,
        }
