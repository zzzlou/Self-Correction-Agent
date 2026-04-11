from rllm.agents.agent import Episode
from rllm.workflows.workflow import TerminationEvent, TerminationReason, Workflow


class MultiTurnWorkflow(Workflow):
    def __init__(
        self,
        agent_cls,
        env_cls,
        agent_args=None,
        env_args=None,
        max_steps=5,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Initialize mutable defaults
        agent_args = dict(agent_args) if agent_args is not None else {}
        env_args = dict(env_args) if env_args is not None else {}

        self.agent = agent_cls(**agent_args)
        self.register_agent(self.agent)
        self.env = env_cls(**env_args)
        self.max_steps = max_steps

    async def run(self, task: dict, uid: str, **kwargs) -> Episode | None:
        """Execute a multi-step workflow"""

        observation, info = await self.run_in_executor(self.reset, task=task, uid=uid)  # returns observation and info from the environment

        self.agent.update_from_env(observation, 0, False, info)

        for _ in range(1, self.max_steps + 1):
            response = (await self.get_model_response(self.agent, **kwargs)).text
            action = self.agent.update_from_model(response)

            next_obs, reward, done, info = await self.run_in_executor(self.env.step, action)
            self.agent.update_from_env(next_obs, reward, done, info)

            if self._termination_buffer is not None:
                raise TerminationEvent(self._termination_buffer)

            if done:
                raise TerminationReason.ENV_DONE

        raise TerminationReason.MAX_TURNS_EXCEEDED
