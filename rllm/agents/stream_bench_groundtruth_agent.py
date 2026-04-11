import copy
from typing import Any, Dict, List

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory
from rllm.environments.base.base_env import BaseEnv

class StreamBenchGroundTruthAgent(BaseAgent):
    """
    一个“作弊”的 Agent，用于在 rllm 框架中测试 StreamBenchEnv。

    它伪装成一个 LLM Agent，但它会：
    1. 在 `update_from_env` 中，从 observation 字典里偷看 "label_text" (正确答案)。
    2. 在 `update_from_model` 中，完全忽略 LLM 的（无效）回复，
       并直接返回偷看到的 "label_text" 作为 action。
    
    这会浪费 API 调用（引擎仍然会调用 LLM），但它允许我们在
    不修改 AgentExecutionEngine 的情况下测试环境的端到端流程。
    """

    def __init__(self, **kwargs):
        self.reset()

    def reset(self) -> None:
        """重置 Agent 的内部状态。"""
        self._trajectory = Trajectory()
        self.current_observation: Any = None
        self.ground_truth: Any = None # 用于存储“偷看”到的答案
        
        # 我们必须提供一个系统提示，否则引擎的第一次 LLM 调用会失败
        self.messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": "You are a ground truth agent. You will ignore all user input "
                           "and your output will be programmatically controlled."
            }
        ]

    def update_from_env(
        self,
        observation: Any,
        reward: float,
        done: bool,
        info: dict,
        **kwargs
    ):
        """
        从环境中接收一个新的 observation。
        这是我们“偷看”正确答案的地方。
        """
        
        if observation is not None:
            if "ground_truth_label" not in observation:
                raise ValueError(
                    "StreamBenchEnv observation is missing 'ground_truth_label'. "
                    "GroundTruthAgent cannot function."
                )
            self.ground_truth = observation["ground_truth_label"]
        
        self.current_observation = str(observation)

        # 2. 将当前状态添加到 trajectory (如果不是第一次)
        if self._trajectory.steps:
            last_step = self._trajectory.steps[-1]
            last_step.reward = reward
            last_step.done = done
            last_step.info = info

        # 3. 添加一个“假的”用户消息，以保持聊天记录的交替
        #    引擎会把这个消息连同历史记录一起发送给 LLM
        time_step = info.get("time_step", 0)
        
        # 只有在有新问题时才添加用户消息 (即 done=False)
        if not done:
            user_prompt = f"Time step {time_step}. Received new observation."
            self.messages.append({"role": "user", "content": user_prompt})

    def update_from_model(self, response: str, **kwargs) -> Action:
        
        # 1. 这就是我们的“作弊” action

        action_str = f"{self.ground_truth}." if self.ground_truth else ""
        
        # 2. 记录这一步
        thought = "GroundTruthAgent: Using stored ground truth answer."
        
        new_step = Step(
            chat_completions=copy.deepcopy(self.chat_completions),
            thought=thought,
            action=action_str,
            model_response=response, # 记录下无效的 LLM 回复
            observation=self.current_observation
        )
        self._trajectory.steps.append(new_step)

        # 3. 添加一个“假的”助手消息，以保持聊天记录的交替
        self.messages.append({"role": "assistant", "content": "[Returning Ground Truth]"})
        
        # 4. 返回包含正确答案的 Action
        return Action(action=action_str)

    @property
    def chat_completions(self) -> List[Dict[str, str]]:
        """返回引擎将发送给 LLM 的消息。"""
        messages = copy.deepcopy(self.messages)
        return messages

    @property
    def trajectory(self) -> Trajectory:
        """返回完整的轨迹。"""
        return self._trajectory