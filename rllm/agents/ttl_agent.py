import copy
import re
from typing import Any, Dict, List, Tuple, Optional
from rllm.agents.agent import Action, BaseAgent, Step, Trajectory
import sys


class TTLAgent(BaseAgent):
    def __init__(self, system_prompt, **kwargs):
        self.system_prompt = system_prompt
        self.current_obs_info = {}
        self.reset()

    def reset(self):
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self._trajectory = Trajectory()
        self.current_obs_info = {}

    def update_from_model(self, response: str, **kwargs) -> Action:
        # 解析 <learn> 标签
        # 这里不需要解析 thought 或 answer，因为这个 Agent 不做题
        learn_matches = re.findall(r"<learn>(.*?)</learn>", response, flags=re.DOTALL)
        rule = "\n".join([m.strip() for m in learn_matches]) if learn_matches else ""
        
        # 记录 Step 用于 RL 训练
        new_step = Step(
            chat_completions=copy.deepcopy(self.messages),
            model_response=response,
            action=Action(action=rule), # Action 就是这条 Rule
            info=copy.deepcopy(self.current_obs_info)
        )
        self._trajectory.steps.append(new_step)
        return Action(action=rule)

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        # observation 这里是一个字符串，包含了上一题的 Question + Answer + Ground Truth
        # 这就是 Meta-Agent 的 State
        self.current_obs_info = info
        user_prompt = f"Here is the trajectory of the previous task attempt:\n{observation}\n\nPlease reflect on it and give a feedback on how to improve"
        
        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    @property
    def chat_completions(self) -> List[Dict[str, str]]:
        return copy.deepcopy(self.messages)

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory