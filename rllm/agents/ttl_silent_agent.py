from rllm.agents.agent import BaseAgent, Action, Trajectory, Step
import copy

class SilentAgent(BaseAgent):
    """一个永远只返回空动作的 Agent"""
    def __init__(self, **kwargs):
        self.reset()
        
    def reset(self):
        self._trajectory = Trajectory()
        self.messages = [] # 保持为空，或者放一个假的 system prompt

    def update_from_model(self, response: str, **kwargs) -> Action:
        # 无论 Engine 传什么 response 进来，直接无视
        # 返回空字符串，Env 会检测 if new_feedback: ... 从而跳过添加历史
        return Action(action="") 

    def update_from_env(self, observation, reward, done, info, **kwargs):
        # 记录 dummy step，保证 Engine 的 logging 正常工作
        self._trajectory.steps.append(Step(
            chat_completions=[], 
            model_response="", 
            action=Action(action=""), 
            info=info
        ))
        # 这里的 observation 就是上一局的 Log，我们直接丢弃，不放入 memory

    @property
    def chat_completions(self):
        return []

    @property
    def trajectory(self):
        return self._trajectory