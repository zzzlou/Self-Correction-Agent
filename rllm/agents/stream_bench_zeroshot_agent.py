import copy
import re
from typing import Any, Dict, List, Tuple, Optional
from rllm.agents.agent import Action, BaseAgent, Step, Trajectory
import sys
class StreamBenchZeroShotAgent(BaseAgent):
    def __init__(self, system_prompt: str, **kwargs):
        self.system_prompt = system_prompt
        self.reset()

    def reset(self) -> None:
        self._trajectory = Trajectory()
        self.current_observation: Any = None
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]

    def _parse_model_response(self, response: str) -> Tuple[str, Optional[str], str]:
        """
        解析 thought, learn, action。
        支持多个 <learn> 标签。
        使用 <answer>...</answer> 提取最终答案。
        """
        thought = ""
        learn_content = None
        action_str = ""
        
        # 定义颜色代码 (绿色高亮)
        COLOR_GREEN = "\033[92m"
        COLOR_RESET = "\033[0m"

        # 1. Extract ALL Learn tags (支持多个)
        # re.findall 会返回一个列表，包含所有匹配到的内容
        learn_matches = re.findall(r"<learn>(.*?)</learn>", response, flags=re.DOTALL)
        
        num_learns = len(learn_matches)
        # print(f"{COLOR_GREEN}[Parser] Detected {num_learns} <learn> tag(s){COLOR_RESET}")
        if num_learns > 0:

            # 将多个 learn 内容合并成一个字符串，用换行符分隔
            learn_content = "\n".join([m.strip() for m in learn_matches])

        # 2. Extract Think (通常只有一个，取第一个即可)
        think_match = re.search(r"<think>(.*?)</think>", response, flags=re.DOTALL)
        if think_match:
            thought = think_match.group(1).strip()

        # 3. Extract Action (<answer> 格式)
        answer_match = re.search(r"<answer>(.*?)</answer>", response, flags=re.DOTALL)
        
        if answer_match:
            action_str = answer_match.group(1).strip()
        else:
            cleaned_response = response
            if thought: 
                cleaned_response = re.sub(r"<think>.*?</think>", "", cleaned_response, flags=re.DOTALL)
            action_str = cleaned_response.strip()
        return thought, learn_content, action_str

    def update_from_model(self, response: str, **kwargs) -> Action:
        """
        解析 LLM 输出，更新内部记忆 (Learned Rules)，并返回 Action 给 Env。
        """
        # 1. 解析
        thought, learn_content, action_str = self._parse_model_response(response)
        

        # 3. 记录 Step (用于训练的数据)
        # 注意：此时 self.chat_completions 是生成这个 response 时的 Prompt (含旧 Rules)
        new_step = Step(
            chat_completions=copy.deepcopy(self.chat_completions),
            thought=thought,
            action=action_str,
            model_response=response,
            observation=self.current_observation
        )
        self._trajectory.steps.append(new_step)

        # 4. 临时追加 assistant msg (为了应付 Engine 检查，马上会被 update_from_env 清除)
        self.messages.append({"role": "assistant", "content": response})
        
        return Action(action=action_str)

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        """
        接收新题(observation) 和 上一题反馈(info)。
        构造包含 Rules + Feedback + NewQuestion 的完整 Prompt。
        """
        # 1. 更新 Trajectory 最后一步的 reward/info
        if self._trajectory.steps:
            last_step = self._trajectory.steps[-1]
            last_step.reward = reward
            last_step.done = done
            last_step.info = info

        if not done:
            self.current_observation = observation
            
            # --- C. 构造 New Question 部分 ---
            # 假设 observation 是一个包含 'prompt_zeroshot' 的字典，或者直接就是字符串
            if isinstance(observation, dict):
                new_question_text = observation.get('prompt_zeroshot', str(observation))
            else:
                new_question_text = str(observation)

            # --- D. 最终拼装 & 重置 Context ---
            full_user_prompt = f"{new_question_text}"
            
            
            # 【强制重置】只保留 System + 当前构造好的 Full Prompt
            self.messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": full_user_prompt}
            ]

    @property
    def chat_completions(self) -> List[Dict[str, str]]:
        return copy.deepcopy(self.messages)

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory