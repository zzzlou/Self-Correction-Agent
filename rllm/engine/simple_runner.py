import asyncio
import time
from typing import Dict, Any, Type, List

from rllm.agents.agent import BaseAgent
from rllm.environments.base.base_env import BaseEnv
from rllm.environments.jericho.openai_helpers import chat_completion_with_retries

class SimpleRunner:
    """
    一个轻量级的 Pipeline Runner，替代 AgentExecutionEngine。
    """
    def __init__(
        self,
        agent_class: Type[BaseAgent],
        env_class: Type[BaseEnv],
        agent_args: Dict[str, Any],
        env_args: Dict[str, Any],
        meta_llm_config: Dict[str, Any], # 专门用于 Meta Agent 的 LLM 配置
        log=True,
    ):
        self.agent_class = agent_class
        self.env_class = env_class
        self.agent_args = agent_args
        self.env_args = env_args
        self.meta_llm_config = meta_llm_config
        self.log = log

    async def execute_tasks(self, tasks: List[Dict],max_concurrent=15) -> List[Any]:
        """执行所有任务"""
        if self.log:
            print(f"🐌 [Serial Runner] Starting execution of {len(tasks)} tasks sequentially...")

        results = []
        start_time_all = time.time()
        import traceback
        for i, task in enumerate(tasks):
            try:
                # 直接调用，不加任何魔法
                traj = self.run_single_task(i, task)
                if traj:
                    results.append(traj)
            except Exception as e:
                print(f"❌ Task {i} Failed: {e}")
                traceback.print_exc() # 打印报错，方便调试
        
        total_time = time.time() - start_time_all
        if self.log:
            print(f"🏁 All tasks finished in {total_time:.2f}s")
            
        return results

    async def run_single_task(self, task_id: int, task_config: Dict,):
        if self.log:
            print(f"\n🎬 [SimpleRunner] Start running Task {task_id}...")
        
        # 1. 准备 Env 参数 (合并 Task Config 和 Env Args)
        # Env.from_dict 会处理 meta_cfg, actor_cls 等参数的提取
        full_env_args = {**task_config, **self.env_args}
        env = self.env_class.from_dict(full_env_args)
        
        # 2. 初始化 Meta Agent
        # agent_args 应该包含 system_prompt
        agent = self.agent_class(**self.agent_args)
        
        # 3. 初始 Observation (Round 0)
        obs, info = env.reset()
        agent.reset()
        
        # 将 Round 0 的结果喂给 Meta-Agent
        agent.update_from_env(observation=obs, reward=0, done=False, info=info)
        
        # 4. 获取 Meta Loop 次数 (从 env 的 meta_cfg 里读，或者统一配置)
        # 这里假设 env 里的 meta_cfg 已经保存了 max_episodes
        max_meta_steps = env.meta_cfg.get("max_episodes", 3)
        
        for step in range(max_meta_steps):
            
            # --- A. Meta Agent 思考 (LLM API Call) ---
            # 获取当前对话历史
            messages = agent.chat_completions
            
            # 调用 LLM (使用 simple_runner 传入的 meta_llm_config)
            if self.log:
                print(f"--- 🔄 Task {task_id} | Optimization Step {step+1}/{max_meta_steps} ---")
                print(f"   🤖 Meta-Agent is giving test time guidance")
            response_text = chat_completion_with_retries(
                model=self.meta_llm_config["model"],
                messages=messages,
                temperature=self.meta_llm_config.get("temperature", 0.7),
            ).choices[0].message.content
            
            # --- B. Meta Agent 生成 Action (解析 XML) ---
            # update_from_model 会解析 <learn> 标签并存入 trajectory
            action = agent.update_from_model(response_text)
            
            if self.log:
                print(f"   💡 Feedback Generated: {action.action}...") # 打印前100个字符
            
            # --- C. 环境执行 (Actor 根据反馈玩游戏) ---
            # env.step 内部会运行一整局游戏
            next_obs, reward, done, info = env.step(action.action)
            
            # --- D. 闭环 (将结果反馈给 Meta Agent) ---
            agent.update_from_env(observation=next_obs, reward=reward, done=done, info=info)
            
            if agent.trajectory.steps:
                last_step = agent.trajectory.steps[-1]
                last_step.reward = reward
                last_step.info.update(info)
                
            if done:
                if self.log:
                    print(f"   ✅ Task {task_id} finished (Max episodes reached).")
                break
        if self.log:
            print(f"🏁 [SimpleRunner] Task {task_id} Complete. Final Reward: {agent.trajectory.reward}")
        agent.trajectory.task = task_config
        agent.trajectory.full_history = agent.messages
        return agent.trajectory