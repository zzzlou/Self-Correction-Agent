# 必须定义在类外面，确保可被 Pickle
from rllm.engine.simple_runner import SimpleRunner
import multiprocessing
import concurrent.futures
import multiprocessing
import traceback
from typing import Dict, Any, List

from rllm.environments.jericho.openai_helpers import chat_completion_with_retries, init_global_client

def _worker_run_full_episode(
    task_id: int, 
    task_config: Dict, 
    env_cls: Any, 
    agent_cls: Any, 
    env_args: Dict, 
    agent_args: Dict, 
    meta_llm_cfg: Dict, 
    api_cfg: Dict,
    log: bool,
):
    """
    Worker 进程逻辑：一次性跑完整个 Trajectory
    """
    try:
        # 🔥 关键 1：在子进程内重新初始化 API Client
        if api_cfg:
            init_global_client(base_url=api_cfg.get('base_url'), api_key=api_cfg.get('api_key'))
        
        # 1. 实例化
        full_env_args = {**task_config, **env_args}
        env = env_cls.from_dict(full_env_args)
        agent = agent_cls(**agent_args)
        
        # 2. Reset
        obs, info = env.reset()
        agent.reset()
        agent.update_from_env(observation=obs, reward=0, done=False, info=info)
        
        # 3. 跑完整个 Meta Loop
        max_meta_steps = env.meta_cfg.get("max_episodes", 3)
        
        for step in range(max_meta_steps):
            messages = agent.chat_completions
            
            if log:
                print(f"--- 🔄 Task {task_id} | Step {step+1}/{max_meta_steps} ---")

            # 调用 LLM (注意：这里不能用 self.meta_llm_config，要用传进来的 meta_llm_cfg)
            response_text = chat_completion_with_retries(
                model=meta_llm_cfg["model"],
                messages=messages,
                temperature=meta_llm_cfg.get("temperature", 0.7),
                extra_body=meta_llm_cfg.get("extra_body", None)
            ).choices[0].message.content
            
            # --- 解析 Action ---
            action = agent.update_from_model(response_text)
            
            if log:
                # 打印前 100 字符，防止日志爆炸
                print(f"   💡 [Task {task_id}] Feedback: {action.action[:100]}...")
            
            # --- 环境执行 ---
            next_obs, reward, done, info = env.step(action.action)
            
            # --- 闭环 ---
            agent.update_from_env(observation=next_obs, reward=reward, done=done, info=info)
            
            if agent.trajectory.steps:
                last_step = agent.trajectory.steps[-1]
                last_step.reward = reward
                last_step.info.update(info)
                
            if done:
                if log:
                    print(f"   ✅ Task {task_id} finished (Max episodes reached).")
                break

        if log:
            print(f"🏁 [Task {task_id}] Complete. Final Reward: {agent.trajectory.reward}")
            
        agent.trajectory.task = task_config
        # agent.trajectory.full_history = agent.messages # 视情况是否需要
        agent.trajectory.full_history = agent.messages
        return agent.trajectory
        
    except Exception as e:
        print(f"❌ Worker {task_id} failed: {e}")
        traceback.print_exc()
        return None

class SimpleRunnerMP(SimpleRunner):
    def execute_tasks(self, tasks, api_cfg, max_concurrent=32):
        
        
        # 使用 Spawn 模式更安全 (避免 C 库死锁)
        ctx = multiprocessing.get_context("spawn")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_concurrent, mp_context=ctx) as executor:
            futures = [
                executor.submit(
                    _worker_run_full_episode,
                    i, task, 
                    self.env_class, self.agent_class, 
                    self.env_args, self.agent_args, 
                    self.meta_llm_config,
                    api_cfg,
                    self.log,
                )
                for i, task in enumerate(tasks)
            ]
            
            results = []
            for f in concurrent.futures.as_completed(futures):
                if f.result(): results.append(f.result())
                
        return results