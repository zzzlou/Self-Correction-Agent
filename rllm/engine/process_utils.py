# process_utils.py
import os
from rllm.environments.jericho.openai_helpers import init_global_client

def _worker_reset_wrapper(env, client_config):
    """
    1. 重新连接 OpenAI (因为 Global 变量在 Worker 里是 None)
    2. 执行 env.reset
    """
    pid = os.getpid()
    print(f"🔥 [Worker Process {pid}] Received RESET task for Env {env.idx if hasattr(env, 'idx') else '?'}")
    # 这里的 client_config 就是你传进来的 base_url 和 api_key
    init_global_client(**client_config)
    
    # 执行正常的 reset
    # 注意：此时 env 是主进程传过来的副本，拥有独立的 actor 和 memory
    log, info = env.reset()
    
    # 把更新后的 env (里面可能更新了 best_score 等) 和结果一起传回去
    return env, log, info

def _worker_step_wrapper(env, action, client_config):
    """
    1. 确保 OpenAI 连接
    2. 执行 env.step
    """
    init_global_client(**client_config)
    
    log, reward, done, info = env.step(action)
    
    # 同样，必须把 env 传回去，否则主进程不知道 actor memory 变了
    return env, log, reward, done, info

def _worker_compute_reward_wrapper(env):
    """如果需要计算最终奖励"""
    reward = env.compute_final_reward()
    return env, reward

def _worker_close_wrapper(env):
    """关闭环境"""
    if hasattr(env, 'close'):
        env.close()
    return None