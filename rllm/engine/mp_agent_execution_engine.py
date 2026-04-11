# process_engine.py
import asyncio
import concurrent.futures
import logging
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor

import torch

from rllm.agents.agent import Action, BaseAgent, Trajectory
from rllm.agents.utils import (
    convert_messages_to_tokens_and_masks,
    get_recent_assistant_user_messages,
)
from rllm.environments.base.base_env import BaseEnv
from rllm.environments.env_utils import (
    compute_mc_return,
    compute_trajectory_reward,
)
from rllm.misc import colorful_print
from rllm.parser import ChatTemplateParser

logger = logging.getLogger(__name__)

import asyncio
import time
import concurrent.futures
import torch
import multiprocessing
from rllm.misc import colorful_print
from rllm.engine.agent_execution_engine import AgentExecutionEngine
# 导入上面定义的 wrapper
from rllm.engine.process_utils import _worker_reset_wrapper, _worker_step_wrapper, _worker_compute_reward_wrapper, _worker_close_wrapper

class MPAgentExecutionEngine(AgentExecutionEngine):
    def __init__(self, client_config=None, **kwargs):
        super().__init__(**kwargs)
        # 我们需要保存 client 配置传给 worker
        self.client_config = client_config or {}
        
        # 覆盖 executor 为 进程池
        # mp_context="spawn" 是为了避免在 Linux 上 fork 导致死锁 (特别是涉及 CUDA 或 C库时)
        # 如果报错，可以尝试去掉 mp_context 参数使用默认 fork
        try:
            ctx = multiprocessing.get_context("spawn")
            self.executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.n_parallel_agents, 
                mp_context=ctx
            )
        except Exception:
            print("Warning: 'spawn' context not supported, falling back to default.")
            self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.n_parallel_agents)

    async def trajectory_generator(self, *args, **kwargs):
        raise NotImplementedError("Currently Not Supported")
    
    async def run_agent_trajectory_async(self, idx, application_id, seed=0, mode="Text", **kwargs):
        """Run a single agent's trajectory asynchronously"""
        agent = self.agents[idx]
        env = self.envs[idx]
        # env_id = env.env_id

        termination_reason = None
        prompt_token_len = 0
        prompt_tokens = []
        response_token_len = 0
        response_tokens = []
        response_masks = []
        total_time = 0.0
        reward_time = None
        llm_time = 0.0
        env_time = 0.0
        reward = 0.0

        # for step return
        episode_steps = []

        # ----------------- MODIFIED RESET -----------------
        loop = asyncio.get_event_loop()
        
        # 将 env 对象本身 pickle 发送过去，并接收更新后的 env 回来
        updated_env, observation, info = await loop.run_in_executor(
            self.executor, 
            _worker_reset_wrapper, 
            self.envs[idx], 
            self.client_config
        )
        self.envs[idx] = updated_env # <--- 关键：同步状态！
        info["max_steps"] = self.max_steps

        # Reset agent logic (保持不变)
        agent.reset()
        agent.update_from_env(
            observation=observation,  # Raw observation from environment
            reward=0.0,
            done=False,
            info=info,
        )
        
        messages = agent.chat_completions
        prompt_tokens, _ = convert_messages_to_tokens_and_masks(messages, tokenizer=self.tokenizer, parser=self.chat_parser, contains_first_msg=True, contains_generation_msg=True)
        prompt_token_len = len(prompt_tokens)
        # Note, this should never happen!
        if prompt_token_len > self.max_prompt_length:
            agent.reset()
            raise Exception(f"Trajectory {idx}: initial prompt length {prompt_token_len} already exceeded max_prompt_length {self.max_prompt_length}, retrying")

        for step_idx in range(self.max_steps):
            if self.log:
                print(f"===========STEP:{step_idx}===========")
            
            # 1. LLM Generation (Keep in Main Process or Async)
            # LLM 请求通常是 I/O 密集型，放在主进程用 asyncio 处理即可，不需要放到 worker 进程
            prompt_messages = agent.chat_completions.copy()
            
            if not self.enforce_max_prompt_length:
                max_tokens = self.max_response_length - response_token_len
            else:
                max_tokens = self.max_response_length

                # since max prompt is enforced, we filter out too long prompts.
                prompt_str = self.chat_parser.parse(prompt_messages, add_generation_prompt=True, is_first_msg=True)
                prompt_len = len(self.tokenizer.encode(prompt_str, add_special_tokens=False))
                if prompt_len > self.max_prompt_length:
                    termination_reason = "PROMPT_TRUNCATION"
                    break

            kwargs["max_tokens"] = max_tokens

            start_time = time.time()
            # 这里调用 LLM API，不需要改动，依旧在主进程异步并发
            if self.log:
                print("="*50)
                print(f"[DEBUG] PROMPT: {prompt_messages}")
                print("="*50)
            response = await self.get_model_response(prompt_messages, application_id, **kwargs)
            if self.log:
                print("="*50)
                print(f"[DEBUG] LLM 原始回复 (RESPONSE): {response}")
                print("="*50)
            delta_time = time.time() - start_time
            llm_time += delta_time
            total_time += delta_time
            
            # Update steps record
            prompt_response_pair = {
                "prompt": self.chat_parser.parse(prompt_messages, add_generation_prompt=True, is_first_msg=True),
                "response": response,
            }
            episode_steps.append(prompt_response_pair)

            action: Action = agent.update_from_model(response)
            action = action.action

            # ----------------- MODIFIED STEP -----------------
            start_time = time.time()
            try:
                # 再次将 env 传给进程，并接收更新后的 env
                updated_env, next_observation, reward, done, info = await asyncio.wait_for(
                    loop.run_in_executor(
                        self.executor, 
                        _worker_step_wrapper, 
                        self.envs[idx], 
                        action,
                        self.client_config
                    ), 
                    timeout=(self.trajectory_timeout - total_time)
                )
                self.envs[idx] = updated_env # <--- 关键：同步状态！
            except asyncio.TimeoutError:
                termination_reason = "ENV_TIMEOUT"
                if step_idx == 0:
                    colorful_print(f"Warning: Trajectory {idx} completed due to: {termination_reason} before able to perform 1 complete action. This might cause unexpected behavior. Consider increasing trajectory timeout limit.\n", "red")
                reward = 0

                cur_step = agent.get_current_state()
                done = True
                cur_step.done = done
                break

            delta_time = time.time() - start_time
            env_time += delta_time
            total_time += delta_time
            info["max_steps"] = self.max_steps
            info["cur_tokens"] = response_token_len
            
            # Update agent internal state.
            agent.update_from_env(
                observation=next_observation,
                reward=reward,
                done=done,
                info=info,
            )

            
            cur_step = agent.get_current_state()
            cur_step.reward = reward
            cur_step.done = done
            cur_step.info.update(info)

            chat_completions_messages = agent.chat_completions
            assistant_message, env_messages = get_recent_assistant_user_messages(chat_completions_messages)

            # Check and convert to tokens if necessary
            assert assistant_message is not None or mode != "Token", "Assistant messages is none when accumulating token trajectories which should be conversations. This should not happen."
            assert env_messages is not None or mode != "Token", "Environment messages is none when accumulating token trajectories which should be conversations. This should not happen."
            assistant_msg_tokens, assistant_msg_masks = [], []
            env_msg_tokens, env_msg_masks = [], []
            if assistant_message:
                assistant_msg_tokens, assistant_msg_masks = convert_messages_to_tokens_and_masks([assistant_message], tokenizer=self.tokenizer, parser=self.chat_parser, contains_first_msg=False, contains_generation_msg=False)
            if env_messages:
                env_msg_tokens, env_msg_masks = convert_messages_to_tokens_and_masks(env_messages, tokenizer=self.tokenizer, parser=self.chat_parser, contains_first_msg=False, contains_generation_msg=True)

            # Update repsonse token length
            response_token_len += len(assistant_msg_tokens) + len(env_msg_tokens)
            # Reached maximum number of tokens for the trajectory
            if not self.enforce_max_prompt_length and response_token_len >= self.max_response_length:
                # Truncation length
                truncation_length = self.max_response_length - response_token_len
                # breakpoint()
                # Truncate the response and masks
                if truncation_length < 0:
                    truncated_response_tokens = (assistant_msg_tokens + env_msg_tokens)[:truncation_length]
                    truncated_response_masks = (assistant_msg_masks + env_msg_masks)[:truncation_length]
                else:
                    # Edge case where the response is exactly the max response length.
                    truncated_response_tokens = assistant_msg_tokens + env_msg_tokens
                    truncated_response_masks = assistant_msg_masks + env_msg_masks
                # Update token collections
                response_tokens.extend(truncated_response_tokens)
                response_masks.extend(truncated_response_masks)

                cur_step = agent.get_current_state()
                if response_token_len - len(env_msg_tokens) > self.max_response_length:
                    cur_step.reward = 0.0
                cur_step.done = True
                termination_reason = "TRUNCATION"
                # handle returning
                break

            # Update the token version of trajectory
            response_tokens.extend(assistant_msg_tokens)
            response_masks.extend(assistant_msg_masks)
            observation = next_observation

            if total_time >= self.trajectory_timeout:
                termination_reason = "TIMEOUT"
                cur_step = agent.get_current_state()
                done = True
                cur_step.done = done
                break
            if done:
                termination_reason = "ENV_DONE"
                break
                
            response_tokens.extend(env_msg_tokens)
            response_masks.extend(env_msg_masks)

            if step_idx == self.max_steps - 1:
                termination_reason = "MAX_STEPS"

        masked_out = False
        if self.overlong_filter:
            if termination_reason == "TRUNCATION" or termination_reason == "MAX_STEPS" or termination_reason == "TIMEOUT":
                # Mask out the entire response for overlong trajectories if the reward is 0.
                response_masks = [0] * len(response_masks)
                masked_out = True
        
        # ----------------- MODIFIED FINAL REWARD -----------------
        if hasattr(self.envs[idx], "compute_final_reward"):
            cur_step = agent.get_current_state()
            start_time = time.time()
            updated_env, reward = await loop.run_in_executor(
                self.executor, 
                _worker_compute_reward_wrapper,
                self.envs[idx]
            )
            self.envs[idx] = updated_env
            reward_time = time.time() - start_time
            cur_step.reward = reward
           
        await loop.run_in_executor(
            self.executor, 
            _worker_close_wrapper, # 使用 wrapper
            self.envs[idx]         # 传入当前的 env
        )
        # 这里怎么close？需要做类似await loop.run_in_executor(self.executor, env.close)吗？
        
        trajectory = agent.trajectory
        compute_trajectory_reward(trajectory)
        final_reward = trajectory.reward
        if termination_reason:
            if final_reward > 0:
                color = "green"
            else:
                color = "yellow"
            colorful_print(
                f"Trajectory {idx} completed due to: {termination_reason}. Reward is {final_reward}. \n",
                color,
            )
            if masked_out:
                colorful_print(f"Trajectory {idx} is masked out due to overlong filter.", "red")

        
        compute_mc_return(trajectory, gamma=self.gamma)
        
        if mode == "Text":
            return trajectory
        elif mode == "Token":
            token_result = {
                "prompt_tokens": torch.tensor(prompt_tokens, dtype=torch.long),
                "response_tokens": torch.tensor(response_tokens, dtype=torch.long),
                "response_masks": torch.tensor(response_masks, dtype=torch.long),
                "trajectory_reward": trajectory.reward,
                "idx": self.envs[idx].idx,
                "chat_completions": agent.chat_completions,
                "metrics": {
                    # Total number of steps taken in the trajectory
                    "steps": len(trajectory.steps),
                    # Time to calculate reward
                    "reward_time": reward_time,
                    # Total time spent in environment execution (env.step)
                    "env_time": env_time,
                    # Time to calculate response tokens
                    "llm_time": llm_time,
                    # Total time spent in the trajectory
                    "total_time": total_time,
                },
            }
            return token_result
        elif mode == "Conversation":
            return agent.chat_completions
        elif mode == "Step":
            steps_result = {
                "steps": episode_steps,
                "trajectory_reward": trajectory.reward,
                "idx": env.idx,
                "mc_returns": [step.mc_return for step in trajectory.steps][: len(episode_steps)],
            }
            return steps_result