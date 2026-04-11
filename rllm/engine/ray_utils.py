import ray
from rllm.environments.jericho.openai_helpers import init_global_client

@ray.remote(num_cpus=1)
class RemoteEnvWorker:
    def __init__(self, idx, env_class, env_args, client_config):
        self.idx = idx
        # 1. 初始化 Client (只做一次！不用每次step都连)
        if client_config:
            init_global_client(**client_config)
            
        # 2. 初始化环境
        # 兼容 from_dict 和 直接构造
        if hasattr(env_class, 'from_dict'):
            self.env = env_class.from_dict(env_args)
        else:
            self.env = env_class(**env_args)
        
        # 保持 idx 属性，方便 logging
        self.env_class = env_class
        self.env.idx = idx

    def step(self, action):
        return self.env.step(action)
    
    def compute_final_reward(self):
        if hasattr(self.env, "compute_final_reward"):
            return self.env.compute_final_reward()
        return 0.0

    def close(self):
        self.env.close()
        
    def get_idx(self):
        return self.idx
    def reset(self, config=None): #这里和jerichoEnv.reset() 不太一样
        if config:
            self.env = self.env_class.from_dict(config)
        return self.env.reset()
        