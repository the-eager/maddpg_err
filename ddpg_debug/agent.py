import numpy as np
import torch
import os
from maddpg.maddpg import MADDPG
import torch.autograd as autograd
from asyncio.log import logger

USE_CUDA = torch.cuda.is_available()
torch.cuda.set_device(0)
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

class Agent:
    def __init__(self, agent_id, args):
        self.args = args
        self.agent_id = agent_id
        self.policy = MADDPG(args, agent_id)

    def select_action(self, o, noise_rate, epsilon):
        # logger.info(f"o:{o}")
        if np.random.uniform() < epsilon: #贪心策略
            u = np.random.uniform(-self.args.high_action, self.args.high_action, self.args.action_shape[self.agent_id]) # 随机动作，取值范围+- high_action
        else:
            inputs = Variable(torch.tensor(o, dtype=torch.float32).unsqueeze(0))
            pi = self.policy.actor_network(inputs).squeeze(0)
            # print('{} : {}'.format(self.name, pi))
            u = pi.cpu().numpy()
            noise = noise_rate * self.args.high_action * np.random.randn(*u.shape)  # gaussian noise
            u += noise
            u = np.clip(u, -self.args.high_action, self.args.high_action) # 截取函数，防止数值超出范围
            # logger.info(f"u.copy():{u.copy()}")
        return u.copy()

    def learn(self, transitions, other_agents):
        self.policy.train(transitions, other_agents)

