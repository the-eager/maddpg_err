from tqdm import tqdm
from agent import Agent
from common.replay_buffer import Buffer
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import logging
from UI.display import *
from common.arguments import get_args
import numpy as np
import random
import torch
import numpy as np
import inspect
import functools
import copy
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"

logger = logging.getLogger()
formatter = logging.Formatter('%(asctime)s - %(message)s')
fhandler = logging.FileHandler('multi_UAV.log', 'w')
fhandler.setLevel(logging.INFO)# DEBUG
fhandler.setFormatter(formatter)

chandler = logging.StreamHandler()
chandler.setLevel(logging.INFO)
chandler.setFormatter(formatter)

logger.addHandler(fhandler)
logger.addHandler(chandler)
logger.setLevel(logging.INFO)


class Runner:
    def __init__(self, args, env):
        self.args = args
        self.noise = args.noise_rate
        self.epsilon = args.epsilon
        self.episode_limit = args.max_episode_len # 200
        self.env = env
        self.agents = self._init_agents()
        self.buffer = Buffer(args)
        self.save_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def _init_agents(self):
        agents = []
        for i in range(self.args.n_agents):
            agent = Agent(i, self.args)
            agents.append(agent)
        return agents

    def run(self):
        returns = []
        eva_count = 0
        for time_step in tqdm(range(self.args.time_steps)):
        # for time_step in tqdm(range(4000)):
            # reset the environment
            if time_step % self.episode_limit == 0:
                s = self.env.reset()
                state_info = [-1,-1,-1,-1,-1,-1 ,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0]
            
            u = []
            actions = []
            with torch.no_grad():
                for agent_id, agent in enumerate(self.agents):
                    action = agent.select_action(s[agent_id], self.noise, self.epsilon)
                    u.append(action)
                    actions.append(action)
            
            
            logger.info(f'==>time_step: {time_step}')   

            for i in range(self.args.n_agents, self.args.n_players): #补充动作空间　n_agents　是AI数，n_players 是所有玩家数
                actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0]) 

            # logger.info(f"actions:{actions}")
            # logger.info(f"state_info,{state_info}")

            s_next, r, done, state_info, connect_topology, connect_BS = self.env.step(actions,state_info)
            # obs_n, reward_n, done_n, state_info, connect_topology, landmark_trans_matrix
            
            if state_info[17] == 1: # 游戏结束标志位是1 直接开始下一轮
                continue
            

            self.buffer.store_episode(s[:self.args.n_agents], u, r[:self.args.n_agents], s_next[:self.args.n_agents])
            s = s_next
            if self.buffer.current_size >= self.args.batch_size: # 设置调整BUffer
                transitions = self.buffer.sample(self.args.batch_size)
                for agent in self.agents:
                    other_agents = self.agents.copy()
                    other_agents.remove(agent)
                    agent.learn(transitions, other_agents)
            
            
            if time_step > 0 and time_step % self.args.evaluate_rate == 0: #每400次进入一次评估

                # logger.info(f'==>time_step: {time_step}')
                # logger.info(f'noise: {self.noise}')
                # logger.info(f'epsilon: {self.epsilon}')
                eva_count += 1
                

                returns.append(self.evaluate(eva_count)) # 进入评估函数
                plt.figure()
                plt.subplots(constrained_layout=True)
                plt.plot(range(len(returns)), returns)
                # logger.info(f"{range(len(returns))}, {returns}")

                plt.xlabel('episode * ' + str(self.args.evaluate_rate / self.episode_limit)) #400次评估一次，200次才一轮游戏，所以平均rew图就2个点
                plt.ylabel('average returns')
                plt.savefig(self.save_path + f'/plt.png', format='png')
            self.noise = max(0.05, self.noise - 0.0000005)
            self.epsilon = max(0.05, self.noise - 0.0000005)
            np.save(self.save_path + f'/returns.pkl', returns)

    def evaluate(self,eva_count):
        returns = []
        for episode in range(self.args.evaluate_episodes): # 进入以后，评估5个episod
            # reset the environment
            s = self.env.reset()
            state_info = [-1, -1, -1, -1, -1, -1 ,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0]
            rewards = 0
            UAV1_loaction = []
            UAV2_loaction = []
            UAV3_loaction = []

            UAV1_data = []
            UAV2_data = []
            UAV3_data = []

            all_topology = []
            all_connect_BS = []

            UAV1_loaction2 = []
            for time_step in range(self.args.evaluate_episode_len): # 评估时每轮游戏100步
                self.env.render()
                actions = []
                with torch.no_grad():
                    for agent_id, agent in enumerate(self.agents):
                        action = agent.select_action(s[agent_id], 0, 0)
                        actions.append(action)
                
                for i in range(self.args.n_agents, self.args.n_players):
                    actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
                s_next, r, done, state_info, connect_topology, landmark_trans_matrix = self.env.step(actions,state_info)
                rewards += r[0]
                s = s_next
                print(s_next)
                UAV1_loaction.append(s_next[0][2:4])
                UAV2_loaction.append(s_next[1][2:4])
                UAV3_loaction.append(s_next[2][2:4])

                UAV1_data.append(copy.copy(self.args.agents[0].state.d_data))
                UAV2_data.append(copy.copy(self.args.agents[1].state.d_data))
                UAV3_data.append(copy.copy(self.args.agents[2].state.d_data))
                all_topology.append(connect_topology)
                sums = 0
                for lists in landmark_trans_matrix:
                    sums += sum(lists)
                if sums!=0:
                    all_connect_BS.append(landmark_trans_matrix)
            
            display_trajectory(UAV1_loaction, UAV2_loaction, UAV3_loaction, self.args.BS, self.save_path,
                               self.args.landmark,eva_count) #把这一轮游戏画了一个图
            # plot_topology(UAV1_loaction, UAV2_loaction, UAV3_loaction, self.args.BS, self.save_path, self.args.landmark,
            #               all_topology, all_connect_BS)
            display_data_change(UAV1_data, UAV2_data, UAV3_data, self.save_path)
            returns.append(rewards)

   
            #logger.info(f'Returns is: {rewards}')
            logger.info(f'served_id is: {state_info}')
            logger.info(f'rewards is: {rewards}')
            # logger.info(f'UAV2_data is: {UAV2_data}')
            # logger.info(f'UAV3_data is: {UAV3_data}')
            # logger.info(f'landmark_trans_matrix is: {all_connect_BS}')

            #logger.info(f'one_step_reward is: {r}')
        return sum(returns) / self.args.evaluate_episodes # 5次求一次平均奖励


def make_env(args):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(args.scenario_name + ".py").Scenario()

    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    # env = MultiAgentEnv(world)
    args.n_players = env.n  # 包含敌人的所有玩家个数
    args.n_agents = env.n - args.num_adversaries  # 需要操控的玩家个数，虽然敌人也可以控制，但是双方都学习的话需要不同的算法
    args.agents = env.agents
    args.landmark = env.landmark
    args.BS = env.BS

    args.obs_shape = [env.observation_space[i].shape[0] for i in range(args.n_agents)]  # 每一维代表该agent的obs维度
    action_shape = []
    for content in env.action_space:
        action_shape.append(content.n)
    args.action_shape = action_shape[:args.n_agents]  # 每一维代表该agent的act维度
    args.high_action = 1
    args.low_action = -1
    return env, args



if __name__ == '__main__':
    # get the params
    args = get_args()
    env, args = make_env(args)
    runner = Runner(args, env)
    if args.evaluate:
        returns = runner.evalluate()
        print('Average returns is', returns)
    else:
        runner.run()
