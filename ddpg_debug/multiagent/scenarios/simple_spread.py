from asyncio.log import logger
import numpy as np
from multiagent.core import World, Agent, Landmark, BaseStation
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 3
        num_landmarks = 17
        num_BS = 1
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.trans = True
            agent.size = 0.1
            agent.state.d_data = 0
            #agent.size = 0.001
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.trans = True
            landmark.state.d_data = 0
            #landmark.size = 0.1
        world.BS = [BaseStation() for i in range(num_BS)]
        for i, Base_station in enumerate(world.BS):
            Base_station.name = 'BS %d' % i
            Base_station.collide = False
            Base_station.movable = False
            Base_station.trans = False
            Base_station.trans = False
            Base_station.size = 0.2
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        np.random.seed(1)
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        for i, BS in enumerate(world.BS):
            BS.color = np.array([0.25, 0.74, 0.13])
        landmark_location = [[-0.8,0.8], [-0.5, 0.5], [-0.8, -0.0], [-0.3, -0.3], [-0.4, 0.4], [0,-0.8], [0.2, 0.6], \
            [0.2,0.1], [0.7, 0.3],[-0.3,0.7],[-0.5,0],[0.5,0.7],[0.4,0.3],[-0.2,-0.4],[-0.7,-0.4],[0.8,0.5],[-0.5,-0.7]]
        UAV_loaction = [[-0.5, 0.5], [-0.3, -0.3], [0.2, 0.6]]
        BS_location = [[1., 1.]]
        land_data = [5 for _ in range(len(world.landmarks))] #初始化每个用户的初始数据量

        # set random initial states
        for i,agent in enumerate(world.agents):  # 初始化agnet站位
            agent.state.p_pos = np.array(UAV_loaction[i]) #无人机固定位出发
            #agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.state.d_data = 0
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.array(landmark_location[i]) #用户固定位
            #landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
            landmark.state.d_data = land_data[i]
        for i, base in enumerate(world.BS):# 基站固定位
            base.state.p_pos = np.array(BS_location[i])

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def outside_boundary(self, agent):
        if agent.state.p_pos[0] > 1.25 or agent.state.p_pos[0] < -1.25 or agent.state.p_pos[1] > 1.25 or agent.state.p_pos[1] < -1.25:
            return True
        else:
            return False

    def reward(self, agent, world, input_list): # 针对一个agent计算reward
        # for _, agent in enumerate(world.agents):
        #     logger.info(f"agent.state.d_data{agent.state.d_data}")
        # for i, landmark in enumerate(world.landmarks):
        #     logger.info(f"landmark{i} {landmark.state.d_data}") 

        done = input_list[17]
        served_sensor = input_list[:17]
        energy_consumptions = input_list[18:21]
        

        # logger.info(f"forever_reward:{forever_reward}")
        # logger.info(f":served_sensor:{served_sensor}")

        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        dists = []

        # for id,l in enumerate(world.landmarks):
        #     if id in served_sensor: # 如果这个服务点已经被服务过
        #         if self.is_collision(agent, l): # 但是无人机依旧在这个区域
        #             rew -= 0.5 #那就直接扣分
        
        
        # 无人机距离惩罚
        for id, l in enumerate(world.landmarks):
            if id in served_sensor:
                continue
            # 计算所有land和每个UAV的最短距离
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            # logger.info(f"dists:{dists}")
            rew -= min(dists) #每个未服务的user都会有reward损失
            logger.info(f"dists:{min(dists)}")
            if self.is_collision(agent, l): # 找到用户就加个5
                logger.info(f"get!!! {agent} {l}")
                rew += 70
                
                # if l.state.d_data == 0: # 这个用户数据已经传输结束了，再放入served_sensor里面
                    # rew += 3
                served_sensor.append(id)
                pop_id = served_sensor.pop(0) #保证数组长度不变
                logger.info(f"served_sensor {served_sensor}")

        
        # 无人机能耗惩罚
        def energy_model(v): # 只和速度相关的
            '''
            能量消耗
            期望速度最小（有限时间内速度最小即飞行距离最短），防止高速大范围迂回
            f(v) = P_0(1+\frac{3v^2}{U_t^2})+P_1(\sqrt{1+\frac{v^4}{4U_i^4}} - \frac{U^2}{2U_i^2})^{\frac{1}{2}}+1/2\rho_1\rho_2v^3
            '''
            P_0 = 9.1827  # w
            U_t = 120 # 60
            P_1 = 11.5274  # w
            U_i = 4.03 # 2.4868
            rho_1 = 0.6 * 0.503 # 0.5017 * 0.2827
            rho_2 = 0.05 * 1.225 # 0.0832 * 1.205
            fv = P_0 * (1 + (3 * v ** 2) / (U_t ** 2)) + P_1 * (
                        np.square(1 + (v ** 4) / (4 * U_i ** 4)) - (v ** 2) / (2 * U_i ** 2)) ** (
                             1 / 2) + 1 / 2 * rho_1 * rho_2 * v ** 3
            return fv
        
        rew_energy = energy_model(np.sqrt(np.sum(np.square(agent.state.p_vel))))*0.01 # 能量消耗计入reward 倍率转换是0.01 
        rew_energy = float((rew_energy - 0.34831888287009044)/0.04759972578209208)
        # rew -= rew_energy
        #　rew_energy = (rew_energy - 0.62)*1000 # 调参加偏置 分别计算各自无人机能耗
        energy_consumptions.append(rew_energy) # 第一个无人机能耗加在最后面
        pop_id = energy_consumptions.pop(0) #保证数组长度不变
        logger.info(f"vel {np.sqrt(np.sum(np.square(agent.state.p_vel)))}")
        logger.info(f"rew_energy:{rew_energy}")
        


        # 无人机相撞惩罚
        if agent.collide: 
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                
        # 超出边界惩罚
        boundary_reward = -10 if self.outside_boundary(agent) else 0
        rew += boundary_reward

        # 检测本轮任务是否结束
        charge_number = 0
        if -1 not in served_sensor:
            for a in world.agents:
                if self.is_collision(a, world.BS[0]):
                    charge_number += 1
            if charge_number == 3:
                done = 1

        return rew, served_sensor+[done]+energy_consumptions

    def observation(self, agent, world, state_info):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for id,entity in enumerate(world.landmarks):  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        #　entity_pos　＝　用户位置　－　无人机位置　表示当前无人机距离各个landmark的距离
        
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        # logger.info(f"p_vel:{agent.state.p_vel}")
        # logger.info(f"p_pos:{agent.state.p_pos}")
        # logger.info(f"entity_pos:{entity_pos}")
        # logger.info(f"other_pos:{other_pos}")
        # logger.info(f"observation:{np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos)}")
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos)
