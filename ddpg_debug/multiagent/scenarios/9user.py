"""
Created on  Apr 9 2021
@author: wangmeng
@modified: Apr 23 2021
@marks:
4-22日实验结果记录

随机分布20个sensor，多UAV目标是让AOI最小（每次训练sensor位置随机出现）

程序版本V2.0-环境版本V1.1：
动作：UAV移动距离
状态：UAV位置状态+所有sensor位置+所有sensor的AOI+其它UAV的位置
奖励： 所有sensor的AOI均值 + 出界惩罚 +服务区域重叠惩罚

一些有效的trick：

1、Reward设置保证单调递增可以有效的加速收敛
2、适度的摒弃无关动作和状态，有利于算法的探索，多智能体的引入会相当程度的影响学习效果
"""
import numpy as np
from env.core import World, Agent, Landmark, BaseStation
from env.scenario import BaseScenario
class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 3
        num_landmarks = 3
        num_BS = 1
        world.collaborative = False
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.movable = True
            agent.trans = True
            # [-1,1]
            agent.size = 0.001
            agent.state.d_data = 0
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.trans = True
            landmark.size = 0.1
            landmark.state.d_data = 0

        world.BS = [BaseStation() for i in range(num_BS)]
        for i, Base_station in enumerate(world.BS):
            Base_station.name = 'BS %d' % i
            Base_station.collide = False
            Base_station.movable = False
            Base_station.trans = False
            Base_station.trans = False
            Base_station.size = 1
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        np.random.seed(1)  # fix seed
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])

        for i, BS in enumerate(world.BS):
            BS.color = np.array([0.25, 0.25, 0])
        # set random initial states
        # 定义固定坐标位置
        #UAV_loaction = [[-25., 25.], [10., 37.]]
        UAV_loaction = [[-0.5, 0.7], [-0.3,-0.5], [0.2, 0.8]]
        #landmark_location = [[-22., 21.], [-41., -4.], [-14., -26.], [21., -7.], [34., 15.], [8., 30.]]
        landmark_location = [[-0.8,0.8], [-0.5, 0.5], [-0.8, -0.0], [-0.3, -0.3], [0.4, -0.4], [0,-0.8], [0.2, 0.6], [0.2,0.1], [0.7, 0.3]]
        BS_location = [[1., 1.]]
        land_data = [0.03 for _ in range(len(world.landmarks))]

        for i, agent in enumerate(world.agents):
            agent.state.p_pos = np.array(UAV_loaction[i])
            #agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)#速度
            agent.state.c = np.zeros(world.dim_c)
            agent.state.d_data = 0
            agent.state.power = 0

        for i, landmark in enumerate(world.landmarks):
            #landmark.state.p_pos = np.array(landmark_location[i])
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
            landmark.state.d_data = land_data[i]

        for i, base in enumerate(world.BS):
            base.state.p_pos = np.array(BS_location[i])


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

    def dist(self,agent1,agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        return dist


    def reward(self, agent, world, inputx):
        goal_landmark = inputx[:3]#目标id
        agent_cumulate_rew = inputx[3:6]#记录某一时刻服务了多少用户
        dist_last_num = inputx[6:9]
        served_sensor = inputx[9:]#已完成id

        UAV_serve_range = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

        # 初始化reward
        rew = 0

        #检测当前UAV_id
        i = 0
        agent_id = -1
        for a in  world.agents:
            if a  == agent:
                agent_id = i
            i += 1


        #检测是否有传感器被服务，若被服务，进入服务记录
        userve_dis = {}

        update_id = -1
        for i, land in enumerate(world.landmarks):
            if i not in served_sensor + goal_landmark:
                userve_dis[i] = self.dist(land, agent)
        if len(userve_dis) > 0:
            goal_id_dis = min(userve_dis.values())
            goal_id = list(userve_dis.keys())[list(userve_dis.values()).index(goal_id_dis)]
            if self.is_collision(agent, world.landmarks[goal_id]):
                rew += 5
                served_sensor.append(goal_id)
                update_id = served_sensor.pop(0)
                goal_landmark[agent_id] = goal_id
                dist_last_num[agent_id] = goal_id_dis

        # if self.is_collision(agent, world.landmarks[goal_landmark[agent_id]]):
        #     rew += 5
        #     server_history.append(goal_landmark[agent_id])
        #     # agent_index0 = dists.index(min(dists))
        #     agent_cumulate_rew[agent_id] += 1
        #
        #     #构成循环，当大于服务总数量3时，会将最早服务的sensor从服务列表中移除
        #     served_area = []
        #     if agent_cumulate_rew[agent_id] >= 3:
        #         for i in server_history:
        #             if i in UAV_serve_range[agent_id]:
        #                 served_area.append(i)
        #
        #
        #     if len(served_area)>=3:
        #         server_history.remove(served_area[0])
        #
        #
        #     # 单智能体测试
        #     # if agent_cumulate_rew[agent_id] >= 6:
        #     #     for i in server_history:
        #     #         if i in [0, 1, 2, 3, 4, 5]:
        #     #             server_history.remove(i)
        #     #             break
        #
        #     dist_last = 5
        #     # 将距离最近的landmark作为下一时刻的目标
        #     for j, l_last in enumerate(world.landmarks):
        #         if j in server_history + goal_landmark:  # 注意这里是服务非目标、非已服务的landmark,加上goal——landmark是为了防止两个UAV服务同一个用户
        #             continue
        #         if j in UAV_serve_range[agent_id]:
        #             if np.sqrt(np.sum(np.square(world.agents[agent_id].state.p_pos - l_last.state.p_pos))) < dist_last:
        #                 dist_last = np.sqrt(np.sum(np.square(world.agents[agent_id].state.p_pos - l_last.state.p_pos)))
        #                 goal_landmark[agent_id] = j
        #                 dist_last_num[agent_id] = dist_last

        #距离惩罚项
        dist = np.sqrt(np.sum(np.square(agent.state.p_pos - world.landmarks[goal_landmark[agent_id]].state.p_pos)))
        rew -= dist
        #rew -= agent.state.d_data*10
        #补偿距离rew，保证reward单调，加快探索速度
        #rew += dist_last_num[agent_id] #保证reward单调递增，防止吃到food后的下一个动作出现reward骤降

        # 服务sensor数量奖励
        #rew += agent_cumulate_rew[agent_id]  # len(server_history)

        #超出边界惩罚
        boundary_reward = -1 if self.outside_boundary(agent) else 0
        rew += boundary_reward

        #碰撞惩罚
        if agent.collide:
            for a in world.agents:
                if a != agent:
                    if self.is_collision(a, agent):
                        rew -= 1
        #相关参数显示
        # print(server_history)
        # print(rew)
        # print('goal_landmark:', goal_landmark)
        # print('dsitance:',np.sqrt(np.sum(np.square(agent.state.p_pos - world.landmarks[goal_landmark[agent_id]].state.p_pos))))
        # print('reward:',agent_id, rew)

        return rew, goal_landmark + agent_cumulate_rew + dist_last_num + served_sensor

    def observation(self, agent, world, input_list):
        #UAV_id检测
        i = 0
        agent_id = -1
        for a in world.agents:
            if a == agent:
                agent_id = i
            i += 1
        if agent_id==1:
            entity_pos = [world.landmarks[5].state.p_pos - agent.state.p_pos]
        elif agent_id == 2:
            entity_pos = [world.landmarks[2].state.p_pos - agent.state.p_pos]
        else:
            entity_pos = [world.landmarks[0].state.p_pos - agent.state.p_pos]

        #开始服务过程
        if len(input_list) != 0:
            gaol_mark = input_list[0:3]
            entity_pos=[world.landmarks[gaol_mark[agent_id]].state.p_pos - agent.state.p_pos]

        comm = [] #UAV之间交流的信息
        other_pos = []

        #与其它UAV之间的距离
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)

        # print(len([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)) #9
        # print(np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm))
        # print(agent.state.p_vel)#[0. 0.]当前速度
        # print(agent.state.p_pos)#[-0.5  0.5]当前UAV
        # print(entity_pos)#[array([ 0.06, -0.08]), array([-0.32, -0.58]), array([ 0.22, -1.02]), array([ 0.92, -0.64]), array([ 1.18, -0.2 ]), array([0.66, 0.1 ])]
        # print(other_pos)#[array([0.7 , 0.24])]另一个UAV
        #return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)

        return np.concatenate([agent.state.p_pos] + entity_pos)


