from asyncio.log import logger
import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from multiagent.multi_discrete import MultiDiscrete

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_viewer=True):

        self.world = world
        self.agents = self.world.policy_agents
        self.landmark = self.world.landmarks
        self.BS = world.BS
        '''
         注意! 这里为了简化模型，把三维空间建模为二维，但是agent高度仍然是一个很重要的信息，
         在此保留，如果切换三维，涉及该变量的部分需要删除
        '''
        # hyperparameter
        self.h = 1  # agent high
        self.P_land = [1 for _ in range(len(world.landmarks))]  # landmark transmit power
        self.P_agent = [1 for _ in range(len(world.agents))]  # agent transmit power
        self.noise = 1  # transmit noise
        self.t = 1  # time slot duration

        # set required vectorized gym env property
        self.n = len(world.policy_agents)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        # environment parameters
        self.discrete_action_space = True
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        # if true, agent will transmit data
        self.data_trans = True
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = world.discrete_action if hasattr(world, 'discrete_action') else False
        # if true, every agent has the same reward
        self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False
        print('share_reward',self.shared_reward)
        self.time = 0

        # configure spaces
        self.action_space = []
        self.observation_space = []
        for agent in self.agents:
            total_action_space = []
            # physical action space
            if self.discrete_action_space:
                u_action_space = spaces.Discrete(world.dim_p * 2 + 1)
            else:
                u_action_space = spaces.Box(low=-agent.u_range, high=+agent.u_range, shape=(world.dim_p,), dtype=np.float32)
            if agent.movable:
                total_action_space.append(u_action_space)
            # communication action space
            if self.discrete_action_space:
                c_action_space = spaces.Discrete(world.dim_c)
            else:
                c_action_space = spaces.Box(low=0.0, high=1.0, shape=(world.dim_c,), dtype=np.float32)
            if not agent.silent:
                total_action_space.append(c_action_space)
            # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])
            # observation space
            obs_dim = len(observation_callback(agent, self.world, []))
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
            agent.action.c = np.zeros(self.world.dim_c)

        # rendering
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n
        self._reset_render()

    def step(self, action_n, state_info):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}

        # self.agents = self.world.policy_agents
        # set action for each agent
        for i, agent in enumerate(self.agents): # 统一化格式，方便导入库
            self._set_action(action_n[i], agent, self.action_space[i],state_info)
            # logger.info(f"i:{i}env-action_n:{action_n[i]}")
            logger.info(f"agent:{agent.action.u[0]} {agent.action.u[1]}")
        
        # advance world state
        self.world.step()

        # data transmit system
        landmark_trans_matrix = [[0 for _ in range(len(self.agents))] for _ in range(len(self.landmark))]
        connect_topology = [[0 for _ in range(len(self.agents))] for _ in range(len(self.agents))]
        connect_BS = [[0 for _ in range(len(self.agents))] for _ in range(len(self.BS))]

        # # update landmark trans matrix
        landmark_trans_matrix = self.updata_landmark_matrix()
        id_record = []
        for id,landmark in enumerate(landmark_trans_matrix):
            if landmark[0] == 1:
                id_record.append(id)
            if landmark[1] == 1:
                id_record.append(id)
            if landmark[2] == 1:
                id_record.append(id)
        logger.info(f"UAV {id_record}")

    

        #  agent全部传输数据给基站
        connect_topology = [[0, 0, 0], [0, 0, 0], [0, 0, 0]] # 暂时无人机之间无交流
        connect_BS = [[1, 1, 1]]

        # lantrans: n_land*n_agent  [2][6]  agenttrans: n_agemt*n_BS [1][2]
        lan_trans, agent_trans = self.updata_trans_data_matrix()
        logger.info(f" agent_trans {agent_trans}")
        # logger.info(f"UAV0 {lan_trans[0].index(max(lan_trans[0]))} UAV1 {lan_trans[1].index(max(lan_trans[1]))} UAV2 {lan_trans[2].index(max(lan_trans[2]))}")
        
        

        # set state for each entity
        for entity in self.agents + self.landmark:
            logger.info(f"data_state {entity} {entity.state.d_data}")
            # calculate entity transmit data
            trans_data = self.analytical_topology_matrix(entity, lan_trans, agent_trans, landmark_trans_matrix,
                                                         connect_topology, connect_BS) # 数据传输
            # set data action for each agent
            self._set_action_data(trans_data, entity)
            

        self.world.data_step() # 更新世界信息

        # record observation for each agent
        
        for index_i , agent in enumerate(self.agents):
            logger.info(f"agent{index_i} {agent}")
            obs_n.append(self._get_obs(agent, state_info))
            # logger.info(f"_get_obs(agent, state_info):{self._get_obs(agent, state_info)}")
            

            
            r, state_info = self._get_reward(agent, state_info)
            logger.info(f"env-_get_reward-r:{r}")
            # logger.info(f"_get_reward-state_info:{state_info}")
            logger.info(f"env-state_info:{state_info}")

            reward_n.append(r)
            done_n.append(self._get_done(agent))

            info_n['n'].append(self._get_info(agent))

        # all agents get total reward in cooperative case
        reward = np.sum(reward_n)
        if self.shared_reward:
            reward_n = [reward] * self.n # 所有无人机优化目标都是降低 shared_reward ，reward_n 是三者相等

        return obs_n, reward_n, done_n, state_info, connect_topology, landmark_trans_matrix

    def reset(self):
        # reset world
        self.reset_callback(self.world)
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n = []
        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_n.append(self._get_obs(agent,[]))
        return obs_n

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent, state_info):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world, state_info)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent):
        if self.done_callback is None:
            return False
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, agent, state_info):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world, state_info)

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, state_info, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)
        # process action
        if isinstance(action_space, MultiDiscrete):
            act = []
            size = action_space.high - action_space.low + 1
            index = 0
            for s in size:
                act.append(action[index:(index+s)])
                index += s
            action = act
            # logger.info(f"MultiDiscrete,true")
        else:
            action = [action]
            # logger.info(f"MultiDiscrete,Flase")


        if agent.movable:
            # physical action
            if self.discrete_action_input:
                agent.action.u = np.zeros(self.world.dim_p)
                # process discrete action
                if action[0] == 1: agent.action.u[0] = -1.0
                if action[0] == 2: agent.action.u[0] = +1.0
                if action[0] == 3: agent.action.u[1] = -1.0
                if action[0] == 4: agent.action.u[1] = +1.0
            else:
                if self.force_discrete_action:
                    d = np.argmax(action[0]) #将最大值赋值为１，其他变为０，把动作的概率变为确定动作值
                    action[0][:] = 0.0
                    action[0][d] = 1.0
                if self.discrete_action_space:
                    import random
                    # if random.random() < 0.1:
                    if 0.01 < 0.1:
                        agent.action.u[0] += action[0][1] - action[0][2]
                        agent.action.u[1] += action[0][3] - action[0][4]
                        norm = np.linalg.norm(np.array(agent.action.u),ord=2)
                        #　logger.info(f"self.discrete_action_space,ture")
                        if not (np.isnan(norm) or  np.isnan(agent.action.u[0]) or np.isnan(agent.action.u[1])): # 求速度的模降速
                            if not np.isnan(agent.action.u[0]/norm): agent.action.u[0] = agent.action.u[0]/norm
                            if not np.isnan(agent.action.u[1]/norm): agent.action.u[1] = agent.action.u[1]/norm
                    else:
                        distance = []
                        served_sensor = state_info[:17]
                        for id, l in enumerate(self.world.landmarks):
                            if id in served_sensor:
                                distance.append(999) # 添加一个极大值
                            else:
                                distance.append(np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos))))
                            # 计算所有land和每个UAV的最短距离

                        index_id = -1
                        for id,l in enumerate(self.world.landmarks):
                            if id == distance.index(min(distance)):
                                target = l.state.p_pos
                                index_id = id
                        logger.info(index_id)
                        v = list(map(lambda x: x[0]-x[1], zip(target, agent.state.p_pos)))
                        agent.action.u[0] = v[0]
                        agent.action.u[1] = v[1]
                        norm = np.linalg.norm(np.array(agent.action.u),ord=2)
                        if not (np.isnan(norm) or  np.isnan(agent.action.u[0]) or np.isnan(agent.action.u[1])): # 求速度的模降速
                            if not np.isnan(agent.action.u[0]/norm): agent.action.u[0] = agent.action.u[0]/norm
                            if not np.isnan(agent.action.u[1]/norm): agent.action.u[1] = agent.action.u[1]/norm
                        
                        if -1 not in served_sensor:
                            # 所有用户都已经找到了，而且BS只有一个的时候
                            logger.info(f"======================================> {self.world.BS[0].state.p_pos}")
                            v = list(map(lambda x: x[0]-x[1], zip(self.world.BS[0].state.p_pos, agent.state.p_pos)))
                            agent.action.u[0] = v[0]
                            agent.action.u[1] = v[1]
                            norm = np.linalg.norm(np.array(agent.action.u),ord=2)
                            if not (np.isnan(norm) or  np.isnan(agent.action.u[0]) or np.isnan(agent.action.u[1])): # 求速度的模降速
                                if not np.isnan(agent.action.u[0]/norm): agent.action.u[0] = agent.action.u[0]/norm
                                if not np.isnan(agent.action.u[1]/norm): agent.action.u[1] = agent.action.u[1]/norm
                            
              
                else:
                    agent.action.u = action[0]
                    # logger.info(f"self.discrete_action_space,false")
            
            sensitivity = 5.0

            if agent.accel is not None:
                logger.info(f"agent.accel is not None")
                sensitivity = agent.accel
            agent.action.u *= sensitivity
            action = action[1:]
        
        if not agent.silent:
            # communication action
            if self.discrete_action_input:
                agent.action.c = np.zeros(self.world.dim_c)
                agent.action.c[action[0]] = 1.0
            else:
                agent.action.c = action[0]
            action = action[1:]
        # make sure we used all elements of action
        assert len(action) == 0
        
    def _set_action_data(self, trans_data, entity):
        '''
        @ modified: June 14 2021
        :param trans_data:
        :param entity:
        :return:
        '''
        # data transmit
        if self.data_trans:
            entity.action.dinput = trans_data[0]
            entity.action.doutput = trans_data[1]
        else:
            entity.action.dinput = 0.0
            entity.action.doutput = 0.0

    def _set_land_state(self, trans_data, land_mark):
        land_mark.action.dinput = trans_data[0]
        land_mark.action.doutput = trans_data[1]

    # calulate transmit loss
    # 上一时刻
    def path_loss(self, agent1, agent2):
        '''
        这里计算的是动作更新后的传输矩阵，但是要注意的是此时动作并未被写入，需要在这里加上，后面集中写入
        同时建模过程简化为了二维空间，如果进行三维空间建模，这里的h需要删除
        :param agent1:
        :param agent2:
        :return:
        '''
        # large scale path loss
        # try:
        #     agent1_pos = agent1.state.p_pos + agent1.action.u
        # except:
        #     agent1_pos = agent1.state.p_pos
        #
        # try:
        #     agent2_pos = agent2.state.p_pos + agent2.action.u
        # except:
        #     agent2_pos = agent2.state.p_pos

        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos))) + self.h**2 #单位时间的数据传输量 
        path_loas = 0.001*dist**(-2) #需要修改
        return path_loas

    def updata_trans_data_matrix(self):
        '''
        @ modified: June 15 2021
        更新传输矩阵，这里传输功率，等信道参数仍需修改
        --------------------------------------------
        land_trans:
        --------------------------------------------
               land1   land2  land3  .... land6
        agent1
        agent2
        ---------------------------------------------
        agent_trans:
        --------------------------------------------
                agent1  agent2
        BS
        agent1
        agent2
        ---------------------------------------------
        '''
        #########
        # transmit matrix
        land_trans = [[0 for _ in range(len(self.landmark))] for _ in range(len(self.agents))] # 2*3
        agent_trans = [[0 for _ in range(len(self.agents))] for _ in range(len(self.BS + self.agents))] # 3*2
        # landmark transmit data to agent
        for i, land_mark in enumerate(self.landmark):
            for j, agent in enumerate(self.agents):
                path_loss = self.path_loss(land_mark, agent)
                D_tr = np.log2(1 + path_loss * self.P_land[i] / self.noise)*self.t
                land_trans[j][i] = D_tr

        # agent transmit data to other agent or BS
        for i, entity in enumerate(self.BS + self.agents):
            for j, agent in enumerate(self.agents):
                if entity == agent:
                    continue
                path_loss = self.path_loss(entity, agent)
                D_tr = np.log2(1 + path_loss * self.P_agent[j] / self.noise)*self.t
                agent_trans[i][j] = D_tr
        return land_trans, agent_trans

    def analytical_topology_matrix(self,  entity, land_trans, agent_trans, landmark_trans_matrix, connect_topology, connect_BS):
        '''
        @ modified: June 15 2021
        landmark_trans_matrix = [[1,0],[0,0],[0,0],[0,0],[0,0],[0,1]]
        connect_topology = [[0,0],[0,0]]
        connect_BS = [[1,1]]
        ---------------------------------------
        landmark_trans_matrix: n_land*n
                   agent1  agent2
        landmark1     1       0
        landmark2     0       1
        ...
        ---------------------------------------
        connect_topology： n*n
        agent1 transmit data to agent2
                agent1  agent2  .....
        agent1      0      1
        agent2      0      0
        ---------------------------------------
        connect_BS: 1*n
                agent1  agent2
        BS        0       1
        ---------------------------------------
        '''
        trans_data = [0,0]

        # agent
        if entity in self.agents:
            i = self.agents.index(entity) 
            # collect landmark data from service range
            for j_0 in range(len(landmark_trans_matrix)): # 输入数据来自landmark
                if landmark_trans_matrix[j_0][i] != 0:
                    trans_data[0] += min(land_trans[i][j_0], self.landmark[j_0].state.d_data)# unidirectional transmission
                    #trans_data[0] += self.landmark[j_0].state.d_data # all data transmit immediately
                    logger.info(f"form landmark {i} {trans_data[0]}")

            # collect data from other agent
            for j_0 in range(len(connect_topology)): # 输入数据来自其他无人机
                if connect_topology[j_0][i] != 0:
                    trans_data[0] += min(agent_trans[i+1][j_0], self.agents[j_0].state.d_data)

            # transmit data to BS 0
            if connect_BS[0][i] != 0: # 输出数据给BS
                trans_data[1] = min(agent_trans[0][i], self.agents[i].state.d_data + trans_data[0] )  # output data
                logger.info(f"output bs {i} {trans_data[0]}")

            # transmit data to other agent
            for j_0 in range(len(connect_topology)): #输出数据给其他无人机
                if connect_topology[i][j_0] != 0:
                    trans_data[1] = min(agent_trans[j_0+1][i], self.agents[i].state.d_data + trans_data[0])

        # landmark
        if entity in self.landmark:
            i = self.landmark.index(entity)
            # data update
            trans_data[0] = 0  # input data
            # transmit data to agent
            for j_0 in range(len(landmark_trans_matrix[0])):
                if landmark_trans_matrix[i][j_0] != 0:
                    trans_data[1] = min(land_trans[j_0][i], self.landmark[i].state.d_data)  # output data

        return trans_data

    # check whether the two entities are covered
    # 上一时刻
    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False


    def updata_landmark_matrix(self):
        '''
        ---------------------------------------
        landmark_trans_matrix: n_land*n
                   agent1  agent2
        landmark1     1       0
        landmark2     0       1
        ...
        ---------------------------------------
        '''
        landmark_trans_matrix = [[0 for _ in range(len(self.agents))] for _ in range(len(self.landmark))]
        # landmark transmit data to agent
        for i, land_mark in enumerate(self.landmark):  # 第i行（第i个用户） 第j列（第j个无人机）
            for j, agent in enumerate(self.agents):
                if self.is_collision(land_mark,agent):
                    landmark_trans_matrix[i][j] = 1

        return landmark_trans_matrix

    # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    # render environment
    def render(self, mode='human'):
        if mode == 'human':
            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            message = ''
            for agent in self.world.agents:
                comm = []
                for other in self.world.agents:
                    if other is agent: continue
                    if np.all(other.state.c == 0):
                        word = '_'
                    else:
                        word = alphabet[np.argmax(other.state.c)]
                    message += (other.name + ' to ' + agent.name + ': ' + word + '   ')
            print(message)

        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                #from gym.envs.classic_control import rendering
                from multiagent import rendering
                self.viewers[i] = rendering.Viewer(700,700)

        # create rendering geometry
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            #from gym.envs.classic_control import rendering
            from multiagent import rendering
            self.render_geoms = []
            self.render_geoms_xform = []
            for entity in self.world.entities:
                geom = rendering.make_circle(entity.size)
                xform = rendering.Transform()
                if 'agent' in entity.name:
                    geom.set_color(*entity.color, alpha=0.5)
                else:
                    geom.set_color(*entity.color)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            # add geoms to viewer
            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)

        results = []
        for i in range(len(self.viewers)):
            from multiagent import rendering
            # update bounds to center around agent
            cam_range = 1
            if self.shared_viewer:
                pos = np.zeros(self.world.dim_p)
            else:
                pos = self.agents[i].state.p_pos
            self.viewers[i].set_bounds(pos[0]-cam_range,pos[0]+cam_range,pos[1]-cam_range,pos[1]+cam_range)
            # update geometry positions
            for e, entity in enumerate(self.world.entities):
                self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
            # render to display or array
            results.append(self.viewers[i].render(return_rgb_array = mode=='rgb_array'))

        return results

    # create receptor field locations in local coordinate frame
    def _make_receptor_locations(self, agent):
        receptor_type = 'polar'
        range_min = 0.05 * 2.0
        range_max = 1.00
        dx = []
        # circular receptive field
        if receptor_type == 'polar':
            for angle in np.linspace(-np.pi, +np.pi, 8, endpoint=False):
                for distance in np.linspace(range_min, range_max, 3):
                    dx.append(distance * np.array([np.cos(angle), np.sin(angle)]))
            # add origin
            dx.append(np.array([0.0, 0.0]))
        # grid receptive field
        if receptor_type == 'grid':
            for x in np.linspace(-range_max, +range_max, 5):
                for y in np.linspace(-range_max, +range_max, 5):
                    dx.append(np.array([x,y]))
        return dx


# vectorized wrapper for a batch of multi-agent environments
# assumes all environments have the same observation and action space
class BatchMultiAgentEnv(gym.Env):
    metadata = {
        'runtime.vectorized': True,
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, env_batch):
        self.env_batch = env_batch

    @property
    def n(self):
        return np.sum([env.n for env in self.env_batch])

    @property
    def action_space(self):
        return self.env_batch[0].action_space

    @property
    def observation_space(self):
        return self.env_batch[0].observation_space

    def step(self, action_n, time):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        i = 0
        for env in self.env_batch:
            obs, reward, done, _ = env.step(action_n[i:(i+env.n)], time)
            i += env.n
            obs_n += obs
            # reward = [r / len(self.env_batch) for r in reward]
            reward_n += reward
            done_n += done
        return obs_n, reward_n, done_n, info_n

    def reset(self):
        obs_n = []
        for env in self.env_batch:
            obs_n += env.reset()
        return obs_n

    # render environment
    def render(self, mode='human', close=True):
        results_n = []
        for env in self.env_batch:
            results_n += env.render(mode, close)
        return results_n
