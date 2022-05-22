"""
Created on  Apr 9 2021
@author: wangmeng
@version: v1.3
@modified: Apr 27 2021
@marks:
加入AOI可视化
"""
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import math
def median_and_percentile(x, axis, lower=1, upper=100):
    """
    计算中位数和指定的百分位数，目前设定：下界为20%，上界为80%
    :param x: 输入数据
    :param axis: 需要计算的轴
    :param lower: 指定百分位数的下界
    :param upper: 指定百分位数的上界
    :return: 中位数、平均数、下界值、上界值
    """
    assert (lower >= 0 and upper <= 100)
    median = np.median(x, axis)
    mean = np.mean(x, axis)
    low_per = np.percentile(x, lower, axis)
    up_per = np.percentile(x, upper, axis)
    return median, mean, low_per, up_per

def stack_data(x, window, stride):
    """
    按上述的算法将一维数组按窗口大小滑动堆叠构成二维数组
    :param x: 原始数据, 一维数组
    :param window: 窗口大小
    :param stride: 步长
    :return: 二维数组
    """
    n = len(x)
    assert n >= window
    y = []
    for i in range((n - window + 1) // stride):
        y.append(x[i * stride: i * stride + window])
    return np.asarray(y)

def plot_energy(UAV0_location, UAV1_location, UAV2_location, save_path, landmarks,show_flag=False):
    # import enviroments.scenarios as scenarios
    #
    # # load scenario from script
    # scenario = scenarios.load(scenarios_name +".py").Scenario()
    # # create world
    # world = scenario.make_world()
    landmark_location =[]

    for entity in landmarks:  # world.entities:
        landmark_location.append(entity.state.p_pos)

    UAV1_dis = []
    UAV2_dis = []
    UAV3_dis = []
    energy_cost_sum = []
    UAV1x = []
    UAV1y = []
    UAV2x = []
    UAV2y = []
    UAV3x = []
    UAV3y = []
    for i in UAV0_location:
        UAV1x.append(i[0])
        UAV1y.append(i[1])
        UAV1_dis.append(np.sqrt((i[0]**2 + i[1]**2)))
    for i in UAV1_location:
        UAV2x.append(i[0])
        UAV2y.append(i[1])
        UAV2_dis.append(np.sqrt((i[0] ** 2 + i[1] ** 2)))
    for i in UAV2_location:
        UAV3x.append(i[0])
        UAV3y.append(i[1])
        UAV3_dis.append(np.sqrt((i[0] ** 2 + i[1] ** 2)))

    for i in range(len(UAV1_dis)):
        energy_cost_sum.append(UAV1_dis[i]+UAV2_dis[i]+UAV3_dis[i])
    print(energy_cost_sum)
    #landmark_location = [[-22., 21.], [-41., -4.], [-17., -17.], [21., -7.], [34., 15.], [8., 30.]]
    x = []
    y = []
    for i in landmark_location:
        x.append(i[0])
        y.append(i[1])
    colors_UAV = ['r', 'g', 'b']

    plt.figure()

    # plt.scatter(UAV1x, UAV1y, s=1, c=colors_UAV[0], linewidths=5, alpha=0.5, marker='o')
    # plt.scatter(UAV2x, UAV2y, s=1, c=colors_UAV[1], linewidths=5, alpha=0.5, marker='o')
    # plt.scatter(UAV3x, UAV3y, s=1, c=colors_UAV[2], linewidths=5, alpha=0.5, marker='o')
    plt.plot(np.asarray(range(len(UAV1_dis))), UAV1_dis, 'r--', label='UAV-1')
    plt.plot(np.asarray(range(len(UAV2_dis))), UAV2_dis, 'g-', label='UAV-2')
    plt.plot(np.asarray(range(len(UAV3_dis))), UAV3_dis, 'b:', label='UAV-3')
    #plt.plot(np.asarray(range(len(energy_cost_sum))), energy_cost_sum, 'r--', label='energy_cost')
    # plt.plot(UAV2x, UAV2y, 'g-', label='UAV2')
    # plt.plot(UAV3x, UAV3y, 'b:', label='UAV3')
    #
    # plt.scatter([UAV1x[0]], [UAV1y[0]], s=1, c=colors_UAV[0], linewidths=10, alpha=1, marker='+', label='start point')
    # plt.scatter([UAV2x[0]], [UAV2y[0]], s=1, c=colors_UAV[1], linewidths=10, alpha=1, marker='+')
    # plt.scatter([UAV3x[0]], [UAV3y[0]], s=1, c=colors_UAV[2], linewidths=10, alpha=1, marker='+')
    #
    #
    # #landmark 坐标
    # plt.scatter(x, y, s=1200, alpha=0.2, marker='o')
    # plt.scatter(x, y, s=10, alpha=0.5, marker='o', label='Ground users')
    #
    # #BS 坐标
    # BS_location = BS[0].state.p_pos
    # plt.scatter(BS_location[0], BS_location[1], s=200, alpha=0.5, marker='o', label='Base station (BS)')
    # R = BS[0].size
    # theta = np.arange(0, 2 * np.pi, 0.01)
    # # x_BS = BS_location[0] + R * np.cos(theta) #0.5为半径
    # # y_BS = BS_location[1] + R * np.sin(theta)
    # # plt.scatter(x_BS, y_BS, s=5, alpha=0.5, marker='_')
    # my_x_ticks = np.arange(-1.1, 1.1, 0.2)
    # my_y_ticks = np.arange(-1.1, 1.1, 0.2)
    # plt.xticks(my_x_ticks)
    # plt.yticks(my_y_ticks)
    #
    # plt.xlim(-1.1, 1.1)
    # plt.ylim(-1.1,1.1)
    front = {'size':15}
    plt.xlabel('Time slot (t)',front)
    # 设置Y轴标签
    plt.ylabel('Energy consumption (e_i(t))',front)
    #plt.title('trajectory')
    plt.grid(linestyle=":")
    plt.legend(loc="best", frameon=False, fontsize=8)

    #plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    # plt.savefig('test_xx.png', dpi=200, bbox_inches='tight', transparent=False)
    if show_flag:
        plt.show()
    else:
        plt.savefig(save_path + '/eps-energy.eps', dpi=600, format='eps', bbox_inches = 'tight')
        plt.savefig(save_path + '/fig-energy.png', dpi=600, format='png', bbox_inches = 'tight')
    plt.close()



def display_trajectory(UAV0_location, UAV1_location, UAV2_location, BS, save_path, landmarks,use_bayesian = False,show_flag=False):
    # import enviroments.scenarios as scenarios
    #
    # # load scenario from script
    # scenario = scenarios.load(scenarios_name +".py").Scenario()
    # # create world
    # world = scenario.make_world()
    landmark_location =[]

    for entity in landmarks:  # world.entities:
        landmark_location.append(entity.state.p_pos)

    UAV1x = []
    UAV1y = []
    UAV2x = []
    UAV2y = []
    UAV3x = []
    UAV3y = []
    for i in UAV0_location:
        UAV1x.append(i[0])
        UAV1y.append(i[1])
    for i in UAV1_location:
        UAV2x.append(i[0])
        UAV2y.append(i[1])
    for i in UAV2_location:
        UAV3x.append(i[0])
        UAV3y.append(i[1])
    #landmark_location = [[-22., 21.], [-41., -4.], [-17., -17.], [21., -7.], [34., 15.], [8., 30.]]
    x = []
    y = []
    for i in landmark_location:
        x.append(i[0])
        y.append(i[1])
    colors_UAV = ['r', 'g', 'b']

    plt.figure()

    plt.scatter(UAV1x, UAV1y, s=1, c=colors_UAV[0], linewidths=5, alpha=0.5, marker='o')
    plt.scatter(UAV2x, UAV2y, s=1, c=colors_UAV[1], linewidths=5, alpha=0.5, marker='o')
    plt.scatter(UAV3x, UAV3y, s=1, c=colors_UAV[2], linewidths=5, alpha=0.5, marker='o')

    plt.plot(UAV1x, UAV1y, 'r--')
    plt.plot(UAV2x, UAV2y, 'g-')
    plt.plot(UAV3x, UAV3y, 'b:')

    plt.scatter([UAV1x[0]], [UAV1y[0]], s=1, c=colors_UAV[0], linewidths=10, alpha=1, marker='+')
    plt.scatter([UAV2x[0]], [UAV2y[0]], s=1, c=colors_UAV[1], linewidths=10, alpha=1, marker='+')
    plt.scatter([UAV3x[0]], [UAV3y[0]], s=1, c=colors_UAV[2], linewidths=10, alpha=1, marker='+')

    plt.text(UAV1x[0]-0.1, UAV1y[0]+0.05, 'UAV-1', fontsize=10, color='k')
    plt.text(UAV2x[0]-0.1, UAV2y[0]-0.1, 'UAV-2', fontsize=10, color='k')
    plt.text(UAV3x[0]+0.03, UAV3y[0]-0.08, 'UAV-3', fontsize=10, color='k')

    #landmark 坐标
    plt.scatter(x, y, s=1200, alpha=0.2, marker='o')
    plt.scatter(x, y, s=10, alpha=0.5, marker='o')
    plt.text(x[0] - 0.03, y[0] + 0.08, '0', fontsize=10, color='k')
    plt.text(x[1] - 0.03, y[1] + 0.08, '1', fontsize=10, color='k')
    plt.text(x[2] - 0.03, y[2] + 0.08, '2', fontsize=10, color='k')
    plt.text(x[3] - 0.03, y[3] + 0.08, '3', fontsize=10, color='k')
    plt.text(x[4] - 0.03, y[4] + 0.08, '4', fontsize=10, color='k')
    plt.text(x[5] - 0.03, y[5] + 0.08, '5', fontsize=10, color='k')
    plt.text(x[6] - 0.03, y[6] + 0.08, '6', fontsize=10, color='k')
    plt.text(x[7] - 0.03, y[7] + 0.08, '7', fontsize=10, color='k')
    plt.text(x[8] - 0.03, y[8] + 0.08, '8', fontsize=10, color='k')

    plt.text(x[4] + 0.03, y[4] - 0.08, 'Ground user', fontsize=10, color='k')

    #BS 坐标
    BS_location = BS[0].state.p_pos
    plt.scatter(BS_location[0], BS_location[1], s=200, alpha=0.5, marker='o')
    plt.text(BS_location[0] - 0.04, BS_location[1] - 0.12, 'BS', fontsize=10, color='k')
    R = BS[0].size
    theta = np.arange(0, 2 * np.pi, 0.01)
    # x_BS = BS_location[0] + R * np.cos(theta) #0.5为半径
    # y_BS = BS_location[1] + R * np.sin(theta)
    # plt.scatter(x_BS, y_BS, s=5, alpha=0.5, marker='_')
    my_x_ticks = np.arange(-1.1, 1.1, 0.2)
    my_y_ticks = np.arange(-1.1, 1.1, 0.2)
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)

    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1,1.1)
    front = {'size': 15}
    plt.xlabel('X coordinate (km)',front)
    # 设置Y轴标签
    plt.ylabel('Y coordinate (km)',front)
    #plt.title('trajectory')
    plt.grid(linestyle=":")
    plt.legend(loc="lower right", frameon=False, fontsize=8)
    # ax = plt.axes()
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    #ax.spines['bottom'].set_visible(False)
    #ax.spines['left'].set_visible(False)

    #plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    # plt.savefig('test_xx.png', dpi=200, bbox_inches='tight', transparent=False)
    if show_flag:
        plt.show()
    else:
        plt.savefig(save_path + '/eps-trajectory_B%(use_Bayesian)s.eps' % {'use_Bayesian': use_bayesian}, dpi=600, format='eps', bbox_inches = 'tight')
        plt.savefig(save_path + '/fig-trajectory_B%(use_Bayesian)s.png' % {'use_Bayesian': use_bayesian}, dpi=600, format='png', bbox_inches = 'tight')
    plt.close()

def plot_topology(UAV0_location, UAV1_location, UAV2_location, BS, save_path, landmarks, all_topology, all_connect_BS, use_bayesian = False,heuristic_id = 1,show_flag=False):
    landmark_location = []

    for entity in landmarks:  # world.entities:
        landmark_location.append(entity.state.p_pos)
    plt.figure()



    UAV1x = []
    UAV1y = []
    UAV2x = []
    UAV2y = []
    UAV3x = []
    UAV3y = []
    for i in UAV0_location:
        UAV1x.append(i[0])
        UAV1y.append(i[1])
    for i in UAV1_location:
        UAV2x.append(i[0])
        UAV2y.append(i[1])
    for i in UAV2_location:
        UAV3x.append(i[0])
        UAV3y.append(i[1])
    # landmark_location = [[-22., 21.], [-41., -4.], [-17., -17.], [21., -7.], [34., 15.], [8., 30.]]
    x = []
    y = []
    for i in landmark_location:
        x.append(i[0])
        y.append(i[1])
    colors_UAV = ['r', 'g', 'b']

    plt.plot(UAV1x, UAV1y, 'r--')
    plt.plot(UAV2x, UAV2y, 'g-')
    plt.plot(UAV3x, UAV3y, 'b:')

    plt.scatter([UAV1x[0]], [UAV1y[0]], s=1, c=colors_UAV[0], linewidths=10, alpha=1, marker='+')
    plt.scatter([UAV2x[0]], [UAV2y[0]], s=1, c=colors_UAV[1], linewidths=10, alpha=1, marker='+')
    plt.scatter([UAV3x[0]], [UAV3y[0]], s=1, c=colors_UAV[2], linewidths=10, alpha=1, marker='+')
    plt.text(UAV1x[0] - 0.1, UAV1y[0] + 0.05, 'UAV-1', fontsize=10, color='k')
    plt.text(UAV2x[0] - 0.1, UAV2y[0] - 0.1, 'UAV-2', fontsize=10, color='k')
    plt.text(UAV3x[0] + 0.03, UAV3y[0] - 0.08, 'UAV-3', fontsize=10, color='k')

    plt.scatter(UAV1x, UAV1y, s=1, c=colors_UAV[0], linewidths=5, alpha=0.5, marker='o', label='U2B model')
    plt.scatter(UAV2x, UAV2y, s=1, c=colors_UAV[1], linewidths=5, alpha=0.5, marker='o')
    plt.scatter(UAV3x, UAV3y, s=1, c=colors_UAV[2], linewidths=5, alpha=0.5, marker='o')

    #全连接

    for k in range(len(UAV0_location)):
        UAV_loc  = []
        UAV_loc.append(UAV0_location[k])
        UAV_loc.append(UAV1_location[k])
        UAV_loc.append(UAV2_location[k])
        for i, UAV_topology in enumerate(all_topology[k]):
            for j, is_connect in enumerate(UAV_topology):
                if is_connect:
                    #plt.plot([UAV_loc[i][0],UAV_loc[j][0]],[UAV_loc[i][1],UAV_loc[j][1]], color='r',linestyle="--")
                    plt.scatter([UAV_loc[i][0],UAV_loc[j][0]],[UAV_loc[i][1],UAV_loc[j][1]], s=1, c='k', linewidths=5, alpha=1, marker='*')

        for i, BS_topology in enumerate(all_connect_BS[k]):
            for j, is_connect in enumerate(BS_topology):
                if is_connect:
                    # plt.plot([UAV_loc[j][0], BS[0].state.p_pos[0]], [UAV_loc[j][1],BS[0].state.p_pos[1]], color='y',
                    #          linestyle="-.")
                    pass
                    #plt.scatter([UAV_loc[i][0],UAV_loc[j][0]],[UAV_loc[i][1],UAV_loc[j][1]], color='b')
    plt.scatter([UAV_loc[i][0], UAV_loc[j][0]], [UAV_loc[i][1], UAV_loc[j][1]], s=1, c='k', linewidths=5, alpha=1,
                marker='*', label='U2U model')
    # landmark 坐标
    plt.scatter(x, y, s=1200, alpha=0.2, marker='o')
    plt.scatter(x, y, s=10, alpha=0.5, marker='o')
    plt.text(x[0] - 0.03, y[0] + 0.08, '0', fontsize=10, color='k')
    plt.text(x[1] - 0.03, y[1] + 0.08, '1', fontsize=10, color='k')
    plt.text(x[2] - 0.03, y[2] + 0.08, '2', fontsize=10, color='k')
    plt.text(x[3] - 0.03, y[3] + 0.08, '3', fontsize=10, color='k')
    plt.text(x[4] - 0.03, y[4] + 0.08, '4', fontsize=10, color='k')
    plt.text(x[5] - 0.03, y[5] + 0.08, '5', fontsize=10, color='k')
    plt.text(x[6] - 0.03, y[6] + 0.08, '6', fontsize=10, color='k')
    plt.text(x[7] - 0.03, y[7] + 0.08, '7', fontsize=10, color='k')
    plt.text(x[8] - 0.03, y[8] + 0.08, '8', fontsize=10, color='k')

    plt.text(x[4] + 0.03, y[4] - 0.08, 'Ground user', fontsize=10, color='k')

    # BS 坐标
    BS_location = BS[0].state.p_pos
    plt.scatter(BS_location[0], BS_location[1], s=200, alpha=0.5, marker='o')
    plt.text(BS_location[0] - 0.04, BS_location[1] - 0.12, 'BS', fontsize=10, color='k')
    # R = BS[0].size
    # theta = np.arange(0, 2 * np.pi, 0.01)
    # x_BS = BS_location[0] + R * np.cos(theta) #0.5为半径
    # y_BS = BS_location[1] + R * np.sin(theta)
    # plt.scatter(x_BS, y_BS, s=5, alpha=0.5, marker='_')
    my_x_ticks = np.arange(-1.1, 1.1, 0.2)
    my_y_ticks = np.arange(-1.1, 1.1, 0.2)
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)

    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    front = {'size': 15}
    plt.xlabel('X coordinate (km)',front)
    # 设置Y轴标签
    plt.ylabel('Y coordinate (km)',front)
    # plt.title('trajectory')
    plt.grid(linestyle=":")
    plt.legend(loc="lower right", frameon=False, fontsize=12)
    # plt.savefig('test_xx.png', dpi=200, bbox_inches='tight', transparent=False)
    if show_flag:
        plt.show()
    else:
        plt.savefig(save_path + '/eps-topology_B%(use_Bayesian)s_N%(use_noise)s.eps'% {'use_Bayesian': use_bayesian,'use_noise':heuristic_id}, dpi=600, format='eps', bbox_inches = 'tight')
        plt.savefig(save_path + '/fig-topology_B%(use_Bayesian)s_N%(use_noise)s.png'% {'use_Bayesian': use_bayesian,'use_noise':heuristic_id}, dpi=600, format='png', bbox_inches = 'tight')
    plt.close()

def display_data_remain(agents, landmarks, save_path,show_flag=False):
    data_memory = []
    for entity in agents + landmarks:  # world.entities:
        data_memory.append(entity.state.d_data)

    plt.figure()

    plt.bar(range(len(data_memory)), data_memory)
    plt.legend(loc=0, ncol=2)
    front = {'size': 15}
    plt.xlabel('Task completion time T (time slot)',front)
    # 设置Y轴标签
    plt.ylabel('data buffer remain ',front)
    plt.title('data_plot')
    plt.grid(True)
    if show_flag:
        plt.show()
    else:
        plt.savefig(save_path + '/fig-data_remain.png',dpi=600, format='eps', bbox_inches = 'tight')
    plt.close()

def display_data_change(UAV1_data, UAV2_data,UAV3_data, save_path,use_bayesian=False,heuristic_id = 1,show_flag=False):
    plt.figure()
    x = range(len(UAV1_data))
    from asyncio.log import logger
    logger.info(f"-UAV1_data {UAV1_data} -UAV2_data {UAV2_data} -UAV3_data {UAV3_data}")
    plt.plot(x, UAV1_data, 'r--', label='UAV-1')
    plt.plot(x, UAV2_data, 'g-', label='UAV-2')
    plt.plot(x, UAV3_data, 'b:', label='UAV-3')
    front = {'size': 15}
    plt.xlabel('Time slot',front)
    plt.ylabel('Buffer size',front)
    #plt.title('UAV data change')
    plt.grid(True)
    plt.legend(loc="best", frameon=False, fontsize=14)
    #plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    if show_flag:
        plt.show()
    else:
        plt.savefig(save_path + '/eps-data_buffer_B%(use_Bayesian)s_N%(use_noise)s.eps'% {'use_Bayesian': use_bayesian,'use_noise':heuristic_id}, dpi=600, format='eps', bbox_inches = 'tight')
        plt.savefig(save_path + '/fig-data_buffer_B%(use_Bayesian)s_N%(use_noise)s.png'% {'use_Bayesian': use_bayesian,'use_noise':heuristic_id}, dpi=600, format='png', bbox_inches = 'tight')
    plt.close('all')

def display_AOI(AOI, save_path):
    x = []
    sensor1 = []
    sensor2 = []
    sensor3 = []
    sensor4 = []
    sensor5 = []
    sensor6 = []

    i=0
    for AOI_item in AOI:
        i+=1
        sensor1.append(AOI_item[0])
        sensor2.append(AOI_item[1])
        sensor3.append(AOI_item[2])
        sensor4.append(AOI_item[3])
        sensor5.append(AOI_item[4])
        sensor6.append(AOI_item[5])
        x.append(i)



    plt.figure()
    plt.plot(x, sensor1, label = "line 1")
    plt.plot(x, sensor2, label="line 2")
    plt.plot(x, sensor3, label="line 3")
    plt.plot(x, sensor4, label="line 4")
    plt.plot(x, sensor5, label="line 5")
    plt.plot(x, sensor6, label="line 6")
    plt.legend(loc=0, ncol=2)

    plt.xlabel('times')
    # 设置Y轴标签
    plt.ylabel('AOI')
    plt.title('AOI')
    plt.grid(True)

    # plt.savefig('test_xx.png', dpi=200, bbox_inches='tight', transparent=False)
    #plt.show()
    plt.savefig(save_path + '/plt3.png', dpi=600, format='eps', bbox_inches = 'tight')

def display_mean(y_axis, x_axis, x_title, y_title, line, title ):
    """
    对要绘制的数组收敛图进行平滑和中值处理,并保存为pdf
    """
    # ------------迭代收敛比较图------------#
    y_axis1_ = stack_data(y_axis, window=8, stride=2)
    optm_m_1, optm_mean_1, optm_l_1, optm_u_1 = median_and_percentile(y_axis1_, axis=1)
    optm_x_1 = np.asarray(range(len(optm_m_1)))
    x_ep1 = optm_x_1 * x_axis / len(optm_x_1)

    fig = plt.figure()
    spl = fig.add_subplot(111)
    spl.plot(x_ep1, optm_mean_1, color='b', label=line)
    spl.fill_between(x_ep1, optm_u_1, optm_l_1, facecolor='b', alpha=0.3)
    plt.xlabel(x_title, fontsize=14)
    plt.ylabel(y_title, fontsize=14)
    spl.legend(loc="upper left", frameon=False, fontsize=14)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.title(title)
    fig.tight_layout()
    pp = PdfPages('checkpoints/Convergence01.pdf')
    plt.savefig(pp, format='pdf')
    pp.close()
    plt.show()

def display_queue_size(data, x_label, save_path, show_flag=False):
    plt.figure()
    color = ['r--', 'g-', 'b:', 'k-']
    label = ['algorithm-1', 'algorithm-2', 'algorithm-3', 'algorithm-4']
    lens = len(data)
    for i in range(lens):
        plt.plot(x_label, data[i], color[i], label=label[i])
    front = {'size': 15}
    plt.xlabel('Workload completion time T (time slot)', front)
    plt.ylabel('Max queue size (M bits)', front)
    # plt.title('UAV data change')
    plt.grid(True)
    plt.legend(loc="best", frameon=False, fontsize=14)
    # plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    if show_flag:
        plt.show()
    else:
        plt.savefig(save_path + '/eps-Dynamics_of_queue_size.eps', dpi=600, format='eps', bbox_inches = 'tight')
        plt.savefig(save_path + '/fig-Dynamics_of_queue_size.png', dpi=600, format='png', bbox_inches = 'tight')
    plt.close('all')

def display_finish_time(data,x_label, save_path, show_flag=False):
    plt.figure()
    color = ['r--', 'g-', 'b:', 'k-']
    label = ['algorithm-1', 'algorithm-2', 'algorithm-3', 'algorithm-4']
    lens = len(data)
    for i in range(lens):
        plt.plot(x_label, data[i], color[i], label=label[i])
    front = {'size': 15}
    plt.xlabel('IoT users data size (Mbits)', front)
    plt.ylabel('Workload completion time T (time slot)', front)
    # plt.title('UAV data change')
    plt.grid(True)
    plt.legend(loc="best", frameon=False, fontsize=14)
    # plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    if show_flag:
        plt.show()
    else:
        plt.savefig(save_path + '/eps-Workload_completion_time.eps', dpi=600, format='eps', bbox_inches = 'tight')
        plt.savefig(save_path + '/fig-Workload_completion_time.png', dpi=600, format='png', bbox_inches = 'tight')
    plt.close('all')

def plot_user_data_remain(landmarks,lan_data,save_path,show_flag=False):
    plt.figure()
    color = ['r--', 'g--', 'b--', 'k--','r:', 'g:', 'b:', 'k:','r-']
    label = ['user-0','user-1', 'user-2', 'user-3', 'user-4','user-5', 'user-6', 'user-7', 'user-8']
    lens = len(landmarks)

    for i in range(lens):
        plt.plot(range(0,len(lan_data[0])), lan_data[i], color[i], label=label[i])
    front = {'size': 15}
    plt.xlabel('Time slot', front)
    plt.ylabel('User remain data (M)', front)
    # plt.title('UAV data change')
    plt.grid(True)
    plt.legend(loc="best", frameon=False, fontsize=14)
    # plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    if show_flag:
        plt.show()
    else:
        plt.savefig(save_path + '/eps-User_data_remain.eps', dpi=600, format='eps', bbox_inches = 'tight')
        plt.savefig(save_path + '/fig-User_data_remain.png', dpi=600, format='png', bbox_inches = 'tight')
    plt.close('all')

def display_choice_rate(Choice_rate, save_path):
    plt.figure()
    front = {'size': 15}
    y_axis1 = Choice_rate
    y_axis1_ = stack_data(y_axis1, window=1, stride=1)*0.1
    
    optm_m_1, optm_mean_1, optm_l_1, optm_u_1 = median_and_percentile(y_axis1_, axis = 1)
    optm_x_1 = np.asarray(range(len(optm_m_1)))
    x_ep1 = optm_x_1 * 200000 / len(optm_x_1)
    log_1 = []
    log_1_u = []
    log_1_l = []
    temp = 0
    for i in range(len(optm_mean_1)):
        temp = 10 * math.log10(optm_mean_1[i])
        log_1.append(temp)
    # for i in range(len(optm_mean_1)):
    #     temp = 10 * math.log10(optm_u_1[i])
    #     log_1_u.append(temp)
    # for i in range(len(optm_mean_1)):
    #     temp = 10 * math.log10(optm_l_1[i])
    #     log_1_l.append(temp)

    plt.plot(x_ep1, y_axis1_, color='k')
    #plt.fill_between(x_ep1, log_1_u, log_1_l, facecolor='k', alpha=0.3)
    plt.xlabel('Epoch', front)
    # 设置Y轴标签
    plt.ylabel('Choice action rate', front)
    plt.grid(True)#linestyle=":")
    #plt.legend(loc="lower right", frameon=False, fontsize=8)
    plt.savefig(save_path + '/eps-choice_rate.eps', dpi=600, format='eps', bbox_inches='tight')
    plt.savefig(save_path + '/fig-choice_rate.png', dpi=600, format='png', bbox_inches='tight')
    plt.close()


def plot_energy(data,save_path,show_flag=False):
    front = {'size': 15}
    plt.figure()
    y_axis1 = data[10:-40]
    y_axis1_ = stack_data(y_axis1, window=1, stride=1)
    optm_m_1, optm_mean_1, optm_l_1, optm_u_1 = median_and_percentile(y_axis1_, axis = 1)
    optm_x_1 = np.asarray(range(len(optm_m_1)))
    x_ep1 = optm_x_1 * 200000 / len(optm_x_1)
    log_1 = []
    log_1_u = []
    log_1_l = []
    temp = 0

    for i in range(len(optm_mean_1)):
        temp = 10 * math.log10(optm_mean_1[i])
        log_1.append(temp)
    for i in range(len(optm_mean_1)):
        temp = 10 * math.log10(optm_u_1[i])
        log_1_u.append(temp)
    for i in range(len(optm_mean_1)):
        temp = 10 * math.log10(optm_l_1[i])
        log_1_l.append(temp)

    plt.plot(x_ep1, log_1, color='k')
    plt.fill_between(x_ep1, log_1_u, log_1_l, facecolor='k', alpha=0.3)

    plt.xlabel('Epoch', front)
    plt.ylabel('System total energy consume', front)
    # plt.title('UAV data change')
    plt.grid(True)
    plt.legend(loc="best", frameon=False, fontsize=14)
    if show_flag:
        plt.show()
    else:
        plt.savefig(save_path + '/eps-Total_energy_consume.eps', dpi=600, format='eps', bbox_inches = 'tight')
        plt.savefig(save_path + '/fig-Total_energy_consume.png', dpi=600, format='png', bbox_inches = 'tight')
    plt.close('all')