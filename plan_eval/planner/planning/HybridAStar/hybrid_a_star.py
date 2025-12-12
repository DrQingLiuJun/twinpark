"""

Hybrid A* path planning

author: Meng Liu (@QingLiu)

"""

import time
import heapq
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
import sys
import os
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from HybridAStar.dynamic_programming_heuristic import calc_distance_heuristic
from ReedsSheppPath import reeds_shepp_path_planning as rs
from HybridAStar.car import move, check_car_collision, MAX_STEER, WB, plot_car, BUBBLE_R
from HybridAStar.PolynomialInterpolation import Polynomial5Interpolation
from HybridAStar.PolynomialInterpolation import velocity_planning
from HybridAStar.search import takeClosest
from HybridAStar.smooth import path_optimization

## 参数设置
XY_GRID_RESOLUTION = 0.755                # [m]XY平面网格的分辨率，即路径规划中考虑的每个网格的大小。
YAW_GRID_RESOLUTION = np.deg2rad(10.0)  # [rad]偏航角（yaw）网格的分辨率，即考虑的每个偏航角的大小。这里将角度转换为弧度。
MOTION_RESOLUTION = 0.3               # [m] path interpolate resolution路径插值分辨率，即规划出的路径上相邻两个点之间的距离。
N_STEER = 10                            # number of steer command可用的转向命令数量，用于规划车辆的转向。

SB_COST = 5.0                           # switch back penalty cost切换方向的惩罚成本，表示在路径规划中切换方向的成本。
BACK_COST = 2.0                         # backward penalty cost后退的惩罚成本，表示在路径规划中后退的成本。
STEER_CHANGE_COST = 5.0                 # steer angle change penalty cost转向角度变化的惩罚成本，表示在路径规划中改变转向角度的成本。
STEER_COST = 5.0                        # steer angle change penalty cost转向的惩罚成本，表示在路径规划中转向的成本。
YAW_COST = 0                            # 20.0偏航角度的惩罚成本，表示在路径规划中偏航角度的成本。
H_COST = 10.0                           # Heuristic cost启发式成本，表示在路径规划中启发式搜索的成本。

show_animation = False   # 是否显示路径规划的动画


class Node:
    def __init__(self, x_ind, y_ind, yaw_ind, direction,
                 x_list, y_list, yaw_list, directions,
                 steer=0.0, parent_index=None, cost=None):
        self.x_index = x_ind
        self.y_index = y_ind
        self.yaw_index = yaw_ind    # 车辆当前朝向的离散化角度位置
        self.direction = direction  # 前进方向、后退方向
        self.x_list = x_list
        self.y_list = y_list
        self.yaw_list = yaw_list
        self.directions = directions
        self.steer = steer
        self.parent_index = parent_index
        self.cost = cost

# 规划的路径
class Path:
    def __init__(self, x_list, y_list, yaw_list, direction_list, cost):
        self.x_list = x_list
        self.y_list = y_list
        self.yaw_list = yaw_list
        self.direction_list = direction_list
        self.cost = cost

# 车辆的轨迹信息 
# 实际运动路径
class Trajactory:
    def __init__(self, x_list, y_list, yaw_list, v_list, omega_list) :
        self.x_list = x_list
        self.y_list = y_list
        self.yaw_list = yaw_list
        self.v_list = v_list
        self.omega_list = omega_list

# 配置环境参数
class Config:
    def __init__(self, ox, oy, xy_resolution, yaw_resolution):
        # 计算环境边界
        min_x_m = min(ox)
        min_y_m = min(oy) # 环境中的障碍物的 x 和 y 坐标列表
        max_x_m = max(ox)
        max_y_m = max(oy)

        ox.append(min_x_m)
        oy.append(min_y_m)
        ox.append(max_x_m)
        oy.append(max_y_m)

        # 环境的 x 和 y 方向的最小和最大网格索引
        self.min_x = round(min_x_m / xy_resolution)
        self.min_y = round(min_y_m / xy_resolution)
        self.max_x = round(max_x_m / xy_resolution)
        self.max_y = round(max_y_m / xy_resolution)

        # 环境的宽度和高度（以网格为单位）
        self.x_w = round(self.max_x - self.min_x)
        self.y_w = round(self.max_y - self.min_y)

        # 偏航角度的最小和最大网格索引
        self.min_yaw = round(- math.pi / yaw_resolution) - 1
        self.max_yaw = round(math.pi / yaw_resolution)
        self.yaw_w = round(self.max_yaw - self.min_yaw)

#生成器函数, 计算机器人的运动输入
def calc_motion_inputs():
    for steer in np.concatenate((np.linspace(-MAX_STEER, MAX_STEER,#生成一个包含从 -MAX_STEER 到 MAX_STEER 范围内均匀分布的转向角的数组
                                             N_STEER), [0.0])):
        for d in [1, -1]:
            yield [steer, d]#将每个运动输入作为生成器的一部分进行返回

# 生成当前节点的邻居节点
def get_neighbors(current, goal_node, config, ox, oy, kd_tree):# kd_tree KD树，用于加速搜索
    for steer, d in calc_motion_inputs():
        node = calc_next_node(current, goal_node, steer, d, config, ox, oy, kd_tree)
        if node and verify_index(node, config):
            yield node

# 计算给定当前节点、转向、前进方向的情况下的下一个节点
def calc_next_node(current, goal_node, steer, direction, config, ox, oy, kd_tree):
    # 获取当前节点的位置和方向
    x, y, yaw = current.x_list[-1], current.y_list[-1], current.yaw_list[-1]

    # ? original version 
    # arc_l = XY_GRID_RESOLUTION * 1.5  
    # x_list, y_list, yaw_list = [], [], []
    # for _ in np.arange(0, arc_l, MOTION_RESOLUTION): # ? why
    #     x, y, yaw = move(x, y, yaw, MOTION_RESOLUTION * direction, steer)
    #     x_list.append(x)
    #     y_list.append(y)
    #     yaw_list.append(yaw)

    # ! modified by zzs 2023/12/12
    arc_l = XY_GRID_RESOLUTION * 1.5 # 车辆沿着当前朝向（yaw）移动的一个距离，车辆在当前步骤中移动的路径长度
    x_list, y_list, yaw_list = [], [], []
    x, y, yaw = move(x, y, yaw, arc_l * direction, steer)
    x_list.append(x)
    y_list.append(y)
    yaw_list.append(yaw)
    # ! modified by zzs 2023/12/12

    if not check_car_collision(x_list, y_list, yaw_list, ox, oy, kd_tree):
        return None

    # 将连续空间中的位置和方向转换为离散的网格索引
    d = direction == 1
    x_ind = round(x / XY_GRID_RESOLUTION)
    y_ind = round(y / XY_GRID_RESOLUTION)
    yaw_ind = round(yaw / YAW_GRID_RESOLUTION)

    added_cost = 0.0

    if d != current.direction:
        added_cost += SB_COST

    # steer penalty
    added_cost += STEER_COST * abs(steer)

    # steer change penalty
    added_cost += STEER_CHANGE_COST * abs(current.steer - steer)

    # ! modified by zzs 2023/12/19
    # 使算法更倾向于选择偏航角变化较小的路径
    added_cost += YAW_COST * abs(yaw - goal_node.yaw_list[-1])

    cost = current.cost + added_cost + arc_l

    node = Node(x_ind, y_ind, yaw_ind, d, x_list,
                y_list, yaw_list, [d],
                parent_index=calc_index(current, config),
                cost=cost, steer=steer)

    return node

# 判断两个节点 n1 和 n2 是否处于相同的网格中
def is_same_grid(n1, n2):
    if n1.x_index == n2.x_index \
            and n1.y_index == n2.y_index \
            and n1.yaw_index == n2.yaw_index:
        return True
    return False

# 在当前节点和目标节点之间搜索可行路径
def analytic_expansion(current, goal, ox, oy, kd_tree):
    # 获取当前节点和目标节点的位置和朝向
    start_x = current.x_list[-1]
    start_y = current.y_list[-1]
    start_yaw = current.yaw_list[-1]

    goal_x = goal.x_list[-1]
    goal_y = goal.y_list[-1]
    goal_yaw = goal.yaw_list[-1]

    max_curvature = math.tan(MAX_STEER) / WB 
    paths = rs.calc_paths(start_x, start_y, start_yaw,
                          goal_x, goal_y, goal_yaw,
                          max_curvature, step_size=MOTION_RESOLUTION)

    if not paths:
        return None

    best_path, best = None, None

    for path in paths:
        if check_car_collision(path.x, path.y, path.yaw, ox, oy, kd_tree):
            cost = calc_rs_path_cost(path)
            if not best or best > cost:
                best = cost
                best_path = path

    return best_path


# 根据解析扩展得到的路径更新当前节点，并返回更新后的节点信息。
def update_node_with_analytic_expansion(current, goal,c, ox, oy, kd_tree):
    path = analytic_expansion(current, goal, ox, oy, kd_tree)

    if path:
        if show_animation:
            plt.plot(path.x, path.y)

        # 提取路径中除起始点外的所有点的 x 坐标、y 坐标和偏航角。
        f_x = path.x[1:]
        f_y = path.y[1:]
        f_yaw = path.yaw[1:]

        f_cost = current.cost + calc_rs_path_cost(path)
        f_parent_index = calc_index(current, c)

        fd = []
        for d in path.directions[1:]:
            fd.append(d >= 0)

        f_steer = 0.0
        f_path = Node(current.x_index, current.y_index, current.yaw_index,
                      current.direction, f_x, f_y, f_yaw, fd,
                      cost=f_cost, parent_index=f_parent_index, steer=f_steer)
        return True, f_path

    return False, None


def calc_rs_path_cost(reed_shepp_path):
    cost = 0.0
    for length in reed_shepp_path.lengths:
        if length >= 0:  # forward
            cost += length
        else:  # back
            cost += abs(length) * BACK_COST

    # switch back penalty
    for i in range(len(reed_shepp_path.lengths) - 1):
        # switch back
        if reed_shepp_path.lengths[i] * reed_shepp_path.lengths[i + 1] < 0.0:
            cost += SB_COST

    # steer penalty
    for course_type in reed_shepp_path.ctypes:
        if course_type != "S":  # curve
            cost += STEER_COST * abs(MAX_STEER)

    # ==steer change penalty
    # calc steer profile
    n_ctypes = len(reed_shepp_path.ctypes)
    u_list = [0.0] * n_ctypes
    for i in range(n_ctypes):
        if reed_shepp_path.ctypes[i] == "R":
            u_list[i] = - MAX_STEER
        elif reed_shepp_path.ctypes[i] == "L":
            u_list[i] = MAX_STEER

    for i in range(len(reed_shepp_path.ctypes) - 1):
        cost += STEER_CHANGE_COST * abs(u_list[i + 1] - u_list[i])

    return cost


def hybrid_a_star_planning(start, goal, ox, oy, xy_resolution, yaw_resolution):
    """
    start: start node
    goal: goal node
    ox: x position list of Obstacles [m]
    oy: y position list of Obstacles [m]
    xy_resolution: grid resolution [m]
    yaw_resolution: yaw angle resolution [rad]
    """


    start[2], goal[2] = rs.pi_2_pi(start[2]), rs.pi_2_pi(goal[2])

    tox, toy = ox[:], oy[:]
    hox, hoy = ox[:], oy[:]
    hox = [ihox / xy_resolution for ihox in hox]
    hoy = [ihoy / xy_resolution for ihoy in hoy]
    # T1 = time.perf_counter()
    obstacle_kd_tree = cKDTree(np.vstack((tox, toy)).T)
    obstacle_kd_tree_h_dp = cKDTree(np.vstack((hox, hoy)).T)
    # T2 = time.perf_counter()
    # print('程序运行时间:%s毫秒' % ((T2 - T1)*1000))

    config = Config(tox, toy, xy_resolution, yaw_resolution)

    start_node = Node(round(start[0] / xy_resolution),
                      round(start[1] / xy_resolution),
                      round(start[2] / yaw_resolution), True,
                      [start[0]], [start[1]], [start[2]], [True], cost=0)
    goal_node = Node(round(goal[0] / xy_resolution),
                     round(goal[1] / xy_resolution),
                     round(goal[2] / yaw_resolution), True,
                     [goal[0]], [goal[1]], [goal[2]], [True])
     
    openList, closedList = {}, {}

    h_dp = calc_distance_heuristic(
        goal_node.x_list[-1], goal_node.y_list[-1],
        ox, oy, xy_resolution, BUBBLE_R, obstacle_kd_tree_h_dp)

    pq = []
    openList[calc_index(start_node, config)] = start_node
    heapq.heappush(pq, (calc_cost(start_node, h_dp, config),
                        calc_index(start_node, config)))
    final_path = None
    # plt.pause(15)
    while True:
        if not openList:
            print("Error: Cannot find path, No open set")
            return [], [], []

        cost, c_id = heapq.heappop(pq)
        if c_id in openList:
            current = openList.pop(c_id)
            closedList[c_id] = current
        else:
            continue

        if show_animation:  # pragma: no cover            
            plt.plot(current.x_list[-1], current.y_list[-1], "xc") 
            plt.grid(False)
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            if len(closedList.keys()) % 10 == 0:
                plt.pause(0.3)

        is_updated, final_path = update_node_with_analytic_expansion(
            current, goal_node, config, ox, oy, obstacle_kd_tree)

        if is_updated:
            print("path found")
            break

        for neighbor in get_neighbors(current, goal_node, config, ox, oy,
                                      obstacle_kd_tree):
            neighbor_index = calc_index(neighbor, config)
            if neighbor_index in closedList:
                continue
            if neighbor_index not in openList \
                    or openList[neighbor_index].cost > neighbor.cost:
                heapq.heappush(
                    pq, (calc_cost(neighbor, h_dp, config),
                         neighbor_index))
                openList[neighbor_index] = neighbor

    path = get_final_path(closedList, final_path)

    return path


def calc_cost(n, h_dp, c):
    ind = (n.y_index - c.min_y) * c.x_w + (n.x_index - c.min_x)
    if ind not in h_dp:
        return n.cost + 999999999  # collision cost
    return n.cost + H_COST * h_dp[ind].cost


def get_final_path(closed, goal_node):
    reversed_x, reversed_y, reversed_yaw = \
        list(reversed(goal_node.x_list)), list(reversed(goal_node.y_list)), \
        list(reversed(goal_node.yaw_list))
    direction = list(reversed(goal_node.directions))
    nid = goal_node.parent_index
    final_cost = goal_node.cost

    while nid:
        n = closed[nid]
        reversed_x.extend(list(reversed(n.x_list)))
        reversed_y.extend(list(reversed(n.y_list)))
        reversed_yaw.extend(list(reversed(n.yaw_list)))
        direction.extend(list(reversed(n.directions)))

        nid = n.parent_index

    reversed_x = list(reversed(reversed_x))
    reversed_y = list(reversed(reversed_y))
    reversed_yaw = list(reversed(reversed_yaw))
    direction = list(reversed(direction))

    # adjust first direction
    direction[0] = direction[1]

    path = Path(reversed_x, reversed_y, reversed_yaw, direction, final_cost)

    return path


def verify_index(node, c):
    x_ind, y_ind = node.x_index, node.y_index
    if c.min_x <= x_ind <= c.max_x and c.min_y <= y_ind <= c.max_y:
        return True

    return False

# 计算给定节点在状态空间中的索引
def calc_index(node, c):
    ind = (node.yaw_index - c.min_yaw) * c.x_w * c.y_w + \
          (node.y_index - c.min_y) * c.x_w + (node.x_index - c.min_x)

    if ind <= 0:
        print("Error(calc_index):", ind)

    return ind

# 计算角速度
def calculate_angular_velocity(yaw_list, timestep):
    """通过差分yaw角直接计算角速度 ω = Δyaw / Δt"""
    omega_list = [0.0]
    
    for i in range(1, len(yaw_list)):
        dyaw = yaw_list[i] - yaw_list[i-1]
        dyaw = np.arctan2(np.sin(dyaw), np.cos(dyaw))  # 归一化
        omega = dyaw / timestep
        omega_list.append(omega)
    
    return omega_list


# 据给定的路径生成轨迹
def generate_trajectory(path):
    
    index, s = [], []
    pos = np.zeros(1)
    vel = np.zeros(1)
    num = len(path.x_list)

    # 1. 识别换向点 (Cusps)
    for i in range(1, len(path.direction_list)):
        if path.direction_list[i] != path.direction_list[i-1]:
            index.append(i-1)
    index.append(len(path.direction_list)-1)

    # 2. 生成每一段的速度规划 (S-V Profile)
    for i in range(0, len(index)):
        dir = np.power(-1, i) 
        if not path.direction_list[0]:
            dir = -dir 
            
        # 设定动力学限制
        target_v = 0.5  # 期望巡航速度
        target_a = 0.8  # 期望加速度

        if i > 0:
            # --- 换向段/中间段 ---
            dist_val = (index[i] - index[i-1]) * MOTION_RESOLUTION
            s.append(dir * dist_val)
            
            # 自适应速度限制：防止距离过短导致时间计算异常
            v_reachable = np.sqrt(abs(dist_val) * target_a)
            v_cal = min(target_v, v_reachable * 0.95)
            
            # vel_start=0, vel_end=0
            polynomial5_trajectory = velocity_planning(s[i], num, 0, 0, 0, 0, v_cal, target_a)
            
            pos = np.append(pos, polynomial5_trajectory[:,0]) 
            vel = np.append(vel, polynomial5_trajectory[:,1])
            vel[-1] = 0 
            
        else:
            # --- 起步段 ---
            dist_val = index[i] * XY_GRID_RESOLUTION * 1.5
            s.append(dir * dist_val)
            
            # 自适应速度限制
            v_reachable = np.sqrt(abs(dist_val) * target_a)
            v_cal = min(target_v, v_reachable * 0.95)

            # vel_start=0
            polynomial5_trajectory = velocity_planning(s[i], num, 0, 0, 0, 0, v_cal, target_a)
            
            pos = np.append(pos, polynomial5_trajectory[:,0])
            # pos = np.delete(pos, 0) 

            vel = np.append(vel, polynomial5_trajectory[:,1])
            vel[-1] = 0 
            # vel = np.delete(vel, 0) 

            
    vel = list(vel)
    pos = list(pos)

    # 3. 将 S-V 曲线重采样回路径点 (Resampling)
    for j in range(0, len(index)):
        dir = np.power(-1, j) # Simplified assumption, check logic if needed
        if not path.direction_list[0]:
            dir = -dir

        if j == 0:
            pos1 = pos[0:num-1]
            vel1 = vel[0:num-1]
            if dir == -1:
                pos1 = list(reversed(pos1))
                vel1 = list(reversed(vel1))
            vel_final = np.zeros(1)
            for k in range (1, index[j]+1): 
                dis = k * XY_GRID_RESOLUTION * 1.5 * dir
                index_pos = takeClosest(pos1, dis)
                if index_pos == 0: index_pos = 1
                denominator = pos1[index_pos] - pos1[index_pos-1]
                if abs(denominator) < 1e-6: denominator = 1e-6 # Avoid div by zero
                a1 = (dis - pos1[index_pos-1]) / denominator
                vel_final = np.append(vel_final, vel1[index_pos-1] + (vel1[index_pos] - vel1[index_pos-1]) * a1)
            vel_final[-1] = 0 
        elif j > 0 and j != len(index)-1:
            pos1 = pos[j*num:(j+1)*num-1]
            vel1 = vel[j*num:(j+1)*num-1]
            if dir == -1:
                pos1 = list(reversed(pos1))
                vel1 = list(reversed(vel1))
            for k in range(1, index[j]-index[j-1]+1):
                dis = k * MOTION_RESOLUTION * 1 * dir
                index_pos = takeClosest(pos1, dis)
                if index_pos == 0: index_pos = 1
                denominator = pos1[index_pos] - pos1[index_pos-1]
                if abs(denominator) < 1e-6: denominator = 1e-6
                a2 = (dis - pos1[index_pos-1]) / denominator
                vel_final = np.append(vel_final, vel1[index_pos-1] + (vel1[index_pos] - vel1[index_pos-1]) * a2)
            vel_final[-1] = 0 
        else:
            pos1 = pos[j*num:]
            vel1 = vel[j*num:]
            if dir == -1:
                pos1 = list(reversed(pos1))
                vel1 = list(reversed(vel1))
            for k in range(1, index[j] -index[j-1]+1):
                dis = k * MOTION_RESOLUTION * 1 * dir
                index_pos = takeClosest(pos1, dis)
                if index_pos == 0: index_pos = 1
                denominator = pos1[index_pos] - pos1[index_pos-1]
                if abs(denominator) < 1e-6: denominator = 1e-6
                a3 = (dis - pos1[index_pos-1]) / denominator
                vel_final = np.append(vel_final, vel1[index_pos-1] + (vel1[index_pos] - vel1[index_pos-1]) * a3)
            vel_final[-1] = 0 

    vel_final = list(vel_final)

    omega_list = calculate_angular_velocity(path.yaw_list, 0.1)
    
    trajactory = Trajactory(path.x_list, path.y_list, path.yaw_list, vel_final, omega_list)

    return trajactory

# 将轨迹写入文本文件中
def WriteMap(trajactory):
    file_name = "Parallel_parking"
    file_path = os.getcwd() + "/data/{}.txt".format(file_name)
    file = open(file_path, "w")
    for i in range(0, len(trajactory.yaw_list)):
        if trajactory.yaw_list[i] < 0:
            trajactory.yaw_list[i] = 2 * math.pi + trajactory.yaw_list[i]
        # trajactory.yaw_list[i] = np.rad2deg(trajactory.yaw_list[i])
        file.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(trajactory.x_list[i], trajactory.y_list[i], trajactory.yaw_list[i], 0, 0, trajactory.v_list[i])) #  x 坐标、y 坐标、偏航角、以及速度信息
    pass

# 绘制车位边线
def plot_parking_spaces():
    # 车库长度30cm，宽度18cm
    # a —— —— —— —— —— b 
    # |                |
    # |                |
    # c —— —— —— —— —— d

    # 左列车位
    goal1 = [167, 210.1, np.deg2rad(0)]

    a = [goal1[0]-15, goal1[1]+9]
    b = [goal1[0]+15, goal1[1]+9]
    c = [goal1[0]-15, goal1[1]-9]
    d = [goal1[0]+15, goal1[1]-9]

    plt.plot([a[0], b[0]], [a[1], b[1]], 'g')  # 上边
    plt.plot([b[0], d[0]], [b[1], d[1]], 'g')  # 右边
    plt.plot([d[0], c[0]], [d[1], c[1]], 'g')  # 下边
    plt.plot([c[0], a[0]], [c[1], a[1]], 'g')  # 左边

    plt.plot([a[0], b[0]], [a[1]-21, b[1]-21], 'g')  # 上边
    plt.plot([b[0], d[0]], [b[1]-21, d[1]-21], 'g')  # 右边
    plt.plot([d[0], c[0]], [d[1]-21, c[1]-21], 'g')  # 下边
    plt.plot([c[0], a[0]], [c[1]-21, a[1]-21], 'g')  # 左边

    plt.plot([a[0], b[0]], [a[1]-42, b[1]-42], 'g')  # 上边
    plt.plot([b[0], d[0]], [b[1]-42, d[1]-42], 'g')  # 右边
    plt.plot([d[0], c[0]], [d[1]-42, c[1]-42], 'g')  # 下边
    plt.plot([c[0], a[0]], [c[1]-42, a[1]-42], 'g')  # 左边

    # 右列车位
    goal2 = [247.1, 210.1, np.deg2rad(0)]

    a = [goal2[0]-15, goal2[1]+9]
    b = [goal2[0]+15, goal2[1]+9]
    c = [goal2[0]-15, goal2[1]-9]
    d = [goal2[0]+15, goal2[1]-9]

    plt.plot([a[0], b[0]], [a[1], b[1]], 'g')  # 上边
    plt.plot([b[0], d[0]], [b[1], d[1]], 'g')  # 右边
    plt.plot([d[0], c[0]], [d[1], c[1]], 'g')  # 下边
    plt.plot([c[0], a[0]], [c[1], a[1]], 'g')  # 左边

    plt.plot([a[0], b[0]], [a[1]-21, b[1]-21], 'g')  # 上边
    plt.plot([b[0], d[0]], [b[1]-21, d[1]-21], 'g')  # 右边
    plt.plot([d[0], c[0]], [d[1]-21, c[1]-21], 'g')  # 下边
    plt.plot([c[0], a[0]], [c[1]-21, a[1]-21], 'g')  # 左边

    plt.plot([a[0], b[0]], [a[1]-42, b[1]-42], 'g')  # 上边
    plt.plot([b[0], d[0]], [b[1]-42, d[1]-42], 'g')  # 右边
    plt.plot([d[0], c[0]], [d[1]-42, c[1]-42], 'g')  # 下边
    plt.plot([c[0], a[0]], [c[1]-42, a[1]-42], 'g')  # 左边

def main():
    print("Start Hybrid A* planning")

    # 设定起点和终点
    start = [207.05, 167, np.deg2rad(90)]
    goal  = [167, 210.1-21, np.deg2rad(0)]
    # 终点 备选位置
    # 左1车位：[167, 210.1, np.deg2rad(0)]
    # 左2车位：[167, 210.1-21, np.deg2rad(0)]
    # 左3车位：[167, 210.1-42, np.deg2rad(0)]
    # 右1车位：[247.1, 210.1, np.deg2rad(180)]
    # 右2车位：[247.1, 210.1-21, np.deg2rad(180)]
    # 右3车位：[247.1, 210.1-42, np.deg2rad(180)]

    ox, oy = [], []

    if show_animation: 
        # 绘制停车位
        plot_parking_spaces()
    
    # 绘制停车场方位图
    # 入口
    for i in range(int((244.5 - 222.6)/XY_GRID_RESOLUTION)):
        ox.append(192.1)
        oy.append(222.6 + XY_GRID_RESOLUTION*i)
    for i in range(int((244.5 - 222.6)/XY_GRID_RESOLUTION)):
        ox.append(221.5)
        oy.append(222.6 + XY_GRID_RESOLUTION*i)    
    #主要区域
    for i in range(int((195.1 - 146.4)/XY_GRID_RESOLUTION)):
        ox.append(146.4 + XY_GRID_RESOLUTION*i)
        oy.append(222.6)
    for i in range(int((267.4 - 221.5)/XY_GRID_RESOLUTION)):
        ox.append(267.4 - XY_GRID_RESOLUTION*i)
        oy.append(222.6)

    for i in range(int((222.6 - 152.4)/XY_GRID_RESOLUTION)):
        ox.append(146.4)
        oy.append(152.4 + XY_GRID_RESOLUTION*i)
    for i in range(int((222.6 - 152.4)/XY_GRID_RESOLUTION)):
        ox.append(267.4)
        oy.append(152.4 + XY_GRID_RESOLUTION*i)

    for i in range(int((195.1 - 146.4)/XY_GRID_RESOLUTION)):
        ox.append(146.4 + XY_GRID_RESOLUTION*i)
        oy.append(152.4)
    for i in range(int((267.4 - 221.5)/XY_GRID_RESOLUTION)):
        ox.append(221.5 + XY_GRID_RESOLUTION*i)
        oy.append(152.4)

    # 出口
    for i in range(int((244.5 - 222.6)/XY_GRID_RESOLUTION)):
        ox.append(192.1)
        oy.append(152.4 - XY_GRID_RESOLUTION*i)
    for i in range(int((244.5 - 222.6)/XY_GRID_RESOLUTION)):
        ox.append(221.5)
        oy.append(152.4 - XY_GRID_RESOLUTION*i)  

    print("start : ", start)
    print("goal : ", goal)

    if show_animation:
        plt.plot(ox, oy, ".k")
        rs.plot_arrow(start[0], start[1], start[2], fc='g')
        rs.plot_arrow(goal[0], goal[1], goal[2])

        plt.grid(False)
        plt.axis("equal")
    # plt.show()
    
    # T1 = time.perf_counter()
    path = hybrid_a_star_planning(
        start, goal, ox, oy, XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION)
    # T2 = time.perf_counter()
    # print('程序运行时间:%s毫秒' % ((T2 - T1)*1000))
    x = path.x_list
    y = path.y_list
    yaw = path.yaw_list
    if show_animation:
        i = 10
        times = 0
        # plt.cla()

        for i_x, i_y, i_yaw in zip(x, y, yaw):
            if times == 0:
                plt.plot(ox, oy, ".k")
                plt.plot(x, y, "-r", label="Hybrid A* path")
                plt.grid(False)
                plt.axis("equal")
                # plot_car(i_x, i_y, i_yaw)
            times = times + 1
            if i != 0:
                # plt.cla()
                i = i - 1
            else:
                plt.plot(ox, oy, ".k")
                plt.plot(x, y, "-r", label="Hybrid A* path")
                plt.grid(False)
                plt.axis("equal")
                # plot_car(i_x, i_y, i_yaw)
                # plt.pause(0.0001)
                i = 10
            if times == len(x):
                plt.plot(ox, oy, ".k")
                plt.plot(x, y, "-r", label="Hybrid A* path")
                plt.grid(False)
                plt.axis("equal")
                # plot_car(i_x, i_y, i_yaw)

    path_opti = path_optimization(path, ox, oy)
    x_opti = path_opti.x_list
    y_opti = path_opti.y_list
    yaw_opti = path_opti.yaw_list
    if show_animation:
        plt.pause(3)
        i = 5
        times = 0
        # plt.cla()
        for i_x, i_y, i_yaw in zip(x_opti, y_opti, yaw_opti):
            if times == 0:
                plt.plot(ox, oy, ".k")
                plt.plot(x_opti, y_opti, "-y", label="Hybrid A* path")
                plt.grid(False)
                plt.axis("equal")
                # plot_car(i_x, i_y, i_yaw)
            times = times + 1
            if i != 0:
                # plt.cla()
                i = i - 1
            else:
                plt.plot(ox, oy, ".k")
                plt.plot(x_opti, y_opti, "-y", label="Hybrid A* path")
                plt.grid(False)
                plt.axis("equal")
                # plot_car(i_x, i_y, i_yaw)
                plt.pause(0.0001)
                i = 5
            if times == len(x):
                plt.plot(ox, oy, ".k")
                plt.plot(x_opti, y_opti, "-y", label="final path")
                plt.grid(False)
                plt.axis("equal")
                # plot_car(i_x, i_y, i_yaw)
                
    trajectory = generate_trajectory(path)
    
    if show_animation: 
        plt.pause(3)
        for i_x, i_y, i_yaw in zip(x_opti, y_opti, yaw_opti):
            plt.cla()
            plot_parking_spaces()
            plt.plot(ox, oy, ".k")
            plt.plot(x_opti, y_opti, "-y", label="final path")
            plt.grid(True)
            plt.axis("equal")
            plot_car(i_x, i_y, i_yaw)
            plt.grid(False)
            plt.pause(0.01)

        plt.show()
    print(__file__ + " done!!")
    # WriteMap(trajectory)
    # print(yaw)
    # print(yaw_opti)
    print("finished")

if __name__ == '__main__':
    main()