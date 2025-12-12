"""

Car model for Hybrid A* path planning

author: Zheng Zh (@Zhengzh)

"""
import time
import sys
import pathlib
root_dir = pathlib.Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

from math import cos, sin, tan, pi

import matplotlib.pyplot as plt
import numpy as np

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.angle import rot_mat_2d

# 车长4.955，宽2.857
WB = 3.368  # rear to front wheel轴距
W = 2.857  # width of car
# LF = 3.2775  # distance from rear to vehicle front end 4.34594
# LB = 1.6775  # distance from rear to vehicle back end 0.60906 

LF = 4.34594  # distance from rear to vehicle front end 4.34594
LB = 0.60906   # distance from rear to vehicle 

MAX_STEER = np.deg2rad(45) # [rad] maximum steering angle

BUBBLE_DIST = (LF - LB) / 2.0  # distance from rear to center of vehicle.
BUBBLE_R = np.hypot((LF + LB) / 2.0, W / 2.0)  # bubble radius

# vehicle rectangle vertices
VRX = [LF, LF, -LB, -LB, LF]
VRY = [W / 2, -W / 2, -W / 2, W / 2, W / 2]

# 检查车辆路径上是否与障碍物发生碰撞
def check_car_collision(x_list, y_list, yaw_list, ox, oy, kd_tree):
    # T1 = time.perf_counter()

    for i_x, i_y, i_yaw in zip(x_list, y_list, yaw_list):
        cx = i_x + BUBBLE_DIST * cos(i_yaw)
        cy = i_y + BUBBLE_DIST * sin(i_yaw)
        # 使用 kd_tree 数据结构来查找距离点 (cx, cy) 一定距离范围内的所有障碍物的索引
        ids = kd_tree.query_ball_point([cx, cy], BUBBLE_R+0.3)

        if not ids:
            continue

        # original version
        if not rectangle_check(i_x, i_y, i_yaw,
                               [ox[i] for i in ids], [oy[i] for i in ids]):
            return False  # no collision

    # T2 = time.perf_counter()
    # print('程序运行时间:%s毫秒' % ((T2 - T1)*1000))
    return True  # collision

def rectangle_check(x, y, yaw, ox, oy):
    # transform obstacles to base link frame
    rot = rot_mat_2d(yaw)
    for iox, ioy in zip(ox, oy):
        tx = iox - x
        ty = ioy - y
        converted_xy = np.stack([tx, ty]).T @ rot
        rx, ry = converted_xy[0], converted_xy[1]
         
        #oringal version
        # if not (rx > LF or rx < -LB or ry > W / 2.0 or ry < -W / 2.0):
        if not (rx > LF+0.3 or rx < -LB-0.3 or ry > (W / 2.0)+0.3 or ry < (-W / 2.0)-0.3):
            return False  # no collision

    return True  # collision


def plot_arrow(x, y, yaw, length=6, width=2, fc="r", ec="k"):
    """Plot arrow."""
    if not isinstance(x, float):
        for (i_x, i_y, i_yaw) in zip(x, y, yaw):
            plot_arrow(i_x, i_y, i_yaw)
    else:
        plt.arrow(x, y, length * cos(yaw), length * sin(yaw),
                  fc=fc, ec=ec, head_width=width, head_length=width, alpha=0.4)
        
def bubble_center(x, y, yaw, use="center"):
    """
    use = "center": 以几何中心为圆心（当前粗筛同样用几何中心：BUBBLE_DIST 偏移）
    use = "rear":   以后轴为圆心
    """
    if use == "rear":
        return x, y
    # use == "center"
    return x + BUBBLE_DIST * cos(yaw), y + BUBBLE_DIST * sin(yaw)

def plot_bubble(x, y, yaw, use="center", lw=1.5, alpha=0.25):
    cx, cy = bubble_center(x, y, yaw, use=use)
    ax = plt.gca()
    circ = plt.Circle((cx, cy), BUBBLE_R, fill=True, alpha=alpha)
    ax.add_patch(circ)
    # 画圆边框更清楚一些
    circ2 = plt.Circle((cx, cy), BUBBLE_R, fill=False, linewidth=lw)
    ax.add_patch(circ2)
    # 画个小点标示圆心
    ax.plot(cx, cy, 'ko', markersize=3)

def plot_car(x, y, yaw):
    car_color = 'b'
    c, s = cos(yaw), sin(yaw)
    rot = rot_mat_2d(-yaw)
    car_outline_x, car_outline_y = [], []
    for rx, ry in zip(VRX, VRY):
        converted_xy = np.stack([rx, ry]).T @ rot
        car_outline_x.append(converted_xy[0]+x)
        car_outline_y.append(converted_xy[1]+y)

    plot_arrow(x, y, yaw)

    plt.plot(car_outline_x, car_outline_y, car_color)
    
    # 碰撞气泡
    plot_bubble(x, y, yaw, use="center")

# 将给定的角度限制在 [-π, π] 的范围内，并保持角度的周期性
def pi_2_pi(angle):
    return (angle + pi) % (2 * pi) - pi


def move(x, y, yaw, distance, steer, L=WB):
    x += distance * cos(yaw)
    y += distance * sin(yaw)
    yaw += pi_2_pi(distance * tan(steer) / L)  # distance/2
    return x, y, yaw

def sign(a):
    if a == 0:
        return 0
    elif a > 0:
        return 1
    elif a < 0:
        return -1
     
def main():
    x, y, yaw = 0., 0., 1.
    plt.axis('equal')
    plot_car(x, y, yaw)
    plt.show()


if __name__ == '__main__':
    main()