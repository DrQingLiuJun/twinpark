#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Velocity-Based MPPI Controller
MPPI直接输出期望速度和转向角，由Ackermann控制器执行底层控制
修复版本：解决速度波动和记录问题
"""

import torch
import sys
import rospy
import csv
import os
import math
import numpy as np
from datetime import datetime
from typing import Tuple
from scipy.signal import savgol_filter
import time

from vehicle_msgs.msg import VehicleState, Trajectory, AckermannCmd
from std_msgs.msg import Float32MultiArray

# Add ROS path if needed
ros_path = '/opt/ros/melodic/lib/python2.7/dist-packages'
if ros_path not in sys.path:
    sys.path.append(ros_path)


class VelocityMPPIController:
    """
    MPPI控制器 - 输出期望速度和转向角
    控制量: [steer, velocity] 而非 [steer, acceleration]
    """
    def __init__(
        self,
        delta_t: float = 0.05,
        wheel_base: float = 3.368,
        max_steer_abs: float = 0.785,
        max_speed_abs: float = 2.0,
        horizon_step_T: int = 30,       
        number_of_samples_K: int = 500,
        param_exploration: float = 0.05,
        param_lambda: float = 50.0,
        param_alpha: float = 0.2,       
        sigma: np.ndarray = np.array([[0.1, 0.0], [0.0, 0.25]]),
        stage_cost_weight: np.ndarray = np.array([15.0, 40.0, 40.0, 80.0]), 
        terminal_cost_weight: np.ndarray = np.array([30.0, 80.0, 80.0, 100.0]),
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rospy.loginfo(f"Velocity MPPI running on: {self.device}")

        # Parameters
        self.dim_x = 4  # [x, y, yaw, v]
        self.dim_u = 2  # [steer, velocity]
        self.T = horizon_step_T
        self.K = number_of_samples_K
        self.param_lambda = param_lambda
        self.param_alpha = param_alpha
        # 大幅降低gamma，减少对前馈的依赖
        self.param_gamma = self.param_lambda * (1.0 - self.param_alpha) * 0.1  # 降低到原来的10%

        # Weights
        self.stage_cost_weight = torch.tensor(stage_cost_weight, dtype=torch.float32, device=self.device)
        self.terminal_cost_weight = torch.tensor(terminal_cost_weight, dtype=torch.float32, device=self.device)
        self.Sigma = torch.tensor(sigma, dtype=torch.float32, device=self.device)
        self.Sigma_inv = torch.linalg.inv(self.Sigma)

        self.max_steer_abs = max_steer_abs
        self.max_speed_abs = max_speed_abs
        self.delta_t = delta_t
        self.wheel_base = wheel_base

        # Trajectory & Tracking
        self.ref_path = None          
        self.ref_times = None  
        self.current_ref_point_debug = None
        
        # Output Smoothing
        self.prev_steer_cmd = 0.0
        self.prev_vel_cmd = 0.0
        self.max_dsteer = 0.15
        self.max_dvel = 0.20
        
        self.u_prev = torch.zeros((self.T, self.dim_u), dtype=torch.float32, device=self.device)

    def set_reference_path(self, trajectory: Trajectory):
        num_points = len(trajectory.x)
        if num_points < 5: return

        path = np.zeros((num_points, 7))
        path[:, 0] = trajectory.x
        path[:, 1] = trajectory.y
        path[:, 2] = trajectory.yaw
        path[:, 3] = trajectory.vx
        
        # Calculate Times
        dists = np.sqrt(np.diff(path[:,0])**2 + np.diff(path[:,1])**2)
        vels = np.abs(path[:,3])
        v_avg = np.maximum((vels[:-1] + vels[1:]) / 2.0, 0.2)
        dt_list = dists / v_avg
        self.ref_times = np.zeros(num_points)
        self.ref_times[1:] = np.cumsum(dt_list)
        self.total_duration = self.ref_times[-1]
        self.total_idx_debug = num_points

        # Smooth velocity
        try:
            path[:, 3] = savgol_filter(path[:, 3], window_length=9, polyorder=2)
        except: pass 

        # Calculate Steer
        dt_diff = np.maximum(np.diff(self.ref_times), 0.02)
        yaw_unwrapped = np.unwrap(path[:, 2])
        dyaw = np.diff(yaw_unwrapped)
        path[1:, 4] = dyaw / dt_diff 
        path[0, 4] = path[1, 4]

        v_safe = path[:, 3].copy()
        mask_zero = np.abs(v_safe) < 0.1
        v_safe[mask_zero] = 0.1 * np.sign(v_safe[mask_zero] + 1e-6)
        
        tan_steer = self.wheel_base * path[:, 4] / v_safe
        path[:, 5] = np.arctan(tan_steer)
        path[:, 5] = np.clip(path[:, 5], -self.max_steer_abs, self.max_steer_abs)
        path[:, 6] = path[:, 3] 

        self.ref_path = path
        self.current_ref_time = 0.0 
        self.goal_reached = False
        rospy.loginfo(f"Ref Path Updated. Duration: {self.total_duration:.2f}s")

    def _sync_time(self, observed_x):
        if self.ref_path is None: return 0.0
        if self.goal_reached: return self.total_duration

        # 1. Standard Sync
        ref_state, idx_debug = self._interpolate_state(self.current_ref_time)
        self.current_idx_debug = idx_debug
        ref_x, ref_y, ref_yaw, ref_v = ref_state[0], ref_state[1], ref_state[2], ref_state[3]

        dx = observed_x[0] - ref_x
        dy = observed_x[1] - ref_y
        
        # Longitudinal Error: (Car - Ref) projected on path direction
        # Positive = Car is Ahead. Negative = Car is Behind.
        lon_error = dx * math.cos(ref_yaw) + dy * math.sin(ref_yaw)
        
        # 根本修复：基于车辆实际速度调整参考时间推进
        # 如果车速度慢，参考点也慢；车速度快，参考点也快
        vehicle_v = observed_x[3]
        
        # 速度比率：实际速度/参考速度
        if abs(ref_v) > 0.05:
            v_ratio = abs(vehicle_v) / abs(ref_v)
            v_ratio = max(0.3, min(1.5, v_ratio))  # 限制在合理范围
        else:
            v_ratio = 0.5  # 参考速度接近0时，慢速推进
        
        # 位置误差修正（小增益）
        k_sync = 0.15  # 很小的增益，只做微调
        pos_correction = 1.0 + k_sync * lon_error
        pos_correction = max(0.8, min(1.2, pos_correction))
        
        # 综合：速度匹配为主，位置修正为辅
        time_rate = v_ratio * pos_correction
        time_rate = max(0.2, min(1.3, time_rate))
        
        self.state_debug = "RUN"

        # 2. CUSP DETECTION LOGIC
        current_sign = np.sign(ref_v) if abs(ref_v) > 0.05 else 0
        found_cusp = False
        target_sign = 0
        
        # Search ahead for velocity flip
        check_steps = 30 
        for k in range(1, check_steps):
            t_check = self.current_ref_time + k * 0.1
            if t_check > self.total_duration: break
            
            s_check, _ = self._interpolate_state(t_check)
            v_check = s_check[3]
            sign_check = np.sign(v_check) if abs(v_check) > 0.05 else 0
            
            if sign_check != 0 and sign_check != current_sign:
                if current_sign != 0: 
                    found_cusp = True
                    target_sign = sign_check
                    break

        if found_cusp:
            # Calculate Time-To-Stop (Approx)
            time_to_stop = 100.0
            for k in range(1, 30):
                t_chk = self.current_ref_time + k * 0.1
                s_chk, _ = self._interpolate_state(t_chk)
                if abs(s_chk[3]) < 0.05:
                     time_to_stop = k * 0.1
                     break
            
            # STATE: LOCK (Approaching Stop < 0.5s) - 只在非常接近停止点时才锁定
            if time_to_stop < 0.5:
                self.state_debug = "LOCK"
                time_rate = 0.0 # Freeze reference
                
                vehicle_v = observed_x[3]
                dist_to_ref = math.sqrt(dx*dx + dy*dy)
                
                # --- JUMP CONDITIONS (优化版) ---
                
                # Condition 1: Perfect Stop (理想停止)
                cond_stop = abs(vehicle_v) < 0.08 and dist_to_ref < 0.4
                
                # Condition 2: Overshoot (超调修正) - 放宽条件
                cond_overshoot = lon_error > 0.15 and abs(vehicle_v) < 0.5
                
                # Condition 3: Timeout (超时保护) - 0.8秒
                if not hasattr(self, 'lock_start_time'):
                    self.lock_start_time = rospy.Time.now()
                lock_duration = (rospy.Time.now() - self.lock_start_time).to_sec()
                cond_timeout = lock_duration > 0.8
                
                if cond_stop or cond_overshoot or cond_timeout:
                    self.state_debug = "JUMP"
                    rospy.logwarn_throttle(1.0, f"CUSP JUMP! (Stop:{cond_stop}, Over:{cond_overshoot}, Timeout:{cond_timeout})")
                    
                    # 重置锁定计时器
                    self.lock_start_time = rospy.Time.now()
                    
                    # 向前搜索新档位起始点
                    jump_t = self.current_ref_time
                    while jump_t < self.total_duration:
                        jump_t += 0.1
                        s, _ = self._interpolate_state(jump_t)
                        # 找到速度符号匹配且速度足够大的点
                        if (np.sign(s[3]) == target_sign) and (abs(s[3]) > 0.1):
                            break
                    
                    self.current_ref_time = jump_t
                    return self.current_ref_time
            else:
                # 不在LOCK状态时，重置计时器
                self.lock_start_time = rospy.Time.now()

        # Normal integration
        dt_increment = time_rate * self.delta_t
        
        # 检查是否接近轨迹终点
        time_to_end = self.total_duration - self.current_ref_time
        
        if time_to_end > 1.0:
            # 距离终点还远，保证最小前进速率
            min_dt = 0.01  # 最小时间增量
            dt_increment = max(dt_increment, min_dt)
        elif time_to_end > 0.1:
            # 接近终点，减速推进
            dt_increment = min(dt_increment, time_to_end * 0.5)
        else:
            # 非常接近终点，停留在终点
            # 此时_get_ref_horizon会根据车辆位置给予驱动速度
            dt_increment = 0.0
        
        self.current_ref_time += dt_increment
        self.current_ref_time = min(self.current_ref_time, self.total_duration)
        
        return self.current_ref_time

    def _interpolate_state(self, t):
        t = np.clip(t, 0, self.total_duration)
        idx = np.searchsorted(self.ref_times, t)
        idx = np.clip(idx, 1, len(self.ref_times)-1)
        
        t0 = self.ref_times[idx-1]
        t1 = self.ref_times[idx]
        ratio = (t - t0) / (t1 - t0 + 1e-6)
        p0 = self.ref_path[idx-1]
        p1 = self.ref_path[idx]
        
        out = p0 + ratio * (p1 - p0)
        yaw0 = p0[2]
        yaw1 = p1[2]
        diff = yaw1 - yaw0
        while diff > np.pi: diff -= 2*np.pi
        while diff < -np.pi: diff += 2*np.pi
        out[2] = yaw0 + ratio * diff
        return out, idx

    def _get_ref_horizon(self, start_time, observed_x=None):
        ref_interp = np.zeros((self.T, 7))
        for i in range(self.T):
            t = start_time + i * self.delta_t
            
            # Force Horizon Freeze during LOCK
            if self.state_debug == "LOCK":
                t = start_time
            
            state, _ = self._interpolate_state(t)
            ref_interp[i, :] = state
            
            # Force Zero Velocity during LOCK
            if self.state_debug == "LOCK":
                ref_interp[i, 3] = 0.0
                ref_interp[i, 6] = 0.0
            
            # 关键修复：到达轨迹终点后，根据距离给予持续驱动速度
            if t >= self.total_duration:
                # 获取目标位置（轨迹终点）
                goal_state = self.ref_path[-1]
                goal_x, goal_y, goal_yaw = goal_state[0], goal_state[1], goal_state[2]
                
                if observed_x is not None:
                    # 计算车辆到目标的距离
                    dx = goal_x - observed_x[0]
                    dy = goal_y - observed_x[1]
                    dist_to_goal = np.sqrt(dx*dx + dy*dy)
                    
                    # 计算纵向距离（沿目标朝向）
                    lon_dist = dx * np.cos(goal_yaw) + dy * np.sin(goal_yaw)
                    
                    # 根据距离设置驱动速度 - 更激进
                    if dist_to_goal > 0.08:  # 距离大于8cm，继续驱动
                        # 速度与距离成正比，更大的增益
                        target_speed = np.clip(lon_dist * 1.5, -0.8, 0.8)
                        
                        # 确保最小驱动速度
                        if abs(target_speed) < 0.15:
                            target_speed = 0.15 if lon_dist > 0 else -0.15
                        
                        ref_interp[i, 3] = target_speed
                        ref_interp[i, 6] = target_speed
                    else:
                        # 距离很近，停止
                        ref_interp[i, 3] = 0.0
                        ref_interp[i, 6] = 0.0
                else:
                    # 没有观测状态，默认停止
                    ref_interp[i, 3] = 0.0
                    ref_interp[i, 6] = 0.0
                
        return torch.tensor(ref_interp, dtype=torch.float32, device=self.device)

    def calc_control_input(self, observed_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
        if self.ref_path is None:
            return np.zeros(2), np.zeros((self.T, 2)), 0.0, 0.0

        # Sync Time State
        ref_time = self._sync_time(observed_x)
        
        # Check Goal - 更严格的停止条件
        ref_end = self.ref_path[-1]
        dist_to_goal = np.linalg.norm(self.ref_path[-1, :2] - observed_x[:2])
        
        # 只有当距离很近且车辆已停稳时才认为到达
        if dist_to_goal < 0.12 and abs(observed_x[3]) < 0.05:
            self.goal_reached = True
            return np.array([0.0, 0.0]), np.zeros((self.T, 2)), 0.0, 0.0
        
        ref_data = self._get_ref_horizon(ref_time, observed_x)
        ref_states = ref_data[:, :5]  # [x, y, yaw, v, omega]
        
        # u_guide: [steer, velocity]
        u_guide = torch.zeros((self.T, 2), dtype=torch.float32, device=self.device)
        u_guide[:, 0] = ref_data[:, 5]  # steer
        u_guide[:, 1] = ref_data[:, 6]  # velocity (参考速度)
        
        self.current_ref_point_debug = ref_states[0].cpu().numpy()

        x0 = torch.tensor(observed_x, dtype=torch.float32, device=self.device)
        epsilon = self._calc_epsilon(self.Sigma, self.K, self.T, self.dim_u)
        
        u_mean = u_guide.unsqueeze(0).repeat(self.K, 1, 1)
        v = u_mean + epsilon
        v[:, :, 0] = torch.clamp(v[:, :, 0], -self.max_steer_abs, self.max_steer_abs)
        v[:, :, 1] = torch.clamp(v[:, :, 1], -self.max_speed_abs, self.max_speed_abs)

        S = torch.zeros(self.K, dtype=torch.float32, device=self.device)
        x = x0.unsqueeze(0).repeat(self.K, 1)

        for t in range(self.T):
            x = self._F(x, v[:, t, :])
            diff = v[:, t, :] - u_guide[t].unsqueeze(0)
            temp = diff @ self.Sigma_inv
            control_cost = self.param_gamma * (temp * diff).sum(dim=1)
            state_cost = self._c_tracking(x, v[:, t, :], ref_states[t])
            S += state_cost + control_cost

        S += self._phi_tracking(x, ref_states[-1])

        rho = torch.min(S)
        eta = torch.sum(torch.exp((-1.0 / self.param_lambda) * (S - rho)))
        w = (1.0 / eta) * torch.exp((-1.0 / self.param_lambda) * (S - rho))
        
        w_expanded = w.view(self.K, 1, 1)
        w_epsilon = torch.sum(w_expanded * epsilon, dim=0)
        u_optimal_seq = u_guide + w_epsilon
        
        raw_steer = u_optimal_seq[0, 0].item()
        raw_vel = u_optimal_seq[0, 1].item()
        
        # 增强平滑输出
        d_steer = np.clip(raw_steer - self.prev_steer_cmd, -self.max_dsteer, self.max_dsteer)
        final_steer = self.prev_steer_cmd + d_steer
        
        d_vel = np.clip(raw_vel - self.prev_vel_cmd, -self.max_dvel, self.max_dvel)
        final_vel = self.prev_vel_cmd + d_vel
        
        self.prev_steer_cmd = final_steer
        self.prev_vel_cmd = final_vel
        
        optimal_input = np.array([final_steer, final_vel])
        
        return optimal_input, u_optimal_seq.cpu().numpy(), rho.item(), S.mean().item()

    def _c_tracking(self, x, u, ref_target):
        """阶段代价函数"""
        ref_x, ref_y, ref_yaw, ref_v = ref_target[0], ref_target[1], ref_target[2], ref_target[3]
        E_Vx = ref_x - x[:, 0]
        E_Vy = ref_y - x[:, 1]
        vir_yaw = x[:, 2]
        
        e_Vy = -torch.sin(vir_yaw) * E_Vx + torch.cos(vir_yaw) * E_Vy 
        e_Vx = torch.cos(vir_yaw) * E_Vx + torch.sin(vir_yaw) * E_Vy  
        e_Vyaw = ref_yaw - x[:, 2] 
        e_Vyaw = torch.atan2(torch.sin(e_Vyaw), torch.cos(e_Vyaw))
        e_Vv = ref_v - x[:, 3]

        # ASYMMETRIC COST: 倒车时加重横向惩罚
        is_reverse = (ref_v < -0.05)
        lat_weight = torch.where(is_reverse, self.stage_cost_weight[1] * 2.5, self.stage_cost_weight[1])
        yaw_weight = self.stage_cost_weight[2]
        
        cost = self.stage_cost_weight[0] * e_Vx**2 + \
               lat_weight * e_Vy**2 + \
               yaw_weight * e_Vyaw**2 + \
               self.stage_cost_weight[3] * e_Vv**2
        return cost

    def _phi_tracking(self, x, ref_target):
        """终端代价函数"""
        ref_x, ref_y, ref_yaw, ref_v = ref_target[0], ref_target[1], ref_target[2], ref_target[3]
        E_Vx = ref_x - x[:, 0]
        E_Vy = ref_y - x[:, 1]
        vir_yaw = x[:, 2]
        
        e_Vy = -torch.sin(vir_yaw) * E_Vx + torch.cos(vir_yaw) * E_Vy
        e_Vx = torch.cos(vir_yaw) * E_Vx + torch.sin(vir_yaw) * E_Vy
        e_Vyaw = torch.atan2(torch.sin(ref_yaw - x[:, 2]), torch.cos(ref_yaw - x[:, 2]))
        e_Vv = ref_v - x[:, 3]
        
        # 终端代价
        is_reverse = (ref_v < -0.05)
        lat_weight = torch.where(is_reverse, self.terminal_cost_weight[1] * 2.5, self.terminal_cost_weight[1])
        yaw_weight = self.terminal_cost_weight[2]
        
        cost = self.terminal_cost_weight[0] * e_Vx**2 + \
               lat_weight * e_Vy**2 + \
               yaw_weight * e_Vyaw**2 + \
               self.terminal_cost_weight[3] * e_Vv**2
        return cost


    def _F(self, x_t, v_t):
        """
        状态转移函数 - 基于速度控制，增加平滑性
        控制量: [steer, velocity]
        """
        x, y, yaw, v = x_t[:, 0], x_t[:, 1], x_t[:, 2], x_t[:, 3]
        u_steer, u_vel = v_t[:, 0], v_t[:, 1]
        
        # 速度响应模型
        tau = 0.3
        alpha = self.delta_t / (tau + self.delta_t)
        new_v = v + alpha * (u_vel - v)
        
        new_x = x + v * torch.cos(yaw) * self.delta_t
        new_y = y + v * torch.sin(yaw) * self.delta_t
        new_yaw = yaw + v / self.wheel_base * torch.tan(u_steer) * self.delta_t
        return torch.stack([new_x, new_y, new_yaw, new_v], dim=1)

    def _calc_epsilon(self, sigma, size_sample, size_time_step, size_dim_u):
        mean = torch.zeros(size_dim_u, device=self.device)
        dist = torch.distributions.MultivariateNormal(mean, covariance_matrix=sigma)
        return dist.sample((size_sample, size_time_step))


class VelocityMPPINode:
    """
    MPPI控制节点 - 发布期望速度和转向角到 /ackermann_cmd 话题
    """
    def __init__(self):
        rospy.init_node('velocity_mppi_node', anonymous=False)
        self.load_parameters()
        
        self.mppi = VelocityMPPIController(
            delta_t=self.delta_t,
            wheel_base=self.wheelbase,
            max_steer_abs=math.radians(self.max_steer_deg),
            max_speed_abs=self.max_speed,
            horizon_step_T=self.horizon_T,
            number_of_samples_K=self.num_samples_K,
            param_exploration=self.exploration,
            param_lambda=self.lambda_param,
            param_alpha=self.alpha_param,
            sigma=np.array([[self.sigma_steer, 0.0], [0.0, self.sigma_vel]]),
            stage_cost_weight=self.stage_cost_weight,
            terminal_cost_weight=self.terminal_cost_weight,
        )
        
        self.current_state = None
        self.catch_traj = False
        self.catch_phy = False
        self.phy_state_data = [0.0] * 20 
        
        # 发布 AckermannCmd 到 carla_bridge
        self.ackermann_pub = rospy.Publisher('/ackermann_cmd', AckermannCmd, queue_size=10)
        
        self.state_sub = rospy.Subscriber('/vehicle_state', VehicleState, self.state_callback)
        self.planned_traj_sub = rospy.Subscriber('/planned_trajectory', Trajectory, self.trajectory_callback)
        self.phy_state_sub = rospy.Subscriber('/xtark/phy_state', Float32MultiArray, self.phy_state_callback)
        
        self.control_rate = rospy.Rate(self.publish_rate)
        
        self._init_csv_logger()
        self.last_debug_time = rospy.Time.now()
        
        rospy.loginfo("Velocity MPPI Node initialized - Publishing to /ackermann_cmd")
    
    def load_parameters(self):
        """加载参数，使用 hybrid_mppi_control 参数组"""
        # 使用 hybrid_mppi_control 参数组
        self.delta_t = rospy.get_param('~hybrid_mppi_control/delta_t', 0.05)
        self.horizon_T = rospy.get_param('~hybrid_mppi_control/horizon_T', 30) 
        self.num_samples_K = rospy.get_param('~hybrid_mppi_control/num_samples_K', 500)
        self.exploration = rospy.get_param('~hybrid_mppi_control/exploration', 0.05)
        
        self.lambda_param = rospy.get_param('~hybrid_mppi_control/lambda', 50.0)
        self.alpha_param = rospy.get_param('~hybrid_mppi_control/alpha', 0.2)
        
        self.sigma_steer = rospy.get_param('~hybrid_mppi_control/sigma_steer', 0.08)
        self.sigma_vel = rospy.get_param('~hybrid_mppi_control/sigma_vel', 0.25)
        
        self.wheelbase = rospy.get_param('~hybrid_mppi_control/wheelbase', 3.368)
        self.max_steer_deg = rospy.get_param('~hybrid_mppi_control/max_steer', 45.0)
        self.max_speed = rospy.get_param('~hybrid_mppi_control/max_speed_abs', 1.5) 
        self.publish_rate = rospy.get_param('~hybrid_mppi_control/publish_rate', 20.0)
        
        # 加载权重参数
        stage_weights = rospy.get_param('~hybrid_mppi_control/stage_cost_weight', [10.0, 40.0, 40.0, 80.0])
        terminal_weights = rospy.get_param('~hybrid_mppi_control/terminal_cost_weight', [20.0, 80.0, 80.0, 20.0])
        
        self.stage_cost_weight = np.array(stage_weights)
        self.terminal_cost_weight = np.array(terminal_weights)
        
        rospy.loginfo(f"Loaded MPPI parameters: horizon_T={self.horizon_T}, K={self.num_samples_K}, lambda={self.lambda_param}")
    
    def _init_csv_logger(self):
        log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_filename = os.path.join(log_dir, f'velocity_mppi_log_{timestamp_str}.csv')

        self.csv_header = [
            # 1. Virtual Car & Reference (Existing 24 cols)
            'timestamp', 'ref_x', 'ref_y', 'ref_yaw', 'ref_v', 'ref_w',
            'vir_x', 'vir_y', 'vir_yaw', 'vir_v', 'vir_vx', 'vir_vy', 'vir_w',
            'e_Vx', 'e_Vy', 'e_Vyaw', 'e_Vv', 'e_Vw',
            'u_Vsteer', 'u_Vaccel', 'cmd_gear', 'comp_time_ms', 'min_cost', 'mean_cost',
            
            # 2. Physical Car (New 20 cols)
            'ref_Px', 'ref_Py', 'ref_Pyaw', 'ref_Pv', 'ref_Pw',
            'phy_x', 'phy_y', 'phy_yaw', 'phy_v', 'phy_w',
            'e_Px', 'e_Py', 'e_Pyaw', 'e_Pv', 'e_Pw',
            'u_Pv', 'u_Pw',
            'phy_v_hat', 'phy_f_hat', 'phy_tau'
        ]
        with open(self.csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.csv_header)
        
    def state_callback(self, msg):
        self.current_state = msg
    
    def trajectory_callback(self, msg):
        self.catch_traj = True
        self.mppi.set_reference_path(msg)
    
    def phy_state_callback(self, msg):
        self.catch_phy = True
        self.phy_state_data = msg.data[:20] 
    
    def compute_control_cmd(self):
        if self.current_state is None or not self.catch_traj:
            return None
        
        observed_x = np.array([
            self.current_state.x,
            self.current_state.y,
            self.current_state.yaw,
            self.current_state.vx 
        ])
        
        start_time = time.time()
        optimal_input, _, min_cost, mean_cost = self.mppi.calc_control_input(observed_x)
        comp_time = (time.time() - start_time) * 1000 # 转换为毫秒
        
        u_steer = optimal_input[0]  # 转向角 (rad)
        u_vel = optimal_input[1]    # 期望速度 (m/s)
        
        # --- FIXED GEAR SANITY CHECK ---
        # Instead of using an obsolete index, we use the CURRENT ref velocity
        if self.mppi.current_ref_point_debug is not None:
            ref_v = self.mppi.current_ref_point_debug[3]
            
            # If reference clearly says "Go Backward" (< -0.05), enforce negative command
            if ref_v < -0.05:
                if u_vel > 0: u_vel = -abs(u_vel)
            
            # If reference clearly says "Go Forward" (> 0.05), enforce positive command
            elif ref_v > 0.05:
                if u_vel < 0: u_vel = abs(u_vel)
        # -------------------------------
        
        # 创建 AckermannCmd 消息
        cmd = AckermannCmd()
        cmd.target_steer = u_steer
        cmd.target_speed = u_vel
        cmd.stamp = rospy.Time.now()
        
        if (rospy.Time.now() - self.last_debug_time).to_sec() > 0.5:
            self._print_console_debug(observed_x, u_steer, u_vel, comp_time, min_cost, mean_cost)
            self.last_debug_time = rospy.Time.now()

        self._log_to_csv(observed_x, u_steer, u_vel, comp_time, min_cost, mean_cost)
        return cmd

    def _print_console_debug(self, observed_x, u_steer, u_vel, comp_time, min_cost, mean_cost):
        ref_point = self.mppi.current_ref_point_debug
        if ref_point is None: 
            return

        ref_x, ref_y, ref_yaw, ref_v = ref_point[0], ref_point[1], ref_point[2], ref_point[3]
        vir_x, vir_y, vir_yaw, vir_v = observed_x[0], observed_x[1], observed_x[2], observed_x[3]
        
        e_Vx = ref_x - vir_x
        e_Vy = ref_y - vir_y
        
        e_Vx_local = np.cos(vir_yaw) * e_Vx + np.sin(vir_yaw) * e_Vy
        e_Vy_local = -np.sin(vir_yaw) * e_Vx + np.cos(vir_yaw) * e_Vy
        e_Vyaw = np.arctan2(np.sin(ref_yaw - vir_yaw), np.cos(ref_yaw - vir_yaw))
        e_Vv = ref_v - vir_v
        
        mode = 'REVERSE' if u_vel < 0 else 'FORWARD'
        
        t_ref = self.mppi.current_ref_time
        # New: Progress display (Index / Total)
        idx = self.mppi.current_idx_debug
        total = self.mppi.total_idx_debug
        prog_str = f"{idx}/{total}"
        
        rospy.loginfo_throttle(0.5,
            "\n" + "="*65 + "\n"
            f" [Time-MPPI] Mode: {mode} | Prog: {prog_str} | Comp: {comp_time:.1f}ms | Cost: {min_cost:.2f}\n"
            f" REF : X={ref_x:6.2f} | Y={ref_y:6.2f} | Yaw={math.degrees(ref_yaw):6.1f}° | V={ref_v:5.2f}\n"
            f" CUR : X={vir_x:6.2f} | Y={vir_y:6.2f} | Yaw={math.degrees(vir_yaw):6.1f}° | V={vir_v:5.2f}\n"
            f" ERR : Lon={e_Vx_local:5.2f} | Lat={e_Vy_local:5.2f} | Yaw={math.degrees(e_Vyaw):5.1f}° | V={e_Vv:5.2f}\n"
            f" OUT : Steer={math.degrees(u_steer):5.1f}° | Vel={u_vel:5.2f} m/s\n"
            + "="*65
        )

    def _log_to_csv(self, observed_x, u_steer, u_vel, comp_time, min_cost, mean_cost):
        try:
            ref_point = self.mppi.current_ref_point_debug
            if ref_point is None:
                return
            
            ref_x, ref_y, ref_yaw, ref_v, ref_w = ref_point[0], ref_point[1], ref_point[2], ref_point[3], ref_point[4]
            vir_x, vir_y, vir_yaw, vir_vx = observed_x[0], observed_x[1], observed_x[2], observed_x[3]
            
            vir_vy = getattr(self.current_state, 'vy', 0.0)
            vir_v = np.sign(vir_vx) * np.sqrt(vir_vx**2 + vir_vy**2)
            vir_w = getattr(self.current_state, 'omega', 0.0)
            
            E_Vx = ref_x - vir_x
            E_Vy = ref_y - vir_y
            e_Vx = np.cos(vir_yaw) * E_Vx + np.sin(vir_yaw) * E_Vy
            e_Vy = -np.sin(vir_yaw) * E_Vx + np.cos(vir_yaw) * E_Vy
            e_Vyaw = np.arctan2(np.sin(ref_yaw - observed_x[2]), np.cos(ref_yaw - observed_x[2]))
            e_Vv = ref_v - vir_vx
            e_Vw = 0.0 - vir_w
            
            target_gear = -1 if u_vel < 0 else 1
            
            row_vir = [
                rospy.Time.now().to_sec(),
                ref_x, ref_y, ref_yaw, ref_v, ref_w,
                vir_x, vir_y, vir_yaw, vir_v, vir_vx, vir_vy, vir_w,
                e_Vx, e_Vy, e_Vyaw, e_Vv, e_Vw,
                u_steer, u_vel, target_gear, comp_time, min_cost, mean_cost
            ]
            
            # 2. Physical Car Data (Appended 20 cols)
            row_phy = self.phy_state_data

            if not self.catch_phy:
                # rospy.logwarn_throttle(5.0, f"[CSV] No physical vehicle data received.")
                pass

            # Combine
            full_row = row_vir + list(row_phy)
            
            with open(self.csv_filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(full_row)
        except Exception as e:
            rospy.logwarn(f"CSV logging error: {e}")
    
    def run(self):
        rospy.loginfo(f"Starting Velocity MPPI at {self.publish_rate} Hz")
        while not rospy.is_shutdown():
            cmd = self.compute_control_cmd()
            if cmd is not None:
                self.ackermann_pub.publish(cmd)
            try:
                self.control_rate.sleep()
            except rospy.ROSInterruptException:
                break


def main():
    try:
        node = VelocityMPPINode()
        node.run()
    except rospy.ROSInterruptException: 
        pass


if __name__ == '__main__':
    main()