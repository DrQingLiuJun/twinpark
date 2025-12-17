#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Robust Hybrid A*-Guided MPPI Controller (Final Fix v2)
Fix: Removed index forcing. Added Physics-based "Kick-Start" logic to overcome deadzone.
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

from vehicle_msgs.msg import VehicleState, ControlCmd, Trajectory
from std_msgs.msg import Float32MultiArray

# Add ROS path if needed
ros_path = '/opt/ros/melodic/lib/python2.7/dist-packages'
if ros_path not in sys.path:
    sys.path.append(ros_path)

class RobustMPPIController:
    def __init__(
        self,
        delta_t: float = 0.05,
        wheel_base: float = 3.368,
        max_steer_abs: float = 0.785,
        max_accel_abs: float = 2.0,
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
        rospy.loginfo(f"Robust MPPI running on: {self.device}")

        # Parameters
        self.dim_x = 4 
        self.dim_u = 2 
        self.T = horizon_step_T
        self.K = number_of_samples_K
        self.param_lambda = param_lambda
        self.param_alpha = param_alpha
        self.param_gamma = self.param_lambda * (1.0 - self.param_alpha)

        # Weights
        self.stage_cost_weight = torch.tensor(stage_cost_weight, dtype=torch.float32, device=self.device)
        self.terminal_cost_weight = torch.tensor(terminal_cost_weight, dtype=torch.float32, device=self.device)
        self.Sigma = torch.tensor(sigma, dtype=torch.float32, device=self.device)
        self.Sigma_inv = torch.linalg.inv(self.Sigma)

        self.max_steer_abs = max_steer_abs
        self.max_accel_abs = max_accel_abs
        self.delta_t = delta_t
        self.wheel_base = wheel_base

        # Trajectory & Tracking
        self.ref_path = None          
        self.ref_times = None
        self.last_closest_idx = 0     
        self.current_ref_point_debug = None
        
        # Output Smoothing
        self.prev_steer_cmd = 0.0
        self.prev_accel_cmd = 0.0
        self.max_dsteer = 0.15 
        self.max_daccel = 0.2
        
        self.u_prev = torch.zeros((self.T, self.dim_u), dtype=torch.float32, device=self.device)

    def set_reference_path(self, trajectory: Trajectory):
        num_points = len(trajectory.x)
        if num_points < 5:
            return

        path = np.zeros((num_points, 7))
        path[:, 0] = trajectory.x
        path[:, 1] = trajectory.y
        path[:, 2] = trajectory.yaw
        path[:, 3] = trajectory.vx 
        
        if hasattr(trajectory, 't') and len(trajectory.t) == num_points and trajectory.t[-1] > 0.1:
             self.ref_times = np.array(trajectory.t)
        else:
            dists = np.sqrt(np.diff(path[:,0])**2 + np.diff(path[:,1])**2)
            vels = np.abs(path[:,3])
            v_avg = np.maximum((vels[:-1] + vels[1:]) / 2.0, 0.2)
            dt_list = dists / v_avg
            self.ref_times = np.zeros(num_points)
            self.ref_times[1:] = np.cumsum(dt_list)

        dt_diff = np.diff(self.ref_times)
        dt_diff = np.maximum(dt_diff, 0.02)

        try:
            path[:, 3] = savgol_filter(path[:, 3], window_length=9, polyorder=2)
        except:
            pass 

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

        dv = np.diff(path[:, 3])
        path[1:, 6] = dv / dt_diff
        path[0, 6] = path[1, 6]
        path[:, 6] = np.clip(path[:, 6], -self.max_accel_abs, self.max_accel_abs)
        
        try:
            path[:, 5] = savgol_filter(path[:, 5], window_length=11, polyorder=2) 
            path[:, 6] = savgol_filter(path[:, 6], window_length=11, polyorder=2) 
        except:
            pass

        self.ref_path = path
        self.last_closest_idx = 0
        rospy.loginfo(f"Ref Path Processed. {num_points} pts.")

    def _update_tracking_index(self, observed_x):
        """
        Improved Monotonic Tracking with direction change handling.
        Key improvements:
        1. Detect direction change points (where ref_v changes sign)
        2. Force index advancement when near direction change points
        3. Use time-based advancement as fallback
        """
        if self.ref_path is None:
            return 0.0, 0

        if self.last_closest_idx >= len(self.ref_path) - 1:
            return self.ref_times[-1], len(self.ref_path) - 1

        search_start = self.last_closest_idx
        search_end = min(len(self.ref_path), search_start + 80)  # 扩大搜索范围

        best_idx = search_start
        min_dist_sq = float('inf')
        
        for i in range(search_start, search_end):
            dx = self.ref_path[i, 0] - observed_x[0]
            dy = self.ref_path[i, 1] - observed_x[1]
            d_sq = dx*dx + dy*dy
            if d_sq < min_dist_sq:
                min_dist_sq = d_sq
                best_idx = i
        
        # 检测是否在换向点附近
        # 换向点特征：ref_v 接近 0 且前后符号不同
        current_ref_v = self.ref_path[best_idx, 3]
        is_near_direction_change = False
        
        # 检查当前点是否是换向点（速度接近0）
        if abs(current_ref_v) < 0.15:
            # 检查前后速度符号
            look_back = max(0, best_idx - 5)
            look_ahead = min(len(self.ref_path) - 1, best_idx + 5)
            v_before = self.ref_path[look_back, 3]
            v_after = self.ref_path[look_ahead, 3]
            
            # 如果前后速度符号不同，说明是换向点
            if v_before * v_after < 0:
                is_near_direction_change = True
        
        # 如果在换向点附近且车辆速度很低，强制推进索引
        current_v = observed_x[3]
        if is_near_direction_change and abs(current_v) < 0.1:
            # 计算到当前参考点的距离
            dist_to_ref = np.sqrt(min_dist_sq)
            
            # 如果距离足够近（< 0.5m），强制推进到换向点之后
            if dist_to_ref < 0.5:
                # 找到换向点之后的第一个有速度的点
                for i in range(best_idx, min(best_idx + 30, len(self.ref_path))):
                    if abs(self.ref_path[i, 3]) > 0.1:
                        best_idx = i
                        rospy.loginfo_throttle(1.0, f"Direction change detected! Advancing index to {best_idx}, ref_v={self.ref_path[i, 3]:.2f}")
                        break
        
        # 防止索引后退（单调递增）
        if best_idx < self.last_closest_idx:
            best_idx = self.last_closest_idx
        
        self.last_closest_idx = best_idx
        return self.ref_times[best_idx], best_idx

    def _get_ref_horizon(self, start_idx):
        ref_interp = np.zeros((self.T, 7))
        for t in range(self.T):
            curr_idx = min(start_idx + t, len(self.ref_path) - 1)
            ref_interp[t, :] = self.ref_path[curr_idx, :]
            if curr_idx == len(self.ref_path) - 1:
                ref_interp[t, 3] = 0.0 
                ref_interp[t, 6] = 0.0 
        return torch.tensor(ref_interp, dtype=torch.float32, device=self.device)

    def calc_control_input(self, observed_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
        if self.ref_path is None:
            return np.zeros(2), np.zeros((self.T, 2)), 0.0, 0.0

        current_ref_time, current_idx = self._update_tracking_index(observed_x)
        
        ref_data = self._get_ref_horizon(current_idx)
        ref_states = ref_data[:, :5]
        u_guide = ref_data[:, 5:7].clone() 
        
        self.current_ref_point_debug = ref_states[0].cpu().numpy()

        x0 = torch.tensor(observed_x, dtype=torch.float32, device=self.device)
        epsilon = self._calc_epsilon(self.Sigma, self.K, self.T, self.dim_u)
        
        u_mean = u_guide.unsqueeze(0).repeat(self.K, 1, 1)
        v = u_mean + epsilon
        v[:, :, 0] = torch.clamp(v[:, :, 0], -self.max_steer_abs, self.max_steer_abs)
        v[:, :, 1] = torch.clamp(v[:, :, 1], -self.max_accel_abs, self.max_accel_abs)

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
        raw_accel = u_optimal_seq[0, 1].item()
        
        d_steer = np.clip(raw_steer - self.prev_steer_cmd, -self.max_dsteer, self.max_dsteer)
        final_steer = self.prev_steer_cmd + d_steer
        
        d_accel = np.clip(raw_accel - self.prev_accel_cmd, -self.max_daccel, self.max_daccel)
        final_accel = self.prev_accel_cmd + d_accel
        
        self.prev_steer_cmd = final_steer
        self.prev_accel_cmd = final_accel
        
        optimal_input = np.array([final_steer, final_accel])
        
        return optimal_input, u_optimal_seq.cpu().numpy(), rho.item(), S.mean().item()

    def _c_tracking(self, x, u, ref_target):
        ref_x, ref_y, ref_yaw, ref_v = ref_target[0], ref_target[1], ref_target[2], ref_target[3]
        E_Vx = ref_x - x[:, 0]
        E_Vy = ref_y - x[:, 1]
        vir_yaw = x[:, 2]
        
        e_Vy = -torch.sin(vir_yaw) * E_Vx + torch.cos(vir_yaw) * E_Vy 
        e_Vx = torch.cos(vir_yaw) * E_Vx + torch.sin(vir_yaw) * E_Vy  
        e_Vyaw = ref_yaw - x[:, 2] 
        e_Vyaw = torch.atan2(torch.sin(e_Vyaw), torch.cos(e_Vyaw))
        e_Vv = ref_v - x[:, 3]

        cost = self.stage_cost_weight[0] * e_Vx**2 + \
               self.stage_cost_weight[1] * e_Vy**2 + \
               self.stage_cost_weight[2] * e_Vyaw**2 + \
               self.stage_cost_weight[3] * e_Vv**2
        return cost

    def _phi_tracking(self, x, ref_target):
        ref_x, ref_y, ref_yaw, ref_v = ref_target[0], ref_target[1], ref_target[2], ref_target[3]
        E_Vx = ref_x - x[:, 0]
        E_Vy = ref_y - x[:, 1]
        vir_yaw = x[:, 2]
        e_Vy = -torch.sin(vir_yaw) * E_Vx + torch.cos(vir_yaw) * E_Vy
        e_Vx = torch.cos(vir_yaw) * E_Vx + torch.sin(vir_yaw) * E_Vy
        e_Vyaw = torch.atan2(torch.sin(ref_yaw - x[:, 2]), torch.cos(ref_yaw - x[:, 2]))
        e_Vv = ref_v - x[:, 3]
        
        cost = self.terminal_cost_weight[0] * e_Vx**2 + \
               self.terminal_cost_weight[1] * e_Vy**2 + \
               self.terminal_cost_weight[2] * e_Vyaw**2 + \
               self.terminal_cost_weight[3] * e_Vv**2
        return cost

    def _F(self, x_t, v_t):
        x, y, yaw, v = x_t[:, 0], x_t[:, 1], x_t[:, 2], x_t[:, 3]
        u_Vsteer, u_Vaccel = v_t[:, 0], v_t[:, 1]
        new_v = v + u_Vaccel * self.delta_t
        new_x = x + v * torch.cos(yaw) * self.delta_t
        new_y = y + v * torch.sin(yaw) * self.delta_t
        new_yaw = yaw + v / self.wheel_base * torch.tan(u_Vsteer) * self.delta_t
        return torch.stack([new_x, new_y, new_yaw, new_v], dim=1)

    def _calc_epsilon(self, sigma, size_sample, size_time_step, size_dim_u):
        mean = torch.zeros(size_dim_u, device=self.device)
        dist = torch.distributions.MultivariateNormal(mean, covariance_matrix=sigma)
        return dist.sample((size_sample, size_time_step))

class MPPIControlNode:
    def __init__(self):
        rospy.init_node('mppi_control_node', anonymous=False)
        self.load_parameters()
        
        self.mppi = RobustMPPIController(
            delta_t=self.delta_t,
            wheel_base=self.wheelbase,
            max_steer_abs=math.radians(self.max_steer_deg),
            max_accel_abs=self.max_accel,
            horizon_step_T=self.horizon_T,
            number_of_samples_K=self.num_samples_K,
            param_exploration=self.exploration,
            param_lambda=self.lambda_param,
            param_alpha=self.alpha_param,
            sigma=np.array([[self.sigma_steer, 0.0], [0.0, self.sigma_accel]]),
        )
        
        self.current_state = None
        self.catch_traj = False
        self.catch_phy = False
        self.phy_state_data = [0.0] * 20 
        
        self.control_pub = rospy.Publisher('/control_cmd', ControlCmd, queue_size=10)
        self.state_sub = rospy.Subscriber('/vehicle_state', VehicleState, self.state_callback)
        self.planned_traj_sub = rospy.Subscriber('/planned_trajectory', Trajectory, self.trajectory_callback)
        self.phy_state_sub = rospy.Subscriber('/xtark/phy_state', Float32MultiArray, self.phy_state_callback)
        
        self.control_rate = rospy.Rate(self.publish_rate)
        
        self._init_csv_logger()
        self.last_debug_time = rospy.Time.now()
        self._last_gear = 1
        
        rospy.loginfo("Robust MPPI Node initialized")
    
    def _init_csv_logger(self):
        log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_filename = os.path.join(log_dir, f'robust_mppi_log_{timestamp_str}.csv')

        self.csv_header = [
            'timestamp', 'ref_x', 'ref_y', 'ref_yaw', 'ref_v', 'ref_w',
            'vir_x', 'vir_y', 'vir_yaw', 'vir_v', 'vir_vx', 'vir_vy', 'vir_w',
            'e_Vx', 'e_Vy', 'e_Vyaw', 'e_Vv', 'e_Vw',
            'u_Vsteer', 'u_Vaccel', 'cmd_gear', 'comp_time_ms', 'min_cost', 'mean_cost',
            'ref_Px', 'ref_Py', 'ref_Pyaw', 'ref_Pv', 'ref_Pw',
            'phy_x', 'phy_y', 'phy_yaw', 'phy_v', 'phy_w',
            'e_Px', 'e_Py', 'e_Pyaw', 'e_Pv', 'e_Pw',
            'u_Pv', 'u_Pw',
            'phy_v_hat', 'phy_f_hat', 'phy_tau'
        ]
        with open(self.csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.csv_header)
    
    def load_parameters(self):
        self.delta_t = rospy.get_param('~mppi_control_node/delta_t', 0.05)
        self.horizon_T = rospy.get_param('~mppi_control_node/horizon_T', 30) 
        self.num_samples_K = rospy.get_param('~mppi_control_node/num_samples_K', 500)
        self.exploration = rospy.get_param('~mppi_control_node/exploration', 0.05)
        
        self.lambda_param = rospy.get_param('~mppi_control_node/lambda', 50.0)
        self.alpha_param = rospy.get_param('~mppi_control_node/alpha', 0.2)
        
        self.sigma_steer = rospy.get_param('~mppi_control_node/sigma_steer', 0.08) 
        self.sigma_accel = rospy.get_param('~mppi_control_node/sigma_accel', 0.25)
        
        self.wheelbase = rospy.get_param('~mppi_control_node/wheelbase', 3.368)
        self.max_steer_deg = rospy.get_param('~mppi_control_node/max_steer', 45.0)
        self.max_accel = rospy.get_param('~mppi_control_node/max_accel', 2.0)
        self.publish_rate = rospy.get_param('~mppi_control_node/publish_rate', 20.0)
        
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
        comp_time = (time.time() - start_time) * 1000 
        
        target_gear = self._last_gear
        if self.mppi.current_ref_point_debug is not None:
            ref_v = self.mppi.current_ref_point_debug[3]
            if ref_v < -0.05: target_gear = -1
            elif ref_v > 0.05: target_gear = 1
            self._last_gear = target_gear
        
        cmd = ControlCmd()
        u_Vsteer = optimal_input[0]
        max_steer_rad = math.radians(self.max_steer_deg)
        cmd.steer = np.clip(u_Vsteer / max_steer_rad, -1.0, 1.0)
        
        u_Vaccel = optimal_input[1]
        
        cmd.reverse = (target_gear == -1)
        cmd.gear = target_gear
        
        # --- KICK-START / ANTI-STICTION LOGIC ---
        # "推车一把": If car is stuck (v ~ 0) but reference wants speed,
        # Force a minimum throttle to overcome static friction.
        
        current_v = observed_x[3]
        ref_v = self.mppi.current_ref_point_debug[3] if self.mppi.current_ref_point_debug is not None else 0.0
        
        # 检查未来几个点的速度，判断是否需要启动
        future_wants_move = False
        if self.mppi.ref_path is not None:
            look_ahead_idx = min(self.mppi.last_closest_idx + 10, len(self.mppi.ref_path) - 1)
            future_ref_v = self.mppi.ref_path[look_ahead_idx, 3]
            if abs(future_ref_v) > 0.1:
                future_wants_move = True
        
        # Thresholds
        STOP_THRESH = 0.08
        MOVE_REQ_THRESH = 0.08  # 降低阈值，更容易触发启动
        MIN_KICK_THROTTLE = 0.22  # 22% throttle boost (Adjust based on vehicle weight)
        
        is_stuck = abs(current_v) < STOP_THRESH
        wants_to_move = abs(ref_v) > MOVE_REQ_THRESH or future_wants_move
        
        # 检测是否在换向点：当前速度接近0，但未来需要反向运动
        at_direction_change = False
        if self.mppi.ref_path is not None and is_stuck:
            idx = self.mppi.last_closest_idx
            # 检查前后速度符号
            look_back = max(0, idx - 5)
            look_ahead = min(len(self.mppi.ref_path) - 1, idx + 10)
            v_before = self.mppi.ref_path[look_back, 3]
            v_after = self.mppi.ref_path[look_ahead, 3]
            
            if v_before * v_after < 0:  # 符号不同，是换向点
                at_direction_change = True
                # 根据未来速度方向决定档位
                if v_after < -0.05:
                    target_gear = -1
                    cmd.reverse = True
                    cmd.gear = -1
                    rospy.loginfo_throttle(1.0, f"At direction change point! Switching to REVERSE, v_after={v_after:.2f}")
                elif v_after > 0.05:
                    target_gear = 1
                    cmd.reverse = False
                    cmd.gear = 1
                    rospy.loginfo_throttle(1.0, f"At direction change point! Switching to FORWARD, v_after={v_after:.2f}")
        
        if target_gear == 1:
            if u_Vaccel > 0: 
                raw_throttle = min(u_Vaccel/self.max_accel, 1.0)
                # Apply Kick
                if is_stuck and wants_to_move:
                    cmd.throttle = max(raw_throttle, MIN_KICK_THROTTLE)
                else:
                    cmd.throttle = raw_throttle
                cmd.brake = 0.0
            else: 
                cmd.throttle = 0.0; cmd.brake = min(-u_Vaccel/self.max_accel, 1.0)
        else:  # target_gear == -1 (倒车)
            if u_Vaccel < 0: 
                raw_throttle = min(-u_Vaccel/self.max_accel, 1.0)
                # Apply Kick - 倒车启动时也需要推一把
                if is_stuck and wants_to_move:
                    cmd.throttle = max(raw_throttle, MIN_KICK_THROTTLE)
                else:
                    cmd.throttle = raw_throttle
                cmd.brake = 0.0
            else: 
                cmd.throttle = 0.0; cmd.brake = min(u_Vaccel/self.max_accel, 1.0)
        
        # 在换向点，如果车已停稳但需要反向运动，给一个启动油门
        if at_direction_change and is_stuck and wants_to_move:
            cmd.throttle = max(cmd.throttle, MIN_KICK_THROTTLE)
            cmd.brake = 0.0
            rospy.loginfo_throttle(0.5, f"Kick-start at direction change! throttle={cmd.throttle:.2f}, gear={target_gear}")

        # Force Stop Logic at Destination (if index is near end)
        if self.mppi.ref_path is not None:
            if self.mppi.last_closest_idx >= len(self.mppi.ref_path) - 2:
                cmd.throttle = 0.0
                cmd.brake = 1.0
                rospy.loginfo_throttle(1.0, "Goal Reached. Holding.")

        if (rospy.Time.now() - self.last_debug_time).to_sec() > 0.5:
            self._print_console_debug(observed_x, u_Vsteer, u_Vaccel, cmd)
            self.last_debug_time = rospy.Time.now()

        self._log_to_csv(observed_x, u_Vsteer, u_Vaccel, target_gear, comp_time, min_cost, mean_cost)
        return cmd

    def _print_console_debug(self, observed_x, u_Vsteer, u_Vaccel, cmd):
        ref_point = self.mppi.current_ref_point_debug
        if ref_point is None: return

        ref_x, ref_y, ref_yaw, ref_v = ref_point[0], ref_point[1], ref_point[2], ref_point[3]
        vir_x, vir_y, vir_yaw, vir_v = observed_x[0], observed_x[1], observed_x[2], observed_x[3]
        
        e_Vx = ref_x - vir_x
        e_Vy = ref_y - vir_y
        
        e_Vx_local = np.cos(vir_yaw) * e_Vx + np.sin(vir_yaw) * e_Vy
        e_Vy_local = -np.sin(vir_yaw) * e_Vx + np.cos(vir_yaw) * e_Vy
        e_Vyaw = np.arctan2(np.sin(ref_yaw - vir_yaw), np.cos(ref_yaw - vir_yaw))
        e_Vv = ref_v - vir_v
        
        rospy.loginfo_throttle(0.5,
            "\n" + "="*65 + "\n"
            f" [MPPI STATUS] Mode: {'REVERSE' if cmd.gear < 0 else 'FORWARD'}\n"
            f" REF : X={ref_x:6.2f} | Y={ref_y:6.2f} | Yaw={math.degrees(ref_yaw):6.1f}° | V={ref_v:5.2f}\n"
            f" CUR : X={vir_x:6.2f} | Y={vir_y:6.2f} | Yaw={math.degrees(vir_yaw):6.1f}° | V={vir_v:5.2f}\n"
            f" ERR : Lon={e_Vx_local:5.2f} | Lat={e_Vy_local:5.2f} | Yaw={math.degrees(e_Vyaw):5.1f}° | V={e_Vv:5.2f}\n"
            f" OUT : Steer={math.degrees(u_Vsteer):5.1f}° | Accel={u_Vaccel:5.2f}\n"
            f" CMD : Thr={cmd.throttle:4.2f} | Brk={cmd.brake:4.2f} | Str={cmd.steer:5.2f} | G={cmd.gear}\n"
            + "="*65
        )

    def _log_to_csv(self, observed_x, u_Vsteer, u_Vaccel, target_gear, comp_time, min_cost, mean_cost):
        try:
            ref_point = self.mppi.current_ref_point_debug
            if ref_point is None:
                return
            
            ref_x, ref_y, ref_yaw, ref_v, ref_w = ref_point[0], ref_point[1], ref_point[2], ref_point[3], 0.0
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
            
            row_vir = [
                rospy.Time.now().to_sec(),
                ref_x, ref_y, ref_yaw, ref_v, ref_w,
                vir_x, vir_y, vir_yaw, vir_v, vir_vx, vir_vy, vir_w,
                e_Vx, e_Vy, e_Vyaw, e_Vv, e_Vw,
                u_Vsteer, u_Vaccel, target_gear, comp_time, min_cost, mean_cost
            ]
            row_phy = self.phy_state_data
            full_row = row_vir + list(row_phy)
            
            with open(self.csv_filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(full_row)
        except Exception as e:
            pass
    
    def run(self):
        rospy.loginfo(f"Starting Robust MPPI at {self.publish_rate} Hz")
        while not rospy.is_shutdown():
            cmd = self.compute_control_cmd()
            if cmd is not None:
                self.control_pub.publish(cmd)
            try:
                self.control_rate.sleep()
            except rospy.ROSInterruptException:
                break

def main():
    try:
        node = MPPIControlNode()
        node.run()
    except rospy.ROSInterruptException: pass

if __name__ == '__main__':
    main()