#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MPPI Controller Node for TwinPark System (Time-Based Tracking Version)
Model Predictive Path Integral Control for trajectory tracking
"""
import torch
import sys

# Add ROS path if needed
ros_path = '/opt/ros/melodic/lib/python2.7/dist-packages'
if ros_path not in sys.path:
    sys.path.append(ros_path)

import rospy
import csv
import os
import math
import numpy as np
from datetime import datetime
from typing import Tuple

from vehicle_msgs.msg import VehicleState, ControlCmd, Trajectory


class MPPIController:
    """MPPI Controller using PyTorch with Time-Based Trajectory Tracking"""
    
    def __init__(
        self,
        delta_t: float = 0.05,
        wheel_base: float = 3.368,
        max_steer_abs: float = 0.785,
        max_accel_abs: float = 2.0,
        horizon_step_T: int = 20,
        number_of_samples_K: int = 1000,
        param_exploration: float = 0.05,
        param_lambda: float = 100.0,
        param_alpha: float = 0.98,
        sigma: np.ndarray = np.array([[0.6, 0.0], [0.0, 0.8]]),
        stage_cost_weight: np.ndarray = np.array([5.0, 15.0, 10.0, 5.0]),
        terminal_cost_weight: np.ndarray = np.array([10.0, 30.0, 20.0, 5.0]),
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rospy.loginfo(f"MPPI Controller running on: {self.device}")

        # Parameters
        self.dim_x = 4 
        self.dim_u = 2 
        self.T = horizon_step_T
        self.K = number_of_samples_K
        self.param_exploration = param_exploration
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

        # Reference Path storage
        self.ref_path_tensor = None # Tensor on device
        self.ref_path_np = None     # Numpy for CPU interpolation
        self.ref_times = None       # Calculated time stamps for each point
        
        # [关键] 轨迹播放器计时器
        self.trajectory_start_time = None 
        
        # Debug info for external nodes
        self.current_ref_point_debug = None
        
        self.u_prev = torch.zeros((self.T, self.dim_u), dtype=torch.float32, device=self.device)

    def set_reference_path(self, trajectory: Trajectory):
        """
        1. Store trajectory.
        2. Reconstruct time profile (t) from distance (s) and velocity (v).
        """
        num_points = len(trajectory.x)
        if num_points < 2:
            return

        path_np = np.zeros((num_points, 4))
        path_np[:, 0] = trajectory.x
        path_np[:, 1] = trajectory.y
        path_np[:, 2] = trajectory.yaw
        path_np[:, 3] = trajectory.vx
        
        # 保存 Numpy 版本用于插值
        self.ref_path_np = path_np
        self.ref_path_tensor = torch.tensor(path_np, dtype=torch.float32, device=self.device)
        
        # --- 重建时间戳 (Time Reconstruction) ---
        dists = np.sqrt(np.diff(path_np[:,0])**2 + np.diff(path_np[:,1])**2)
        vels = path_np[:,3]
        
        # 计算每段的平均速度，防止除以0
        v_avg = np.maximum(np.abs(vels[:-1] + vels[1:]) / 2.0, 0.1)
        dt_list = dists / v_avg
        
        self.ref_times = np.zeros(num_points)
        self.ref_times[1:] = np.cumsum(dt_list)
        
        # 重置播放器
        self.trajectory_start_time = None 
        
        total_time = self.ref_times[-1]
        rospy.loginfo(f"Reference path updated. Total Time: {total_time:.2f}s, Points: {num_points}")

    def _get_ref_states_for_horizon(self, current_time):
        """
        根据当前时间，插值获取未来 Horizon T 步的参考状态
        Returns: Tensor (T, 4)
        """
        # 生成查询时间点: [t, t+dt, ..., t+(T-1)dt]
        query_times = current_time + np.arange(self.T) * self.delta_t
        
        # 限制不超过轨迹终点时间 (到达终点后，参考点就停在那里)
        max_time = self.ref_times[-1]
        query_times = np.clip(query_times, 0, max_time)
        
        # 线性插值 (Linear Interpolation)
        rx = np.interp(query_times, self.ref_times, self.ref_path_np[:,0])
        ry = np.interp(query_times, self.ref_times, self.ref_path_np[:,1])
        rv = np.interp(query_times, self.ref_times, self.ref_path_np[:,3])
        
        # Yaw 角需要特殊处理 (插值 sin/cos)
        r_sin = np.interp(query_times, self.ref_times, np.sin(self.ref_path_np[:,2]))
        r_cos = np.interp(query_times, self.ref_times, np.cos(self.ref_path_np[:,2]))
        ryaw = np.arctan2(r_sin, r_cos)
        
        # 组合成 Tensor
        ref_np = np.stack([rx, ry, ryaw, rv], axis=1)
        return torch.tensor(ref_np, dtype=torch.float32, device=self.device)

    def calc_control_input(self, observed_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
        if self.ref_path_np is None:
            return np.zeros(2), np.zeros((self.T, 2)), 0.0, 0.0

        # --- 1. 时间同步与更新 (Time Sync) ---
        now = rospy.Time.now().to_sec()
        
        # 如果是第一次运行，或者重置了，进行初始化同步
        if self.trajectory_start_time is None:
            # 找到离当前车辆最近的点，作为起始时间
            dists = np.linalg.norm(self.ref_path_np[:,:2] - observed_x[:2], axis=1)
            closest_idx = np.argmin(dists)
            start_offset_time = self.ref_times[closest_idx]
            
            # 设置“虚拟开始时间”
            self.trajectory_start_time = now - start_offset_time
            rospy.loginfo(f"Trajectory tracking started at progress: {start_offset_time:.2f}s")

        # 计算当前处于轨迹的第几秒 (Reference point walks by itself!)
        time_from_start = now - self.trajectory_start_time
        
        # 获取未来 T 步的参考轨迹
        ref_seq = self._get_ref_states_for_horizon(time_from_start)
        
        # [关键] 保存当前时刻的参考点供外部使用
        self.current_ref_point_debug = ref_seq[0].cpu().numpy() # [x, y, yaw, v]

        # --- 2. MPPI 优化 ---
        x0 = torch.tensor(observed_x, dtype=torch.float32, device=self.device)
        
        # Sample noise
        epsilon = self._calc_epsilon(self.Sigma, self.K, self.T, self.dim_u)
        v = self.u_prev.unsqueeze(0) + epsilon
        
        # Exploration & Clamp
        split_idx = int((1.0 - self.param_exploration) * self.K)
        if split_idx < self.K:
            v[split_idx:, :, :] = epsilon[split_idx:, :, :]
        v[:, :, 0] = torch.clamp(v[:, :, 0], -self.max_steer_abs, self.max_steer_abs)
        v[:, :, 1] = torch.clamp(v[:, :, 1], -self.max_accel_abs, self.max_accel_abs)

        # Rollout
        S = torch.zeros(self.K, dtype=torch.float32, device=self.device)
        x = x0.unsqueeze(0).repeat(self.K, 1)

        # Main Loop
        for t in range(self.T):
            x = self._F(x, v[:, t, :])
            
            u_t = self.u_prev[t].unsqueeze(0)
            v_t = v[:, t, :]
            temp = u_t @ self.Sigma_inv
            control_cost = self.param_gamma * (temp * v_t).sum(dim=1)

            # [核心修改] 传入 t 时刻对应的参考点
            state_cost = self._c_tracking(x, v[:, t, :], ref_seq[t])
            
            S += state_cost + control_cost

        # Terminal Cost
        S += self._phi_tracking(x, ref_seq[-1])

        # Compute Weights & Update
        rho = torch.min(S)
        eta = torch.sum(torch.exp((-1.0 / self.param_lambda) * (S - rho)))
        w = (1.0 / eta) * torch.exp((-1.0 / self.param_lambda) * (S - rho))
        
        min_cost = rho.item()
        mean_cost = S.mean().item()

        w_expanded = w.view(self.K, 1, 1)
        w_epsilon = torch.sum(w_expanded * epsilon, dim=0)
        w_epsilon = self._moving_average_filter(w_epsilon)
        self.u_prev = self.u_prev + w_epsilon
        optimal_input = self.u_prev[0].clone()
        
        self.u_prev[:-1] = self.u_prev[1:].clone()
        self.u_prev[-1] = self.u_prev[-1].clone()

        return optimal_input.cpu().numpy(), self.u_prev.cpu().numpy(), min_cost, mean_cost

    def _c_tracking(self, x, u, ref_target):
        """Tracking Cost Function"""
        ref_x, ref_y, ref_yaw, ref_v = ref_target[0], ref_target[1], ref_target[2], ref_target[3]
        
        dx = x[:, 0] - ref_x
        dy = x[:, 1] - ref_y
        
        e_lat = -torch.sin(ref_yaw) * dx + torch.cos(ref_yaw) * dy
        e_lon = torch.cos(ref_yaw) * dx + torch.sin(ref_yaw) * dy
        
        e_yaw = x[:, 2] - ref_yaw
        e_yaw = torch.atan2(torch.sin(e_yaw), torch.cos(e_yaw))
        
        e_v = x[:, 3] - ref_v

        # Cost Weights # 新增：纵向位置误差惩罚
        cost = self.stage_cost_weight[0] * e_lon**2 + \
            self.stage_cost_weight[1] * e_lat**2 + \
            self.stage_cost_weight[2] * e_yaw**2 + \
            self.stage_cost_weight[3] * e_v**2
                

        # Control regularization
        accel, steer = u[:, 1], u[:, 0]
        w_u = 0.01 
        u_reg = accel**2 + 2.0 * steer**2
        cost += w_u * u_reg
        
        return cost

    def _phi_tracking(self, x, ref_target):
        """Terminal Cost for Tracking"""
        ref_x, ref_y, ref_yaw, ref_v = ref_target[0], ref_target[1], ref_target[2], ref_target[3]
        
        dx = x[:, 0] - ref_x
        dy = x[:, 1] - ref_y
        
        e_lat = -torch.sin(ref_yaw) * dx + torch.cos(ref_yaw) * dy
        e_lon = torch.cos(ref_yaw) * dx + torch.sin(ref_yaw) * dy
        e_yaw = torch.atan2(torch.sin(x[:, 2] - ref_yaw), torch.cos(x[:, 2] - ref_yaw))
        e_v = x[:, 3] - ref_v
        
        cost = self.terminal_cost_weight[0] * e_lon**2 + \
                self.terminal_cost_weight[1] * e_lat**2 + \
                self.terminal_cost_weight[2] * e_yaw**2 + \
                self.terminal_cost_weight[3] * e_v**2
               
        return cost

    def _F(self, x_t, v_t):
        x, y, yaw, v = x_t[:, 0], x_t[:, 1], x_t[:, 2], x_t[:, 3]
        steer, accel = v_t[:, 0], v_t[:, 1]
        new_v = v + accel * self.delta_t
        new_x = x + new_v * torch.cos(yaw) * self.delta_t
        new_y = y + new_v * torch.sin(yaw) * self.delta_t
        new_yaw = yaw + new_v / self.wheel_base * torch.tan(steer) * self.delta_t
        return torch.stack([new_x, new_y, new_yaw, new_v], dim=1)

    def _calc_epsilon(self, sigma, size_sample, size_time_step, size_dim_u):
        mean = torch.zeros(size_dim_u, device=self.device)
        dist = torch.distributions.MultivariateNormal(mean, covariance_matrix=sigma)
        return dist.sample((size_sample, size_time_step))

    def _moving_average_filter(self, xx, window_size=10):
        xx_t = xx.permute(1, 0).unsqueeze(0)
        kernel = torch.ones(1, 1, window_size, device=self.device) / window_size
        kernel = kernel.repeat(self.dim_u, 1, 1)
        pad = window_size // 2
        xx_padded = torch.nn.functional.pad(xx_t, (pad, pad), mode='replicate')
        out = torch.nn.functional.conv1d(xx_padded, kernel, groups=self.dim_u)
        if window_size % 2 == 0:
            out = out[:, :, :-1]
        return out.squeeze(0).permute(1, 0)


class MPPIControlNode:
    """MPPI Control Node"""
    
    def __init__(self):
        rospy.init_node('mppi_control_node', anonymous=False)
        self.load_parameters()
        
        self.mppi = MPPIController(
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
            stage_cost_weight=np.array(self.stage_cost_weight),
            terminal_cost_weight=np.array(self.terminal_cost_weight),
        )
        
        self.current_state = None
        self.trajectory_received = False
        
        self.control_pub = rospy.Publisher('/control_cmd', ControlCmd, queue_size=10)
        self.state_sub = rospy.Subscriber('/vehicle_state', VehicleState, self.state_callback)
        self.traj_sub = rospy.Subscriber('/reference_state', VehicleState, self.reference_callback)
        self.planned_traj_sub = rospy.Subscriber('/planned_trajectory', Trajectory, self.trajectory_callback)
        
        self.control_rate = rospy.Rate(self.publish_rate)
        self._init_csv_logger()
        rospy.loginfo("MPPI Control Node initialized")
    
    def _init_csv_logger(self):
        log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_filename = os.path.join(log_dir, f'mppi_log_{timestamp_str}.csv')
        self.csv_header = [
            'timestamp', 'ref_x', 'ref_y', 'ref_yaw', 'ref_v',
            'act_x', 'act_y', 'act_yaw', 'act_v', 'act_vx', 'act_vy',
            'error_lon', 'error_lat', 'error_yaw', 'error_v',
            'cmd_steer', 'cmd_accel', 'cmd_gear', 'comp_time_ms', 'min_cost', 'mean_cost'
        ]
        with open(self.csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.csv_header)
        rospy.loginfo(f"[MPPI] CSV logging to: {self.csv_filename}")
    
    def load_parameters(self):
        # MPPI parameters
        self.delta_t = rospy.get_param('~mppi_control_node/delta_t', 0.1)
        self.horizon_T = rospy.get_param('~mppi_control_node/horizon_T', 10)
        self.num_samples_K = rospy.get_param('~mppi_control_node/num_samples_K', 500)
        self.exploration = rospy.get_param('~mppi_control_node/exploration', 0.15)
        self.lambda_param = rospy.get_param('~mppi_control_node/lambda', 15.0)
        self.alpha_param = rospy.get_param('~mppi_control_node/alpha', 0.85)
        self.sigma_steer = rospy.get_param('~mppi_control_node/sigma_steer', 0.6)
        self.sigma_accel = rospy.get_param('~mppi_control_node/sigma_accel', 0.8)
        self.stage_cost_weight = rospy.get_param('~mppi_control_node/stage_cost_weight', [5.0, 20.0, 8.0, 5.0])
        self.terminal_cost_weight = rospy.get_param('~mppi_control_node/terminal_cost_weight', [10.0, 40.0, 15.0, 3.0])
        self.wheelbase = rospy.get_param('~mppi_control_node/wheelbase', 3.368)
        self.max_steer_deg = rospy.get_param('~mppi_control_node/max_steer', 45.0)
        self.max_accel = rospy.get_param('~mppi_control_node/max_accel', 2.0)
        self.publish_rate = rospy.get_param('~mppi_control_node/publish_rate', 20.0)
        
    def state_callback(self, msg):
        self.current_state = msg
    
    def reference_callback(self, msg):
        pass
    
    def trajectory_callback(self, msg):
        self.mppi.set_reference_path(msg)
        self.trajectory_received = True
    
    def compute_control_cmd(self):
        if self.current_state is None or not self.trajectory_received:
            return None
        
        # 1. 准备状态
        observed_x = np.array([
            self.current_state.x,
            self.current_state.y,
            self.current_state.yaw,
            self.current_state.vx 
        ])
        
        # 2. 计算 MPPI 控制量 (这会更新内部时间进度)
        try:
            import time
            start_time = time.time()
            optimal_input, _, min_cost, mean_cost = self.mppi.calc_control_input(observed_x)
            comp_time = (time.time() - start_time) * 1000 
        except Exception as e:
            rospy.logerr(f"MPPI calculation failed: {e}")
            return None
        
        # 3. 挡位判断逻辑 (使用 Controller 刚刚算出来的当前时刻参考点)
        if not hasattr(self, '_last_gear'):
            self._last_gear = 1
        target_gear = self._last_gear
        
        # 获取当前参考点速度 (由时间驱动)
        if self.mppi.current_ref_point_debug is not None:
            ref_v = self.mppi.current_ref_point_debug[3] # [x, y, yaw, v]
            
            # 使用简单的迟滞比较
            if ref_v < -0.08:
                target_gear = -1
            elif ref_v > 0.08:
                target_gear = 1
            # 速度接近0时保持上一挡位
            
            self._last_gear = target_gear
        
        # 4. 解析命令
        cmd = ControlCmd()
        steer_rad = optimal_input[0]
        max_steer_rad = math.radians(self.max_steer_deg)
        cmd.steer = np.clip(steer_rad / max_steer_rad, -1.0, 1.0)
        
        accel = optimal_input[1]
        
        # 平滑加速度
        if not hasattr(self, '_last_accel'):
            self._last_accel = 0.0
        alpha_accel = 0.3
        accel = alpha_accel * accel + (1 - alpha_accel) * self._last_accel
        self._last_accel = accel
        
        cmd.reverse = (target_gear == -1)
        cmd.gear = target_gear
        
        accel_deadzone = 0.05
        # 简单踏板逻辑
        if target_gear == 1:
            if accel > accel_deadzone:
                cmd.throttle = min(accel / self.max_accel, 1.0); cmd.brake = 0.0
            elif accel < -accel_deadzone:
                cmd.throttle = 0.0; cmd.brake = min(-accel / self.max_accel, 1.0)
            else:
                cmd.throttle = 0.0; cmd.brake = 0.0
        else:
            if accel < -accel_deadzone:
                cmd.throttle = min(-accel / self.max_accel, 1.0); cmd.brake = 0.0
            elif accel > accel_deadzone:
                cmd.throttle = 0.0; cmd.brake = min(accel / self.max_accel, 1.0)
            else:
                cmd.throttle = 0.0; cmd.brake = 0.0

        current_v = self.current_state.vx
        if (target_gear == -1 and current_v > 0.2) or (target_gear == 1 and current_v < -0.2):
            cmd.throttle = 0.0
            cmd.brake = 0.8

        self._log_to_csv(observed_x, steer_rad, accel, target_gear, comp_time, min_cost, mean_cost)
        return cmd
    
    def _log_to_csv(self, observed_x, steer_rad, accel, target_gear, comp_time, min_cost, mean_cost):
        """Log control data using Time-Based reference"""
        try:
            # 直接使用 Controller 算好的当前参考点
            ref_point = self.mppi.current_ref_point_debug
            if ref_point is None:
                return
            
            ref_x, ref_y, ref_yaw, ref_v = ref_point[0], ref_point[1], ref_point[2], ref_point[3]
            
            # 速度分解
            act_vx = observed_x[3]
            act_vy = getattr(self.current_state, 'vy', 0.0)
            act_v = np.sign(act_vx) * np.sqrt(act_vx**2 + act_vy**2)
            
            # 误差计算
            dx = observed_x[0] - ref_x
            dy = observed_x[1] - ref_y
            
            error_lon = np.cos(ref_yaw) * dx + np.sin(ref_yaw) * dy
            error_lat = -np.sin(ref_yaw) * dx + np.cos(ref_yaw) * dy
            error_yaw = np.arctan2(np.sin(observed_x[2] - ref_yaw), np.cos(observed_x[2] - ref_yaw))
            error_v = act_vx - ref_v
            
            row = [
                rospy.Time.now().to_sec(),
                ref_x, ref_y, ref_yaw, ref_v,
                observed_x[0], observed_x[1], observed_x[2], act_v, act_vx, act_vy,
                error_lon, error_lat, error_yaw, error_v,
                steer_rad, accel, target_gear, comp_time, min_cost, mean_cost
            ]
            
            with open(self.csv_filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
                
        except Exception as e:
            pass
    
    def run(self):
        rospy.loginfo(f"Starting MPPI control loop at {self.publish_rate} Hz")
        while not rospy.is_shutdown():
            cmd = self.compute_control_cmd()
            if cmd is not None:
                self.control_pub.publish(cmd)
            try:
                self.control_rate.sleep()
            except rospy.ROSInterruptException:
                break
        rospy.loginfo("MPPI Control node shutting down")


def main():
    try:
        node = MPPIControlNode()
        node.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()