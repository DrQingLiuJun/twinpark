#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MPPI Controller Node for TwinPark System
Model Predictive Path Integral Control for trajectory tracking
"""
import torch

import sys
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
    """MPPI Controller using PyTorch for GPU acceleration"""
    
    def __init__(
        self,
        delta_t: float = 0.05,
        wheel_base: float = 3.368,
        max_steer_abs: float = 0.785,  # 45 degrees
        max_accel_abs: float = 2.0,
        horizon_step_T: int = 20,
        number_of_samples_K: int = 1000,
        param_exploration: float = 0.05,
        param_lambda: float = 100.0,
        param_alpha: float = 0.98,
        sigma: np.ndarray = np.array([[0.6, 0.0], [0.0, 0.8]]),
        stage_cost_weight: np.ndarray = np.array([15.0, 10.0, 5.0]),
        terminal_cost_weight: np.ndarray = np.array([30.0, 20.0, 5.0]),
    ):
        # Device Config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rospy.loginfo(f"MPPI Controller running on: {self.device}")

        # Parameters
        self.dim_x = 4  # [x, y, yaw, v]
        self.dim_u = 2  # [steer, accel]
        self.T = horizon_step_T
        self.K = number_of_samples_K
        self.param_exploration = param_exploration
        self.param_lambda = param_lambda
        self.param_alpha = param_alpha
        self.param_gamma = self.param_lambda * (1.0 - self.param_alpha)

        # Weights & Limits (Move to Tensor)
        self.stage_cost_weight = torch.tensor(stage_cost_weight, dtype=torch.float32, device=self.device)
        self.terminal_cost_weight = torch.tensor(terminal_cost_weight, dtype=torch.float32, device=self.device)
        self.Sigma = torch.tensor(sigma, dtype=torch.float32, device=self.device)
        self.Sigma_inv = torch.linalg.inv(self.Sigma)

        self.max_steer_abs = max_steer_abs
        self.max_accel_abs = max_accel_abs
        self.delta_t = delta_t
        self.wheel_base = wheel_base

        # Reference Path (will be updated from trajectory)
        self.ref_path = None
        self.prev_waypoints_idx = 0

        # Control sequence
        self.u_prev = torch.zeros((self.T, self.dim_u), dtype=torch.float32, device=self.device)

    def set_reference_path(self, trajectory: Trajectory):
        """Update reference path from trajectory message"""
        num_points = len(trajectory.x)
        ref_path_np = np.zeros((num_points, 4))
        ref_path_np[:, 0] = trajectory.x
        ref_path_np[:, 1] = trajectory.y
        ref_path_np[:, 2] = trajectory.yaw
        ref_path_np[:, 3] = trajectory.vx
        
        self.ref_path = torch.tensor(ref_path_np, dtype=torch.float32, device=self.device)
        self.prev_waypoints_idx = 0
        rospy.loginfo(f"Reference path updated with {num_points} points")

    def calc_control_input(self, observed_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate optimal control input using MPPI
        
        Args:
            observed_x: Current state [x, y, yaw, v]
            
        Returns:
            optimal_input: [steer, accel]
            u_sequence: Full control sequence for horizon
        """
        if self.ref_path is None:
            rospy.logwarn("No reference path available")
            return np.zeros(2), np.zeros((self.T, 2))

        # Convert observation to tensor
        x0 = torch.tensor(observed_x, dtype=torch.float32, device=self.device)
        
        # Update nearest waypoint index
        prev_idx = self.prev_waypoints_idx
        self._update_nearest_waypoint_index(observed_x[0], observed_x[1])
        
        # Debug: Check reference waypoint and distance
        if self.prev_waypoints_idx < self.ref_path.shape[0]:
            ref_point = self.ref_path[self.prev_waypoints_idx].cpu().numpy()
            dist_to_ref = np.sqrt((observed_x[0] - ref_point[0])**2 + (observed_x[1] - ref_point[1])**2)
            
            if not hasattr(self, '_mppi_debug_counter'):
                self._mppi_debug_counter = 0
            self._mppi_debug_counter += 1
            
            # Log waypoint changes
            if prev_idx != self.prev_waypoints_idx:
                rospy.loginfo("[MPPI] Waypoint advanced: %d -> %d (dist=%.2f)", 
                             prev_idx, self.prev_waypoints_idx, dist_to_ref)
            
            # Simple log every iteration
            if self._mppi_debug_counter % 10 == 0:
                rospy.loginfo("[MPPI] Iter %d: wp=%d/%d, dist=%.2f, v=%.2f/%.2f",
                             self._mppi_debug_counter, self.prev_waypoints_idx,
                             self.ref_path.shape[0]-1, dist_to_ref,
                             observed_x[3], ref_point[3])
            
            # Detailed debug every 10 iterations (0.5 second at 20Hz)
            if self._mppi_debug_counter % 10 == 0:
                rospy.loginfo("=" * 80)
                rospy.loginfo("[MPPI DEBUG] Iteration %d", self._mppi_debug_counter)
                rospy.loginfo("-" * 80)
                
                # Reference state
                rospy.loginfo("Reference [%d/%d]: x=%.2f, y=%.2f, yaw=%.3f (%.1f°), v=%.2f",
                             self.prev_waypoints_idx, self.ref_path.shape[0]-1,
                             ref_point[0], ref_point[1], ref_point[2], 
                             math.degrees(ref_point[2]), ref_point[3])
                
                # Current state
                rospy.loginfo("Actual:            x=%.2f, y=%.2f, yaw=%.3f (%.1f°), v=%.2f",
                             observed_x[0], observed_x[1], observed_x[2],
                             math.degrees(observed_x[2]), observed_x[3])
                
                # Errors
                e_x = observed_x[0] - ref_point[0]
                e_y = observed_x[1] - ref_point[1]
                e_yaw = math.degrees(np.arctan2(np.sin(observed_x[2] - ref_point[2]), 
                                                np.cos(observed_x[2] - ref_point[2])))
                e_v = observed_x[3] - ref_point[3]
                
                # Calculate lateral error
                e_lat = -np.sin(ref_point[2]) * e_x + np.cos(ref_point[2]) * e_y
                
                rospy.loginfo("Error:             ex=%.3f, ey=%.3f, e_lat=%.3f, e_yaw=%.1f°, ev=%.2f",
                             e_x, e_y, e_lat, e_yaw, e_v)
                rospy.loginfo("Distance to ref:   %.3f m", dist_to_ref)
                
                # Estimate curvature at current reference point
                kappa = self._estimate_curvature(self.prev_waypoints_idx, window=2)
                steer_ref_rad = np.arctan(self.wheel_base * kappa)
                steer_ref_deg = math.degrees(steer_ref_rad)
                max_steer_deg = math.degrees(self.max_steer_abs)
                rospy.loginfo("Path curvature:    κ=%.4f, steer_ref=%.1f° (max=±%.1f°)", 
                             kappa, steer_ref_deg, max_steer_deg)

        # Sample noise: epsilon ~ (K, T, dim_u)
        import time
        t_start = time.time()
        epsilon = self._calc_epsilon(self.Sigma, self.K, self.T, self.dim_u)

        # Prepare Control Input: v = u_prev + epsilon
        v = self.u_prev.unsqueeze(0) + epsilon
        
        # Apply exploration noise
        split_idx = int((1.0 - self.param_exploration) * self.K)
        if split_idx < self.K:
            v[split_idx:, :, :] = epsilon[split_idx:, :, :]

        # Clamp inputs
        v[:, :, 0] = torch.clamp(v[:, :, 0], -self.max_steer_abs, self.max_steer_abs)
        v[:, :, 1] = torch.clamp(v[:, :, 1], -self.max_accel_abs, self.max_accel_abs)

        # Rollout (Forward Simulation)
        S = torch.zeros(self.K, dtype=torch.float32, device=self.device)
        x = x0.unsqueeze(0).repeat(self.K, 1)

        # Main Loop over Time Steps
        for t in range(self.T):
            x = self._F(x, v[:, t, :])

            # Control cost (MPPI standard formulation)
            u_t = self.u_prev[t].unsqueeze(0)
            v_t = v[:, t, :]
            temp = u_t @ self.Sigma_inv
            control_cost = self.param_gamma * (temp * v_t).sum(dim=1)

            # State cost (now includes control regularization and steering reference)
            state_cost = self._c(x, v[:, t, :])
            S += state_cost + control_cost

        # Terminal Cost
        S += self._phi(x)

        # Compute Weights
        rho = torch.min(S)
        eta = torch.sum(torch.exp((-1.0 / self.param_lambda) * (S - rho)))
        w = (1.0 / eta) * torch.exp((-1.0 / self.param_lambda) * (S - rho))
        
        # Debug: Check if negative acceleration samples have lower cost
        if hasattr(self, '_mppi_debug_counter') and self._mppi_debug_counter % 10 == 0:
            # Check first timestep acceleration distribution
            accel_samples = v[:, 0, 1]  # (K,) acceleration at t=0
            neg_accel_mask = accel_samples < 0
            pos_accel_mask = accel_samples >= 0
            
            if neg_accel_mask.sum() > 0 and pos_accel_mask.sum() > 0:
                neg_cost_mean = S[neg_accel_mask].mean().item()
                pos_cost_mean = S[pos_accel_mask].mean().item()
                neg_count = neg_accel_mask.sum().item()
                pos_count = pos_accel_mask.sum().item()
                rospy.loginfo("[MPPI ACCEL DEBUG] neg_accel: count=%d, mean_cost=%.2f | pos_accel: count=%d, mean_cost=%.2f",
                             neg_count, neg_cost_mean, pos_count, pos_cost_mean)

        # Update Control Sequence
        w_expanded = w.view(self.K, 1, 1)
        w_epsilon = torch.sum(w_expanded * epsilon, dim=0)
        w_epsilon = self._moving_average_filter(w_epsilon)

        self.u_prev = self.u_prev + w_epsilon
        
        # Shift Control Sequence
        optimal_input = self.u_prev[0].clone()
        
        # Calculate computation time
        t_end = time.time()
        comp_time_ms = (t_end - t_start) * 1000
        
        # Debug: Log control statistics
        if hasattr(self, '_mppi_debug_counter') and self._mppi_debug_counter % 10 == 0:
            rospy.loginfo("-" * 80)
            rospy.loginfo("Cost:              min=%.2f, mean=%.2f, max=%.2f",
                         S.min().item(), S.mean().item(), S.max().item())
            rospy.loginfo("Optimal control:   steer=%.3f rad (%.1f°), accel=%.2f m/s²",
                         optimal_input[0].item(), 
                         math.degrees(optimal_input[0].item()),
                         optimal_input[1].item())
            rospy.loginfo("Control sequence:  steer=[%.3f, %.3f] rad, accel=[%.3f, %.3f] m/s²",
                         self.u_prev[:, 0].min().item(), self.u_prev[:, 0].max().item(),
                         self.u_prev[:, 1].min().item(), self.u_prev[:, 1].max().item())
            rospy.loginfo("Computation time:  %.1f ms (target: <50ms for 20Hz)", comp_time_ms)
            rospy.loginfo("=" * 80)
        
        # Warning if computation is too slow
        if comp_time_ms > 50:
            rospy.logwarn_throttle(5.0, "[MPPI] Computation too slow: %.1f ms (target <50ms)", comp_time_ms)
        
        self.u_prev[:-1] = self.u_prev[1:].clone()
        self.u_prev[-1] = self.u_prev[-1].clone()

        return optimal_input.cpu().numpy(), self.u_prev.cpu().numpy()

    def _calc_epsilon(self, sigma, size_sample, size_time_step, size_dim_u):
        mean = torch.zeros(size_dim_u, device=self.device)
        dist = torch.distributions.MultivariateNormal(mean, covariance_matrix=sigma)
        return dist.sample((size_sample, size_time_step))
    
    def _lateral_error(self, x, y, ref_x, ref_y, ref_yaw):
        """
        Calculate lateral error (Frenet coordinate)
        Positive: left of path, Negative: right of path
        
        Formula: e_lat = -sin(yaw_ref) * dx + cos(yaw_ref) * dy
        where dx = x - ref_x, dy = y - ref_y
        """
        dx = x - ref_x
        dy = y - ref_y
        # Normal vector: n = [-sin(yaw_ref), cos(yaw_ref)]
        e_lat = -torch.sin(ref_yaw) * dx + torch.cos(ref_yaw) * dy
        return e_lat
    
    def _estimate_curvature(self, idx, window=2):
        """
        Estimate curvature at reference point idx
        κ ≈ dψ/ds (change in heading per unit arc length)
        
        Returns: curvature as a Python float (for compatibility)
        """
        if self.ref_path is None:
            return 0.0
        
        n = self.ref_path.shape[0]
        i0 = max(0, idx - window)
        i1 = min(n - 1, idx + window)
        
        if i0 >= i1:
            return 0.0
        
        # Calculate arc length
        dx = self.ref_path[i1, 0] - self.ref_path[i0, 0]
        dy = self.ref_path[i1, 1] - self.ref_path[i0, 1]
        ds = torch.sqrt(dx**2 + dy**2) + 1e-6
        
        # Calculate heading change
        yaw0 = self.ref_path[i0, 2]
        yaw1 = self.ref_path[i1, 2]
        dyaw = torch.atan2(torch.sin(yaw1 - yaw0), torch.cos(yaw1 - yaw0))
        
        kappa = (dyaw / ds).item()  # Convert to Python float
        return kappa

    def _F(self, x_t, v_t):
        """
        Kinematic Bicycle Model with reverse support
        
        Key change: velocity can be negative for reverse driving
        The model naturally handles reverse motion when v < 0
        """
        x, y, yaw, v = x_t[:, 0], x_t[:, 1], x_t[:, 2], x_t[:, 3]
        steer, accel = v_t[:, 0], v_t[:, 1]
        
        # Update velocity (can be negative for reverse)
        new_v = v + accel * self.delta_t
        # No clamping here - let velocity be negative for reverse
        
        # Position update: works for both forward (v>0) and reverse (v<0)
        new_x = x + new_v * torch.cos(yaw) * self.delta_t
        new_y = y + new_v * torch.sin(yaw) * self.delta_t
        
        # Heading update: same formula works for both directions
        new_yaw = yaw + new_v / self.wheel_base * torch.tan(steer) * self.delta_t
        
        return torch.stack([new_x, new_y, new_yaw, new_v], dim=1)

    def _c(self, x_t, u_t):
        """
        Stage cost with lateral error, reverse handling, and steering reference
        
        Key improvements from MPPI_parkingPNC.py:
        1. Use lateral error (Frenet coordinate)
        2. Add extra penalty for reverse segments
        3. Add steering angle reference based on path curvature
        4. Add control regularization
        """
        x, y, yaw, v = x_t[:, 0], x_t[:, 1], x_t[:, 2], x_t[:, 3]
        steer, accel = u_t[:, 0], u_t[:, 1]

        # Find nearest waypoint
        SEARCH_LEN = 200
        start_idx = self.prev_waypoints_idx
        end_idx = min(start_idx + SEARCH_LEN, self.ref_path.shape[0])
        path_seg = self.ref_path[start_idx:end_idx]
        
        if len(path_seg) == 0:
            return torch.zeros_like(x)

        dx = x.unsqueeze(1) - path_seg[:, 0].unsqueeze(0)
        dy = y.unsqueeze(1) - path_seg[:, 1].unsqueeze(0)
        d2 = dx**2 + dy**2
        min_inds = torch.argmin(d2, dim=1)
        
        # Get absolute indices in full path
        abs_inds = start_idx + min_inds
        abs_inds = torch.clamp(abs_inds, 0, self.ref_path.shape[0] - 1)
        
        ref_vals = path_seg[min_inds]
        ref_x, ref_y, ref_yaw, ref_v = ref_vals[:, 0], ref_vals[:, 1], ref_vals[:, 2], ref_vals[:, 3]

        # Use lateral error (perpendicular to path)
        e_lat = self._lateral_error(x, y, ref_x, ref_y, ref_yaw)
        
        # Heading error
        yaw_diff = torch.atan2(torch.sin(yaw - ref_yaw), torch.cos(yaw - ref_yaw))
        
        # Velocity error
        v_error = v - ref_v
        
        # Boost velocity tracking when vehicle is nearly stopped but should be moving
        # This prevents the "stuck" situation where lateral error dominates
        v_abs = torch.abs(v)
        ref_v_abs = torch.abs(ref_v)
        is_stuck = (v_abs < 0.15) & (ref_v_abs > 0.05)  # Nearly stopped but should move
        v_weight_boost = torch.where(is_stuck, 5.0, 1.0)  # 5x boost when stuck

        # Base cost: lateral error + heading error + velocity error (with boost)
        cost = self.stage_cost_weight[0] * e_lat**2 + \
               self.stage_cost_weight[1] * yaw_diff**2 + \
               self.stage_cost_weight[2] * v_weight_boost * v_error**2

        # Control regularization (penalize large control inputs)
        # Reduced weight to allow vehicle to accelerate when needed
        w_u = 0.005  # Very small weight for control regularization
        u_reg = accel**2 + 0.5 * steer**2
        cost = cost + w_u * u_reg
        
        # For reverse segments, encourage negative acceleration
        # Look ahead to determine if we should be reversing
        # Check if any nearby reference points have negative velocity
        lookahead_for_reverse = 5
        n_path = self.ref_path.shape[0]
        is_reverse_seg = torch.zeros_like(ref_v, dtype=torch.bool)
        
        for i in range(lookahead_for_reverse):
            check_idx = torch.clamp(abs_inds + i, 0, n_path - 1)
            ref_v_ahead = self.ref_path[check_idx, 3]
            is_reverse_seg = is_reverse_seg | (ref_v_ahead < -0.05)
        
        # Penalize wrong direction acceleration
        accel_direction_penalty = torch.where(
            is_reverse_seg & (accel > 0.1),  # Should reverse but accelerating forward
            3.0 * accel**2,
            torch.zeros_like(accel)
        )
        # Also encourage negative accel when should reverse
        accel_direction_bonus = torch.where(
            is_reverse_seg & (accel < -0.1),
            -0.5 * accel**2,  # Negative cost = bonus
            torch.zeros_like(accel)
        )
        cost = cost + accel_direction_penalty + accel_direction_bonus

        # Steering reference based on path curvature
        # Simplified: use a single reference for all samples (much faster)
        # Lower weight to let lateral error dominate (prevents cutting corners)
        w_delta = 2.0  # Weight for steering reference (reduced from 5.0)
        
        # Calculate steering reference for the current reference point only
        # (not for each sample, which would be too slow)
        if len(abs_inds) > 0:
            # Use the median index as representative
            median_idx = abs_inds[len(abs_inds)//2].item()
            kappa = self._estimate_curvature(median_idx, window=2)
            steer_ref = np.arctan(self.wheel_base * kappa)
            steer_ref = np.clip(steer_ref, -self.max_steer_abs, self.max_steer_abs)
            steer_ref_tensor = torch.tensor(steer_ref, device=self.device)
            
            # Apply to all samples (vectorized)
            steer_error = torch.atan2(torch.sin(steer - steer_ref_tensor), 
                                     torch.cos(steer - steer_ref_tensor))
            cost = cost + w_delta * steer_error**2

        # Extra penalty for reverse segments (ref_v < 0)
        is_reverse = ref_v < -0.01
        reverse_penalty = torch.where(
            is_reverse,
            0.5 * self.stage_cost_weight[0] * e_lat**2 + 
            0.5 * self.stage_cost_weight[1] * yaw_diff**2,
            torch.zeros_like(cost)
        )
        cost = cost + reverse_penalty

        return cost

    def _phi(self, x_T):
        """
        Terminal cost with lateral error
        
        Terminal cost emphasizes accurate final position and heading
        """
        x, y, yaw, v = x_T[:, 0], x_T[:, 1], x_T[:, 2], x_T[:, 3]

        SEARCH_LEN = 200
        start_idx = self.prev_waypoints_idx
        end_idx = min(start_idx + SEARCH_LEN, self.ref_path.shape[0])
        path_seg = self.ref_path[start_idx:end_idx]
        
        dx = x.unsqueeze(1) - path_seg[:, 0].unsqueeze(0)
        dy = y.unsqueeze(1) - path_seg[:, 1].unsqueeze(0)
        min_inds = torch.argmin(dx**2 + dy**2, dim=1)
        ref_vals = path_seg[min_inds]
        ref_x, ref_y, ref_yaw, ref_v = ref_vals[:, 0], ref_vals[:, 1], ref_vals[:, 2], ref_vals[:, 3]
        
        # Use lateral error for terminal cost as well
        e_lat = self._lateral_error(x, y, ref_x, ref_y, ref_yaw)
        
        # Heading error
        yaw_diff = torch.atan2(torch.sin(yaw - ref_yaw), torch.cos(yaw - ref_yaw))
        
        # Velocity error
        v_error = v - ref_v
        
        # Terminal cost: emphasize position and heading accuracy
        cost = self.terminal_cost_weight[0] * e_lat**2 + \
               self.terminal_cost_weight[1] * yaw_diff**2 + \
               self.terminal_cost_weight[2] * v_error**2
               
        return cost

    def _moving_average_filter(self, xx, window_size=10):
        """Moving average filter using Conv1d"""
        xx_t = xx.permute(1, 0).unsqueeze(0)
        kernel = torch.ones(1, 1, window_size, device=self.device) / window_size
        kernel = kernel.repeat(self.dim_u, 1, 1)
        pad = window_size // 2
        xx_padded = torch.nn.functional.pad(xx_t, (pad, pad), mode='replicate')
        out = torch.nn.functional.conv1d(xx_padded, kernel, groups=self.dim_u)
        if window_size % 2 == 0:
            out = out[:, :, :-1]
        return out.squeeze(0).permute(1, 0)

    def _update_nearest_waypoint_index(self, x, y):
        """
        Update nearest waypoint index - find globally nearest point
        
        This ensures the reference point is always the closest one,
        preventing issues when vehicle overshoots or gets off track.
        """
        SEARCH_LEN = 50  # Look ahead 50 points
        
        prev_idx = self.prev_waypoints_idx
        n = self.ref_path.shape[0]
        end_idx = min(prev_idx + SEARCH_LEN, n)
        
        if end_idx <= prev_idx:
            return
        
        path_seg = self.ref_path[prev_idx:end_idx]
        
        # Calculate distances to all points in search window
        dx = x - path_seg[:, 0]
        dy = y - path_seg[:, 1]
        d2 = dx**2 + dy**2
        distances = torch.sqrt(d2)
        
        # Find nearest point
        min_idx = torch.argmin(d2).item()
        min_dist = distances[min_idx].item()
        
        # Always update to nearest point
        new_idx = prev_idx + min_idx
        
        if new_idx != prev_idx:
            if abs(new_idx - prev_idx) > 2:
                rospy.loginfo("[MPPI] Waypoint jump: %d -> %d (dist=%.2f)", 
                             prev_idx, new_idx, min_dist)
            self.prev_waypoints_idx = new_idx


class MPPIControlNode:
    """MPPI Control Node"""
    
    def __init__(self):
        rospy.init_node('mppi_control_node', anonymous=False)
        
        # Load parameters
        self.load_parameters()
        
        # Initialize MPPI controller
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
        
        # State
        self.current_state = None
        self.trajectory_received = False
        
        # Publishers
        self.control_pub = rospy.Publisher('/control_cmd', ControlCmd, queue_size=10)
        
        # Subscribers
        self.state_sub = rospy.Subscriber('/vehicle_state', VehicleState, self.state_callback)
        self.traj_sub = rospy.Subscriber('/reference_state', VehicleState, self.reference_callback)
        self.planned_traj_sub = rospy.Subscriber('/planned_trajectory', Trajectory, self.trajectory_callback)
        
        # Control timer
        self.control_rate = rospy.Rate(self.publish_rate)
        
        # CSV data logging
        self._init_csv_logger()
        
        rospy.loginfo("MPPI Control Node initialized")
    
    def _init_csv_logger(self):
        """Initialize CSV file for data logging"""
        # Create logs directory if not exists
        log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Create CSV file with timestamp
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_filename = os.path.join(log_dir, f'mppi_log_{timestamp_str}.csv')
        
        # CSV header
        self.csv_header = [
            'timestamp', 'ref_x', 'ref_y', 'ref_yaw', 'ref_v',
            'act_x', 'act_y', 'act_yaw', 'act_v',
            'error_lat', 'error_yaw', 'error_v',
            'cmd_steer', 'cmd_accel', 'cmd_gear', 'comp_time_ms'
        ]
        
        # Write header
        with open(self.csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.csv_header)
        
        rospy.loginfo(f"[MPPI] CSV logging to: {self.csv_filename}")
    
    def load_parameters(self):
        """Load parameters from ROS parameter server"""
        # MPPI parameters
        self.delta_t = rospy.get_param('~delta_t', 0.1)
        self.horizon_T = rospy.get_param('~horizon_T', 10)
        self.num_samples_K = rospy.get_param('~num_samples_K', 500)
        self.exploration = rospy.get_param('~exploration', 0.15)
        self.lambda_param = rospy.get_param('~lambda', 15.0)
        self.alpha_param = rospy.get_param('~alpha', 0.85)
        
        # Noise parameters
        self.sigma_steer = rospy.get_param('~sigma_steer', 0.6)
        self.sigma_accel = rospy.get_param('~sigma_accel', 0.8)
        
        # Cost weights [lateral_error, yaw, v]
        # Changed from [x, y, yaw, v] to [e_lat, yaw, v] for better reverse handling
        # Balanced weights for smooth tracking without being too conservative
        self.stage_cost_weight = rospy.get_param('~stage_cost_weight', [20.0, 8.0, 5.0])
        self.terminal_cost_weight = rospy.get_param('~terminal_cost_weight', [40.0, 15.0, 3.0])
        
        # Vehicle parameters
        self.wheelbase = rospy.get_param('~wheelbase', 3.368)
        self.max_steer_deg = rospy.get_param('~max_steer', 45.0)
        self.max_accel = rospy.get_param('~max_accel', 2.0)
        
        # Control rate
        self.publish_rate = rospy.get_param('~publish_rate', 20.0)
        
        rospy.loginfo("Parameters loaded:")
        rospy.loginfo(f"  Horizon: T={self.horizon_T}, K={self.num_samples_K}")
        rospy.loginfo(f"  Lambda={self.lambda_param}, Alpha={self.alpha_param}")
    
    def state_callback(self, msg):
        """Callback for vehicle state"""
        self.current_state = msg
    
    def reference_callback(self, msg):
        """Callback for reference state (not used in MPPI, uses full trajectory)"""
        pass
    
    def trajectory_callback(self, msg):
        """Callback for planned trajectory"""
        self.mppi.set_reference_path(msg)
        self.trajectory_received = True
    
    def compute_control_cmd(self):
        """Compute control command using MPPI with Reverse Support"""
        if self.current_state is None or not self.trajectory_received:
            return None
        
        # 1. 准备状态向量
        observed_x = np.array([
            self.current_state.x,
            self.current_state.y,
            self.current_state.yaw,
            self.current_state.vx  # 注意：这里需要是有符号速度（前进为正，倒车为负）
        ])
        
        # 2. 获取参考点的速度，用于判断目标挡位
        # 向前看几个点来决定方向，避免在速度为0的点卡住
        if not hasattr(self, '_last_gear'):
            self._last_gear = 1  # 初始为前进
        
        target_gear = self._last_gear  # 默认保持上一次的挡位
        ref_v = 0.0
        
        if self.mppi.ref_path is not None and self.mppi.prev_waypoints_idx < self.mppi.ref_path.shape[0]:
            # 向前看最多10个点，找到第一个非零速度的点
            lookahead = 10
            n_points = self.mppi.ref_path.shape[0]
            
            for i in range(lookahead):
                idx = min(self.mppi.prev_waypoints_idx + i, n_points - 1)
                v_at_idx = self.mppi.ref_path[idx, 3].item()
                
                if abs(v_at_idx) > 0.05:  # 找到非零速度
                    ref_v = v_at_idx
                    break
            
            # 如果没找到非零速度，使用当前点
            if abs(ref_v) < 0.01:
                ref_v = self.mppi.ref_path[self.mppi.prev_waypoints_idx, 3].item()
            
            # 挡位判断 - 添加迟滞防止频繁切换
            # 只有当速度明显指向另一个方向时才切换
            if ref_v < -0.08:  # 明确需要倒车
                target_gear = -1
            elif ref_v > 0.08:  # 明确需要前进
                target_gear = 1
            # 否则保持当前挡位 (迟滞)
            
            self._last_gear = target_gear

        # 3. 计算 MPPI 优化控制量
        try:
            import time
            start_time = time.time()
            optimal_input, _ = self.mppi.calc_control_input(observed_x)
            comp_time = (time.time() - start_time) * 1000 
        except Exception as e:
            rospy.logerr(f"MPPI calculation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        # 4. 解析控制命令 (关键修改部分)
        cmd = ControlCmd()
        
        # --- 转向控制 ---
        # MPPI输出的是转向角（弧度），需要归一化到[-1, 1]
        steer_rad = optimal_input[0]
        max_steer_rad = math.radians(self.max_steer_deg)
        cmd.steer = np.clip(steer_rad / max_steer_rad, -1.0, 1.0)
        
        # Debug: Log steering details
        if not hasattr(self, '_steer_debug_counter'):
            self._steer_debug_counter = 0
        self._steer_debug_counter += 1
        if self._steer_debug_counter % 20 == 0:
            rospy.loginfo(f"[Steer] MPPI output: {steer_rad:.3f} rad ({math.degrees(steer_rad):.1f}°) -> cmd.steer: {cmd.steer:.3f}")
        
        # --- 纵向控制 (油门/刹车/换挡) ---
        accel = optimal_input[1]
        current_v = self.current_state.vx
        
        # 平滑加速度输出 (低通滤波)
        if not hasattr(self, '_last_accel'):
            self._last_accel = 0.0
        alpha_accel = 0.3  # 平滑系数，越小越平滑
        accel = alpha_accel * accel + (1 - alpha_accel) * self._last_accel
        self._last_accel = accel
        
        cmd.reverse = (target_gear == -1)
        cmd.gear = target_gear
        
        # 死区处理 - 小加速度直接忽略，减少抖动
        accel_deadzone = 0.05
        
        if target_gear == 1:  # 前进模式
            # 特殊处理：车静止且需要前进时，强制给油门启动
            if abs(current_v) < 0.1 and ref_v > 0.05:
                cmd.throttle = 0.3  # 给一个基础油门启动
                cmd.brake = 0.0
            elif accel > accel_deadzone:
                cmd.throttle = min(accel / self.max_accel, 1.0)
                cmd.brake = 0.0
            elif accel < -accel_deadzone:
                cmd.throttle = 0.0
                cmd.brake = min(-accel / self.max_accel, 1.0)
            else:
                # 死区内，保持滑行
                cmd.throttle = 0.0
                cmd.brake = 0.0
                
        else:  # 倒车模式
            # 特殊处理：车静止且需要倒车时，强制给油门启动
            if abs(current_v) < 0.1 and abs(ref_v) > 0.05:
                # 车静止但需要移动，强制给油门
                cmd.throttle = 0.3  # 给一个基础油门启动
                cmd.brake = 0.0
            elif accel < -accel_deadzone:
                cmd.throttle = min(-accel / self.max_accel, 1.0)
                cmd.brake = 0.0
            elif accel > accel_deadzone:
                cmd.throttle = 0.0
                cmd.brake = min(accel / self.max_accel, 1.0)
            else:
                cmd.throttle = 0.0
                cmd.brake = 0.0

        # 挡位切换保护 - 只在速度方向明显错误时才刹车
        if (target_gear == -1 and current_v > 0.2) or (target_gear == 1 and current_v < -0.2):
            cmd.throttle = 0.0
            cmd.brake = 0.8  # 适度刹车

        # Detailed debug logging (every 10 iterations = 0.5 second at 20Hz)
        if not hasattr(self, '_debug_counter'):
            self._debug_counter = 0
        self._debug_counter += 1
        
        if self._debug_counter % 10 == 0:
            rospy.loginfo("=" * 80)
            rospy.loginfo("[CONTROL CMD] Iteration %d", self._debug_counter)
            rospy.loginfo("-" * 80)
            rospy.loginfo("Mode:              %s (gear=%d)", 
                         'REVERSE' if cmd.reverse else 'DRIVE', target_gear)
            rospy.loginfo("Reference v:       %.2f m/s", ref_v)
            rospy.loginfo("Current v:         %.2f m/s", current_v)
            rospy.loginfo("MPPI output:       steer=%.3f rad (%.1f°), accel=%.2f m/s²",
                         steer_rad, math.degrees(steer_rad), accel)
            rospy.loginfo("Control command:   throttle=%.2f, brake=%.2f, steer=%.2f (%.1f°)",
                         cmd.throttle, cmd.brake, cmd.steer, math.degrees(cmd.steer * max_steer_rad))
            rospy.loginfo("=" * 80)

        # CSV data logging
        self._log_to_csv(observed_x, steer_rad, accel, target_gear, comp_time)

        return cmd
    
    def _log_to_csv(self, observed_x, steer_rad, accel, target_gear, comp_time):
        """Log control data to CSV file"""
        try:
            # Get reference point
            if self.mppi.ref_path is None or self.mppi.prev_waypoints_idx >= self.mppi.ref_path.shape[0]:
                return
            
            ref_point = self.mppi.ref_path[self.mppi.prev_waypoints_idx].cpu().numpy()
            
            # Calculate errors
            e_x = observed_x[0] - ref_point[0]
            e_y = observed_x[1] - ref_point[1]
            e_lat = -np.sin(ref_point[2]) * e_x + np.cos(ref_point[2]) * e_y
            e_yaw = np.arctan2(np.sin(observed_x[2] - ref_point[2]), 
                              np.cos(observed_x[2] - ref_point[2]))
            e_v = observed_x[3] - ref_point[3]
            
            # Prepare row data
            row = [
                rospy.Time.now().to_sec(),
                ref_point[0], ref_point[1], ref_point[2], ref_point[3],
                observed_x[0], observed_x[1], observed_x[2], observed_x[3],
                e_lat, e_yaw, e_v,
                steer_rad, accel, target_gear, comp_time
            ]
            
            # Write to CSV
            with open(self.csv_filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
                
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"[MPPI] CSV logging error: {e}")
    
    def run(self):
        """Main control loop"""
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
