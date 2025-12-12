import math
import numpy as np
import torch
from typing import Tuple
import time
from pathtracking_kbm_obav import Vehicle

class MPPIControllerForPathTrackingPyTorch():
    def __init__(
            self,
            delta_t: float = 0.05,
            wheel_base: float = 2.5,
            vehicle_width: float = 3.0,
            vehicle_length: float = 4.0,
            max_steer_abs: float = 0.523,
            max_accel_abs: float = 2.000,
            ref_path: np.ndarray = np.array([[0.0, 0.0, 0.0, 1.0]]),
            horizon_step_T: int = 30,
            number_of_samples_K: int = 1000,
            param_exploration: float = 0.0,
            param_lambda: float = 50.0,
            param_alpha: float = 1.0,
            sigma: np.ndarray = np.array([[0.5, 0.0], [0.0, 0.1]]), 
            stage_cost_weight: np.ndarray = np.array([50.0, 50.0, 1.0, 20.0]),
            terminal_cost_weight: np.ndarray = np.array([50.0, 50.0, 1.0, 20.0]),
            visualize_optimal_traj = True,
            visualze_sampled_trajs = True,
            obstacle_circles: np.ndarray = np.array([[-2.0, 1.0, 1.0]]),
            collision_safety_margin_rate: float = 1.2,
    ) -> None:
        
        # --- Device Config ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] MPPI Controller is running on: {self.device}")

        # --- Parameters ---
        self.dim_x = 4 
        self.dim_u = 2 
        self.T = horizon_step_T 
        self.K = number_of_samples_K 
        self.param_exploration = param_exploration
        self.param_lambda = param_lambda
        self.param_alpha = param_alpha
        self.param_gamma = self.param_lambda * (1.0 - (self.param_alpha))

        # --- Weights & Limits (Move to Tensor) ---
        self.stage_cost_weight = torch.tensor(stage_cost_weight, dtype=torch.float32, device=self.device)
        self.terminal_cost_weight = torch.tensor(terminal_cost_weight, dtype=torch.float32, device=self.device)
        self.Sigma = torch.tensor(sigma, dtype=torch.float32, device=self.device)
        self.Sigma_inv = torch.linalg.inv(self.Sigma) # Pre-compute inverse

        self.max_steer_abs = max_steer_abs
        self.max_accel_abs = max_accel_abs
        self.delta_t = delta_t
        self.wheel_base = wheel_base
        
        # Reference Path (Move to Tensor)
        self.ref_path = torch.tensor(ref_path, dtype=torch.float32, device=self.device)
        self.prev_waypoints_idx = 0

        # Obstacles (Move to Tensor)
        self.obstacle_circles = torch.tensor(obstacle_circles, dtype=torch.float32, device=self.device)
        self.collision_safety_margin_rate = collision_safety_margin_rate

        # Vehicle Shape for Collision Check
        self.vehicle_w = vehicle_width
        self.vehicle_l = vehicle_length
        # Pre-compute relative positions of key points (9 points)
        vw = self.vehicle_w * self.collision_safety_margin_rate
        vl = self.vehicle_l * self.collision_safety_margin_rate
        # Shape: (9, 2)
        self.vehicle_shape_base = torch.tensor([
            [-0.5*vl, 0.0], [-0.5*vl, +0.5*vw], [0.0, +0.5*vw], [+0.5*vl, +0.5*vw],
            [+0.5*vl, 0.0], [+0.5*vl, -0.5*vw], [0.0, -0.5*vw], [-0.5*vl, -0.5*vw], [0.0, 0.0]
        ], dtype=torch.float32, device=self.device)

        # Variables
        self.u_prev = torch.zeros((self.T, self.dim_u), dtype=torch.float32, device=self.device)
        self.visualize_optimal_traj = visualize_optimal_traj
        self.visualze_sampled_trajs = visualze_sampled_trajs

    def calc_control_input(self, observed_x: np.ndarray) -> Tuple[float, np.ndarray]:
        # Convert observation to tensor
        x0 = torch.tensor(observed_x, dtype=torch.float32, device=self.device)
        
        # 1. Update nearest waypoint index (CPU logic is fine for this single operation)
        self._update_nearest_waypoint_index(observed_x[0], observed_x[1])

        # 2. Sample noise: epsilon ~ (K, T, dim_u)
        epsilon = self._calc_epsilon(self.Sigma, self.K, self.T, self.dim_u)

        # 3. Prepare Control Input: v = u_prev + epsilon
        # u_prev shape (T, dim_u) -> broadcast to (K, T, dim_u)
        v = self.u_prev.unsqueeze(0) + epsilon
        
        # Apply exploration noise logic (mix exploitation and exploration)
        # Note: In pure vectorization, masking is faster than if-else loops
        split_idx = int((1.0 - self.param_exploration) * self.K)
        # For the exploration part (indices >= split_idx), v = epsilon
        if split_idx < self.K:
             v[split_idx:, :, :] = epsilon[split_idx:, :, :]

        # Clamp inputs
        v[:, :, 0] = torch.clamp(v[:, :, 0], -self.max_steer_abs, self.max_steer_abs)
        v[:, :, 1] = torch.clamp(v[:, :, 1], -self.max_accel_abs, self.max_accel_abs)

        # 4. Rollout (Forward Simulation)
        # S: State Cost (K,)
        S = torch.zeros(self.K, dtype=torch.float32, device=self.device)
        
        # Initial state for all K samples: (K, dim_x)
        x = x0.unsqueeze(0).repeat(self.K, 1)

        # Main Loop over Time Steps T
        for t in range(self.T):
            # Update State: x_{t+1} = F(x_t, v_t)
            # v[:, t, :] shape is (K, dim_u)
            x = self._F(x, v[:, t, :])

            # Calculate Stage Cost
            # Control cost: u^T * Sigma_inv * v
            # Shape: (K, 2) * (2, 2) * (K, 2)^T -> needs care
            # Efficient: (u @ Sigma_inv * v).sum(dim=1)
            u_t = self.u_prev[t].unsqueeze(0) # (1, 2)
            v_t = v[:, t, :] # (K, 2)
            
            # Control effort cost part
            # Calculating u_prev^T @ Sigma_inv @ v_t for each k
            # (1, 2) @ (2, 2) -> (1, 2)
            temp = u_t @ self.Sigma_inv 
            # (1, 2) * (K, 2) -> (K, 2) -> sum -> (K,)
            control_cost = self.param_gamma * (temp * v_t).sum(dim=1)

            # State cost + Collision cost
            state_cost = self._c(x)
            
            S += state_cost + control_cost

        # 5. Terminal Cost
        S += self._phi(x)

        # 6. Compute Weights
        # w = exp(-1/lambda * (S - rho)) / sum(...)
        rho = torch.min(S)
        eta = torch.sum(torch.exp((-1.0 / self.param_lambda) * (S - rho)))
        w = (1.0 / eta) * torch.exp((-1.0 / self.param_lambda) * (S - rho))

        # 7. Update Control Sequence
        # w: (K,) -> (K, 1, 1) for broadcasting
        w_expanded = w.view(self.K, 1, 1)
        # epsilon: (K, T, dim_u)
        # weighted_epsilon: (T, dim_u)
        w_epsilon = torch.sum(w_expanded * epsilon, dim=0)

        # Apply moving average filter (using 1D convolution for speed)
        w_epsilon = self._moving_average_filter(w_epsilon)

        self.u_prev = self.u_prev + w_epsilon
        
        # 8. Shift Control Sequence for next step
        optimal_input = self.u_prev[0].clone()
        self.u_prev[:-1] = self.u_prev[1:].clone()
        self.u_prev[-1] = self.u_prev[-1].clone() # Repeat last

        # --- Visualization Data Preparation (NumPy conversion) ---
        optimal_traj_np = np.empty(0)
        sampled_traj_list_np = np.empty(0)

        if self.visualize_optimal_traj:
            # Re-run dynamics for optimal u
            opt_x = x0.unsqueeze(0)
            opt_traj_list = []
            for t in range(self.T):
                opt_x = self._F(opt_x, self.u_prev[t].unsqueeze(0))
                opt_traj_list.append(opt_x)
            optimal_traj_np = torch.cat(opt_traj_list, dim=0).cpu().numpy()

        if self.visualze_sampled_trajs:
            # 使用PyTorch并行计算采样轨迹
            # 选择权重最高的前N个样本进行可视化（避免显示太多轨迹）
            num_vis_samples = min(200, self.K)  # 最多显示50条采样轨迹
            
            # 获取权重最高的样本索引
            _, top_indices = torch.topk(w, num_vis_samples)
            
            # 并行计算这些样本的轨迹
            # 初始化状态: (num_vis_samples, dim_x)
            sampled_x = x0.unsqueeze(0).repeat(num_vis_samples, 1)  # (num_vis_samples, dim_x)
            sampled_traj_list = []
            
            for t in range(self.T):
                # 获取对应样本的控制输入
                v_t = v[top_indices, t, :]  # (num_vis_samples, dim_u)
                # 更新状态
                sampled_x = self._F(sampled_x, v_t)  # (num_vis_samples, dim_x)
                sampled_traj_list.append(sampled_x.unsqueeze(1))  # (num_vis_samples, 1, dim_x)
            
            # 拼接所有时间步: (num_vis_samples, T, dim_x)
            sampled_traj_tensor = torch.cat(sampled_traj_list, dim=1)
            sampled_traj_list_np = sampled_traj_tensor.cpu().numpy()

        return optimal_input.cpu().numpy(), self.u_prev.cpu().numpy(), optimal_traj_np, sampled_traj_list_np

    def _calc_epsilon(self, sigma, size_sample, size_time_step, size_dim_u):
        # PyTorch multivariate normal
        mean = torch.zeros(size_dim_u, device=self.device)
        # eps shape: (K, T, 2)
        # Note: We generate independent noise for each time step
        dist = torch.distributions.MultivariateNormal(mean, covariance_matrix=sigma)
        return dist.sample((size_sample, size_time_step))

    def _F(self, x_t, v_t):
        # Batch Dynamics
        # x_t: (K, 4), v_t: (K, 2)
        x, y, yaw, v = x_t[:, 0], x_t[:, 1], x_t[:, 2], x_t[:, 3]
        steer, accel = v_t[:, 0], v_t[:, 1]
        
        # Vectorized Kinematic Bicycle Model
        new_x = x + v * torch.cos(yaw) * self.delta_t
        new_y = y + v * torch.sin(yaw) * self.delta_t
        new_yaw = yaw + v / self.wheel_base * torch.tan(steer) * self.delta_t
        new_v = v + accel * self.delta_t
        
        # Stack back: (K, 4)
        return torch.stack([new_x, new_y, new_yaw, new_v], dim=1)

    def _c(self, x_t):
        # x_t: (K, 4)
        x, y, yaw, v = x_t[:, 0], x_t[:, 1], x_t[:, 2], x_t[:, 3]
        yaw = (yaw + 2.0 * np.pi) % (2.0 * np.pi)

        # Batch Nearest Waypoint Calculation
        # We search within a window around prev_idx
        SEARCH_LEN = 200
        start_idx = self.prev_waypoints_idx
        end_idx = min(start_idx + SEARCH_LEN, self.ref_path.shape[0])
        
        # Path segment: (M, 4)
        path_seg = self.ref_path[start_idx:end_idx] 
        
        if len(path_seg) == 0:
             return torch.zeros_like(x)

        # Compute distances from all K samples to all M path points
        # x: (K,), path_x: (M,)
        # (K, 1) - (1, M) -> (K, M)
        dx = x.unsqueeze(1) - path_seg[:, 0].unsqueeze(0)
        dy = y.unsqueeze(1) - path_seg[:, 1].unsqueeze(0)
        d2 = dx**2 + dy**2 # (K, M)
        
        # Find minimum distance index for each sample
        min_inds = torch.argmin(d2, dim=1) # (K,)
        
        # Gather reference values
        # path_seg: (M, 4) -> ref_vals: (K, 4)
        ref_vals = path_seg[min_inds]
        ref_x, ref_y, ref_yaw, ref_v = ref_vals[:, 0], ref_vals[:, 1], ref_vals[:, 2], ref_vals[:, 3]

        # Angle difference
        yaw_diff = yaw - ref_yaw
        yaw_diff = torch.atan2(torch.sin(yaw_diff), torch.cos(yaw_diff))

        # Cost
        cost = self.stage_cost_weight[0] * (x - ref_x)**2 + \
               self.stage_cost_weight[1] * (y - ref_y)**2 + \
               self.stage_cost_weight[2] * (yaw_diff)**2 + \
               self.stage_cost_weight[3] * (v - ref_v)**2

        # Collision Check (Vectorized)
        collision_cost = self._is_collided_batch(x_t) * 1.0e10
        
        return cost + collision_cost

    def _phi(self, x_T):
        # Terminal cost uses the same logic as stage cost usually
        # but with different weights
        # For simplicity in this structure, calling _c but manually applying terminal weights
        # Note: To be strictly correct with the original code, we need to re-implement 
        # the component-wise calculation if weights differ significantly.
        # Assuming similar logic for brevity, just replacing weights:
        
        cost = self._c(x_T) 
        # Adjust weights scaling (Hack: _c uses stage weights, we want terminal)
        # Since _c logic is identical except weights, we can do:
        # Re-calculating cleanly is better for correctness.
        
        x, y, yaw, v = x_T[:, 0], x_T[:, 1], x_T[:, 2], x_T[:, 3]
        yaw = (yaw + 2.0 * np.pi) % (2.0 * np.pi)

        SEARCH_LEN = 200
        start_idx = self.prev_waypoints_idx
        end_idx = min(start_idx + SEARCH_LEN, self.ref_path.shape[0])
        path_seg = self.ref_path[start_idx:end_idx] 
        dx = x.unsqueeze(1) - path_seg[:, 0].unsqueeze(0)
        dy = y.unsqueeze(1) - path_seg[:, 1].unsqueeze(0)
        min_inds = torch.argmin(dx**2 + dy**2, dim=1)
        ref_vals = path_seg[min_inds]
        ref_x, ref_y, ref_yaw, ref_v = ref_vals[:, 0], ref_vals[:, 1], ref_vals[:, 2], ref_vals[:, 3]
        
        yaw_diff = torch.atan2(torch.sin(yaw-ref_yaw), torch.cos(yaw-ref_yaw))
        
        cost = self.terminal_cost_weight[0] * (x - ref_x)**2 + \
               self.terminal_cost_weight[1] * (y - ref_y)**2 + \
               self.terminal_cost_weight[2] * (yaw_diff)**2 + \
               self.terminal_cost_weight[3] * (v - ref_v)**2
               
        collision_cost = self._is_collided_batch(x_T) * 1.0e10
        return cost + collision_cost

    def _is_collided_batch(self, x_t):
        # x_t: (K, 4)
        K = x_t.shape[0]
        x, y, yaw = x_t[:, 0], x_t[:, 1], x_t[:, 2]

        # 1. Transform Vehicle Keypoints
        # vehicle_shape_base: (9, 2)
        # We need to rotate these 9 points by K different yaws
        
        # Rotation matrices: (K, 2, 2)
        c = torch.cos(yaw)
        s = torch.sin(yaw)
        # R = [[c, -s], [s, c]]
        R = torch.stack([torch.stack([c, -s], dim=1), torch.stack([s, c], dim=1)], dim=1)
        
        # Apply rotation: (K, 2, 2) @ (9, 2).T -> (K, 2, 9) -> transpose -> (K, 9, 2)
        # Easier: (K, 1, 2, 2) broadcasting?
        # Let's use einsum or manual multiplication
        # Rotated Points = R * P
        # shape_x = x * cos - y * sin
        # shape_y = x * sin + y * cos
        
        # vehicle_shape_base: (P=9, 2)
        px = self.vehicle_shape_base[:, 0] # (9,)
        py = self.vehicle_shape_base[:, 1] # (9,)
        
        # Broadcasting: (K, 1) * (1, 9) -> (K, 9)
        rot_x = px.unsqueeze(0) * c.unsqueeze(1) - py.unsqueeze(0) * s.unsqueeze(1)
        rot_y = px.unsqueeze(0) * s.unsqueeze(1) + py.unsqueeze(0) * c.unsqueeze(1)
        
        # Add translation: (K, 9) + (K, 1)
        global_x = rot_x + x.unsqueeze(1)
        global_y = rot_y + y.unsqueeze(1)
        
        # 2. Check Distances to Obstacles
        # Obstacles: (N, 3) -> x, y, r
        obs = self.obstacle_circles
        N = obs.shape[0]
        
        if N == 0:
            return torch.zeros(K, dtype=torch.bool, device=self.device)

        # Distances: check every point (K, 9) against every obstacle (N)
        # We want tensor of shape (K, 9, N)
        
        # global_x: (K, 9) -> (K, 9, 1)
        # obs_x: (N,) -> (1, 1, N)
        dx = global_x.unsqueeze(2) - obs[:, 0].view(1, 1, N)
        dy = global_y.unsqueeze(2) - obs[:, 1].view(1, 1, N)
        dist_sq = dx**2 + dy**2 # (K, 9, N)
        
        # Radii squared
        r_sq = (obs[:, 2]**2).view(1, 1, N)
        
        # Check collision: dist < r
        has_collision = dist_sq < r_sq # (K, 9, N)
        
        # If any point (dim 1) hits any obstacle (dim 2), sample is collided
        # any() over dim 2, then any() over dim 1
        collided_mask = has_collision.any(dim=2).any(dim=1) # (K,)
        
        return collided_mask.float() # Return 1.0 or 0.0

    def _moving_average_filter(self, xx, window_size=10):
        # xx: (T, dim_u)
        # Use Conv1d
        # Input: (Batch, Channel, Length) -> (1, dim_u, T)
        xx_t = xx.permute(1, 0).unsqueeze(0) 
        
        kernel = torch.ones(1, 1, window_size, device=self.device) / window_size
        # Repeat kernel for channels (groups)
        kernel = kernel.repeat(self.dim_u, 1, 1)
        
        # Padding 'same' logic manually or use padding
        pad = window_size // 2
        
        # Replicate padding for boundary conditions similar to original
        # Original code scales boundaries. Conv1d with zero padding reduces magnitude at edges.
        # To strictly match original is complex. Standard reflect/replicate padding is good enough.
        xx_padded = torch.nn.functional.pad(xx_t, (pad, pad), mode='replicate')
        
        # Conv
        out = torch.nn.functional.conv1d(xx_padded, kernel, groups=self.dim_u)
        
        # Slice to match size T
        if window_size % 2 == 0:
            out = out[:, :, :-1] # Crop extra
        
        # Back to (T, dim_u)
        return out.squeeze(0).permute(1, 0)

    def _update_nearest_waypoint_index(self, x, y):
        # Run on CPU / Single item
        SEARCH_IDX_LEN = 200
        prev_idx = self.prev_waypoints_idx
        # ref_path is tensor, bring to cpu for scalar index search or keep on gpu
        # For simplicity and given ref_path is small, GPU is fine.
        
        end_idx = min(prev_idx + SEARCH_IDX_LEN, self.ref_path.shape[0])
        path_seg = self.ref_path[prev_idx:end_idx]
        
        dx = x - path_seg[:, 0]
        dy = y - path_seg[:, 1]
        d2 = dx**2 + dy**2
        min_idx = torch.argmin(d2).item()
        
        self.prev_waypoints_idx = prev_idx + min_idx


def generate_figure8_path(num_points=500, scale=10.0, center_x=15.0, center_y=0.0):
    """
    生成"8"字形参考轨迹
    
    参数:
        num_points: 轨迹点数量
        scale: "8"字的大小缩放
        center_x: 中心点X坐标
        center_y: 中心点Y坐标
    
    返回:
        ref_path: (N, 4) numpy数组 [x, y, yaw, v]
    """
    # 使用Lemniscate曲线（伯努利双纽线）生成"8"字形
    # 参数方程: x = a*cos(t)/(1+sin^2(t)), y = a*sin(t)*cos(t)/(1+sin^2(t))
    t = np.linspace(0, 2*np.pi, num_points)
    
    # 生成"8"字形轨迹
    a = scale
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    denominator = 1 + sin_t**2
    
    x = center_x + a * cos_t / denominator
    y = center_y + a * sin_t * cos_t / denominator
    
    # 计算航向角（切线方向）
    dx = np.gradient(x)
    dy = np.gradient(y)
    yaw = np.arctan2(dy, dx)
    
    # 设置参考速度（可以根据曲率调整）
    v = np.ones_like(x) * 5.0  # 恒定速度3 m/s
    
    # 组合成 (N, 4) 数组
    ref_path = np.stack([x, y, yaw, v], axis=1)
    
    return ref_path


def generate_oval_path(num_points=500, width=30.0, height=20.0, center_x=15.0, center_y=0.0):
    """
    生成椭圆形参考轨迹
    
    参数:
        num_points: 轨迹点数量
        width: 椭圆宽度
        height: 椭圆高度
        center_x: 中心点X坐标
        center_y: 中心点Y坐标
    
    返回:
        ref_path: (N, 4) numpy数组 [x, y, yaw, v]
    """
    t = np.linspace(0, 2*np.pi, num_points)
    
    a = width / 2
    b = height / 2
    
    x = center_x + a * np.cos(t)
    y = center_y + b * np.sin(t)
    
    # 计算航向角
    dx = -a * np.sin(t)
    dy = b * np.cos(t)
    yaw = np.arctan2(dy, dx)
    
    # 参考速度
    v = np.ones_like(x) * 3.0
    
    ref_path = np.stack([x, y, yaw, v], axis=1)
    
    return ref_path


def run_simulation_mppi_pytorch():
    print("[INFO] Start simulation with PyTorch MPPI...")
    
    delta_t = 0.05
    sim_steps = 250  # 增加步数以完成"8"字形轨迹
    
    # 生成"8"字形参考轨迹
    print("[INFO] Generating figure-8 reference path...")
    ref_path = generate_figure8_path(num_points=800, scale=30.0, center_x=0.0, center_y=0.0)
    
    # 障碍物设置（根据"8"字形轨迹调整位置）
    OBSTACLE_CIRCLES = np.array([
        [15.0, +6.0, 5.0],   # 上方障碍物
        [15.0, -6.0, 2.0],   # 下方障碍物
        [20.0, 0.0, 2.0],    # 右侧障碍物
        [10.0, 0.0, 2.0],    # 左侧障碍物
    ])
    
    # 如果想使用椭圆轨迹，可以取消下面这行的注释
    # ref_path = generate_oval_path(num_points=500, width=30.0, height=20.0, center_x=15.0, center_y=0.0)
    
    # 如果想从CSV文件加载，可以取消下面这行的注释
    # ref_path = np.genfromtxt(r'H:\Code\PNC\MPPI\python_simple_mppi\data\ovalpath.csv', delimiter=',', skip_header=1)
    
    print(f"[INFO] Reference path generated with {len(ref_path)} points")
    print(f"[INFO] Path start: ({ref_path[0, 0]:.2f}, {ref_path[0, 1]:.2f}), yaw: {ref_path[0, 2]:.2f}")
    print(f"[INFO] Path range: X=[{ref_path[:, 0].min():.2f}, {ref_path[:, 0].max():.2f}], Y=[{ref_path[:, 1].min():.2f}, {ref_path[:, 1].max():.2f}]")
    
    vehicle = Vehicle(
        wheel_base=2.5,
        max_steer_abs=0.523,
        max_accel_abs=2.000,
        ref_path=ref_path[:, 0:2],
        obstacle_circles=OBSTACLE_CIRCLES,
        visualize=True 
    )
    
    # 使用轨迹起点作为初始位置
    init_x = ref_path[0, 0]
    init_y = ref_path[0, 1]
    init_yaw = ref_path[0, 2]
    init_v = 0.0
    
    print(f"[INFO] Vehicle initial state: x={init_x:.2f}, y={init_y:.2f}, yaw={init_yaw:.2f}, v={init_v:.2f}")
    vehicle.reset(init_state=np.array([init_x, init_y, init_yaw, init_v]))
    
    # Init PyTorch Controller
    mppi = MPPIControllerForPathTrackingPyTorch(
        delta_t=delta_t*2.0,
        wheel_base=2.5,
        max_steer_abs=0.523,
        max_accel_abs=2.000,
        ref_path=ref_path,
        horizon_step_T=20,    
        number_of_samples_K=1000, 
        param_exploration=0.05,
        param_lambda=100.0,
        param_alpha=0.98,
        sigma=np.array([[0.075, 0.0], [0.0, 2.0]]),
        stage_cost_weight=np.array([50.0, 50.0, 1.0, 20.0]),
        terminal_cost_weight=np.array([50.0, 50.0, 1.0, 20.0]),
        visualze_sampled_trajs=True, # Disable for speed in drawing
        obstacle_circles=OBSTACLE_CIRCLES,
        collision_safety_margin_rate=1.2,
    )

    times = []
    for i in range(sim_steps):
        start_time = time.time()
        current_state = vehicle.get_state()
        
        try:
            u, _, optimal_traj, sampled_traj = mppi.calc_control_input(current_state)
        except IndexError:
            break
            
        exec_time = time.time() - start_time
        times.append(exec_time)
        
        print(f"Step {i} | Time: {exec_time*1000:.2f}ms | Steer: {u[0]:.2f} | Accel: {u[1]:.2f}")
        
        # 处理sampled_traj的维度问题
        if sampled_traj.ndim == 3:
            sampled_traj_2d = sampled_traj[:, :, 0:2]
        else:
            sampled_traj_2d = None
            
        # 处理optimal_traj的维度问题
        if optimal_traj.ndim == 2 and optimal_traj.shape[0] > 0:
            optimal_traj_2d = optimal_traj[:, 0:2]
        else:
            optimal_traj_2d = None
            
        vehicle.update(u, delta_t, optimal_traj=optimal_traj_2d, sampled_traj_list=sampled_traj_2d)

    print(f"Average MPPI Comp. Time: {np.mean(times)*1000:.2f} ms")
    vehicle.save_animation("mppi_pytorch_demo.mp4", interval=int(delta_t * 1000), movie_writer="ffmpeg")
    print("[INFO] 动画已保存。")

if __name__ == "__main__":
    run_simulation_mppi_pytorch()