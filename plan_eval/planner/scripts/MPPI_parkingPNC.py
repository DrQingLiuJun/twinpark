import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# Import planning modules from the planning folder
import sys
import os
# Add planning folder to path
planning_path = os.path.join(os.path.dirname(__file__), '../planning')
sys.path.append(planning_path)

from HybridAStar.hybrid_a_star import hybrid_a_star_planning, generate_trajectory
# ===============================
# 车辆与场景参数
# ===============================
WB = 20.7   # 轴距
W  = 14.0   # 车宽

# 设计前后悬，使总车长 = 26
front_overhang = 3.0
rear_overhang  = 2.3   # WB + 3.0 + 2.3 = 26

# 以【后轴中心】为原点时，到车头/车尾的距离
LF = WB + front_overhang   # 后轴 → 车头最前端（必须 >= WB） 23.7
LB = rear_overhang         # 后轴 → 车尾最末端（>= 0） 2.3

MAX_STEER = np.deg2rad(30)     # 最大前轮转角
v_max = 30                      # cm/s
a_max = 40.0                    # 最大纵向加速度 (cm/s^2)
STEER_RATE_MAX = np.deg2rad(90) # 最大转角变化率 (rad/s)

# 碰撞泡泡（几何中心相对后轴的偏移）
SHIFT_CENTER = (LF - LB) / 2.0
BUBBLE_DIST = SHIFT_CENTER
BUBBLE_R = np.hypot((LF + LB) / 2.0, W / 2.0)

# 车辆轮廓（局部坐标系以【后轴中心】为原点）
VRX = [LF, LF, -LB, -LB, LF]
VRY = [W / 2, -W / 2, -W / 2, W / 2, W / 2]

def wrap_angle(a):
    return (a + np.pi) % (2*np.pi) - np.pi

# ===============================
# 车位绘制函数（zorder 低，让车身覆盖）
# ===============================
def plot_parking_spaces(ax):
    goal1 = [167, 210.1, np.deg2rad(0)]
    for offset in [0, -21, -42]:
        a = [goal1[0]-15, goal1[1]+9+offset]
        b = [goal1[0]+15, goal1[1]+9+offset]
        c = [goal1[0]-15, goal1[1]-9+offset]
        d = [goal1[0]+15, goal1[1]-9+offset]
        ax.plot([a[0], b[0]], [a[1], b[1]], 'g', zorder=1)
        ax.plot([b[0], d[0]], [b[1], d[1]], 'g', zorder=1)
        ax.plot([d[0], c[0]], [d[1], c[1]], 'g', zorder=1)
        ax.plot([c[0], a[0]], [c[1], a[1]], 'g', zorder=1)

    goal2 = [247.1, 210.1, np.deg2rad(0)]
    for offset in [0, -21, -42]:
        a = [goal2[0]-15, goal2[1]+9+offset]
        b = [goal2[0]+15, goal2[1]+9+offset]
        c = [goal2[0]-15, goal2[1]-9+offset]
        d = [goal2[0]+15, goal2[1]-9+offset]
        ax.plot([a[0], b[0]], [a[1], b[1]], 'g', zorder=1)
        ax.plot([b[0], d[0]], [b[1], d[1]], 'g', zorder=1)
        ax.plot([d[0], c[0]], [d[1], c[1]], 'g', zorder=1)
        ax.plot([c[0], a[0]], [c[1], a[1]], 'g', zorder=1)

# ===============================
# （新增）构造 Hybrid A* 障碍环境（基于停车场边界）
# ===============================
def build_parking_obstacles():
    # 用稀疏栅格采样边界线段，单位与坐标保持一致（cm）
    XY_RES = 5.0
    ox, oy = [], []
    # 你场景的矩形边界（和 plot_parking_spaces 对应的停车场外围通道）
    # 下面这组数值只是示范（与你之前示例一致的停车场框架）
    for x in np.arange(146.4, 195.1 + 1e-6, XY_RES):  # 上边界左段
        ox.append(x); oy.append(222.6)
    for x in np.arange(221.5, 267.4 + 1e-6, XY_RES):  # 上边界右段
        ox.append(x); oy.append(222.6)
    # for y in np.arange(152.4, 222.6 + 1e-6, XY_RES):  # 左边界
    #     ox.append(146.4); oy.append(y)
    for y in np.arange(152.4, 222.6 + 1e-6, XY_RES):  # 右边界
        ox.append(267.4); oy.append(y)
    for x in np.arange(146.4, 195.1 + 1e-6, XY_RES):  # 下边界左段
        ox.append(x); oy.append(152.4)
    for x in np.arange(221.5, 267.4 + 1e-6, XY_RES):  # 下边界右段
        ox.append(x); oy.append(152.4)
    # # 中间两道“柱子”（入口/出口竖线），形成通道
    # for y in np.arange(222.6, 244.5 + 1e-6, XY_RES):
    #     ox.extend([192.1, 221.5]); oy.extend([y, y])
    # for y in np.arange(152.4, 130.0 - 1e-6, -XY_RES):
    #     ox.extend([192.1, 221.5]); oy.extend([y, y])
    return np.array(ox), np.array(oy)

# ===============================
# 车辆运动学模型（state=[x,y,yaw,v,delta]，u=[a, d_delta]）
# ===============================
def vehicle_step(state, u, dt=0.1):
    x, y, yaw, v, delta = state
    a, d_delta = u

    a = np.clip(a, -a_max, a_max)
    d_delta = np.clip(d_delta, -STEER_RATE_MAX, STEER_RATE_MAX)

    v = np.clip(v + a * dt, -v_max, v_max)
    delta = np.clip(delta + d_delta * dt, -MAX_STEER, MAX_STEER)

    x += v * np.cos(yaw) * dt
    y += v * np.sin(yaw) * dt
    yaw += v / WB * np.tan(delta) * dt
    yaw = wrap_angle(yaw)
    return np.array([x, y, yaw, v, delta])

def lateral_error_xy(x, y, yaw_ref, rx, ry):
    """相对参考位姿 (rx,ry,yaw_ref) 的横向误差，左正右负"""
    dx, dy = x - rx, y - ry
    # 法向 n = [-sin(ψ), cos(ψ)]
    return -np.sin(yaw_ref) * dx + np.cos(yaw_ref) * dy

def estimate_curvature(ref_yaw, ref_s, i, win=2):
    """κ ≈ dψ/ds 的差分估计，用于给出参考转角"""
    i0 = max(0, i - win); i1 = min(len(ref_yaw) - 1, i + win)
    dpsi = wrap_angle(ref_yaw[i1] - ref_yaw[i0])
    ds = max(1e-6, ref_s[i1] - ref_s[i0])
    return dpsi / ds

# ===============================
# （新增）跟踪代价 & 参考索引推进
# ===============================
def track_cost(s_traj, u_traj, ref_x, ref_y, ref_yaw, ref_v, start_idx,
               ref_s=None, use_delta_ref=True,
               w_lat=8.0, w_yaw=220.0, w_v=0.05, w_u=0.02, w_term=4.0, w_delta=0.3):
    """
    状态已是后轴坐标：
      - 仅惩罚横向误差（Frenet 法向）与航向误差
      - 速度项权重小，避免“为追速度而横向飘”
      - （可选）按参考曲率给出 delta_ref 的软约束
    """
    T = s_traj.shape[0]; nref = len(ref_x)
    cost = 0.0
    for i in range(T):
        idx = min(start_idx + i, nref - 1)
        xr, yr, yaw, v, delta = s_traj[i, 0], s_traj[i, 1], s_traj[i, 2], s_traj[i, 3], s_traj[i, 4]
        rx, ry, ryaw, rv = ref_x[idx], ref_y[idx], ref_yaw[idx], ref_v[idx]

        e_lat = lateral_error_xy(xr, yr, ryaw, rx, ry)
        e_yaw = wrap_angle(yaw - ryaw)
        ev    = v - rv
        u_reg = (u_traj[i,0]**2 + 0.5*u_traj[i,1]**2)

        cost += w_lat*(e_lat**2) + w_yaw*(e_yaw**2) + w_v*(ev**2) + w_u*u_reg

        if use_delta_ref and (ref_s is not None):
            kappa = estimate_curvature(ref_yaw, ref_s, idx, win=2)
            delta_ref = np.arctan(WB * kappa)
            cost += w_delta * (wrap_angle(delta - delta_ref))**2

        # 倒车段额外加权（参考速度为负）
        if rv < -1e-3:
            cost += 0.5*w_lat*(e_lat**2) + 0.5*w_yaw*(e_yaw**2)

    # 终端项：仍用横向 + 航向
    idxT = min(start_idx + T - 1, nref - 1)
    xrT, yrT, yawT = s_traj[-1, 0], s_traj[-1, 1], s_traj[-1, 2]
    e_lat_T = lateral_error_xy(xrT, yrT, ref_yaw[idxT], ref_x[idxT], ref_y[idxT])
    e_yaw_T = wrap_angle(yawT - ref_yaw[idxT])
    cost += w_term*(e_lat_T**2 + 0.5*e_yaw_T**2)
    return cost


def update_ref_index(state, ref_x, ref_y, cur_idx, look_ahead=40):
    xr, yr = state[0], state[1]
    n = len(ref_x)
    i_end = min(n - 2, cur_idx + look_ahead)
    best_i, best_d2 = cur_idx, 1e18

    # 在 [cur_idx, i_end] 的折线段上做投影
    px, py = xr, yr
    for i in range(cur_idx, i_end + 1):
        x0, y0 = ref_x[i], ref_y[i]
        x1, y1 = ref_x[i+1], ref_y[i+1]
        vx, vy = x1 - x0, y1 - y0
        L2 = vx*vx + vy*vy + 1e-12
        t = ((px - x0)*vx + (py - y0)*vy) / L2
        t = np.clip(t, 0.0, 1.0)
        qx, qy = x0 + t*vx, y0 + t*vy
        d2 = (px - qx)**2 + (py - qy)**2
        if d2 < best_d2:
            best_d2 = d2
            # 索引前进：如果投影在段内，用 i 或 i+1 作为“下一个参考点”
            best_i = i if t < 0.5 else i + 1

    return min(best_i, n - 1)


# ===============================
# 绘制车辆（含四轮阿克曼）
# ===============================
def draw_car(ax, state, color='b'):
    """
    state: [x_r, y_r, yaw, v, delta]，(x_r, y_r) = 后轴中心。
    车身与四轮的局部坐标均以【后轴中心】为原点。
    """
    x_r, y_r, yaw = state[0], state[1], state[2]
    delta = state[4] if len(state) >= 5 else 0.0

    # 车身：顶点以【后轴中心】为原点 (VRX/VRY)
    Rb = np.array([[np.cos(yaw), -np.sin(yaw)],
                   [np.sin(yaw),  np.cos(yaw)]])
    body_local  = np.array([VRX, VRY])              # 以后轴为原点
    body_world  = (Rb @ body_local).T + np.array([x_r, y_r])
    ax.add_patch(Polygon(body_world, closed=True, fc=color, ec='k', alpha=0.6, zorder=10))

    # 轮胎参数
    wheel_L, wheel_W = 8.0, 3.0
    track = W
    half_track = track / 2.0

    # 四轮中心（以后轴为原点）
    centers_local = {
        "rl": np.array([0.0,       half_track]),
        "rr": np.array([0.0,      -half_track]),
        "fl": np.array([WB,        half_track]),
        "fr": np.array([WB,       -half_track]),
    }

    # Ackermann：由“平均前轮转角 delta”推左右前轮角
    if abs(np.tan(delta)) < 1e-6:
        delta_fl = delta_fr = delta
    else:
        R_turn  = WB / np.tan(delta)
        delta_fl = np.arctan(WB / (R_turn - half_track))
        delta_fr = np.arctan(WB / (R_turn + half_track))

    def draw_wheel(center_local, wheel_yaw, facecolor='#333', alpha=0.9):
        hwL, hwW = wheel_L / 2.0, wheel_W / 2.0
        corners = np.array([[ hwL,  hwW],
                            [ hwL,- hwW],
                            [-hwL,- hwW],
                            [-hwL,  hwW]]).T
        Rw = np.array([[np.cos(wheel_yaw), -np.sin(wheel_yaw)],
                       [np.sin(wheel_yaw),  np.cos(wheel_yaw)]])
        wheel_pts_local = (Rw @ corners).T + center_local
        wheel_pts_world = (Rb @ wheel_pts_local.T).T + np.array([x_r, y_r])
        ax.add_patch(Polygon(wheel_pts_world, closed=True, fc=facecolor, ec='k', alpha=alpha, zorder=10))

    # 后轮与车身同向（相对车体局部转角为 0）
    draw_wheel(centers_local["rl"], 0.0)
    draw_wheel(centers_local["rr"], 0.0)
    # 前轮用 Ackermann 角
    draw_wheel(centers_local["fl"], delta_fl)
    draw_wheel(centers_local["fr"], delta_fr)

def find_first_switch_idx(v):
    s = np.sign(v)
    for i in range(1, len(v)):
        if s[i] != 0 and s[i-1] != 0 and s[i] != s[i-1]:
            return i
    return None

def insert_transition_segment(ref_x, ref_y, ref_yaw, ref_v,
                              switch_idx,
                              L_trans=18.0,    # 过渡段弧长，越大右摆越明显，注意别碰墙
                              N_trans=15,      # 过渡段离散点数
                              oversteer=1.15,  # >1 表示“略多打一把”
                              v_end=0.0):      # 过渡段末速度=0（准备换挡）
    if switch_idx is None or switch_idx >= len(ref_x)-2:
        return ref_x, ref_y, ref_yaw, ref_v

    x0, y0, yaw0 = ref_x[switch_idx], ref_y[switch_idx], ref_yaw[switch_idx]

    # 倒车段的总体朝向（取后方几个点的切向，避免噪声）
    k = min(6, len(ref_x)-1-switch_idx)
    yaw_rev = wrap_angle(np.arctan2(
        ref_y[switch_idx+k] - ref_y[switch_idx],
        ref_x[switch_idx+k] - ref_x[switch_idx]
    ))
    # 希望在换挡前把车头摆到“更朝右”的方向（相对 yaw0 多转一点）
    dpsi = wrap_angle(yaw_rev - yaw0) * oversteer
    if abs(dpsi) < 1e-3:
        return ref_x, ref_y, ref_yaw, ref_v

    # 恒曲率圆弧过渡：kappa = dpsi / L
    kappa = dpsi / L_trans
    ss = np.linspace(L_trans/N_trans, L_trans, N_trans)
    xs, ys, yaws = [], [], []
    if abs(kappa) < 1e-6:
        for s in ss:
            xs.append(x0 + s*np.cos(yaw0))
            ys.append(y0 + s*np.sin(yaw0))
            yaws.append(wrap_angle(yaw0))
    else:
        for s in ss:
            psi = yaw0 + kappa*s
            x = x0 + (np.sin(psi) - np.sin(yaw0)) / kappa
            y = y0 - (np.cos(psi) - np.cos(yaw0)) / kappa
            xs.append(x); ys.append(y); yaws.append(wrap_angle(psi))

    # 速度：正向逐步降到 0，为换挡做准备
    v0 = max(8.0, abs(ref_v[switch_idx]))
    vs = np.linspace(v0, v_end, N_trans)

    # 拼接
    x_new = np.concatenate([ref_x[:switch_idx+1], np.array(xs), ref_x[switch_idx+1:]])
    y_new = np.concatenate([ref_y[:switch_idx+1], np.array(ys), ref_y[switch_idx+1:]])
    yaw_new = np.concatenate([ref_yaw[:switch_idx+1], np.array(yaws), ref_yaw[switch_idx+1:]])
    v_new = np.concatenate([ref_v[:switch_idx+1], np.array(vs), ref_v[switch_idx+1:]])
    return x_new, y_new, yaw_new, v_new

# ===============================
# MPPI/仿真参数
# ===============================
dt = 0.1
T = 30
K = 300
lambda_ = 5.0

# 给定后轴中心的位置，直接使用后轴坐标
start = np.array([207.05, 167.0, np.deg2rad(90), 0.0, 0.0])  # 后轴中心的起点
goal  = np.array([154.3, 189.1, np.deg2rad(0), 0.0, 0.0])  # 后轴中心的目标点

state = start.copy()
u_seq = np.zeros((T, 2))
traj = [state.copy()]

# ===============================
# 先：Hybrid A* 规划 + 生成参考轨迹
# ===============================
print("Planning with Hybrid A* ...")
ox, oy = build_parking_obstacles()
xy_res = 5.0
yaw_res = np.deg2rad(15.0)
path = hybrid_a_star_planning(
    [float(start[0]), float(start[1]), float(start[2])],
    [float(goal[0]),  float(goal[1]),  float(goal[2])],
    list(ox), list(oy), xy_res, yaw_res
)
if path is None:
    raise RuntimeError("Hybrid A* failed to find a path.")
traj_ref = generate_trajectory(path)  # 应返回 x_list/y_list/yaw_list/v_list
ref_x   = np.asarray(traj_ref.x_list, dtype=float)
ref_y   = np.asarray(traj_ref.y_list, dtype=float)
ref_yaw = np.asarray(traj_ref.yaw_list, dtype=float)
ref_yaw = (ref_yaw + np.pi) % (2*np.pi) - np.pi
ref_v   = np.asarray(traj_ref.v_list, dtype=float)
ref_v   = np.clip(ref_v, -v_max, v_max)
# 参考弧长（用于曲率/转角参考）
ds = np.hypot(np.diff(ref_x), np.diff(ref_y))
ref_s = np.concatenate([[0.0], np.cumsum(ds)])


# —— 新增：在换挡点前插入“右打前进”的过渡段 ——
switch_idx = find_first_switch_idx(ref_v)
if switch_idx is not None:
    ref_x, ref_y, ref_yaw, ref_v = insert_transition_segment(
        ref_x, ref_y, ref_yaw, ref_v,
        switch_idx,
        L_trans=18.0,   # 可调：15~25
        N_trans=15,     # 可调
        oversteer=1.15, # 可调：1.05~1.25
        v_end=0.0
    )


# 从当前状态附近的参考点开始
ref_idx = update_ref_index(state, ref_x, ref_y, cur_idx=0, look_ahead=50)

# ===============================
# 动态绘图
# ===============================
plt.ion()
fig, ax = plt.subplots(figsize=(7, 6))

def plot_frame(state, traj, samples, goal, ref_x, ref_y):
    ax.clear()
    plot_parking_spaces(ax)
    # 参考路径（红）
    ax.plot(ref_x, ref_y, 'r-', lw=2, label='Hybrid A* path', zorder=2)
    # 采样轨迹（灰）
    for s_traj in samples:
        ax.plot(s_traj[:, 0], s_traj[:, 1], color='gray', alpha=0.08, zorder=0.5)
    # ---- 实际轨迹（蓝）：按“后轴中心”绘制 ----
    ax.plot([s[0] for s in traj], [s[1] for s in traj], 'b-', lw=2, label='trajectory', zorder=5)

    draw_car(ax, state)
    ax.scatter(goal[0], goal[1], c='r', s=80, label='Goal', zorder=6)
    ax.set_xlim(130, 280); ax.set_ylim(130, 240)
    ax.set_aspect('equal'); ax.grid(False); ax.legend(loc='upper right')
    ax.set_title('Hybrid A* + MPPI Reverse Parking (rate-limited steering)')
    plt.pause(0.001)


# ===============================
# MPPI 主循环（跟踪参考）
# ===============================
for step_idx in range(300):
    noise = np.random.randn(K, T, 2) * np.array([20.0, np.deg2rad(30)])  # [a, d_delta]
    costs = np.zeros(K)
    sampled_trajs = []

    for k in range(K):
        s = state.copy()
        s_traj, u_traj = [], []
        for t in range(T):
            u = u_seq[t] + noise[k, t]
            u[0] = np.clip(u[0], -a_max, a_max)
            u[1] = np.clip(u[1], -STEER_RATE_MAX, STEER_RATE_MAX)
            s = vehicle_step(s, u, dt=dt)
            s_traj.append(s); u_traj.append(u)
        s_traj = np.array(s_traj); u_traj = np.array(u_traj)
        sampled_trajs.append(s_traj)
        costs[k] = track_cost(s_traj, u_traj, ref_x, ref_y, ref_yaw, ref_v,
                      start_idx=ref_idx, ref_s=ref_s, use_delta_ref=True)

    beta = np.min(costs)
    w = np.exp(-(costs - beta) / lambda_); w /= np.sum(w)
    du = np.sum(w[:, None, None] * noise, axis=0)
    u_seq += du

    u_exec = u_seq[0]
    u_exec[0] = np.clip(u_exec[0], -a_max, a_max)
    u_exec[1] = np.clip(u_exec[1], -STEER_RATE_MAX, STEER_RATE_MAX)
    state = vehicle_step(state, u_exec, dt=dt)
    traj.append(state)

    # 控制序列滚动 + 参考索引推进
    u_seq[:-1] = u_seq[1:]; u_seq[-1] = 0.0
    ref_idx = update_ref_index(state, ref_x, ref_y, cur_idx=ref_idx, look_ahead=30)

    plot_frame(state, traj, sampled_trajs, goal, ref_x, ref_y)

    # 抵达参考末端附近则退出
    done_pos = np.hypot(state[0]-ref_x[-1], state[1]-ref_y[-1]) < 2.0
    done_yaw = np.abs(wrap_angle(state[2]-ref_yaw[-1])) < np.deg2rad(5)
    if done_pos and done_yaw:
        print(f"✅ Parking successful at step {step_idx+1}")
        break

plt.ioff()
plt.show()
