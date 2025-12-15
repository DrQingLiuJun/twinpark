import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os

# %% 0. 全局设置与颜色定义
# 设置全局字体为 Arial
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['mathtext.fontset'] = 'cm'  # 使用 Computer Modern 渲染数学公式
rcParams['axes.unicode_minus'] = False # 解决负号显示问题

# 定义颜色 (归一化到 0-1)
COLORS = {
    'yellow': (243/255, 187/255, 20/255),
    'red':    (236/255, 68/255, 52/255),
    'blue':   (95/255, 183/255, 238/255), 
    'dark_blue': (0/255, 110/255, 188/255), 
    'green':  (0/255, 174/255, 101/255),
    'orange': (255/255, 133/255, 0/255),
    'fresh_blue': (0.2, 0.6, 0.8), 
    'alert_red':  (0.9, 0.3, 0.3), 
    'purple':     (148/255, 0/255, 211/255), # Cost 图用的紫色
    'gray':       (0.5, 0.5, 0.5)
}

# 文件路径 (请修改为你本地的实际路径)
FILE_PATH = r'/home/user/dynamic_map/src/twinpark/control_actor/vehicle_control/logs/mppi_log_20251215_122447.csv'

# %% 1. 数据读取与处理
def load_data(file_path):
    if not os.path.exists(file_path):
        print(f"警告: 文件 {file_path} 不存在，生成模拟数据用于演示...")
        N = 1000
        data = pd.DataFrame()
        # 模拟基础数据
        data['xr'] = np.linspace(-10, -27, N)
        data['yr'] = np.linspace(14, -9, N) * 0.5
        data['x_vir'] = data['xr'] + np.sin(np.linspace(0, 10, N)) * 0.1
        data['y_vir'] = data['yr'] + 0.1
        data['steer'] = np.sin(np.linspace(0, 20, N)) * 0.3
        data['accel'] = np.cos(np.linspace(0, 20, N)) * 0.5
        data['ref_v'] = np.abs(np.sin(np.linspace(0, 10, N)))
        # 模拟角速度
        data['ref_w'] = np.cos(np.linspace(0, 10, N)) * 0.5
        data['act_w'] = data['ref_w'] + np.random.normal(0, 0.05, N)
        data['e_w'] = data['ref_w'] - data['act_w']
        
        data['vir_v'] = data['ref_v'] * 0.95
        data['vir_vx'] = data['vir_v']
        data['vir_vy'] = np.zeros(N)
        data['ex'] = np.random.normal(0, 0.05, N)
        data['ey'] = np.sin(np.linspace(0, 10, N)) * 0.1
        data['eth'] = np.random.normal(0, 0.02, N)
        data['ev'] = np.random.normal(0, 0.05, N)
        data['comp_time'] = np.random.normal(35, 5, N)
        # 模拟 Cost 数据
        data['min_cost'] = 10 + np.abs(data['ey']) * 100 + np.random.normal(0, 1, N)
        data['mean_cost'] = data['min_cost'] * 1.5 + 5
        
        raw = {
            'x_ref': data['xr'].values, 'y_ref': data['yr'].values,
            'x_vir': data['x_vir'].values, 'y_vir': data['y_vir'].values,
            'vir_steer': data['steer'].values, 'vir_accel': data['accel'].values,
            'ref_v': data['ref_v'].values, 'vir_v': data['vir_v'].values,
            'ref_w': data['ref_w'].values, 'act_w': data['act_w'].values, # NEW
            'vir_vx': data['vir_vx'].values, 'vir_vy': data['vir_vy'].values,
            'e_Vx': data['ex'].values, 'e_Vy': data['ey'].values, 'e_Vth': data['eth'].values, 'e_Vv': data['ev'].values,
            'e_Vw': data['e_w'].values, # NEW
            'comp_time_ms': data['comp_time'].values,
            'min_cost': data['min_cost'].values, 'mean_cost': data['mean_cost'].values
        }
    else:
        try:
            df = pd.read_csv(file_path)
            # 检查是否包含表头 (简单检查)
            # 注意：如果CSV第一行就是数据，这里可能会报错，建议根据实际情况调整 header=None
            # 这里假设 CSV 包含标准表头，或者我们按列索引读取
            
            # 这里的索引对应你在 MPPIControlNode 中定义的顺序：
            # 0:timestamp, 1:ref_x, 2:ref_y, 3:ref_yaw, 4:ref_v, 5:ref_w
            # 6:act_x, 7:act_y, 8:act_yaw, 9:act_v, 10:act_vx, 11:act_vy, 12:act_w
            # 13:error_lon, 14:error_lat, 15:error_yaw, 16:error_v, 17:error_w
            # 18:cmd_steer, 19:cmd_accel, 20:cmd_gear, 21:comp_time_ms, 22:min_cost, 23:mean_cost
            
            # 使用 iloc 读取，更稳健
            df_vals = df.values # 转为 numpy array 处理
            
            # 如果包含表头，pandas会自动处理，df.iloc[:, i] 就是第 i 列数据
            # 如果第一行被读作 header，且正好是你要的名字，直接用名字也行
            # 这里为了保险，假设用户可能没改 header 或者 header 读取有问题，我们用 iloc
            
            raw = {
                'x_ref': df.iloc[:, 1].values, 
                'y_ref': df.iloc[:, 2].values,
                'x_vir': df.iloc[:, 6].values, # act_x
                'y_vir': df.iloc[:, 7].values, # act_y
                'ref_v': df.iloc[:, 4].values,      
                'ref_w': df.iloc[:, 5].values,  # NEW: ref_w
                'vir_v': df.iloc[:, 9].values,  # act_v
                'vir_vx': df.iloc[:, 10].values,     
                'vir_vy': df.iloc[:, 11].values, 
                'act_w': df.iloc[:, 12].values, # NEW: act_w
                'e_Vx': df.iloc[:, 13].values,  # error_lon
                'e_Vy': df.iloc[:, 14].values,  # error_lat
                'e_Vth': df.iloc[:, 15].values, # error_yaw
                'e_Vv': df.iloc[:, 16].values,  # error_v
                'e_Vw': df.iloc[:, 17].values,  # NEW: error_w
                'vir_steer': df.iloc[:, 18].values, 
                'vir_accel': df.iloc[:, 19].values, 
                'comp_time_ms': df.iloc[:, 21].values,
                'min_cost': df.iloc[:, 22].values,
                'mean_cost': df.iloc[:, 23].values
            }
        except Exception as e:
            print(f"Error reading file: {e}")
            return None

    N = len(raw['e_Vx'])
    dt = 0.05
    raw['dt'] = dt
    raw['time'] = np.arange(N) * dt
    return raw

# %% 综合评分系统
def calculate_score(val, target, limit):
    if val <= target:
        return 100.0
    elif val >= limit:
        return 0.0
    else:
        return 100.0 * (limit - val) / (limit - target)

def calculate_and_print_metrics(data):
    if data is None: return

    # --- 指标计算 ---
    # 1. 跟踪误差
    rmse_lon = np.sqrt(np.mean(data['e_Vx']**2))
    rmse_lat = np.sqrt(np.mean(data['e_Vy']**2))
    rmse_yaw = np.sqrt(np.mean(data['e_Vth']**2))
    rmse_v   = np.sqrt(np.mean(data['e_Vv']**2))
    rmse_w   = np.sqrt(np.mean(data['e_Vw']**2)) # NEW

    max_lon = np.max(np.abs(data['e_Vx']))
    max_lat = np.max(np.abs(data['e_Vy']))
    max_yaw = np.max(np.abs(data['e_Vth']))

    # 2. 终点误差 (最后 5 帧)
    term_idx = -5
    term_lon = np.mean(np.abs(data['e_Vx'][term_idx:]))
    term_lat = np.mean(np.abs(data['e_Vy'][term_idx:]))
    term_yaw = np.mean(np.abs(data['e_Vth'][term_idx:]))

    # 3. 平滑度
    dt = data['dt']
    steer_rate = np.diff(data['vir_steer']) / dt
    mean_steer_rate = np.mean(np.abs(steer_rate))
    jerk = np.diff(data['vir_accel']) / dt
    mean_jerk = np.mean(np.abs(jerk))

    # 4. 代价/稳定性
    avg_min_cost = np.mean(data['min_cost'])
    cost_variance = np.var(data['min_cost'])
    
    # 5. 实时性
    mean_time = np.mean(data['comp_time_ms'])
    max_time = np.max(data['comp_time_ms'])

    # --- 评分逻辑 (权重与阈值可调) ---
    weights = {
        'acc': 0.40,  # 过程精度
        'term': 0.30, # 终点精度
        'comf': 0.20, # 舒适度
        'stab': 0.10  # 稳定性(Cost)
    }

    # 单项打分
    s_lon  = calculate_score(rmse_lon, 0.05, 0.30)
    s_lat  = calculate_score(rmse_lat, 0.03, 0.20) 
    s_yaw  = calculate_score(rmse_yaw, 0.02, 0.10)
    
    s_tlon = calculate_score(term_lon, 0.02, 0.15)
    s_tlat = calculate_score(term_lat, 0.02, 0.10) 
    s_tyaw = calculate_score(term_yaw, 0.01, 0.08) 

    s_str  = calculate_score(mean_steer_rate, 0.5, 3.0)
    s_jerk = calculate_score(mean_jerk, 0.5, 4.0)

    s_cost = calculate_score(avg_min_cost, 50, 200)

    # 加权总分
    score_acc  = (s_lon + s_lat * 2 + s_yaw) / 4 
    score_term = (s_tlon + s_tlat * 2 + s_tyaw) / 4
    score_comf = (s_str + s_jerk) / 2
    score_stab = s_cost

    final_score = (score_acc * weights['acc'] + 
                   score_term * weights['term'] + 
                   score_comf * weights['comf'] + 
                   score_stab * weights['stab'])
    
    grade = 'S' if final_score >= 90 else 'A' if final_score >= 80 else 'B' if final_score >= 70 else 'C' if final_score >= 60 else 'D'

    # --- 打印报表 ---
    print("="*60)
    print(f"{'EXPERIMENT PERFORMANCE METRICS':^60}")
    print("="*60)
    print(f"FINAL SCORE: {final_score:5.1f} / 100   [ Grade: {grade} ]")
    print("-" * 60)
    
    print(f"[1] Tracking Accuracy (RMSE / Max)")
    print(f"  Longitudinal Error (m):   {rmse_lon:.4f} / {max_lon:.4f}")
    print(f"  Lateral Error (m):        {rmse_lat:.4f} / {max_lat:.4f}")
    print(f"  Heading Error (rad):      {rmse_yaw:.4f} / {max_yaw:.4f}  ({np.degrees(rmse_yaw):.2f}° / {np.degrees(max_yaw):.2f}°)")
    print(f"  Velocity Error (m/s):     {rmse_v:.4f}")
    print(f"  Omega Error (rad/s):      {rmse_w:.4f}") # NEW
    print("-" * 60)

    print(f"[2] Terminal Parking Accuracy (Last 5 frames)")
    print(f"  Final Lon. Error (m):     {term_lon:.4f}")
    print(f"  Final Lat. Error (m):     {term_lat:.4f}")
    print(f"  Final Yaw Error (deg):    {np.degrees(term_yaw):.4f}°")
    print("-" * 60)

    print(f"[3] Control Smoothness & Comfort")
    print(f"  Steering Smoothness (MASR): {mean_steer_rate:.4f} rad/s")
    print(f"  Ride Comfort (Mean Jerk):   {mean_jerk:.4f} m/s³")
    print("-" * 60)
    
    print(f"[4] Stability & Cost")
    print(f"  Avg Min Cost:               {avg_min_cost:.2f}")
    print(f"  Cost Variance:              {cost_variance:.2f}")
    print("-" * 60)

    print(f"[5] Real-time Performance")
    print(f"  Mean Computation Time:      {mean_time:.2f} ms")
    print(f"  Max Computation Time:       {max_time:.2f} ms")
    print("="*60)

# %% 绘图函数
def plot_trajectory(data):
    target_xlim = [-27, -10]
    target_ylim = [-9, 14.154]
    data_X_range = target_xlim[1] - target_xlim[0]
    data_Y_range = target_ylim[1] - target_ylim[0]
    dpi = 100
    fig_width = 600 / dpi
    fig_height = (600 * data_Y_range / data_X_range) / dpi
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    p1, = ax.plot(data['x_ref'], data['y_ref'], '-', color=COLORS['green'], linewidth=3, label='Reference')
    p2, = ax.plot(data['x_vir'], data['y_vir'], '-', color=COLORS['blue'], linewidth=3, label='Virtual')
    ax.plot(data['x_ref'][0], data['y_ref'][0], 'p', markersize=14, markerfacecolor=COLORS['green'], markeredgecolor=COLORS['green'], zorder=5)
    ax.plot(data['x_vir'][0], data['y_vir'][0], 'p', markersize=14, markerfacecolor=COLORS['blue'], markeredgecolor=COLORS['blue'], zorder=5)
    ax.plot(data['x_ref'][-1], data['y_ref'][-1], 'o', markersize=8, markerfacecolor=COLORS['green'], markeredgecolor=COLORS['green'], zorder=5)
    ax.plot(data['x_vir'][-1], data['y_vir'][-1], 'o', markersize=8, markerfacecolor=COLORS['blue'], markeredgecolor=COLORS['blue'], zorder=5)
    ax.set_xlabel(r'$X$ Position (m)', fontsize=22)
    ax.set_ylabel(r'$Y$ Position (m)', fontsize=22)
    ax.set_xlim(target_xlim)
    ax.set_ylim(target_ylim)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(False)
    for spine in ax.spines.values(): spine.set_linewidth(2)
    ax.tick_params(axis='both', which='major', labelsize=26, width=2, length=6)
    ax.legend(handles=[p1, p2], loc='upper right', fontsize=26, frameon=False)
    plt.tight_layout()

def plot_controls(data):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    ax1.plot(data['time'], data['vir_steer'], color=COLORS['dark_blue'], linewidth=2.5)
    ax1.grid(True)
    ax1.set_xlim([0, max(data['time'])])
    ax1.set_ylabel(r'$u_{\delta}$(rad)', fontsize=22)
    ax1.tick_params(labelsize=18, width=2)
    for spine in ax1.spines.values(): spine.set_linewidth(2)
    
    ax2.plot(data['time'], data['vir_accel'], color=COLORS['dark_blue'], linewidth=2.5)
    ax2.grid(True)
    ax2.set_xlim([0, max(data['time'])])
    ax2.set_xlabel('Time(s)', fontsize=22)
    ax2.set_ylabel(r'$u_a(\mathrm{m}/\mathrm{s}^2)$', fontsize=22)
    ax2.tick_params(labelsize=18, width=2)
    for spine in ax2.spines.values(): spine.set_linewidth(2)
    plt.tight_layout()

def plot_velocities(data):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    ax1.plot(data['time'], data['ref_v'], color=COLORS['green'], linewidth=2.5)
    ax1.grid(True)
    ax1.set_xlim([0, max(data['time'])])
    ax1.set_ylabel(r'$v_{R}$(m/s)', fontsize=22)
    ax1.tick_params(labelsize=18, width=2)
    for spine in ax1.spines.values(): spine.set_linewidth(2)

    ax2.plot(data['time'], data['vir_v'], color=COLORS['dark_blue'], linewidth=2.5, label=r'$v_{V}$')
    ax2.plot(data['time'], data['vir_vx'], color=COLORS['red'], linewidth=2.5, label=r'$v_{V_x}$')
    ax2.plot(data['time'], data['vir_vy'], color=COLORS['green'], linewidth=2.5, label=r'$v_{V_y}$')
    ax2.grid(True)
    ax2.set_xlim([0, max(data['time'])])
    ax2.set_xlabel('Time(s)', fontsize=18)
    ax2.set_ylabel(r'Velocity(m/s)', fontsize=22)
    ax2.tick_params(labelsize=18, width=2)
    for spine in ax2.spines.values(): spine.set_linewidth(2)
    ax2.legend(fontsize=16, loc='best')
    plt.tight_layout()

def plot_angular_velocities(data):
    """
    NEW: 绘制角速度对比图
    Subplot 1: Reference Omega vs Actual Omega
    Subplot 2: Omega Error
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    
    # 子图1：参考与实际角速度对比
    ax1.plot(data['time'], data['ref_w'], color=COLORS['green'], linewidth=2.5, label=r'$\omega_{ref}$')
    ax1.plot(data['time'], data['act_w'], color=COLORS['dark_blue'], linewidth=2.0, linestyle='--', label=r'$\omega_{act}$')
    ax1.grid(True)
    ax1.set_xlim([0, max(data['time'])])
    ax1.set_ylabel(r'$\omega$ (rad/s)', fontsize=22)
    ax1.tick_params(labelsize=18, width=2)
    for spine in ax1.spines.values(): spine.set_linewidth(2)
    ax1.legend(fontsize=16, loc='upper right', frameon=False)
    
    # 子图2：角速度误差
    ax2.plot(data['time'], data['e_Vw'], color=COLORS['red'], linewidth=2.5)
    ax2.grid(True)
    ax2.set_xlim([0, max(data['time'])])
    ax2.set_xlabel('Time(s)', fontsize=22)
    ax2.set_ylabel(r'$e_{\omega}$ (rad/s)', fontsize=22)
    ax2.tick_params(labelsize=18, width=2)
    for spine in ax2.spines.values(): spine.set_linewidth(2)
    
    plt.tight_layout()

def plot_errors(data):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 6))
    def setup_error_ax(ax, y_label, is_bottom=False):
        ax.grid(True)
        ax.set_xlim([0, max(data['time'])])
        ax.set_ylabel(y_label, fontsize=22)
        ax.tick_params(labelsize=18, width=2)
        for spine in ax.spines.values(): spine.set_linewidth(2)
        if is_bottom: ax.set_xlabel('Time(s)', fontsize=22)
    
    ax1.plot(data['time'], data['e_Vx'], color=COLORS['dark_blue'], linewidth=2.5)
    setup_error_ax(ax1, r'${e}_{Vx}$(m)')
    ax2.plot(data['time'], data['e_Vy'], color=COLORS['dark_blue'], linewidth=2.5)
    setup_error_ax(ax2, r'${e}_{Vy}$(m)')
    ax3.plot(data['time'], data['e_Vth'], color=COLORS['dark_blue'], linewidth=2.5)
    setup_error_ax(ax3, r'${e}_{V\theta}$ (rad)', is_bottom=True) 
    plt.tight_layout()

def plot_realtime(data):
    threshold_ms = 50
    fig = plt.figure(figsize=(10, 7), facecolor='white')
    gs = fig.add_gridspec(2, 1, hspace=0.3)
    
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(data['comp_time_ms'], color=COLORS['fresh_blue'], alpha=0.8, linewidth=2.5, label='Computation Time')
    ax1.axhline(y=threshold_ms, color=COLORS['alert_red'], linestyle='--', linewidth=2.5)
    ax1.text(0, threshold_ms, 'Critical Threshold (50ms)', color=COLORS['alert_red'], fontsize=16, va='bottom', ha='left')
    over_idx = np.where(data['comp_time_ms'] > threshold_ms)[0]
    if len(over_idx) > 0:
        ax1.plot(over_idx, data['comp_time_ms'][over_idx], '.', color=COLORS['alert_red'], markersize=8, label='Violation')
    ax1.set_ylabel('Comp. Time (ms)', fontsize=16)
    ax1.set_xlabel('Iteration Steps', fontsize=16)
    ax1.set_title('Computation Time History', fontsize=20)
    ax1.grid(True, alpha=0.15, linewidth=1)
    ax1.legend(loc='upper right', fontsize=16)
    ax1.set_xlim([0, len(data['comp_time_ms'])])
    ax1.tick_params(labelsize=16)

    ax2 = fig.add_subplot(gs[1])
    counts, bins, patches = ax2.hist(data['comp_time_ms'], bins=50, color=COLORS['fresh_blue'], alpha=0.7, edgecolor='none')
    ax2.axvline(x=threshold_ms, color=COLORS['alert_red'], linestyle='--', linewidth=2.5)
    avg_time = np.mean(data['comp_time_ms'])
    max_time = np.max(data['comp_time_ms'])
    over_rate = np.sum(data['comp_time_ms'] > threshold_ms) / len(data['comp_time_ms']) * 100
    info_str = (f"$\\bf{{Mean\\ Time:}}$ {avg_time:.2f} ms\n"
                f"$\\bf{{Max\\ Time:}}$ {max_time:.2f} ms\n"
                f"$\\bf{{Violation\\ Rate:}}$ {over_rate:.1f}%")
    ax2.text(0.95, 0.85, info_str, transform=ax2.transAxes, fontsize=16,
             ha='right', va='top', bbox=dict(facecolor='white', edgecolor='#CCCCCC', boxstyle='square,pad=0.5'))
    ax2.set_ylabel('Count', fontsize=16)
    ax2.set_xlabel('Computation Time (ms)', fontsize=16)
    ax2.set_title('Time Distribution Histogram', fontsize=18)
    ax2.grid(True, alpha=0.15, linewidth=1)
    ax2.tick_params(labelsize=16)

def plot_costs(data):
    """
    绘制 MPPI 代价收敛曲线
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(data['time'], data['mean_cost'], color=COLORS['gray'], alpha=0.6, linewidth=1.5, label='Mean Cost (Sampling Avg)')
    ax.plot(data['time'], data['min_cost'], color=COLORS['purple'], linewidth=2.5, label='Min Cost (Optimal Traj)')
    
    ax.fill_between(data['time'], data['min_cost'], data['mean_cost'], color=COLORS['purple'], alpha=0.1)

    ax.set_ylabel('MPPI Cost', fontsize=18)
    ax.set_xlabel('Time (s)', fontsize=18)
    ax.set_title('Controller Optimization Landscape', fontsize=20)
    ax.set_xlim([0, max(data['time'])])
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', fontsize=14)
    ax.tick_params(labelsize=14)
    
    plt.tight_layout()
    plt.show()

# %% 执行
if __name__ == "__main__":
    # 1. 加载数据
    data_dict = load_data(FILE_PATH)
    
    if data_dict is not None:
        # 2. 计算评分与指标
        calculate_and_print_metrics(data_dict)
        
        # 3. 绘图
        plot_trajectory(data_dict)
        plot_controls(data_dict)
        plot_velocities(data_dict)
        plot_angular_velocities(data_dict) # NEW PLOT
        plot_errors(data_dict)
        plot_realtime(data_dict)
        plot_costs(data_dict)