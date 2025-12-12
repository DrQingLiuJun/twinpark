import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os

# %% 0. 全局设置与颜色定义
# 设置全局字体为 Arial，与 MATLAB 保持一致
rcParams['font.family'] = 'Arial'
rcParams['mathtext.fontset'] = 'cm'  # 使用 Computer Modern 渲染数学公式 (类似 MATLAB 的 Latex)
rcParams['axes.unicode_minus'] = False # 解决负号显示问题

# 定义颜色 (归一化到 0-1)
COLORS = {
    'yellow': (243/255, 187/255, 20/255),
    'red':    (236/255, 68/255, 52/255),
    'blue':   (95/255, 183/255, 238/255), # 轨迹图用的浅蓝
    'dark_blue': (0/255, 110/255, 188/255), # 曲线图用的深蓝
    'green':  (0/255, 174/255, 101/255),
    'orange': (255/255, 133/255, 0/255),
    'fresh_blue': (0.2, 0.6, 0.8), # 实时性图的主色
    'alert_red':  (0.9, 0.3, 0.3), # 实时性图的警告色
    'pink_fill':  (0.96, 0.81, 0.98) # 包络填充色
}

# 文件路径 (请修改为你本地的实际路径)
FILE_PATH = r'H:\无人驾驶\论文\图片\第2篇\code\Matlab\mppi_log_20251209_201923.csv'

# %% 1. 数据读取与处理
def load_data(file_path):
    if not os.path.exists(file_path):
        print(f"警告: 文件 {file_path} 不存在，生成模拟数据用于演示...")
        # 生成模拟数据以防止代码报错
        N = 1000
        data = pd.DataFrame()
        # 模拟列
        data['xr'] = np.linspace(-10, -27, N)  # x_ref (反向模拟)
        data['yr'] = np.linspace(14, -9, N) * 0.5 # y_ref
        data['x_vir'] = data['xr'] + np.sin(np.linspace(0, 10, N)) # x_vir
        data['y_vir'] = data['yr'] + 0.5 # y_vir
        
        # 模拟控制量与其他
        data['steer'] = np.sin(np.linspace(0, 20, N)) * 0.3
        data['accel'] = np.cos(np.linspace(0, 20, N))
        data['ref_v'] = np.abs(np.sin(np.linspace(0, 10, N)))
        data['vir_v'] = data['ref_v'] * 0.9
        
        # 模拟误差
        data['ex'] = np.random.normal(0, 0.1, N)
        data['ey'] = np.sin(np.linspace(0, 10, N)) * 0.5
        data['eth'] = np.random.normal(0, 0.05, N)
        data['comp_time'] = np.random.normal(35, 10, N) + (np.random.rand(N)>0.95)*30
        
        # 映射到字典以便统一调用
        raw = {
            'x_ref': data['xr'].values, 'y_ref': data['yr'].values,
            'x_vir': data['x_vir'].values, 'y_vir': data['y_vir'].values,
            'vir_steer': data['steer'].values, 'vir_accel': data['accel'].values,
            'ref_v': data['ref_v'].values, 'vir_v': data['vir_v'].values,
            'e_Vx': data['ex'].values, 'e_Vy': data['ey'].values, 'e_Vth': data['eth'].values,
            'comp_time_ms': data['comp_time'].values
        }
    else:
        # 读取 CSV，跳过第一行 (NumHeaderLines=1)
        # 注意：pandas 读取 csv header 默认是第一行，如果你的数据第一行是说明，第二行是头，可能需要 header=1
        # 这里假设 CSV 结构是纯数据或带标准头，按 MATLAB 代码逻辑：readmatrix 跳过 1 行
        # 我们直接读取所有数据，然后按列索引提取
        df = pd.read_csv(file_path, skiprows=1, header=None)
        
        # 列索引映射 (Python 从 0 开始，MATLAB 从 1 开始)
        # xr(1), yr(2), thr(3), x1(5), y1(6)...
        # 对应 pandas 列索引：col 1, col 2 ...
        raw = {
            'x_ref': df.iloc[:, 1].values, 
            'y_ref': df.iloc[:, 2].values,
            'x_vir': df.iloc[:, 5].values,
            'y_vir': df.iloc[:, 6].values,
            'vir_steer': df.iloc[:, 12].values, # col 13
            'vir_accel': df.iloc[:, 13].values, # col 14
            'ref_v': df.iloc[:, 4].values,      # col 5
            'vir_v': df.iloc[:, 8].values,      # col 9
            'e_Vy': df.iloc[:, 9].values,       # col 10
            'e_Vth': df.iloc[:, 10].values,     # col 11
            'e_Vx': df.iloc[:, 11].values,      # col 12
            'comp_time_ms': df.iloc[:, 15].values # col 16
        }

    # 时间轴处理
    N = len(raw['e_Vx'])
    raw['time'] = np.arange(N) * (110 / (N - 1))
    
    return raw

data = load_data(FILE_PATH)

# %% 2. 图1：轨迹图 (严格锁定比例)
def plot_trajectory():
    # 范围定义
    target_xlim = [-27, -10]
    target_ylim = [-9, 14.154]
    data_X_range = target_xlim[1] - target_xlim[0]
    data_Y_range = target_ylim[1] - target_ylim[0]

    # 画布尺寸计算 (为了模拟 MATLAB 的 pixelWidth=600 和自动计算 Height)
    # Matplotlib 使用 inch (dpi=100 时, 600px = 6 inch)
    dpi = 100
    fig_width = 600 / dpi
    fig_height = (600 * data_Y_range / data_X_range) / dpi
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    
    # 绘制轨迹
    p1, = ax.plot(data['x_ref'], data['y_ref'], '-', color=COLORS['green'], linewidth=3, label='Reference')
    p2, = ax.plot(data['x_vir'], data['y_vir'], '-', color=COLORS['blue'], linewidth=3, label='Virtual')
    
    # 标记起点 (五角星)
    ax.plot(data['x_ref'][0], data['y_ref'][0], 'p', markersize=14, 
            markerfacecolor=COLORS['green'], markeredgecolor=COLORS['green'], zorder=5)
    ax.plot(data['x_vir'][0], data['y_vir'][0], 'p', markersize=14, 
            markerfacecolor=COLORS['blue'], markeredgecolor=COLORS['blue'], zorder=5)
    
    # 标记终点 (实心圆)
    ax.plot(data['x_ref'][-1], data['y_ref'][-1], 'o', markersize=8, 
            markerfacecolor=COLORS['green'], markeredgecolor=COLORS['green'], zorder=5)
    ax.plot(data['x_vir'][-1], data['y_vir'][-1], 'o', markersize=8, 
            markerfacecolor=COLORS['blue'], markeredgecolor=COLORS['blue'], zorder=5)
    
    # 坐标轴设置
    ax.set_xlabel(r'$X$ Position (m)', fontsize=22)
    ax.set_ylabel(r'$Y$ Position (m)', fontsize=22)
    
    ax.set_xlim(target_xlim)
    ax.set_ylim(target_ylim)
    
    # 严格锁定比例 (对应 set(gca, 'DataAspectRatio', [1 1 1]))
    ax.set_aspect('equal', adjustable='box')
    
    # 样式
    ax.grid(False)
    # 设置边框粗细和刻度字体
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.tick_params(axis='both', which='major', labelsize=26, width=2, length=6)
    
    # 图例
    ax.legend(handles=[p1, p2], loc='upper right', fontsize=26, frameon=False)
    
    plt.tight_layout()
    # plt.show()

# %% 3. 图2：MPPI控制量 (转向与加速度)
def plot_controls():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    
    # 子图1：转向角
    ax1.plot(data['time'], data['vir_steer'], color=COLORS['dark_blue'], linewidth=2.5)
    ax1.grid(True)
    ax1.set_xlim([0, max(data['time'])])
    ax1.set_ylabel(r'$u_{\delta}$(rad)', fontsize=22)
    ax1.tick_params(labelsize=18, width=2)
    for spine in ax1.spines.values(): spine.set_linewidth(2)
    
    # 子图2：加速度
    ax2.plot(data['time'], data['vir_accel'], color=COLORS['dark_blue'], linewidth=2.5)
    ax2.grid(True)
    ax2.set_xlim([0, max(data['time'])])
    ax2.set_xlabel('Time(s)', fontsize=22)
    ax2.set_ylabel(r'$u_a(\mathrm{m}/\mathrm{s}^2)$', fontsize=22)
    ax2.tick_params(labelsize=18, width=2)
    for spine in ax2.spines.values(): spine.set_linewidth(2)
    
    plt.tight_layout()
    # plt.show()

# %% 4. 图3：速度对比
def plot_velocities():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    
    # 子图1：参考速度
    ax1.plot(data['time'], data['ref_v'], color=COLORS['green'], linewidth=2.5)
    ax1.grid(True)
    ax1.set_xlim([0, max(data['time'])])
    ax1.set_ylabel(r'$v_{R}$(m/s)', fontsize=22)
    ax1.tick_params(labelsize=18, width=2)
    for spine in ax1.spines.values(): spine.set_linewidth(2)

    # 子图2：虚拟车速度
    ax2.plot(data['time'], data['vir_v'], color=COLORS['dark_blue'], linewidth=2.5)
    ax2.grid(True)
    ax2.set_xlim([0, max(data['time'])])
    ax2.set_xlabel('Time(s)', fontsize=18) # 注意：MATLAB 代码中这里没指定 fontsize，但通常需要
    ax2.set_ylabel(r'$v_{V}$(m/s)', fontsize=22)
    ax2.tick_params(labelsize=18, width=2)
    for spine in ax2.spines.values(): spine.set_linewidth(2)

    plt.tight_layout()
    # plt.show()

# %% 5. 图4：轨迹误差
def plot_errors():
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 6))
    
    # 通用设置函数
    def setup_error_ax(ax, y_label, is_bottom=False):
        ax.grid(True)
        ax.set_xlim([0, max(data['time'])])
        ax.set_ylabel(y_label, fontsize=22)
        ax.tick_params(labelsize=18, width=2)
        for spine in ax.spines.values(): spine.set_linewidth(2)
        if is_bottom:
            ax.set_xlabel('Time(s)', fontsize=22)
    
    # Vx Error
    ax1.plot(data['time'], data['e_Vx'], color=COLORS['dark_blue'], linewidth=2.5)
    setup_error_ax(ax1, r'${e}_{Vx}$(m)')
    
    # Vy Error
    ax2.plot(data['time'], data['e_Vy'], color=COLORS['dark_blue'], linewidth=2.5)
    setup_error_ax(ax2, r'${e}_{Vy}$(m)')
    
    # Vtheta Error
    ax3.plot(data['time'], data['e_Vth'], color=COLORS['dark_blue'], linewidth=2.5)
    setup_error_ax(ax3, r'${e}_{V\theta}$ (m)', is_bottom=True) # 注意：MATLAB 代码 label 单位写的是 m，尽管这是角度
    
    plt.tight_layout()
    # plt.show()

# %% 6. 图5：实时性分析 (折线 + 直方图)
def plot_realtime():
    threshold_ms = 50
    
    # 使用 GridSpec 模拟 tiledlayout (2行1列)
    fig = plt.figure(figsize=(10, 7), facecolor='white')
    gs = fig.add_gridspec(2, 1, hspace=0.3)
    
    # -- 子图1：耗时折线图 --
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(data['comp_time_ms'], color=COLORS['fresh_blue'], alpha=0.8, linewidth=2.5, label='Computation Time')
    
    # 阈值线
    ax1.axhline(y=threshold_ms, color=COLORS['alert_red'], linestyle='--', linewidth=2.5)
    ax1.text(0, threshold_ms, 'Critical Threshold (50ms)', color=COLORS['alert_red'], 
             fontsize=16, va='bottom', ha='left')
    
    # 超限点标记
    over_idx = np.where(data['comp_time_ms'] > threshold_ms)[0]
    if len(over_idx) > 0:
        ax1.plot(over_idx, data['comp_time_ms'][over_idx], '.', 
                 color=COLORS['alert_red'], markersize=8, label='Violation') # markersize adjusted for Python
    
    ax1.set_ylabel('Comp. Time (ms)', fontsize=16)
    ax1.set_xlabel('Iteration Steps', fontsize=16)
    ax1.set_title('Computation Time History', fontsize=20)
    ax1.grid(True, alpha=0.15, linewidth=1)
    ax1.legend(loc='upper right', fontsize=16)
    ax1.set_xlim([0, len(data['comp_time_ms'])])
    ax1.tick_params(labelsize=16)

    # -- 子图2：分布直方图 --
    ax2 = fig.add_subplot(gs[1])
    counts, bins, patches = ax2.hist(data['comp_time_ms'], bins=50, color=COLORS['fresh_blue'], 
                                     alpha=0.7, edgecolor='none')
    
    # 阈值线
    ax2.axvline(x=threshold_ms, color=COLORS['alert_red'], linestyle='--', linewidth=2.5)
    
    # 统计信息框
    avg_time = np.mean(data['comp_time_ms'])
    max_time = np.max(data['comp_time_ms'])
    over_rate = np.sum(data['comp_time_ms'] > threshold_ms) / len(data['comp_time_ms']) * 100
    
    info_str = (f"$\\bf{{Mean\\ Time:}}$ {avg_time:.2f} ms\n"
                f"$\\bf{{Max\\ Time:}}$ {max_time:.2f} ms\n"
                f"$\\bf{{Violation\\ Rate:}}$ {over_rate:.1f}%")
    
    # 添加文本框
    ax2.text(0.95, 0.85, info_str, transform=ax2.transAxes, fontsize=16,
             ha='right', va='top', bbox=dict(facecolor='white', edgecolor='#CCCCCC', boxstyle='square,pad=0.5'))
    
    ax2.set_ylabel('Count', fontsize=16)
    ax2.set_xlabel('Computation Time (ms)', fontsize=16)
    ax2.set_title('Time Distribution Histogram', fontsize=18)
    ax2.grid(True, alpha=0.15, linewidth=1)
    ax2.tick_params(labelsize=16)

    plt.show()

# %% 执行所有绘图
if __name__ == "__main__":
    plot_trajectory()
    plot_controls()
    plot_velocities()
    plot_errors()
    plot_realtime()