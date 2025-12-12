import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
import os

# %% 0. 全局设置与颜色定义
# 设置全局字体为 Arial
rcParams['font.family'] = 'Arial'
rcParams['mathtext.fontset'] = 'cm'
rcParams['axes.unicode_minus'] = False

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
}

# 文件路径
FILE_PATH = r'H:\无人驾驶\论文\图片\第2篇\code\Matlab\mppi_log_20251209_201923.csv'

# %% 1. 数据读取与处理
def load_data(file_path):
    if not os.path.exists(file_path):
        print(f"警告: 文件 {file_path} 不存在，生成模拟数据用于演示...")
        N = 1000
        data = pd.DataFrame()
        data['xr'] = np.linspace(-10, -27, N)
        data['yr'] = np.linspace(14, -9, N) * 0.5
        data['x_vir'] = data['xr'] + np.sin(np.linspace(0, 10, N))
        data['y_vir'] = data['yr'] + 0.5
        data['steer'] = np.sin(np.linspace(0, 20, N)) * 0.3
        data['accel'] = np.cos(np.linspace(0, 20, N))
        data['ref_v'] = np.abs(np.sin(np.linspace(0, 10, N)))
        data['vir_v'] = data['ref_v'] * 0.9
        data['ex'] = np.random.normal(0, 0.1, N)
        data['ey'] = np.sin(np.linspace(0, 10, N)) * 0.5
        data['eth'] = np.random.normal(0, 0.05, N)
        data['comp_time'] = np.random.normal(35, 10, N) + (np.random.rand(N)>0.95)*30
        
        raw = {
            'x_ref': data['xr'].values, 'y_ref': data['yr'].values,
            'x_vir': data['x_vir'].values, 'y_vir': data['y_vir'].values,
            'vir_steer': data['steer'].values, 'vir_accel': data['accel'].values,
            'ref_v': data['ref_v'].values, 'vir_v': data['vir_v'].values,
            'e_Vx': data['ex'].values, 'e_Vy': data['ey'].values, 'e_Vth': data['eth'].values,
            'comp_time_ms': data['comp_time'].values
        }
    else:
        df = pd.read_csv(file_path, skiprows=1, header=None)
        raw = {
            'x_ref': df.iloc[:, 1].values, 
            'y_ref': df.iloc[:, 2].values,
            'x_vir': df.iloc[:, 5].values,
            'y_vir': df.iloc[:, 6].values,
            'vir_steer': df.iloc[:, 12].values,
            'vir_accel': df.iloc[:, 13].values,
            'ref_v': df.iloc[:, 4].values,
            'vir_v': df.iloc[:, 8].values,
            'e_Vy': df.iloc[:, 9].values,
            'e_Vth': df.iloc[:, 10].values,
            'e_Vx': df.iloc[:, 11].values,
            'comp_time_ms': df.iloc[:, 15].values
        }

    N = len(raw['e_Vx'])
    raw['time'] = np.arange(N) * (110 / (N - 1))
    return raw

data = load_data(FILE_PATH)

# %% 绘图辅助函数
def set_ax_style(ax, fontsize=8):
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.tick_params(labelsize=fontsize, width=1)
    for spine in ax.spines.values():
        spine.set_linewidth(1)

# %% 2. 绘制各个子图
def draw_trajectory(ax):
    target_xlim = [-27, -10]
    target_ylim = [-9, 14.154]
    
    p1, = ax.plot(data['x_ref'], data['y_ref'], '-', color=COLORS['green'], linewidth=2, label='Reference')
    p2, = ax.plot(data['x_vir'], data['y_vir'], '-', color=COLORS['blue'], linewidth=2, label='Virtual')
    
    ax.plot(data['x_ref'][0], data['y_ref'][0], 'p', markersize=10, 
            markerfacecolor=COLORS['green'], markeredgecolor=COLORS['green'], zorder=5)
    ax.plot(data['x_vir'][0], data['y_vir'][0], 'p', markersize=10, 
            markerfacecolor=COLORS['blue'], markeredgecolor=COLORS['blue'], zorder=5)
    ax.plot(data['x_ref'][-1], data['y_ref'][-1], 'o', markersize=6, 
            markerfacecolor=COLORS['green'], markeredgecolor=COLORS['green'], zorder=5)
    ax.plot(data['x_vir'][-1], data['y_vir'][-1], 'o', markersize=6, 
            markerfacecolor=COLORS['blue'], markeredgecolor=COLORS['blue'], zorder=5)
    
    ax.set_xlabel(r'$X$ Position (m)', fontsize=9)
    ax.set_ylabel(r'$Y$ Position (m)', fontsize=9)
    ax.set_xlim(target_xlim)
    ax.set_ylim(target_ylim)
    ax.set_aspect('equal', adjustable='box')
    
    ax.grid(False)
    for spine in ax.spines.values(): spine.set_linewidth(1)
    ax.tick_params(labelsize=8)
    ax.legend(handles=[p1, p2], loc='upper right', fontsize=8, frameon=False)
    ax.set_title('(A) Global Trajectory Tracking', loc='left', fontsize=10, fontweight='bold')

def draw_velocities(ax):
    ax.plot(data['time'], data['ref_v'], color=COLORS['green'], linewidth=1.5, label=r'$v_R$')
    ax.plot(data['time'], data['vir_v'], '--', color=COLORS['dark_blue'], linewidth=1.5, label=r'$v_V$')
    set_ax_style(ax)
    ax.set_ylabel(r'Velocity (m/s)', fontsize=9)
    ax.set_xlim([0, max(data['time'])])
    ax.set_xticklabels([]) # 隐藏x轴标签
    ax.legend(loc='upper right', fontsize=7, frameon=False, ncol=2)
    ax.set_title('(B) Tracking Performance & Controls', loc='left', fontsize=10, fontweight='bold')

def draw_errors(ax):
    # 双Y轴绘制: 左轴位置误差, 右轴角度误差
    l1, = ax.plot(data['time'], data['e_Vy'], color=COLORS['dark_blue'], linewidth=1.5, label=r'$e_{Vy}$')
    
    ax2 = ax.twinx()
    l2, = ax2.plot(data['time'], data['e_Vth'], ':', color=COLORS['orange'], linewidth=1.5, label=r'$e_{V\theta}$')
    
    set_ax_style(ax)
    ax.set_ylabel(r'Lat. Error (m)', fontsize=9, color=COLORS['dark_blue'])
    ax2.set_ylabel(r'Head. Error (rad)', fontsize=9, color=COLORS['orange'])
    
    ax2.spines['right'].set_color(COLORS['orange'])
    ax2.spines['left'].set_color(COLORS['dark_blue'])
    ax2.tick_params(axis='y', colors=COLORS['orange'], labelsize=8)
    ax.tick_params(axis='y', colors=COLORS['dark_blue'], labelsize=8)
    
    ax.set_xlim([0, max(data['time'])])
    ax.set_xticklabels([]) # 隐藏x轴标签
    
    # 合并图例
    lines = [l1, l2]
    ax.legend(lines, [l.get_label() for l in lines], loc='upper right', fontsize=7, frameon=False, ncol=2)

def draw_controls(ax):
    # 左轴加速度，右轴转向
    l1, = ax.plot(data['time'], data['vir_accel'], color=COLORS['dark_blue'], linewidth=1.5, label=r'$u_a$')
    
    ax2 = ax.twinx()
    l2, = ax2.plot(data['time'], data['vir_steer'], '--', color=COLORS['red'], linewidth=1.5, label=r'$u_{\delta}$')
    
    set_ax_style(ax)
    ax.set_ylabel(r'Accel ($m/s^2$)', fontsize=9, color=COLORS['dark_blue'])
    ax2.set_ylabel(r'Steer (rad)', fontsize=9, color=COLORS['red'])
    
    ax2.spines['right'].set_color(COLORS['red'])
    ax2.spines['left'].set_color(COLORS['dark_blue'])
    ax2.tick_params(axis='y', colors=COLORS['red'], labelsize=8)
    ax.tick_params(axis='y', colors=COLORS['dark_blue'], labelsize=8)
    
    ax.set_xlim([0, max(data['time'])])
    ax.set_xlabel('Time (s)', fontsize=9)
    
    # 合并图例
    lines = [l1, l2]
    ax.legend(lines, [l.get_label() for l in lines], loc='upper right', fontsize=7, frameon=False, ncol=2)

def draw_realtime_history(ax):
    threshold_ms = 50
    ax.plot(data['comp_time_ms'], color=COLORS['fresh_blue'], alpha=0.9, linewidth=1, label='Comp. Time')
    ax.axhline(y=threshold_ms, color=COLORS['alert_red'], linestyle='--', linewidth=1.5)
    
    over_idx = np.where(data['comp_time_ms'] > threshold_ms)[0]
    if len(over_idx) > 0:
        ax.plot(over_idx, data['comp_time_ms'][over_idx], '.', 
                 color=COLORS['alert_red'], markersize=4)
    
    set_ax_style(ax)
    ax.set_ylabel('Time (ms)', fontsize=9)
    ax.set_xlabel('Steps', fontsize=9)
    ax.set_xlim([0, len(data['comp_time_ms'])])
    ax.set_title('(C) Real-time Performance', loc='left', fontsize=10, fontweight='bold')
    ax.text(0.02, 0.9, 'Threshold (50ms)', transform=ax.transAxes, color=COLORS['alert_red'], fontsize=8)

def draw_realtime_hist(ax):
    threshold_ms = 50
    counts, bins, patches = ax.hist(data['comp_time_ms'], bins=40, color=COLORS['fresh_blue'], 
                                     alpha=0.7, edgecolor='none')
    ax.axvline(x=threshold_ms, color=COLORS['alert_red'], linestyle='--', linewidth=1.5)
    
    avg_time = np.mean(data['comp_time_ms'])
    max_time = np.max(data['comp_time_ms'])
    over_rate = np.sum(data['comp_time_ms'] > threshold_ms) / len(data['comp_time_ms']) * 100
    
    info_str = (f"Mean: {avg_time:.1f} ms\n"
                f"Max: {max_time:.1f} ms\n"
                f"Viol.: {over_rate:.1f}%")
    
    ax.text(0.95, 0.85, info_str, transform=ax.transAxes, fontsize=8,
             ha='right', va='top', bbox=dict(facecolor='white', edgecolor='#CCCCCC', boxstyle='square,pad=0.2'))
    
    set_ax_style(ax)
    ax.set_ylabel('Count', fontsize=9)
    ax.set_xlabel('Time (ms)', fontsize=9)
    ax.set_title('(D) Distribution', loc='left', fontsize=10, fontweight='bold')

# %% 3. 主图布局与绘制
def plot_comprehensive_figure():
    # 18.2cm = 7.16 inch
    # 高度自适应，这里设为 9 英寸
    fig = plt.figure(figsize=(7.16, 9), dpi=150)
    
    # 定义网格: 4行，2列
    # 高度比例: 轨迹图占大头(3)，中间三个堆叠图各占1，底部实时性占1.2
    gs = gridspec.GridSpec(4, 2, height_ratios=[2.5, 1, 1, 1], hspace=0.35, wspace=0.25)
    
    # 1. 轨迹图 (占据第一行的两列，或者留空右边放Sim图)
    # 这里我们让它占据整个宽度，因为你还没放Sim图
    ax_traj = fig.add_subplot(gs[0, :])
    draw_trajectory(ax_traj)
    
    # 2. 中间堆叠图 (Velocity, Errors, Controls)
    # 我们将这些图放在第二行到第三行的区域
    # 为了堆叠紧凑，我们使用嵌套 GridSpec
    gs_mid = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1:3, :], hspace=0.1)
    
    ax_vel = fig.add_subplot(gs_mid[0])
    ax_err = fig.add_subplot(gs_mid[1])
    ax_ctrl = fig.add_subplot(gs_mid[2])
    
    draw_velocities(ax_vel)
    draw_errors(ax_err)
    draw_controls(ax_ctrl)
    
    # 3. 底部实时性分析
    ax_hist_line = fig.add_subplot(gs[3, 0])
    ax_hist_dist = fig.add_subplot(gs[3, 1])
    
    draw_realtime_history(ax_hist_line)
    draw_realtime_hist(ax_hist_dist)
    
    plt.show()

# %% 执行
if __name__ == "__main__":
    plot_comprehensive_figure()