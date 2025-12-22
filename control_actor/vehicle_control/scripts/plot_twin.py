#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os

# 0. 全局设置与颜色定义

# 设置全局字体
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['mathtext.fontset'] = 'cm'
rcParams['axes.unicode_minus'] = False

# 定义颜色 (归一化到 0-1)
COLORS = {
    'yellow': (243/255, 187/255, 20/255),
    'red':    (236/255, 68/255, 52/255),    # 物理车颜色
    'blue':   (95/255, 183/255, 238/255),   # 虚拟车颜色
    'dark_blue': (0/255, 110/255, 188/255), 
    'green':  (0/255, 174/255, 101/255),    # 参考轨迹颜色
    'orange': (255/255, 133/255, 0/255),
    'fresh_blue': (0.2, 0.6, 0.8), 
    'alert_red':  (0.9, 0.3, 0.3), 
    'purple':     (148/255, 0/255, 211/255),
    'gray':       (0.5, 0.5, 0.5)
}

# 文件路径 (请修改为你本地的实际路径)
FILE_PATH = r'/home/qingliu/TwinPark/twinpark_ws/src/twinpark/control_actor/vehicle_control/logs/velocity_mppi_log_20251222_230340_best.csv'

# %% 线性映射函数 (User Defined)
def get_mapping_params(lowerLeftSrc, upperRightSrc, lowerLeftDst, upperRightDst):
    """
    计算线性映射的参数 sigma 和 translate
    """
    src_width = upperRightSrc[0] - lowerLeftSrc[0]
    src_height = upperRightSrc[1] - lowerLeftSrc[1]
    dst_width = upperRightDst[0] - lowerLeftDst[0]
    dst_height = upperRightDst[1] - lowerLeftDst[1]
    
    sigma_x = dst_width / src_width
    sigma_y = dst_height / src_height

    translate_x = lowerLeftDst[0] - (lowerLeftSrc[0] * sigma_x)
    translate_y = lowerLeftDst[1] - (lowerLeftSrc[1] * sigma_y)
    
    return sigma_x, sigma_y, translate_x, translate_y

def apply_mapping(x_arr, y_arr, params):
    """
    应用映射到数组
    """
    sigma_x, sigma_y, trans_x, trans_y = params
    mapped_x = x_arr * sigma_x + trans_x
    mapped_y = y_arr * sigma_y + trans_y
    return mapped_x, mapped_y


# %% 1. 数据读取与处理
def load_data(file_path):
    if not os.path.exists(file_path):
        print(f"警告: 文件 {file_path} 不存在")
        return None
    
    try:
        df = pd.read_csv(file_path)

        # 1. 获取原始数据
        raw_ref_x = df.iloc[:, 1].values
        raw_ref_y = df.iloc[:, 2].values
        raw_vir_x = df.iloc[:, 6].values
        raw_vir_y = df.iloc[:, 7].values

        # 2. 定义映射参数 (用户提供)
        src_min = [-24.1, 10.4]
        src_max = [-12.9, -6.7]
        dst_min = [-1.631, 0.527]
        dst_max = [-0.838, -0.546]

        # 3. 计算映射系数
        map_params = get_mapping_params(src_min, src_max, dst_min, dst_max)
        sigma_x, sigma_y, trans_x, trans_y = map_params
        
        # 4. 执行映射 (将 Ref 和 Vir 映射到 Phy 所在的坐标系)
        map_ref_x, map_ref_y = apply_mapping(raw_ref_x, raw_ref_y, map_params)
        map_vir_x, map_vir_y = apply_mapping(raw_vir_x, raw_vir_y, map_params)
        
        raw = {
            # --- Reference (共用) ---
            'ref_x': map_ref_x, 
            'ref_y': map_ref_y,
            'ref_v': df.iloc[:, 4].values * (sigma_x + sigma_y) / 2.0,      
            'ref_w': df.iloc[:, 5].values,

            # --- Virtual Car (Blue) ---
            'vir_x': map_vir_x,
            'vir_y': map_vir_y,
            'vir_v': df.iloc[:, 9].values * (sigma_x + sigma_y) / 2.0,
            'vir_vx': df.iloc[:, 10].values * (sigma_x + sigma_y) / 2.0,     
            'vir_vy': df.iloc[:, 11].values * (sigma_x + sigma_y) / 2.0, 
            'vir_w': df.iloc[:, 12].values, # actual omega
            
            'e_Vx': df.iloc[:, 13].values * sigma_x,
            'e_Vy': df.iloc[:, 14].values * sigma_y,
            'e_Vth': df.iloc[:, 15].values,
            'e_Vv': df.iloc[:, 16].values * (sigma_x + sigma_y) / 2.0,
            'e_Vw': df.iloc[:, 17].values,
            
            'u_Vsteer': df.iloc[:, 18].values, 
            'u_Vv': df.iloc[:, 19].values * 0.1, 

            # --- System Stats ---
            'comp_time_ms': df.iloc[:, 21].values,
            'min_cost': df.iloc[:, 22].values,
            'mean_cost': df.iloc[:, 23].values,

            # --- Physical Car (Red) [New Data] ---
            'phy_x': df.iloc[:, 29].values,
            'phy_y': df.iloc[:, 30].values,
            'phy_yaw': df.iloc[:, 31].values,
            'phy_v': df.iloc[:, 32].values,
            'phy_w': df.iloc[:, 33].values,

            'e_Px': df.iloc[:, 34].values,
            'e_Py': df.iloc[:, 35].values,
            'e_Pyaw': df.iloc[:, 36].values,
            'e_Pv': df.iloc[:, 37].values,
            'e_Pw': df.iloc[:, 38].values,

            'u_Pv': df.iloc[:, 39].values, # Physical Velocity Cmd
            'u_Pw': df.iloc[:, 40].values  # Physical Omega/Steer Cmd
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

    # --- Virtual Metrics ---
    rmse_lon = np.sqrt(np.mean(data['e_Vx']**2))
    rmse_lat = np.sqrt(np.mean(data['e_Vy']**2))
    rmse_yaw = np.sqrt(np.mean(data['e_Vth']**2))
    
    # --- Physical Metrics ---
    rmse_phy_lon = np.sqrt(np.mean(data['e_Px']**2))
    rmse_phy_lat = np.sqrt(np.mean(data['e_Py']**2))
    rmse_phy_yaw = np.sqrt(np.mean(data['e_Pyaw']**2))

    max_lon = np.max(np.abs(data['e_Vx']))
    max_phy_lon = np.max(np.abs(data['e_Px']))

    # 平滑度 (基于 Virtual 控制输出)
    dt = data['dt']
    steer_rate = np.diff(data['u_Vsteer']) / dt
    mean_steer_rate = np.mean(np.abs(steer_rate))
    
    # Cost
    avg_min_cost = np.mean(data['min_cost'])
    
    # 简单的打分逻辑 (基于 Virtual)
    s_lon  = calculate_score(rmse_lon, 0.05, 0.30)
    s_lat  = calculate_score(rmse_lat, 0.03, 0.20) 
    final_score = (s_lon + s_lat)/2 # 简化

    print("="*60)
    print(f"{'EXPERIMENT PERFORMANCE METRICS':^60}")
    print("="*60)
    
    print(f"[1] Virtual Car Tracking Accuracy (Blue)")
    print(f"  RMSE Lon: {rmse_lon:.4f} m, RMSE Lat: {rmse_lat:.4f} m, RMSE Yaw: {rmse_yaw:.4f} rad")
    print("-" * 60)
    
    print(f"[2] Physical Car Tracking Accuracy (Red)")
    print(f"  RMSE Lon: {rmse_phy_lon:.4f} m, RMSE Lat: {rmse_phy_lat:.4f} m, RMSE Yaw: {rmse_phy_yaw:.4f} rad")
    print(f"  Max Lon Error: {max_phy_lon:.4f} m")
    print("-" * 60)

    print(f"[3] System Stats")
    print(f"  Avg Min Cost: {avg_min_cost:.2f}")
    print(f"  Mean Comp Time: {np.mean(data['comp_time_ms']):.2f} ms")
    print("="*60)

def mapping(self, position, lowerLeftSrc, upperRightSrc, lowerLeftDst, upperRightDst):
        src_width = upperRightSrc[0] - lowerLeftSrc[0]
        src_height = upperRightSrc[1] - lowerLeftSrc[1]
        dst_width = upperRightDst[0] - lowerLeftDst[0]
        dst_height = upperRightDst[1] - lowerLeftDst[1]
        
        sigma_x = dst_width / src_width
        sigma_y = dst_height / src_height

        translate_x = lowerLeftDst[0] - (lowerLeftSrc[0] * sigma_x)
        translate_y = lowerLeftDst[1] - (lowerLeftSrc[1] * sigma_y)

        mapped_x = position[0] * sigma_x + translate_x
        mapped_y = position[1] * sigma_y + translate_y

        return [mapped_x, mapped_y], [sigma_x, sigma_y]


# %% 绘图函数
def plot_trajectory(data):
    # 根据数据范围动态调整比例，但保持XY等比例
    target_xlim = [-27, -10]
    target_ylim = [-9, 14.154]
    data_X_range = target_xlim[1] - target_xlim[0]
    data_Y_range = target_ylim[1] - target_ylim[0]
    dpi = 100
    fig_width = 800 / dpi
    fig_height = (800 * data_Y_range / data_X_range) / dpi
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    
    # 1. 参考轨迹 (Green)
    p1, = ax.plot(data['ref_x'], data['ref_y'], '-', color=COLORS['green'], linewidth=4, alpha=0.6, label='Reference')
    
    # 2. 虚拟车轨迹 (Blue)
    p2, = ax.plot(data['vir_x'], data['vir_y'], '-', color=COLORS['blue'], linewidth=3.5, label='Virtual')
    
    # 3. 物理车轨迹 (Red)
    p3, = ax.plot(data['phy_x'], data['phy_y'], '-', color=COLORS['red'], linewidth=3.5, linestyle='-', label='Physical')

    # 起点终点标记
    ax.plot(data['ref_x'][0], data['ref_y'][0], 'p', markersize=14, markerfacecolor=COLORS['green'], markeredgecolor='k', zorder=5)
    ax.plot(data['ref_x'][-1], data['ref_y'][-1], 'o', markersize=10, markerfacecolor=COLORS['green'], markeredgecolor='k', zorder=5)

    ax.set_xlabel(r'$X$ Position (m)', fontsize=22)
    ax.set_ylabel(r'$Y$ Position (m)', fontsize=22)
    # ax.set_xlim(target_xlim) # 如果需要固定范围取消注释
    # ax.set_ylim(target_ylim)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    ax.legend(handles=[p1, p2, p3], loc='best', fontsize=20, framealpha=0.9)
    # plt.tight_layout()

def plot_controls(data):
    """
    Subplot 1: Lateral Control (Steer / Omega)
    Subplot 2: Longitudinal Control (Accel / Velocity)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    
    # --- Subplot 1: Steering / Angular Control ---
    # 都在 ax1 上绘制，取消 twinx
    ax1.plot(data['time'], data['u_Vsteer'], color=COLORS['dark_blue'], linewidth=3.5, label=r'Vir: $u_{\delta}$ (rad)')
    ax1.plot(data['time'], data['u_Pw'], color=COLORS['red'], linewidth=3, label=r'Phy: $u_{\omega}$ (rad/s)')
    
    ax1.grid(True)
    ax1.set_xlim([0, max(data['time'])])
    ax1.set_ylabel(r'Lateral Control', fontsize=18)
    ax1.legend(loc='best', fontsize=14)

    # --- Subplot 2: Longitudinal Control ---
    # 都在 ax2 上绘制，取消 twinx
    ax2.plot(data['time'], data['u_Vv'], color=COLORS['dark_blue'], linewidth=3.5, label=r'Vir: $u_{Vv}$ (m/s)')
    ax2.plot(data['time'], data['u_Pv'], color=COLORS['red'], linewidth=3, label=r'Phy: $u_{Pv}$ (m/s)')
    
    ax2.grid(True)
    ax2.set_xlim([0, max(data['time'])])
    ax2.set_xlabel('Time(s)', fontsize=22)
    ax2.set_ylabel(r'Longitudinal Control (m/s)', fontsize=18)
    ax2.legend(loc='best', fontsize=14)
    
    # 保持默认边框（不隐藏右轴和上轴）
    for spine in ax2.spines.values(): spine.set_linewidth(2)
    
    plt.tight_layout()
    
def plot_velocities(data):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    
    # --- Longitudinal Velocity Comparison ---
    ax1.plot(data['time'], data['vir_vx'], color=COLORS['blue'], linewidth=3, label=r'$v_{Vir}$')
    ax1.plot(data['time'], data['ref_v'], color=COLORS['green'], linewidth=3,  label=r'$v_{Ref}$')
    ax1.plot(data['time'], data['phy_v'], color=COLORS['red'], linewidth=3, label=r'$v_{Phy}$')
    
    ax1.grid(True)
    ax1.set_xlim([0, max(data['time'])])
    ax1.set_ylabel(r'Long. Velocity (m/s)', fontsize=18)
    ax1.legend(fontsize=14, loc='best')
    for spine in ax1.spines.values(): spine.set_linewidth(2)

    # --- Total Velocity / Components ---
    ax2.plot(data['time'], data['vir_v'], color=COLORS['blue'], linewidth=3, label=r'$v_{Vir\_Total}$')
    ax2.plot(data['time'], data['phy_v'], color=COLORS['red'], linewidth=3, label=r'$v_{Phy\_Total}$')
    
    ax2.grid(True)
    ax2.set_xlim([0, max(data['time'])])
    ax2.set_xlabel('Time(s)', fontsize=18)
    ax2.set_ylabel(r'Total Velocity(m/s)', fontsize=18)
    ax2.legend(fontsize=14, loc='best')
    for spine in ax2.spines.values(): spine.set_linewidth(2)
    
    plt.tight_layout()

def plot_angular_velocities(data):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    
    # --- Omega Comparison ---
    ax1.plot(data['time'], data['ref_w'], color=COLORS['green'], linewidth=3, label=r'$\omega_{Ref}$')
    ax1.plot(data['time'], data['vir_w'], color=COLORS['blue'], linewidth=3, label=r'$\omega_{Vir}$')
    ax1.plot(data['time'], data['phy_w'], color=COLORS['red'], linewidth=3, label=r'$\omega_{Phy}$')
    
    ax1.grid(True)
    ax1.set_xlim([0, max(data['time'])])
    ax1.set_ylabel(r'$\omega$ (rad/s)', fontsize=22)
    ax1.legend(fontsize=14, loc='upper right', frameon=False)
    for spine in ax1.spines.values(): spine.set_linewidth(2)
    
    # --- Omega Error ---
    ax2.plot(data['time'], data['e_Vw'], color=COLORS['blue'], linewidth=3, label=r'$e_{\omega\_Vir}$')
    ax2.plot(data['time'], data['e_Pw'], color=COLORS['red'], linewidth=3, label=r'$e_{\omega\_Phy}$')
    
    ax2.grid(True)
    ax2.set_xlim([0, max(data['time'])])
    ax2.set_xlabel('Time(s)', fontsize=22)
    ax2.set_ylabel(r'$e_{\omega}$ (rad/s)', fontsize=22)
    ax2.legend(fontsize=14, loc='upper right')
    for spine in ax2.spines.values(): spine.set_linewidth(2)
    
    plt.tight_layout()

def plot_errors(data):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
    
    def setup_error_ax(ax, y_label, is_bottom=False):
        ax.grid(True)
        ax.set_xlim([0, max(data['time'])])
        ax.set_ylabel(y_label, fontsize=18)
        ax.tick_params(labelsize=14, width=2)
        for spine in ax.spines.values(): spine.set_linewidth(2)
        if is_bottom: ax.set_xlabel('Time(s)', fontsize=18)
        ax.legend(loc='upper right', fontsize=12)
    
    # 纵向误差
    ax1.plot(data['time'], data['e_Vx'], color=COLORS['blue'], linewidth=3, label='Virtual')
    ax1.plot(data['time'], data['e_Px'], color=COLORS['red'], linewidth=3, label='Physical')
    setup_error_ax(ax1, r'Lon Error (m)')
    
    # 横向误差
    ax2.plot(data['time'], data['e_Vy'], color=COLORS['blue'], linewidth=3, label='Virtual')
    ax2.plot(data['time'], data['e_Py'], color=COLORS['red'], linewidth=3,  label='Physical')
    setup_error_ax(ax2, r'Lat Error (m)')
    
    # 航向误差
    ax3.plot(data['time'], data['e_Vth'], color=COLORS['blue'], linewidth=3, label='Virtual')
    ax3.plot(data['time'], data['e_Pyaw'], color=COLORS['red'], linewidth=3, label='Physical')
    setup_error_ax(ax3, r'Yaw Error (rad)', is_bottom=True)
    
    plt.tight_layout()

def plot_realtime(data):
    # 此图主要展示计算耗时，只与控制器本身有关，无需添加物理车数据
    threshold_ms = 50
    fig = plt.figure(figsize=(10, 7), facecolor='white')
    gs = fig.add_gridspec(2, 1, hspace=0.3)
    
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(data['comp_time_ms'], color=COLORS['fresh_blue'], alpha=0.8, linewidth=3.5, label='Computation Time')
    ax1.axhline(y=threshold_ms, color=COLORS['alert_red'], linestyle='--', linewidth=3.5)
    
    ax1.set_ylabel('Comp. Time (ms)', fontsize=16)
    ax1.set_title('Computation Time History', fontsize=20)
    ax1.grid(True, alpha=0.15)
    ax1.legend(loc='upper right', fontsize=16)
    ax1.set_xlim([0, len(data['comp_time_ms'])])

    ax2 = fig.add_subplot(gs[1])
    ax2.hist(data['comp_time_ms'], bins=50, color=COLORS['fresh_blue'], alpha=0.7)
    ax2.axvline(x=threshold_ms, color=COLORS['alert_red'], linestyle='--', linewidth=3.5)
    ax2.set_xlabel('Computation Time (ms)', fontsize=16)
    ax2.set_ylabel('Count', fontsize=16)
    ax2.grid(True, alpha=0.15)
    
    # plt.tight_layout()

def plot_costs(data):
    # 此图展示MPPI内部优化过程，与物理车状态无关
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data['time'], data['mean_cost'], color=COLORS['gray'], alpha=0.6, linewidth=1.5, label='Mean Cost')
    ax.plot(data['time'], data['min_cost'], color=COLORS['purple'], linewidth=3.5, label='Min Cost')
    ax.fill_between(data['time'], data['min_cost'], data['mean_cost'], color=COLORS['purple'], alpha=0.1)
    ax.set_ylabel('MPPI Cost', fontsize=18)
    ax.set_xlabel('Time (s)', fontsize=18)
    ax.set_xlim([0, max(data['time'])])
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', fontsize=14)
    # plt.tight_layout()

# %% 执行
if __name__ == "__main__":
    # 1. 加载数据
    data_dict = load_data(FILE_PATH)
    
    if data_dict is not None:
        # 2. 计算评分与指标
        calculate_and_print_metrics(data_dict)
        
        # 3. 绘图
        plot_trajectory(data_dict)
        plot_controls(data_dict)        # Updated: Double axis for Vir/Phy controls
        plot_velocities(data_dict)      # Updated: Add Phy velocity
        plot_angular_velocities(data_dict) # Updated: Add Phy omega
        plot_errors(data_dict)          # Updated: Add Phy errors
        # plot_realtime(data_dict)
        # plot_costs(data_dict)
        
        plt.show()