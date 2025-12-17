#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CARLA Vehicle Control Interface Test Script
Tests:
1. set_target_velocity + set_target_angular_velocity (Simulated Turning)
"""

import glob
import os
import sys
import random
import time
import math

# ==============================================================================
# -- Find CARLA module ---------------------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

def get_speed(vehicle):
    """
    Compute speed of a vehicle in m/s.
    """
    vel = vehicle.get_velocity()
    return math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

def test_turning_simulation(vehicle, world, target_speed_mps=3.0):
    """
    模拟转弯：同时控制 线速度 和 角速度
    这是在没有 apply_ackermann_control 的版本中模拟真实驾驶的最佳方式。
    """
    print("\n[TEST TURN] Testing Turning with Physics Override...")
    
    # 车辆参数
    wheelbase = 3.0 # 轴距 (米)
    
    # 阶段设定
    duration_straight = 3.0 # 直行 3秒
    duration_turn = 5.0     # 转弯 5秒
    
    start_time = time.time()
    
    while True:
        elapsed = time.time() - start_time
        
        # 1. 状态机：决定当前的控制输入
        if elapsed < duration_straight:
            # 阶段一：直行
            steer_deg = 0.0
            mode = "Straight"
        elif elapsed < duration_straight + duration_turn:
            # 阶段二：右转 (方向盘打20度)
            steer_deg = 20.0 
            mode = "Turning Right"
        else:
            break # 结束
            
        # 2. 获取车辆当前姿态
        transform = vehicle.get_transform()
        yaw_deg = transform.rotation.yaw
        yaw_rad = math.radians(yaw_deg)
        
        # 3. 计算 线速度向量 (Linear Velocity)
        # 让速度永远沿着车头方向，消除侧滑
        vx = target_speed_mps * math.cos(yaw_rad)
        vy = target_speed_mps * math.sin(yaw_rad)
        
        current_vel = vehicle.get_velocity()
        target_linear_vel = carla.Vector3D(x=vx, y=vy, z=current_vel.z) # 保持z轴物理(重力)
        
        # 4. 计算 角速度向量 (Angular Velocity)
        # 公式: omega = (v / L) * tan(delta)
        steer_rad = math.radians(steer_deg)
        if abs(steer_deg) > 0.1:
            yaw_rate_rad_s = (target_speed_mps / wheelbase) * math.tan(steer_rad)
            yaw_rate_deg_s = math.degrees(yaw_rate_rad_s)
        else:
            yaw_rate_deg_s = 0.0
            
        target_angular_vel = carla.Vector3D(x=0, y=0, z=yaw_rate_deg_s)
        
        # 5. 应用控制 (双管齐下)
        vehicle.set_target_velocity(target_linear_vel)
        vehicle.set_target_angular_velocity(target_angular_vel)
        
        # 驱动仿真
        world.tick()
        
        print(f"\rMode: {mode:15s} | Speed: {get_speed(vehicle):.2f} m/s | Steer: {steer_deg:.1f}° | YawRate: {yaw_rate_deg_s:.1f}°/s", end="")

    # Stop
    vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
    vehicle.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
    world.tick()
    print("\n[TEST TURN] Done.")
    time.sleep(1.0)

def main():
    client = None
    vehicle = None
    
    try:
        # 1. 连接 CARLA
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        
        # 2. 设置同步模式
        settings = world.get_settings()
        if not settings.synchronous_mode:
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            world.apply_settings(settings)

        print("Connected to CARLA (Sync Mode).")

        # 3. 生成车辆
        blueprint_library = world.get_blueprint_library()
        veh_bp = blueprint_library.find('vehicle.tesla.model3')
        
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            print("No spawn points found!")
            return
            
        spawned = False
        for i, spawn_point in enumerate(spawn_points):
            spawn_point.location.z = 0.1 
            vehicle = world.try_spawn_actor(veh_bp, spawn_point)
            if vehicle is not None:
                print(f"Spawned {vehicle.type_id} at {spawn_point.location}")
                spawned = True
                break
        
        if not spawned:
            print("Failed to spawn vehicle.")
            return
        
        # 落地稳定
        for _ in range(20):
            world.tick()

        # ==========================================
        # 运行测试：直行 -> 转弯
        # ==========================================
        test_turning_simulation(vehicle, world, target_speed_mps=3.0)

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if vehicle is not None:
            vehicle.destroy()
            print("Vehicle destroyed.")
        
        if client is not None:
            try:
                settings = world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                world.apply_settings(settings)
                print("Settings restored to asynchronous.")
            except:
                print("Failed to restore settings.")

if __name__ == '__main__':
    main()