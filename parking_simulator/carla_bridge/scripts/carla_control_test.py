#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CARLA Vehicle Control Interface Test Script
Tests two methods:
1. set_target_velocity (Physics Override)
2. apply_ackermann_control (Physics-based Ackermann Control)
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

def test_set_target_velocity(vehicle, world, target_speed_mps=5.0, duration=5.0):
    """
    Method 1: set_target_velocity
    这种方法直接设置刚体的线性速度向量。
    优点：无视摩擦和死区，响应瞬间完成。
    缺点：如果设置的方向和车头不一致，车会横着飘（像滑冰一样）。
    """
    print("\n[TEST 1] Testing set_target_velocity (Direct Physics Override)...")
    print(f"Target Speed: {target_speed_mps} m/s")
    
    start_time = time.time()
    
    while time.time() - start_time < duration:
        # 1. 获取当前车辆的姿态（主要是Yaw）
        transform = vehicle.get_transform()
        yaw_rad = math.radians(transform.rotation.yaw)
        
        # 2. 计算速度向量 (让速度方向永远沿着车头方向)
        # 注意：这不会改变车轮转角，只是强行把车推向车头指向的方向
        vx = target_speed_mps * math.cos(yaw_rad)
        vy = target_speed_mps * math.sin(yaw_rad)
        
        # 3. 设置速度向量
        # 注意：z轴速度保留原有物理计算（重力），或者设置为0（如果不考虑下坡）
        current_vel = vehicle.get_velocity()
        target_vel = carla.Vector3D(x=vx, y=vy, z=current_vel.z)
        
        vehicle.set_target_velocity(target_vel)
        
        world.wait_for_tick()
        
        print(f"\rMode: TargetVel | Current Speed: {get_speed(vehicle):.2f} m/s", end="")
    
    # Stop
    vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
    print("\n[TEST 1] Done.")
    time.sleep(1.0)

def test_apply_ackermann_control(vehicle, world, target_speed_mps=5.0, target_steer_rad=0.3, duration=5.0):
    """
    Method 2: apply_ackermann_control
    这是 CARLA 0.9.14+ 推荐的高级控制。
    它使用内置的控制器来调节油门/刹车以达到目标速度。
    """
    print("\n[TEST 2] Testing apply_ackermann_control (Internal PID)...")
    print(f"Target Speed: {target_speed_mps} m/s, Steer: {target_steer_rad} rad")
    
    # 创建阿克曼控制指令
    # steer: 转向角 (弧度)
    # steer_speed: 转向角变化率 (弧度/秒)
    # speed: 目标速度 (m/s)
    # acceleration: 最大加速度 (m/s^2)
    # jerk: 最大加加速度 (m/s^3)
    ackermann_control = carla.VehicleAckermannControl(
        steer=target_steer_rad,
        steer_speed=0.5,
        speed=target_speed_mps,
        acceleration=2.0,
        jerk=0.0
    )
    
    start_time = time.time()
    
    while time.time() - start_time < duration:
        # 发送控制指令
        vehicle.apply_ackermann_control(ackermann_control)
        
        world.wait_for_tick()
        
        # 获取当前的控制状态用于调试
        control = vehicle.get_control()
        print(f"\rMode: Ackermann | Speed: {get_speed(vehicle):.2f}/{target_speed_mps} m/s | Throttle: {control.throttle:.2f}", end="")
        
    # Stop
    stop_control = carla.VehicleAckermannControl(speed=0.0, acceleration=5.0)
    vehicle.apply_ackermann_control(stop_control)
    print("\n[TEST 2] Done.")
    time.sleep(1.0)

def main():
    client = None
    vehicle = None
    
    try:
        # 1. 连接 CARLA
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        
        # 2. 设置同步模式 (保证控制平滑)
        settings = world.get_settings()
        if not settings.synchronous_mode:
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            world.apply_settings(settings)

        print("Connected to CARLA.")

        # 3. 生成车辆
        blueprint_library = world.get_blueprint_library()
        # 选择一辆支持阿克曼的常见车型，如 Model 3
        veh_bp = blueprint_library.find('vehicle.haut.car')
        
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            print("No spawn points found!")
            return
            
        spawn_point = spawn_points[0]
        vehicle = world.try_spawn_actor(veh_bp, spawn_point)
        
        if vehicle is None:
            print("Failed to spawn vehicle.")
            return
            
        print(f"Spawned {vehicle.type_id} at {spawn_point.location}")
        
        # 让车先落地稳定一下
        for _ in range(20):
            world.tick()

        # ==========================================
        # 运行测试
        # ==========================================
        
        # 测试 1: 直接设置速度向量 (适合处理死区)
        test_set_target_velocity(vehicle, world, target_speed_mps=3.0, duration=5.0)
        
        # 休息一下
        time.sleep(1)
        
        # 测试 2: 阿克曼控制 (适合正常行驶)
        # 注意：如果你的 CARLA 版本过低 (<0.9.13)，这个方法可能不存在或报错
        try:
            test_apply_ackermann_control(vehicle, world, target_speed_mps=5.0, target_steer_rad=0.5, duration=8.0)
        except AttributeError:
            print("\n[ERROR] Your CARLA version might be too old for apply_ackermann_control.")
            print("Please upgrade to CARLA 0.9.14+ or use Method 1.")

    except Exception as e:
        print(f"\nAn error occurred: {e}")

    finally:
        # 清理
        if vehicle is not None:
            vehicle.destroy()
            print("Vehicle destroyed.")
        
        if client is not None:
            # 恢复异步模式
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)
            print("Settings restored.")

if __name__ == '__main__':
    main()