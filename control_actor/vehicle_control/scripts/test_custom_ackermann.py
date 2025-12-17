#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for Custom Python Ackermann Controller
"""

import glob
import os
import sys
import time
import math
import carla
from ackermann_controller import AckermannController

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    
    # Sync mode is crucial for PID stability
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    
    vehicle = None
    
    try:
        bp_lib = world.get_blueprint_library()
        bp = bp_lib.find('vehicle.tesla.model3')
        spawn = world.get_map().get_spawn_points()[0]
        spawn.location.x = 0
        spawn.location.y = 0
        spawn.location.z = 0.1
        
        vehicle = world.try_spawn_actor(bp, spawn)
        if not vehicle:
            print("Spawn failed")
            return
            
        # Init Controller
        controller = AckermannController(vehicle)
        
        print("Start Test...")
        
        # Test Sequence
        # 1. Accelerate to 5 m/s, Go Straight
        # 2. Turn Right while keeping 5 m/s
        # 3. Stop
        # 4. Reverse at -3 m/s
        
        start_t = time.time()
        last_print_time = time.time()
        
        while True:
            world.tick()
            current_time = time.time()
            t = current_time - start_t
            
            if t < 3.0:
                # Accelerate
                tgt_v = 2.0
                tgt_steer = 0.0
                mode = "Accel"
            elif t <5.0:
                # Turn Right
                tgt_v = 2.0
                tgt_steer = 0.5 # rad
                mode = "Turn"
            elif t < 8.0:
                # Stop
                tgt_v = 0.0
                tgt_steer = 0.0
                mode = "Stop"
            elif t < 15.0:
                # Reverse
                tgt_v = -1.0
                tgt_steer = -0.5
                mode = "Reverse"
            else:
                break
                
            # EXECUTE CONTROL
            control = controller.run_step(tgt_v, tgt_steer)
            
            v = vehicle.get_velocity()
            speed = math.sqrt(v.x**2 + v.y**2 + v.z**2)
            if control.reverse: speed = -speed
            
            if current_time - last_print_time >= 0.25:
                print(f"\r[{mode}] Tgt: {tgt_v:.1f} m/s | Cur: {speed:.2f} m/s | Thr: {control.throttle:.2f} | Brk: {control.brake:.2f} | Steer: {control.steer:.2f}")
                last_print_time = current_time
    finally:
        settings.synchronous_mode = False
        world.apply_settings(settings)
        if vehicle: vehicle.destroy()
        print("\nDone.")

if __name__ == '__main__':
    main()