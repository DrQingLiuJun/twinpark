#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Custom Python Implementation of Ackermann Control for CARLA 0.9.13
Optimization: Softer gains to prevent oscillation with MPPI.
"""

import math
import time
from collections import deque
import carla

class PIDController:
    def __init__(self, Kp, Ki, Kd, output_min, output_max):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.min = output_min
        self.max = output_max
        self.prev_error = 0.0
        self.integral = 0.0

    def step(self, error, dt):
        if dt < 1e-6: return 0.0
        
        # Integral with dynamic clamping
        if abs(self.Ki) > 1e-5:
            limit = 1.0 / self.Ki # Clamp integral influence to max output
            self.integral = max(-limit, min(limit, self.integral + error * dt))
        else:
            self.integral = 0.0

        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        return max(self.min, min(self.max, output))
    
    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0

class AckermannController:
    def __init__(self, vehicle, 
                 max_steer_deg=45.0, 
                 wheelbase=2.875):
        self.vehicle = vehicle
        self.max_steer_rad = math.radians(max_steer_deg)
        self.wheelbase = wheelbase
        
        # --- PID Tuning (Stability Focused) ---
        # Forward: Smooth tracking
        self.kp_fwd = 1.00  # Reduced from 1.5
        self.ki_fwd = 0.20  
        self.kd_fwd = 0.05 
        
        # Reverse: High damping, low aggression
        self.kp_rev = 0.80  # Reduced from 1.2
        self.ki_rev = 0.30  # Enough to close gap
        self.kd_rev = 0.10  # High damping to stop oscillation
        
        self.speed_pid = PIDController(Kp=self.kp_fwd, Ki=self.ki_fwd, Kd=self.kd_fwd, output_min=-1.0, output_max=1.0)
        self.last_time = time.time()
        
        # Friction Compensation
        self.static_friction = 0.15 

    def run_step(self, target_speed, target_steer_rad):
        current_time = time.time()
        dt = current_time - self.last_time
        if dt > 0.2 or dt <= 0.0: dt = 0.05 
        self.last_time = current_time
        
        # 1. State
        v_vec = self.vehicle.get_velocity()
        current_speed = math.sqrt(v_vec.x**2 + v_vec.y**2 + v_vec.z**2)
        
        # 2. Logic
        control = carla.VehicleControl()
        control.manual_gear_shift = True
        
        # Hysteresis for gear switching
        if target_speed < -0.05:
            control.gear = -1
            control.reverse = True
            current_speed_signed = -current_speed
            # Switch Gains
            self.speed_pid.Kp = self.kp_rev
            self.speed_pid.Ki = self.ki_rev
            self.speed_pid.Kd = self.kd_rev
        elif target_speed > 0.05:
            control.gear = 1
            control.reverse = False
            current_speed_signed = current_speed
            # Switch Gains
            self.speed_pid.Kp = self.kp_fwd
            self.speed_pid.Ki = self.ki_fwd
            self.speed_pid.Kd = self.kd_fwd
        else:
            # Deadband holding
            control.gear = 1
            control.reverse = False
            current_speed_signed = current_speed
            
        # 3. PID
        error = target_speed - current_speed_signed
        cmd = self.speed_pid.step(error, dt)
        
        # 4. Steer
        control.steer = max(-1.0, min(1.0, target_steer_rad / self.max_steer_rad))
        
        # 5. Throttle/Brake
        if abs(target_speed) < 0.01:
            control.throttle = 0.0
            control.brake = 1.0
            self.speed_pid.reset()
        else:
            # Unified Logic: cmd > 0 means "apply force in gear direction"
            # cmd < 0 means "apply force against gear direction" (Brake)
            
            # Note: In Reverse gear, Throttle pushes BACKWARDS. 
            # So if target is -3, current is -2, error is -1.
            # Wait, my previous logic was: target(-3) - current(-2) = -1.
            # PID output negative -> Need more negative speed -> Throttle.
            
            # Let's trust the signed logic:
            # If Reverse:
            # Error = Tgt(-3) - Cur(-2) = -1. PID -> Negative.
            # We need Throttle. So if cmd < 0: Throttle.
            
            if control.reverse:
                if cmd < 0: # Accelerate backwards
                    raw = abs(cmd)
                    control.throttle = min(1.0, raw + self.static_friction)
                    control.brake = 0.0
                else: # Decelerate (moving backwards too fast)
                    control.throttle = 0.0
                    control.brake = min(1.0, cmd)
            else:
                if cmd > 0: # Accelerate forwards
                    control.throttle = min(1.0, cmd + self.static_friction)
                    control.brake = 0.0
                else:
                    control.throttle = 0.0
                    control.brake = min(1.0, abs(cmd))
        
        self.vehicle.apply_control(control)
        return control