#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Vehicle Tracking Control Node
Implements Backstepping control law and PID speed control
"""

import rospy
import math
from vehicle_msgs.msg import VehicleState, ControlCmd
from geometry_msgs.msg import Vector3


class LongitudinalPID:
    """PID controller for longitudinal speed control"""
    
    def __init__(self, kp, ki, kd, max_integral=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_integral = max_integral
        
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None
    
    def compute(self, desired_vx, current_vx):
        """Compute control output"""
        current_time = rospy.Time.now()
        
        # Calculate error
        error = desired_vx - current_vx
        
        # Calculate dt
        if self.prev_time is None:
            dt = 0.02  # Default 50Hz
        else:
            dt = (current_time - self.prev_time).to_sec()
            if dt <= 0:
                dt = 0.02
        
        self.prev_time = current_time
        
        # Integral term with anti-windup
        self.integral += error * dt
        self.integral = max(-self.max_integral, min(self.max_integral, self.integral))
        
        # Derivative term
        if dt > 0:
            derivative = (error - self.prev_error) / dt
        else:
            derivative = 0.0
        
        self.prev_error = error
        
        # PID output
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        
        return output
    
    def reset(self):
        """Reset PID state"""
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None


class BacksteppingController:
    """Backstepping controller for trajectory tracking"""
    
    def __init__(self, k1, k2, k3):
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
    
    def compute_control(self, current_state, reference_state):
        """
        Compute desired velocity and angular velocity
        
        Args:
            current_state: VehicleState message
            reference_state: VehicleState message
        
        Returns:
            ux: desired longitudinal velocity
            uth: desired angular velocity
        """
        # Calculate tracking error in vehicle body frame
        dx = reference_state.x - current_state.x
        dy = reference_state.y - current_state.y
        
        cos_yaw = math.cos(current_state.yaw)
        sin_yaw = math.sin(current_state.yaw)
        
        # Transform error to body frame
        ex = dx * cos_yaw + dy * sin_yaw
        ey = -dx * sin_yaw + dy * cos_yaw
        eth = self.normalize_angle(reference_state.yaw - current_state.yaw)
        
        # Backstepping control law
        cos_eth = math.cos(eth)
        sin_eth = math.sin(eth)
        
        ux = reference_state.vx * cos_eth + self.k1 * ex
        
        # Avoid division by zero
        if abs(ux) < 1e-6:
            ux_safe = math.copysign(1e-6, ux) if ux != 0 else 1e-6
        else:
            ux_safe = ux
        
        # Angular velocity control
        # When eth > 0, we need positive uth to increase yaw
        uth = 0.6 * reference_state.omega + \
              reference_state.vx * self.k2 * ey + \
              reference_state.vx * self.k3 * sin_eth
        
        return ux, uth, ex, ey, eth
    
    @staticmethod
    def normalize_angle(angle):
        """Normalize angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle


class VehicleControlNode:
    """Main vehicle control node"""
    
    def __init__(self):
        rospy.init_node('vehicle_control_node', anonymous=False)
        
        # Load parameters
        self.load_parameters()
        
        # Initialize controllers
        self.backstepping = BacksteppingController(self.k1, self.k2, self.k3)
        self.pid = LongitudinalPID(
            self.pid_kp, 
            self.pid_ki, 
            self.pid_kd, 
            self.pid_max_integral
        )
        
        # State variables
        self.current_state = None
        self.reference_state = None
        
        # Publishers
        self.control_pub = rospy.Publisher(
            '/control_cmd', 
            ControlCmd, 
            queue_size=10
        )
        self.error_pub = rospy.Publisher(
            '/tracking_error', 
            Vector3, 
            queue_size=10
        )
        
        # Subscribers
        self.state_sub = rospy.Subscriber(
            '/vehicle_state', 
            VehicleState, 
            self.state_callback
        )
        self.ref_sub = rospy.Subscriber(
            '/reference_state', 
            VehicleState, 
            self.reference_callback
        )
        
        # Control timer
        self.control_rate = rospy.Rate(self.publish_rate)
        
        rospy.loginfo("Vehicle Control Node initialized")
    
    def load_parameters(self):
        """Load parameters from ROS parameter server"""
        # Backstepping parameters - read from vehicle_control namespace
        self.k1 = rospy.get_param('~vehicle_control/k1', 1.0)
        self.k2 = rospy.get_param('~vehicle_control/k2', 0.5)
        self.k3 = rospy.get_param('~vehicle_control/k3', 0.2)
        
        # PID parameters
        self.pid_kp = rospy.get_param('~vehicle_control/pid/kp', 0.5)
        self.pid_ki = rospy.get_param('~vehicle_control/pid/ki', 0.0)
        self.pid_kd = rospy.get_param('~vehicle_control/pid/kd', 0.1)
        self.pid_max_integral = rospy.get_param('~vehicle_control/pid/max_integral', 1.0)
        
        # Control limits
        self.max_throttle = rospy.get_param('~vehicle_control/max_throttle', 1.0)
        self.max_brake = rospy.get_param('~vehicle_control/max_brake', 1.0)
        self.max_steer = rospy.get_param('~vehicle_control/max_steer', 1.0)
        
        # Vehicle parameters
        self.wheelbase = rospy.get_param('~vehicle_control/wheelbase', 3.368)
        
        # Control rate
        self.publish_rate = rospy.get_param('~vehicle_control/publish_rate', 50.0)
        
        rospy.loginfo("Parameters loaded:")
        rospy.loginfo("  Backstepping: k1=%.2f, k2=%.2f, k3=%.2f", self.k1, self.k2, self.k3)
        rospy.loginfo("  PID: kp=%.2f, ki=%.2f, kd=%.2f", self.pid_kp, self.pid_ki, self.pid_kd)
    
    def state_callback(self, msg):
        """Callback for vehicle state"""
        self.current_state = msg
    
    def reference_callback(self, msg):
        """Callback for reference state"""
        self.reference_state = msg
    
    def compute_steer_angle(self, uth, vx):
        """
        Compute steering angle using simplified proportional model
        
        Args:
            uth: desired angular velocity (rad/s)
            vx: actual longitudinal velocity (m/s)
        
        Returns:
            tuple: (steer_normalized, steer_deg) - normalized steering [-1, 1] and angle in degrees
        """
        # Simplified approach: directly map angular velocity to steering angle
        # This avoids speed-dependent amplification that causes excessive steering at low speeds
        
        # Maximum angular velocity that corresponds to max steering
        # Tuned for typical parking speeds (0.3-1.0 m/s)
        max_omega = 0.8  # rad/s
        
        # Calculate desired steering angle in radians
        # Using proportional relationship: steer_angle proportional to omega
        max_steer_rad = math.radians(45.0)  # Maximum physical steering angle
        steer_rad = (uth / max_omega) * max_steer_rad
        
        # When moving backward, reverse steering direction
        if vx < 0:
            steer_rad = -steer_rad
        
        # Convert to degrees for logging
        steer_deg = math.degrees(steer_rad)
        
        # Linear mapping to normalized range [-1, 1]
        # Map max_steer_rad (45°) to 1.0, and -max_steer_rad to -1.0
        steer_normalized = steer_rad / max_steer_rad
        
        # Clamp to [-1, 1] to respect physical limits
        steer_normalized = max(-1.0, min(1.0, steer_normalized))
        
        return steer_normalized, steer_deg
    
    def compute_control_cmd(self):
        """Compute control command"""
        if self.current_state is None or self.reference_state is None:
            return None
        
        # Backstepping control
        ux, uth, ex, ey, eth = self.backstepping.compute_control(
            self.current_state, 
            self.reference_state
        )
        
        # PID speed control
        pid_output = self.pid.compute(ux, self.current_state.vx)
        
        # Create control command
        cmd = ControlCmd()
        
        # Determine gear and reverse flag based on desired velocity
        if ux >= 0:
            cmd.gear = 1
            cmd.reverse = False
        else:
            cmd.gear = -1
            cmd.reverse = True
            # Adjust PID output for reverse
            pid_output = -pid_output
        
        # Convert PID output to throttle/brake
        if pid_output >= 0:
            cmd.throttle = min(pid_output, self.max_throttle)
            cmd.brake = 0.0
        else:
            cmd.throttle = 0.0
            cmd.brake = min(-pid_output, self.max_brake)
        
        # Compute steering using actual velocity
        # The function handles direction reversal internally
        cmd.steer, steer_deg = self.compute_steer_angle(uth, self.current_state.vx)
        
        # Publish tracking error
        error_msg = Vector3()
        error_msg.x = ex
        error_msg.y = ey
        error_msg.z = eth
        self.error_pub.publish(error_msg)
        
        # Debug logging (every 50 iterations = 1 second at 50Hz)
        if not hasattr(self, '_debug_counter'):
            self._debug_counter = 0
        self._debug_counter += 1
        
        if self._debug_counter % 50 == 0:
            rospy.loginfo("=" * 60)
            rospy.loginfo("Reference: x=%.2f, y=%.2f, yaw=%.3f (%.1f°), vx=%.2f",
                         self.reference_state.x, self.reference_state.y,
                         self.reference_state.yaw, math.degrees(self.reference_state.yaw),
                         self.reference_state.vx)
            rospy.loginfo("Actual:    x=%.2f, y=%.2f, yaw=%.3f (%.1f°), vx=%.2f",
                         self.current_state.x, self.current_state.y,
                         self.current_state.yaw, math.degrees(self.current_state.yaw),
                         self.current_state.vx)
            rospy.loginfo("Error:     ex=%.3f, ey=%.3f, eth=%.3f (%.1f°)",
                         ex, ey, eth, math.degrees(eth))
            rospy.loginfo("Control:   ux=%.3f, uth=%.3f, throttle=%.2f, brake=%.2f, steer=%.2f, steer_deg=%.1f°",
                          ux, uth, cmd.throttle, cmd.brake, cmd.steer, steer_deg)
        
        return cmd
    
    def run(self):
        """Main control loop"""
        rospy.loginfo("Starting control loop at %.1f Hz", self.publish_rate)
        
        while not rospy.is_shutdown():
            cmd = self.compute_control_cmd()
            
            if cmd is not None:
                self.control_pub.publish(cmd)
            
            try:
                self.control_rate.sleep()
            except rospy.ROSInterruptException:
                break
        
        rospy.loginfo("Control node shutting down")


def main():
    try:
        node = VehicleControlNode()
        node.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
