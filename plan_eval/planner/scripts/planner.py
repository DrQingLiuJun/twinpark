#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hybrid A* Planner Node for TwinPark System
Based on the planning algorithm from /planning folder

Uses Reeds-Shepp curves for path planning
"""

import rospy
import math
import numpy as np
import heapq
from scipy.spatial import cKDTree

from vehicle_msgs.msg import VehicleState, Trajectory
from geometry_msgs.msg import PoseStamped

# Import planning modules from the planning folder
import sys
import os
# Add planning folder to path
planning_path = os.path.join(os.path.dirname(__file__), '../planning')
sys.path.append(planning_path)

from HybridAStar import hybrid_a_star
from ReedsSheppPath import reeds_shepp_path_planning as rs


class PlannerNode:
    """Hybrid A* Planner Node using Reeds-Shepp curves"""
    
    def __init__(self):
        rospy.init_node('planner_node', anonymous=False)
        
        # Load parameters
        self.load_parameters()
        
        # Publishers
        self.traj_pub = rospy.Publisher('/planned_trajectory', Trajectory, queue_size=1, latch=True)
        
        # Subscribers
        self.state_sub = rospy.Subscriber('/vehicle_state', VehicleState, self.state_callback)
        
        # State
        self.current_state = None
        self.trajectory_published = False
        
        rospy.loginfo("Parameters loaded, Planner node initialized")
        
    def load_parameters(self):
        """Load parameters from ROS parameter server"""
        # Hybrid A* parameters
        self.xy_resolution = rospy.get_param('~planner_node/xy_resolution', 0.755)
        self.yaw_resolution = rospy.get_param('~planner_node/yaw_resolution', 10.0)
        
        # Vehicle parameters
        self.wheelbase = rospy.get_param('~planner_node/wheelbase', 3.368)
        self.width = rospy.get_param('~planner_node/width', 2.857)
        self.length = rospy.get_param('~planner_node/length', 4.955)
        self.max_steer = rospy.get_param('~planner_node/max_steer', 45.0)
        
        # Speed planning parameters
        self.max_speed = rospy.get_param('~planner_node/max_speed', 2.0)
        self.max_accel = rospy.get_param('~planner_node/max_accel', 1.0)
        self.max_jerk = rospy.get_param('~planner_node/max_jerk', 0.5)
        
        # Goal configuration
        self.goal_x = rospy.get_param('~planner_node/goal_x', 23.8)
        self.goal_y = rospy.get_param('~planner_node/goal_y', 0.1)
        self.goal_yaw = math.radians(rospy.get_param('~planner_node/goal_yaw', 180.0))
        
        # Auto planning
        self.auto_plan = rospy.get_param('~planner_node/auto_plan', True)
        
        # Obstacle map
        self.obstacles_config = rospy.get_param('~planner_node/obstacles', {})
        self.x_bounds = self.obstacles_config.get('x_bounds', [9.476, 27.401])
        self.y_bounds = self.obstacles_config.get('y_bounds', [-10.101, 14.377])
        
        # Convert yaw resolution to radians
        self.yaw_resolution_rad = math.radians(self.yaw_resolution)
        
    def state_callback(self, msg):
        """Callback for vehicle state"""
        self.current_state = msg
        
        # Auto plan on first state received
        if self.auto_plan and not self.trajectory_published:
            self.plan_trajectory()
            
    @staticmethod
    def normalize_angle(angle):
        """Normalize angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle
    
    def create_obstacle_map(self):
        """Create obstacle map from environment bounds"""
        ox, oy = [], []
        
        # Create boundary obstacles
        x_min, x_max = self.x_bounds
        y_min, y_max = self.y_bounds
        
        # Top boundary
        for x in np.arange(x_min, x_max, self.xy_resolution):
            ox.append(x)
            oy.append(y_max)
            
        # Bottom boundary
        for x in np.arange(x_min, x_max, self.xy_resolution):
            ox.append(x)
            oy.append(y_min)
            
        # Left boundary
        for y in np.arange(y_min, y_max, self.xy_resolution):
            ox.append(x_min)
            oy.append(y)
            
        # Right boundary
        for y in np.arange(y_min, y_max, self.xy_resolution):
            ox.append(x_max)
            oy.append(y)
            
        return ox, oy
        
    def plan_trajectory(self):
        """Plan trajectory using Hybrid A*"""
        if self.current_state is None:
            rospy.logwarn("No vehicle state received yet")
            return
            
        rospy.loginfo("Starting trajectory planning...")
        
        # Get start pose from current state
        start_x = self.current_state.x
        start_y = self.current_state.y
        start_yaw = self.current_state.yaw
        
        rospy.loginfo(f"Start: ({start_x:.2f}, {start_y:.2f}, {math.degrees(start_yaw):.1f}°)")
        rospy.loginfo(f"Goal: ({self.goal_x:.2f}, {self.goal_y:.2f}, {math.degrees(self.goal_yaw):.1f}°)")
        
        # Create obstacle map
        ox, oy = self.create_obstacle_map()
        
        # Plan path using Hybrid A*
        start = [start_x, start_y, start_yaw]
        goal = [self.goal_x, self.goal_y, self.goal_yaw]
        
        try:
            path = hybrid_a_star.hybrid_a_star_planning(
                start, goal, ox, oy,
                self.xy_resolution,
                self.yaw_resolution_rad
            )
            
            if path is None or len(path.x_list) == 0:
                rospy.logerr("Failed to find path")
                return
                
            rospy.loginfo(f"Path found with {len(path.x_list)} waypoints")
            
            # Check if path direction matches vehicle direction
            if len(path.yaw_list) > 0:
                yaw_diff = abs(self.normalize_angle(path.yaw_list[0] - start_yaw))
                rospy.loginfo(f"Start yaw: {math.degrees(start_yaw):.1f}°, Path start yaw: {math.degrees(path.yaw_list[0]):.1f}°, Diff: {math.degrees(yaw_diff):.1f}°")
                
                # If difference > 90°, the path is backwards
                if yaw_diff > math.pi / 2:
                    rospy.logwarn("Path direction is reversed! Flipping yaw angles by 180°")
                    for i in range(len(path.yaw_list)):
                        path.yaw_list[i] = self.normalize_angle(path.yaw_list[i] + math.pi)
            
            # Publish trajectory directly from path
            self.publish_trajectory_from_path(path)
            
            self.trajectory_published = True
            
        except Exception as e:
            rospy.logerr(f"Planning failed: {e}")
            import traceback
            traceback.print_exc()
            
    def publish_trajectory_from_path(self, path):
        """Publish trajectory from path using library's generate_trajectory"""
        # Use the library function to generate proper velocity profile
        # It handles direction changes correctly: decelerates to 0, then accelerates in reverse
        trajectory = hybrid_a_star.generate_trajectory(path)
        
        msg = Trajectory()
        num_points = len(trajectory.x_list)
        
        # Log direction changes
        if hasattr(path, 'direction_list'):
            direction_changes = 0
            for i in range(1, len(path.direction_list)):
                if path.direction_list[i] != path.direction_list[i-1]:
                    direction_changes += 1
            rospy.loginfo(f"Path has {direction_changes} direction changes")
        
        # Calculate time stamps based on distance and velocity
        t_list = [0.0]
        for i in range(1, num_points):
            dx = trajectory.x_list[i] - trajectory.x_list[i-1]
            dy = trajectory.y_list[i] - trajectory.y_list[i-1]
            dist = math.sqrt(dx*dx + dy*dy)
            avg_v = abs((trajectory.v_list[i] + trajectory.v_list[i-1]) / 2.0)
            if avg_v > 0.01:
                dt_step = dist / avg_v
            else:
                dt_step = 0.1
            t_list.append(t_list[-1] + dt_step)
        
        msg.t = t_list
        msg.x = trajectory.x_list
        msg.y = trajectory.y_list
        msg.yaw = trajectory.yaw_list
        msg.vx = trajectory.v_list
        msg.omega = trajectory.omega_list
        msg.kappa = [0.0] * num_points
        
        self.traj_pub.publish(msg)
        
        duration = t_list[-1] if t_list else 0.0
        rospy.loginfo(f"Trajectory published! Duration: {duration:.2f}s")
        
    def spin(self):
        """Keep node running"""
        rospy.spin()


def main():
    """Main entry point"""
    try:
        planner = PlannerNode()
        planner.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
