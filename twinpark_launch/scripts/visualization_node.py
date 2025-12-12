#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualization Node for TwinPark System

Subscribes to trajectory and state topics and publishes visualization markers
for RViz display.
"""

import rospy
from geometry_msgs.msg import PoseStamped, Pose
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker
from vehicle_msgs.msg import Trajectory, VehicleState
import math


class VisualizationNode:
    """Node for publishing visualization markers"""
    
    def __init__(self):
        rospy.init_node('visualization_node', anonymous=False)
        
        # Publishers
        self.goal_marker_pub = rospy.Publisher('/goal_marker', Marker, queue_size=1, latch=True)
        self.planned_path_pub = rospy.Publisher('/planned_path', Path, queue_size=1, latch=True)
        self.vehicle_pose_pub = rospy.Publisher('/vehicle_pose', PoseStamped, queue_size=10)
        self.reference_pose_pub = rospy.Publisher('/reference_pose', PoseStamped, queue_size=10)
        
        # Subscribers
        rospy.Subscriber('/planned_trajectory', Trajectory, self.trajectory_callback)
        rospy.Subscriber('/vehicle_state', VehicleState, self.vehicle_state_callback)
        rospy.Subscriber('/reference_state', VehicleState, self.reference_state_callback)
        
        # Get goal from parameters
        self.goal_x = rospy.get_param('planner_node/goal_x', 23.8)
        self.goal_y = rospy.get_param('planner_node/goal_y', 0.1)
        self.goal_yaw = rospy.get_param('planner_node/goal_yaw', 180.0)
        
        rospy.loginfo("Visualization Node initialized")
        rospy.loginfo(f"Goal: ({self.goal_x:.2f}, {self.goal_y:.2f}, {self.goal_yaw:.1f}°)")
        
        # Publish goal marker immediately
        self.publish_goal_marker()
        
    def publish_goal_marker(self):
        """Publish goal position as a marker"""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "goal"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        
        # Position
        marker.pose.position.x = self.goal_x
        marker.pose.position.y = self.goal_y
        marker.pose.position.z = 0.5
        
        # Orientation (convert yaw to quaternion)
        yaw_rad = math.radians(self.goal_yaw)
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = math.sin(yaw_rad / 2.0)
        marker.pose.orientation.w = math.cos(yaw_rad / 2.0)
        
        # Scale
        marker.scale.x = 3.0  # Arrow length
        marker.scale.y = 0.5  # Arrow width
        marker.scale.z = 0.5  # Arrow height
        
        # Color (bright green)
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        marker.lifetime = rospy.Duration(0)  # Never expire
        
        self.goal_marker_pub.publish(marker)
        rospy.loginfo("Goal marker published")
        
    def trajectory_callback(self, msg):
        """Convert trajectory to Path for visualization"""
        path = Path()
        path.header.frame_id = "map"
        path.header.stamp = rospy.Time.now()
        
        # Trajectory message uses arrays: x[], y[], yaw[], etc.
        num_points = len(msg.x)
        
        for i in range(num_points):
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = msg.x[i]
            pose.pose.position.y = msg.y[i]
            pose.pose.position.z = 0.0
            
            # Convert yaw to quaternion
            yaw_rad = msg.yaw[i]
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = math.sin(yaw_rad / 2.0)
            pose.pose.orientation.w = math.cos(yaw_rad / 2.0)
            
            path.poses.append(pose)
        
        self.planned_path_pub.publish(path)
        rospy.loginfo(f"Published planned path with {len(path.poses)} poses")
        
    def vehicle_state_callback(self, msg):
        """Publish vehicle state as pose for visualization"""
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = rospy.Time.now()
        
        pose.pose.position.x = msg.x
        pose.pose.position.y = msg.y
        pose.pose.position.z = 0.0
        
        # Convert yaw to quaternion
        yaw_rad = msg.yaw
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = math.sin(yaw_rad / 2.0)
        pose.pose.orientation.w = math.cos(yaw_rad / 2.0)
        
        self.vehicle_pose_pub.publish(pose)
        
    def reference_state_callback(self, msg):
        """Publish reference state as pose for visualization"""
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = rospy.Time.now()
        
        pose.pose.position.x = msg.x
        pose.pose.position.y = msg.y
        pose.pose.position.z = 0.0
        
        # Convert yaw to quaternion
        yaw_rad = msg.yaw
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = math.sin(yaw_rad / 2.0)
        pose.pose.orientation.w = math.cos(yaw_rad / 2.0)
        
        self.reference_pose_pub.publish(pose)
        
    def spin(self):
        """Keep node running"""
        rospy.spin()


if __name__ == '__main__':
    try:
        node = VisualizationNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
