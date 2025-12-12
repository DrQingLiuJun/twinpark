#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Trajectory Server Node

This node manages trajectory data and publishes interpolated reference states
for the vehicle controller to track.

Features:
- Trajectory subscription and storage
- Linear interpolation for reference state generation
- Time scaling (slow motion / fast forward)
- Trajectory completion detection
"""

import rospy
import numpy as np
from vehicle_msgs.msg import Trajectory, VehicleState
from std_msgs.msg import String


class TrajServer:
    """Trajectory Server for managing and interpolating reference trajectories"""
    
    def __init__(self):
        """Initialize the trajectory server node"""
        rospy.init_node('traj_server', anonymous=False)
        
        # Load parameters
        self.publish_rate = rospy.get_param('~publish_rate', 50.0)  # Hz
        self.time_scale = rospy.get_param('~time_scale', 1.0)  # 1.0 = normal speed
        self.interpolation_method = rospy.get_param('~interpolation_method', 'linear')
        self.loop_trajectory = rospy.get_param('~loop_trajectory', False)
        
        # Trajectory storage
        self.trajectory = None
        self.has_trajectory = False
        self.start_time = None
        self.trajectory_duration = 0.0
        
        # Publishers
        self.ref_pub = rospy.Publisher('/reference_state', VehicleState, queue_size=10)
        self.status_pub = rospy.Publisher('/trajectory_status', String, queue_size=10)
        
        # Subscribers
        self.traj_sub = rospy.Subscriber('/planned_trajectory', Trajectory, 
                                         self.trajectory_callback, queue_size=1)
        
        # Timer for publishing reference states
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.publish_rate), 
                                 self.publish_reference_state)
        
        rospy.loginfo("Trajectory Server initialized")
        rospy.loginfo(f"  Publish rate: {self.publish_rate} Hz")
        rospy.loginfo(f"  Time scale: {self.time_scale}x")
        rospy.loginfo(f"  Interpolation: {self.interpolation_method}")
        rospy.loginfo(f"  Loop trajectory: {self.loop_trajectory}")
    
    def trajectory_callback(self, msg):
        """
        Callback for receiving new trajectory
        
        Args:
            msg: Trajectory message containing time series data
        """
        if len(msg.t) == 0:
            rospy.logwarn("Received empty trajectory, ignoring")
            return
        
        # Store trajectory
        self.trajectory = msg
        self.has_trajectory = True
        self.start_time = rospy.Time.now()
        
        # Calculate trajectory duration
        self.trajectory_duration = msg.t[-1] - msg.t[0]
        
        rospy.loginfo(f"Received new trajectory with {len(msg.t)} points")
        rospy.loginfo(f"  Duration: {self.trajectory_duration:.2f} seconds")
        rospy.loginfo(f"  Start: ({msg.x[0]:.2f}, {msg.y[0]:.2f})")
        rospy.loginfo(f"  End: ({msg.x[-1]:.2f}, {msg.y[-1]:.2f})")
        
        # Publish status
        status_msg = String()
        status_msg.data = "trajectory_received"
        self.status_pub.publish(status_msg)
    
    def interpolate_linear(self, t_query):
        """
        Linear interpolation of trajectory at given time
        
        Args:
            t_query: Query time (seconds from trajectory start)
            
        Returns:
            VehicleState: Interpolated reference state
        """
        traj = self.trajectory
        t_array = np.array(traj.t)
        
        # Find interpolation indices
        if t_query <= t_array[0]:
            # Before trajectory start - return first point
            idx = 0
            alpha = 0.0
        elif t_query >= t_array[-1]:
            # After trajectory end - return last point
            idx = len(t_array) - 2
            alpha = 1.0
        else:
            # Find bracketing indices
            idx = np.searchsorted(t_array, t_query) - 1
            idx = max(0, min(idx, len(t_array) - 2))
            
            # Calculate interpolation weight
            dt = t_array[idx + 1] - t_array[idx]
            if dt > 1e-6:
                alpha = (t_query - t_array[idx]) / dt
            else:
                alpha = 0.0
        
        # Linear interpolation
        ref_state = VehicleState()
        ref_state.stamp = rospy.Time.now()
        
        ref_state.x = (1 - alpha) * traj.x[idx] + alpha * traj.x[idx + 1]
        ref_state.y = (1 - alpha) * traj.y[idx] + alpha * traj.y[idx + 1]
        ref_state.vx = (1 - alpha) * traj.vx[idx] + alpha * traj.vx[idx + 1]
        ref_state.omega = (1 - alpha) * traj.omega[idx] + alpha * traj.omega[idx + 1]
        
        # Interpolate yaw angle (handle wraparound)
        yaw1 = traj.yaw[idx]
        yaw2 = traj.yaw[idx + 1]
        
        # Normalize angle difference to [-pi, pi]
        dyaw = yaw2 - yaw1
        while dyaw > np.pi:
            dyaw -= 2 * np.pi
        while dyaw < -np.pi:
            dyaw += 2 * np.pi
        
        ref_state.yaw = yaw1 + alpha * dyaw
        
        # Normalize final yaw to [-pi, pi]
        while ref_state.yaw > np.pi:
            ref_state.yaw -= 2 * np.pi
        while ref_state.yaw < -np.pi:
            ref_state.yaw += 2 * np.pi
        
        # Set gear based on velocity direction
        if ref_state.vx >= 0:
            ref_state.gear = 1  # Forward
            ref_state.reverse = False
        else:
            ref_state.gear = -1  # Reverse
            ref_state.reverse = True
        
        # vy is typically zero for trajectory tracking
        ref_state.vy = 0.0
        
        return ref_state
    
    def publish_reference_state(self, event):
        """
        Timer callback to publish reference state
        
        Args:
            event: Timer event (unused)
        """
        if not self.has_trajectory:
            return
        
        # Calculate elapsed time with time scaling
        elapsed = (rospy.Time.now() - self.start_time).to_sec()
        t_query = elapsed * self.time_scale
        
        # Check if trajectory is complete
        if t_query > self.trajectory_duration:
            if self.loop_trajectory:
                # Loop back to start
                t_query = t_query % self.trajectory_duration
            else:
                # Trajectory complete
                rospy.loginfo_once("Trajectory execution complete")
                
                # Publish completion status
                status_msg = String()
                status_msg.data = "trajectory_complete"
                self.status_pub.publish(status_msg)
                
                # Continue publishing last point
                t_query = self.trajectory_duration
        
        # Interpolate reference state
        if self.interpolation_method == 'linear':
            ref_state = self.interpolate_linear(t_query)
        else:
            rospy.logwarn_once(f"Interpolation method '{self.interpolation_method}' not supported, using linear")
            ref_state = self.interpolate_linear(t_query)
        
        # Publish reference state
        self.ref_pub.publish(ref_state)
    
    def spin(self):
        """Main loop"""
        rospy.spin()


def main():
    """Main entry point"""
    try:
        server = TrajServer()
        server.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Trajectory Server shutting down")


if __name__ == '__main__':
    main()
