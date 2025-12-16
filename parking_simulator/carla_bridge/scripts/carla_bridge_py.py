#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CARLA Bridge Node - Python Implementation
Connects CARLA simulator with ROS ecosystem
"""

import rospy
import carla
import math
import sys
from vehicle_msgs.msg import VehicleState, ControlCmd, Trajectory
from geometry_msgs.msg import PoseStamped


class CarlaBridgePy:
    """Bridge between CARLA simulator and ROS"""
    
    def __init__(self):
        """Initialize CARLA Bridge node"""
        rospy.init_node('carla_bridge_py', anonymous=False)
        
        # Load parameters
        self.load_parameters()
        
        # CARLA client and vehicle
        self.client = None
        self.world = None
        self.vehicle = None
        self.obstacle_vehicles = [] # List to store obstacle vehicle actors
        
        # ROS publishers and subscribers
        self.state_pub = rospy.Publisher('/vehicle_state', VehicleState, queue_size=10)
        self.control_sub = rospy.Subscriber('/control_cmd', ControlCmd, self.control_callback)
        self.trajectory_sub = rospy.Subscriber('/planned_trajectory', Trajectory, self.trajectory_callback)
        
        # Control command buffer
        self.current_control = ControlCmd()
        self.current_control.throttle = 0.0
        self.current_control.brake = 0.0
        self.current_control.steer = 0.0
        self.current_control.gear = 1
        self.current_control.reverse = False
        
        # Trajectory visualization
        self.planned_trajectory = None
        self.trajectory_drawn = False
        
        # Actual trajectory tracking
        self.actual_trajectory = []  # List of (x_ros, y_ros) tuples
        self.last_actual_pos = None
        
        # Connect to CARLA 
        self.connect_carla()
        
        # Spawn static obstacles
        self.spawn_obstacles()
        
        # Spawn ego vehicle
        self.spawn_vehicle()

        # Set the view
        spectator = self.world.get_spectator()
        bv_transform = carla.Transform(carla.Location(x=17, y=2, z=30), carla.Rotation(yaw=90, pitch=-90))
        spectator.set_transform(bv_transform)
        
        rospy.loginfo("CARLA Bridge initialized successfully")
    
    def load_parameters(self):
        """Load parameters from ROS parameter server"""
        # CARLA connection parameters
        self.host = rospy.get_param('~carla/host', 'localhost')
        self.port = rospy.get_param('~carla/port', 2000)
        self.timeout = rospy.get_param('~carla/timeout', 10.0)
        self.vehicle_filter = rospy.get_param('~carla/vehicle_filter', 'haut')
        
        # Spawn point
        self.spawn_x = rospy.get_param('~carla/spawn_point/x', 18.3)
        self.spawn_y = rospy.get_param('~carla/spawn_point/y', -12.0)
        self.spawn_z = rospy.get_param('~carla/spawn_point/z', 0.05)
        self.spawn_yaw = rospy.get_param('~carla/spawn_point/yaw', 90.0)
        
        # Coordinate transformation
        self.x_scale = rospy.get_param('~carla/coordinate_transform/x_scale', -1.0)
        self.y_scale = rospy.get_param('~carla/coordinate_transform/y_scale', 1.0)
        self.yaw_offset = rospy.get_param('~carla/coordinate_transform/yaw_offset', 180.0)
        
        # Vehicle parameters
        self.wheelbase = rospy.get_param('~carla/wheelbase', 3.368)
        
        # Publishing rate
        self.publish_rate = rospy.get_param('~publish_rate', 50.0)
        
        rospy.loginfo(f"Loaded parameters: host={self.host}, port={self.port}")
    
    def connect_carla(self):
        """Connect to CARLA server"""
        try:
            rospy.loginfo(f"Connecting to CARLA server at {self.host}:{self.port}...")
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(self.timeout)
            self.world = self.client.get_world()
            rospy.loginfo("Successfully connected to CARLA server")
            
            # Set weather
            weather = self.world.get_weather()
            self.world.set_weather(carla.WeatherParameters.CloudyNoon)
            rospy.loginfo("Weather set to CloudyNoon")
            
        except Exception as e:
            rospy.logerr(f"Failed to connect to CARLA: {e}")
            sys.exit(1)
    
    def ros_to_carla(self, x_ros, y_ros, yaw_ros_deg):
        """
        Convert ROS coordinates to CARLA coordinates for spawning
        
        Args:
            x_ros: X in ROS frame
            y_ros: Y in ROS frame
            yaw_ros_deg: Yaw in ROS frame (degrees)
            
        Returns:
            tuple: (x_carla, y_carla, yaw_carla_deg)
        """
        # Inverse transformation: ROS -> CARLA
        x_carla = -x_ros
        y_carla = y_ros
        yaw_carla_deg = 180.0 - yaw_ros_deg
        
        return x_carla, y_carla, yaw_carla_deg
    
    def spawn_vehicle(self):
        """Spawn vehicle in CARLA world"""
        try:
            # Get blueprint library
            blueprint_library = self.world.get_blueprint_library()
            
            # Find vehicle blueprint
            vehicle_bp = None
            for bp in blueprint_library.filter('vehicle.*'):
                if self.vehicle_filter.lower() in bp.id.lower():
                    vehicle_bp = bp
                    break
            
            if vehicle_bp is None:
                rospy.logwarn(f"Vehicle filter '{self.vehicle_filter}' not found, using default vehicle")
                vehicle_bp = blueprint_library.filter('vehicle.*')[0]
            
            rospy.loginfo(f"Using vehicle blueprint: {vehicle_bp.id}")
            
            # Convert ROS spawn point to CARLA coordinates
            x_carla, y_carla, yaw_carla = self.ros_to_carla(
                self.spawn_x, self.spawn_y, self.spawn_yaw
            )
            
            rospy.loginfo(f"Spawn point ROS: ({self.spawn_x:.2f}, {self.spawn_y:.2f}, {self.spawn_yaw:.1f}°)")
            rospy.loginfo(f"Spawn point CARLA: ({x_carla:.2f}, {y_carla:.2f}, {yaw_carla:.1f}°)")
            
            # Create spawn transform
            spawn_transform = carla.Transform(
                carla.Location(x=x_carla, y=y_carla, z=self.spawn_z),
                carla.Rotation(yaw=yaw_carla)
            )
            
            # Spawn vehicle
            self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_transform)
            rospy.loginfo(f"Vehicle spawned successfully")
            
        except Exception as e:
            rospy.logerr(f"Failed to spawn vehicle: {e}")
            sys.exit(1)
    
    def spawn_obstacles(self):
        """Spawn static obstacle vehicles if they don't already exist"""
        try:
            # 1. Fetch obstacle list from ROS Parameter Server
            static_vehicles_param = rospy.get_param('~planner_node/obstacles/static_vehicles', [])
            if not static_vehicles_param:
                rospy.logwarn("No static vehicles found in params (/planner_node/obstacles/static_vehicles). Skipping obstacle spawn.")
                return

            rospy.loginfo(f"Found {len(static_vehicles_param)} static obstacles in configuration.")
            
            blueprint_library = self.world.get_blueprint_library()
            
            # Find vehicle blueprint
            obstacle_bp = None
            for bp in blueprint_library.filter('vehicle.*'):
                if 'haut' in bp.id.lower(): 
                    obstacle_bp = bp
                    break
            
            if obstacle_bp is None:
                obstacle_bp = blueprint_library.filter('vehicle.lincoln.mkz_2017')[0]
                rospy.logwarn("Obstacle filter 'haut' not found, using default Lincoln MKZ")
   
            # 1. Get existing actors in the world to check for duplicates
            existing_actors = self.world.get_actors().filter('vehicle.*')

            for i, vehicle_data in enumerate(static_vehicles_param):
                # Get ROS coordinates from YAML
                name = str(vehicle_data['name'])
                x = float(vehicle_data['x'])
                y = float(vehicle_data['y'])
                yaw = float(vehicle_data['yaw'])
                
                target_location = carla.Location(x=x, y=y, z=0.1)
                
                # Check if an obstacle already exists at this location
                already_exists = False
                for actor in existing_actors.filter('vehicle.*'):
                    # Calculate distance
                    dist = actor.get_location().distance(target_location)
                    if dist < 1.0: # If a vehicle is within 1 meter
                        already_exists = True
                        rospy.loginfo(f"Obstacle {name} already exists at location. Skipping spawn.")
                        # We can add it to our list to manage it if needed, but requirements say don't destroy.
                        # self.obstacle_vehicles.append(actor) 
                        break
                
                if not already_exists:
                    # Set role_name to identify it later if needed
                    obstacle_bp.set_attribute('role_name', name)
                    
                    transform = carla.Transform(
                        target_location,
                        carla.Rotation(yaw=yaw)
                    )
                    
                    actor = self.world.try_spawn_actor(obstacle_bp, transform)
                    if actor:
                        actor.set_autopilot(False)
                        control = carla.VehicleControl()
                        control.hand_brake = True
                        actor.apply_control(control)
                        
                        self.obstacle_vehicles.append(actor)
                        rospy.loginfo(f"Spawned obstacle {name} at ({x}, {y}, {yaw}°)")
                    else:
                        rospy.logwarn(f"Failed to spawn obstacle {name} at ({x}, {y})")

        except Exception as e:
            rospy.logerr(f"Error spawning obstacles: {e}")
    
    def carla_to_ros(self, carla_transform, carla_velocity):
        """
        Convert CARLA coordinates to ROS coordinates
        """
        # Extract CARLA coordinates (vehicle center)
        x_carla_center = carla_transform.location.x
        y_carla_center = carla_transform.location.y
        yaw_carla_deg = carla_transform.rotation.yaw
        
        # Convert yaw to radians
        yaw_carla_rad = math.radians(yaw_carla_deg)
        
        # Convert vehicle center to rear axle center in CARLA coordinates
        x_carla_rear = x_carla_center - (self.wheelbase / 2.0) * math.cos(yaw_carla_rad)
        y_carla_rear = y_carla_center - (self.wheelbase / 2.0) * math.sin(yaw_carla_rad)
        
        # Transform to ROS coordinates
        x_ros = self.x_scale * x_carla_rear
        y_ros = self.y_scale * y_carla_rear
        
        # Transform yaw: yaw_ros = yaw_offset - yaw_carla
        # This ensures: CARLA yaw=0° -> ROS yaw=180°, CARLA yaw=90° -> ROS yaw=90°
        yaw_ros = math.radians(self.yaw_offset) - yaw_carla_rad
        
        # Normalize yaw to [-pi, pi]
        yaw_ros = self.normalize_angle(yaw_ros)
        
        # Transform velocity to vehicle body frame
        # Velocity in body frame should not be affected by world coordinate mirroring
        vx_carla = carla_velocity.x
        vy_carla = carla_velocity.y
        
        # Convert to body frame velocity (always relative to vehicle orientation)
        # vx: forward velocity, vy: lateral velocity
        vx_ros = vx_carla * math.cos(yaw_carla_rad) + vy_carla * math.sin(yaw_carla_rad)
        vy_ros = vx_carla * math.sin(yaw_carla_rad) - vy_carla * math.cos(yaw_carla_rad)
        
        return x_ros, y_ros, yaw_ros, vx_ros, vy_ros
    
    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle
    
    def publish_vehicle_state(self):
        """Publish vehicle state to ROS"""
        if self.vehicle is None:
            return
        
        try:
            # Get vehicle transform and velocity from CARLA
            carla_transform = self.vehicle.get_transform()
            carla_velocity = self.vehicle.get_velocity()
            carla_angular_velocity = self.vehicle.get_angular_velocity()
            
            # Convert to ROS coordinates
            x_ros, y_ros, yaw_ros, vx_ros, vy_ros = self.carla_to_ros(carla_transform, carla_velocity)
            
            # Debug: log first state
            if not hasattr(self, '_first_state_logged'):
                rospy.loginfo(f"CARLA vehicle: yaw={carla_transform.rotation.yaw:.1f}°")
                rospy.loginfo(f"ROS vehicle_state: yaw={math.degrees(yaw_ros):.1f}° ({yaw_ros:.3f} rad)")
                self._first_state_logged = True
            
            # Calculate angular velocity (omega)
            omega = math.radians(carla_angular_velocity.z) * -1
            
            # Record actual trajectory point (only if moved enough)
            min_dist = 0.1  # Minimum distance between recorded points
            if self.last_actual_pos is None:
                self.actual_trajectory.append((x_ros, y_ros))
                self.last_actual_pos = (x_ros, y_ros)
            else:
                dist = math.sqrt((x_ros - self.last_actual_pos[0])**2 + 
                                (y_ros - self.last_actual_pos[1])**2)
                if dist >= min_dist:
                    self.actual_trajectory.append((x_ros, y_ros))
                    self.last_actual_pos = (x_ros, y_ros)
            
            # Create VehicleState message
            state_msg = VehicleState()
            state_msg.x = x_ros
            state_msg.y = y_ros
            state_msg.yaw = yaw_ros
            state_msg.vx = vx_ros
            state_msg.vy = vy_ros
            state_msg.omega = omega
            state_msg.gear = self.current_control.gear
            state_msg.reverse = self.current_control.reverse
            state_msg.stamp = rospy.Time.now()
            
            # Publish
            self.state_pub.publish(state_msg)
            
        except Exception as e:
            rospy.logerr(f"Error publishing vehicle state: {e}")
    
    def control_callback(self, msg):
        """
        Callback for control command
        
        Args:
            msg: ControlCmd message
        """
        self.current_control = msg
        self.apply_control()
    
    def trajectory_callback(self, msg):
        """
        Callback for planned trajectory - draws it in CARLA
        
        Args:
            msg: Trajectory message
        """
        self.planned_trajectory = msg
        self.draw_trajectory_in_carla()
    
    def draw_trajectory_in_carla(self):
        """Draw the planned trajectory in CARLA world"""
        if self.planned_trajectory is None or self.world is None:
            return
        
        try:
            # Get trajectory data
            x_list = self.planned_trajectory.x
            y_list = self.planned_trajectory.y
            
            if len(x_list) < 2:
                rospy.logwarn("Trajectory has less than 2 points, cannot draw")
                return
            
            # rospy.loginfo(f"Drawing trajectory with {len(x_list)} points in CARLA")
            
            # Draw lines between consecutive points
            for i in range(len(x_list) - 1):
                # Convert ROS coordinates back to CARLA coordinates
                # ROS: x_ros = -x_carla, y_ros = y_carla
                # So: x_carla = -x_ros, y_carla = y_ros
                start = carla.Location(
                    x=-x_list[i],      # ROS to CARLA: negate x
                    y=y_list[i],       # ROS to CARLA: keep y
                    z=0.5              # Height above ground for visibility
                )
                end = carla.Location(
                    x=-x_list[i+1],    # ROS to CARLA: negate x
                    y=y_list[i+1],     # ROS to CARLA: keep y
                    z=0.5
                )
                
                # Draw green line with 1.5 second lifetime
                self.world.debug.draw_line(
                    start, 
                    end, 
                    thickness=0.2, 
                    color=carla.Color(r=0, g=255, b=0),  # Green color
                    life_time=1.5
                )
            
            # Draw start point (red sphere)
            start_location = carla.Location(x=-x_list[0], y=y_list[0], z=0.5)
            self.world.debug.draw_point(
                start_location,
                size=0.3,
                color=carla.Color(r=255, g=0, b=0),  # Red
                life_time=1.5
            )
            
            # Draw end point (blue sphere)
            end_location = carla.Location(x=-x_list[-1], y=y_list[-1], z=0.5)
            self.world.debug.draw_point(
                end_location,
                size=0.3,
                color=carla.Color(r=0, g=0, b=255),  # Blue
                life_time=1.5
            )
            
            if not self.trajectory_drawn:
                rospy.loginfo("Trajectory visualization enabled in CARLA")
                rospy.loginfo(f"  Start (ROS): ({x_list[0]:.2f}, {y_list[0]:.2f})")
                rospy.loginfo(f"  End (ROS): ({x_list[-1]:.2f}, {y_list[-1]:.2f})")
                # rospy.loginfo(f"  Start (CARLA): ({-x_list[0]:.2f}, {y_list[0]:.2f})")
                # rospy.loginfo(f"  End (CARLA): ({-x_list[-1]:.2f}, {y_list[-1]:.2f})")
                self.trajectory_drawn = True
                
        except Exception as e:
            rospy.logerr(f"Error drawing trajectory in CARLA: {e}")
    
    def draw_actual_trajectory_in_carla(self):
        """Draw the actual vehicle trajectory in CARLA world (blue line)"""
        if self.world is None or len(self.actual_trajectory) < 2:
            return
        
        try:
            # Draw lines between consecutive points
            for i in range(len(self.actual_trajectory) - 1):
                x_ros_start, y_ros_start = self.actual_trajectory[i]
                x_ros_end, y_ros_end = self.actual_trajectory[i + 1]
                
                # Convert ROS coordinates to CARLA coordinates
                start = carla.Location(
                    x=-x_ros_start,    # ROS to CARLA: negate x
                    y=y_ros_start,     # ROS to CARLA: keep y
                    z=0.6              # Height above ground for visibility
                )
                end = carla.Location(
                    x=-x_ros_end,
                    y=y_ros_end,
                    z=0.6
                )
                
                # Draw blue line with same thickness as reference trajectory
                self.world.debug.draw_line(
                    start,
                    end,
                    thickness=0.2,
                    color=carla.Color(r=0, g=0, b=255),  # Blue color
                    life_time=1.5
                )
                
        except Exception as e:
            rospy.logerr(f"Error drawing actual trajectory in CARLA: {e}")
    
    def apply_control(self):
        """Apply control command to CARLA vehicle"""
        if self.vehicle is None:
            return
        
        try:
            # Create CARLA control
            control = carla.VehicleControl()
            
            # Set throttle and brake
            control.throttle = float(max(0.0, min(1.0, self.current_control.throttle)))
            control.brake = float(max(0.0, min(1.0, self.current_control.brake)))
            
            # Set steering (CARLA uses [-1, 1] range)
            # Note: Negated to match coordinate system transformation
            ros_steer = float(max(-1.0, min(1.0, self.current_control.steer)))
            control.steer = -1 * ros_steer
                      
            # Set gear and reverse
            control.manual_gear_shift = True
            if self.current_control.reverse:
                control.gear = -1
                control.reverse = True
            else:
                control.gear = max(1, self.current_control.gear)
                control.reverse = False
            
            # Apply control to vehicle
            self.vehicle.apply_control(control)
            
        except Exception as e:
            rospy.logerr(f"Error applying control: {e}")
    
    def spin(self):
        """Main loop"""
        rate = rospy.Rate(self.publish_rate)
        
        rospy.loginfo("CARLA Bridge spinning...")
        
        while not rospy.is_shutdown():
            self.publish_vehicle_state()
            
            # Continuously redraw trajectory (since CARLA debug lines have lifetime)
            if self.planned_trajectory is not None:
                self.draw_trajectory_in_carla()
            
            # Draw actual trajectory (blue line)
            self.draw_actual_trajectory_in_carla()
            
            rate.sleep()
    
    def cleanup(self):
        """Cleanup resources"""
        rospy.loginfo("Cleaning up CARLA Bridge...")
        if self.vehicle is not None:
            try:
                self.vehicle.destroy()
                rospy.loginfo("Vehicle destroyed")
            except Exception as e:
                rospy.logerr(f"Error destroying vehicle: {e}")
                
        # NOT destroying obstacle vehicles per request
        rospy.loginfo("Obstacle vehicles persist in simulation")


def main():
    """Main entry point"""
    try:
        bridge = CarlaBridgePy()
        bridge.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        if 'bridge' in locals():
            bridge.cleanup()


if __name__ == '__main__':
    main()
