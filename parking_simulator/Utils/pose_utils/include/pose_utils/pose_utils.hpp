#ifndef POSE_UTILS_HPP
#define POSE_UTILS_HPP

#include <cmath>

namespace pose_utils {

/**
 * @brief Convert vehicle center coordinates to rear axle center coordinates
 * @param x_center X coordinate of vehicle center (m)
 * @param y_center Y coordinate of vehicle center (m)
 * @param yaw Vehicle heading angle (rad)
 * @param wheelbase Vehicle wheelbase (m)
 * @param x_rear Output: X coordinate of rear axle center (m)
 * @param y_rear Output: Y coordinate of rear axle center (m)
 */
inline void centerToRearAxle(double x_center, double y_center, double yaw,
                              double wheelbase,
                              double& x_rear, double& y_rear) {
    x_rear = x_center - (wheelbase / 2.0) * std::cos(yaw);
    y_rear = y_center - (wheelbase / 2.0) * std::sin(yaw);
}

/**
 * @brief Convert rear axle center coordinates to vehicle center coordinates
 * @param x_rear X coordinate of rear axle center (m)
 * @param y_rear Y coordinate of rear axle center (m)
 * @param yaw Vehicle heading angle (rad)
 * @param wheelbase Vehicle wheelbase (m)
 * @param x_center Output: X coordinate of vehicle center (m)
 * @param y_center Output: Y coordinate of vehicle center (m)
 */
inline void rearAxleToCenter(double x_rear, double y_rear, double yaw,
                              double wheelbase,
                              double& x_center, double& y_center) {
    x_center = x_rear + (wheelbase / 2.0) * std::cos(yaw);
    y_center = y_rear + (wheelbase / 2.0) * std::sin(yaw);
}

/**
 * @brief Convert CARLA coordinates to ROS planning coordinates
 * @param x_carla X coordinate in CARLA frame (m)
 * @param y_carla Y coordinate in CARLA frame (m)
 * @param yaw_carla Heading angle in CARLA frame (degrees)
 * @param x_ros Output: X coordinate in ROS frame (m)
 * @param y_ros Output: Y coordinate in ROS frame (m)
 * @param yaw_ros Output: Heading angle in ROS frame (rad)
 */
inline void carlaToRos(double x_carla, double y_carla, double yaw_carla,
                       double& x_ros, double& y_ros, double& yaw_ros) {
    x_ros = -x_carla;
    y_ros = y_carla;
    yaw_ros = M_PI - yaw_carla * M_PI / 180.0;
    
    // Normalize the resulting angle
    while (yaw_ros > M_PI) yaw_ros -= 2.0 * M_PI;
    while (yaw_ros < -M_PI) yaw_ros += 2.0 * M_PI;
}

/**
 * @brief Convert ROS planning coordinates to CARLA coordinates
 * @param x_ros X coordinate in ROS frame (m)
 * @param y_ros Y coordinate in ROS frame (m)
 * @param yaw_ros Heading angle in ROS frame (rad)
 * @param x_carla Output: X coordinate in CARLA frame (m)
 * @param y_carla Output: Y coordinate in CARLA frame (m)
 * @param yaw_carla Output: Heading angle in CARLA frame (degrees)
 */
inline void rosToCarla(double x_ros, double y_ros, double yaw_ros,
                       double& x_carla, double& y_carla, double& yaw_carla) {
    x_carla = -x_ros;
    y_carla = y_ros;
    yaw_carla = (M_PI - yaw_ros) * 180.0 / M_PI;
    
    // Normalize to [0, 360)
    while (yaw_carla >= 360.0) yaw_carla -= 360.0;
    while (yaw_carla < 0.0) yaw_carla += 360.0;
}

/**
 * @brief Normalize angle to [-π, π] range
 * @param angle Input angle (rad)
 * @return Normalized angle in [-π, π] (rad)
 */
inline double normalizeAngle(double angle) {
    while (angle > M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
}

/**
 * @brief Convert Euler angles to quaternion
 * @param roll Roll angle (rad)
 * @param pitch Pitch angle (rad)
 * @param yaw Yaw angle (rad)
 * @param qx Output: Quaternion x component
 * @param qy Output: Quaternion y component
 * @param qz Output: Quaternion z component
 * @param qw Output: Quaternion w component
 */
inline void eulerToQuaternion(double roll, double pitch, double yaw,
                               double& qx, double& qy, double& qz, double& qw) {
    double cy = std::cos(yaw * 0.5);
    double sy = std::sin(yaw * 0.5);
    double cp = std::cos(pitch * 0.5);
    double sp = std::sin(pitch * 0.5);
    double cr = std::cos(roll * 0.5);
    double sr = std::sin(roll * 0.5);
    
    qw = cr * cp * cy + sr * sp * sy;
    qx = sr * cp * cy - cr * sp * sy;
    qy = cr * sp * cy + sr * cp * sy;
    qz = cr * cp * sy - sr * sp * cy;
}

/**
 * @brief Convert quaternion to Euler angles
 * @param qx Quaternion x component
 * @param qy Quaternion y component
 * @param qz Quaternion z component
 * @param qw Quaternion w component
 * @param roll Output: Roll angle (rad)
 * @param pitch Output: Pitch angle (rad)
 * @param yaw Output: Yaw angle (rad)
 */
inline void quaternionToEuler(double qx, double qy, double qz, double qw,
                               double& roll, double& pitch, double& yaw) {
    // Roll (x-axis rotation)
    double sinr_cosp = 2.0 * (qw * qx + qy * qz);
    double cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy);
    roll = std::atan2(sinr_cosp, cosr_cosp);
    
    // Pitch (y-axis rotation)
    double sinp = 2.0 * (qw * qy - qz * qx);
    if (std::abs(sinp) >= 1.0)
        pitch = std::copysign(M_PI / 2.0, sinp); // Use 90 degrees if out of range
    else
        pitch = std::asin(sinp);
    
    // Yaw (z-axis rotation)
    double siny_cosp = 2.0 * (qw * qz + qx * qy);
    double cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz);
    yaw = std::atan2(siny_cosp, cosy_cosp);
}

} // namespace pose_utils

#endif // POSE_UTILS_HPP
