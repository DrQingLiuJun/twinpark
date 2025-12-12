#include "pose_utils/pose_utils.hpp"
#include <iostream>
#include <cmath>
#include <iomanip>

// Simple test helper
bool approxEqual(double a, double b, double epsilon = 1e-6) {
    return std::abs(a - b) < epsilon;
}

void testCenterToRearAxle() {
    std::cout << "Testing centerToRearAxle..." << std::endl;
    
    double x_rear, y_rear;
    double wheelbase = 3.368;
    
    // Test 1: Vehicle facing east (yaw = 0)
    pose_utils::centerToRearAxle(10.0, 5.0, 0.0, wheelbase, x_rear, y_rear);
    if (approxEqual(x_rear, 10.0 - wheelbase/2.0) && approxEqual(y_rear, 5.0)) {
        std::cout << "  ✓ Test 1 passed: yaw=0" << std::endl;
    } else {
        std::cout << "  ✗ Test 1 failed: expected (" << (10.0 - wheelbase/2.0) 
                  << ", 5.0), got (" << x_rear << ", " << y_rear << ")" << std::endl;
    }
    
    // Test 2: Vehicle facing north (yaw = π/2)
    pose_utils::centerToRearAxle(10.0, 5.0, M_PI/2.0, wheelbase, x_rear, y_rear);
    if (approxEqual(x_rear, 10.0) && approxEqual(y_rear, 5.0 - wheelbase/2.0)) {
        std::cout << "  ✓ Test 2 passed: yaw=π/2" << std::endl;
    } else {
        std::cout << "  ✗ Test 2 failed" << std::endl;
    }
}

void testRearAxleToCenter() {
    std::cout << "\nTesting rearAxleToCenter..." << std::endl;
    
    double x_center, y_center;
    double wheelbase = 3.368;
    
    // Test: Inverse of centerToRearAxle
    double x_rear = 8.316;
    double y_rear = 5.0;
    pose_utils::rearAxleToCenter(x_rear, y_rear, 0.0, wheelbase, x_center, y_center);
    
    if (approxEqual(x_center, 10.0, 1e-3) && approxEqual(y_center, 5.0)) {
        std::cout << "  ✓ Test passed: inverse transformation" << std::endl;
    } else {
        std::cout << "  ✗ Test failed: expected (10.0, 5.0), got (" 
                  << x_center << ", " << y_center << ")" << std::endl;
    }
}

void testCarlaToRos() {
    std::cout << "\nTesting carlaToRos..." << std::endl;
    
    double x_ros, y_ros, yaw_ros;
    
    // Test 1: CARLA (10, 5, 0°) -> ROS (-10, 5, π)
    pose_utils::carlaToRos(10.0, 5.0, 0.0, x_ros, y_ros, yaw_ros);
    if (approxEqual(x_ros, -10.0) && approxEqual(y_ros, 5.0) && approxEqual(yaw_ros, M_PI)) {
        std::cout << "  ✓ Test 1 passed: CARLA yaw=0°" << std::endl;
    } else {
        std::cout << "  ✗ Test 1 failed: got (" << x_ros << ", " << y_ros 
                  << ", " << yaw_ros << ")" << std::endl;
    }
    
    // Test 2: CARLA (0, 0, 90°) -> ROS (0, 0, π/2)
    pose_utils::carlaToRos(0.0, 0.0, 90.0, x_ros, y_ros, yaw_ros);
    if (approxEqual(x_ros, 0.0) && approxEqual(y_ros, 0.0) && approxEqual(yaw_ros, M_PI/2.0)) {
        std::cout << "  ✓ Test 2 passed: CARLA yaw=90°" << std::endl;
    } else {
        std::cout << "  ✗ Test 2 failed: got yaw_ros=" << yaw_ros << std::endl;
    }
}

void testNormalizeAngle() {
    std::cout << "\nTesting normalizeAngle..." << std::endl;
    
    // Test 1: Angle > π
    double result = pose_utils::normalizeAngle(3.5 * M_PI);
    if (approxEqual(result, -0.5 * M_PI)) {
        std::cout << "  ✓ Test 1 passed: 3.5π -> -0.5π" << std::endl;
    } else {
        std::cout << "  ✗ Test 1 failed: expected " << (-0.5 * M_PI) 
                  << ", got " << result << std::endl;
    }
    
    // Test 2: Angle < -π
    result = pose_utils::normalizeAngle(-3.5 * M_PI);
    if (approxEqual(result, 0.5 * M_PI)) {
        std::cout << "  ✓ Test 2 passed: -3.5π -> 0.5π" << std::endl;
    } else {
        std::cout << "  ✗ Test 2 failed" << std::endl;
    }
    
    // Test 3: Already normalized
    result = pose_utils::normalizeAngle(0.5);
    if (approxEqual(result, 0.5)) {
        std::cout << "  ✓ Test 3 passed: 0.5 -> 0.5" << std::endl;
    } else {
        std::cout << "  ✗ Test 3 failed" << std::endl;
    }
}

void testEulerQuaternionConversion() {
    std::cout << "\nTesting Euler-Quaternion conversion..." << std::endl;
    
    double roll = 0.1, pitch = 0.2, yaw = 0.3;
    double qx, qy, qz, qw;
    double roll_out, pitch_out, yaw_out;
    
    // Convert Euler -> Quaternion -> Euler
    pose_utils::eulerToQuaternion(roll, pitch, yaw, qx, qy, qz, qw);
    pose_utils::quaternionToEuler(qx, qy, qz, qw, roll_out, pitch_out, yaw_out);
    
    if (approxEqual(roll, roll_out) && approxEqual(pitch, pitch_out) && approxEqual(yaw, yaw_out)) {
        std::cout << "  ✓ Test passed: round-trip conversion" << std::endl;
    } else {
        std::cout << "  ✗ Test failed: " << std::endl;
        std::cout << "    Input:  (" << roll << ", " << pitch << ", " << yaw << ")" << std::endl;
        std::cout << "    Output: (" << roll_out << ", " << pitch_out << ", " << yaw_out << ")" << std::endl;
    }
    
    // Test identity quaternion (0, 0, 0) -> (0, 0, 0, 1)
    pose_utils::eulerToQuaternion(0.0, 0.0, 0.0, qx, qy, qz, qw);
    if (approxEqual(qx, 0.0) && approxEqual(qy, 0.0) && approxEqual(qz, 0.0) && approxEqual(qw, 1.0)) {
        std::cout << "  ✓ Test passed: identity quaternion" << std::endl;
    } else {
        std::cout << "  ✗ Test failed: identity quaternion" << std::endl;
    }
}

int main() {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "=== pose_utils Test Suite ===" << std::endl << std::endl;
    
    testCenterToRearAxle();
    testRearAxleToCenter();
    testCarlaToRos();
    testNormalizeAngle();
    testEulerQuaternionConversion();
    
    std::cout << "\n=== Tests Complete ===" << std::endl;
    
    return 0;
}
