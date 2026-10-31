#!/bin/bash
# Gazebo と ROS 2 をつなぐ「橋」
#   ]  : ROS → Gazebo（速度の指示）
#   [  : Gazebo → ROS（位置・カメラ映像）
ros2 run ros_gz_bridge parameter_bridge \
  "/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist" \
  "/bulldozer/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry" \
  "/bulldozer/camera@sensor_msgs/msg/Image[gz.msgs.Image" \
  "/overhead/image@sensor_msgs/msg/Image[gz.msgs.Image"
