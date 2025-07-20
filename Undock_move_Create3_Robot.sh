#!/bin/bash

ros2 action send_goal /undock irobot_create_msgs/action/Undock "{}"
sleep 5

# Turn
for i in {1..3}
do
  ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist \
    "{linear: {x: 0.2}, angular: {z: 0.3}}"
  sleep 0.05
done

# Drive straight
for i in {1..40}
do
  ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist \
    "{linear: {x: 1.5}, angular: {z: 0.0}}"
done
