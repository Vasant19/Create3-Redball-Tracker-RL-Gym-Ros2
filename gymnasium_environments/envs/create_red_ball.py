import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
from irobot_create_msgs.msg import StopStatus


class RedBall(Node):
    def __init__(self):
        super().__init__('redball')
        self.subscription = self.create_subscription(
            Image,
            'custom_ns/camera1/image_raw',
            self.listener_callback,
            10
        )
        self.subscription  # prevent unused variable warning

        self.br = CvBridge()
        self.target_publisher = self.create_publisher(Image, 'target_redball', 10)
        self.twist_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.create3_is_stopped = True
        self.redball_position = None
        self.last_seen_direction = 1  # default spin direction
        self.turn_gain = 0.5

        self.create_subscription(
            StopStatus,
            '/stop_status',
            self.stop_status_callback,
            10
        )

    def step(self, action_pixel):
        self.create3_is_stopped = False

        angle = (action_pixel - 320) / 320 * (np.pi / 2) * self.turn_gain
        twist = Twist()
        twist.angular.z = angle

        self.twist_publisher.publish(twist)

        while not self.create3_is_stopped:
            rclpy.spin_once(self, timeout_sec=0.1)

        return self.redball_position

    def listener_callback(self, msg):
        frame = self.br.imgmsg_to_cv2(msg)
        hsv_conv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        bright_red_lower_bounds = (110, 100, 100)
        bright_red_upper_bounds = (130, 255, 255)
        bright_red_mask = cv2.inRange(hsv_conv_img, bright_red_lower_bounds, bright_red_upper_bounds)

        blurred_mask = cv2.GaussianBlur(bright_red_mask, (9, 9), 3, 3)

        erode_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilate_element = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
        eroded_mask = cv2.erode(blurred_mask, erode_element)
        dilated_mask = cv2.dilate(eroded_mask, dilate_element)

        detected_circles = cv2.HoughCircles(
            dilated_mask, cv2.HOUGH_GRADIENT, 1, 150,
            param1=100, param2=20, minRadius=2, maxRadius=2000
        )

        if detected_circles is not None:
            circles = sorted(detected_circles[0, :], key=lambda c: c[2], reverse=True)
            best_circle = circles[0]
            x, y, r = int(best_circle[0]), int(best_circle[1]), int(best_circle[2])
            circled_orig = cv2.circle(frame, (x, y), r, (0, 255, 0), thickness=3)

            self.redball_position = x
            self.last_seen_direction = 1 if x < 320 else -1
            self.get_logger().info(f'✅ Red ball detected at x={self.redball_position}')
            self.target_publisher.publish(self.br.cv2_to_imgmsg(circled_orig))
        else:
            self.redball_position = None
            self.get_logger().info('❌ No ball detected')

    def stop_status_callback(self, msg):
        self.create3_is_stopped = msg.is_stopped

    def search_for_ball(self, timeout=5.0):
        self.get_logger().info('🔍 Searching for red ball...')
        twist = Twist()

        # Try spinning in the last known direction, then opposite if needed
        directions = [self.last_seen_direction, -self.last_seen_direction]

        for direction in directions:
            twist.angular.z = 0.8 * direction
            start_time = self.get_clock().now()

            while (self.redball_position is None and
                   (self.get_clock().now() - start_time).nanoseconds / 1e9 < timeout / 2):
                self.twist_publisher.publish(twist)
                rclpy.spin_once(self, timeout_sec=0.1)

            if self.redball_position is not None:
                self.get_logger().info('🎯 Red ball reacquired during search!')
                break

        twist.angular.z = 0.0
        self.twist_publisher.publish(twist)

        if self.redball_position is None:
            self.get_logger().info('⚠️ Red ball NOT found after search')


class CreateRedBallEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode=None):
        super().__init__()
        rclpy.init()

        self.observation_space = spaces.Discrete(641)  # pixel positions 0 to 640
        self.action_space = spaces.Discrete(641)       # same range for actions
        self.render_mode = render_mode
        self.redball = RedBall()
        self.step_count = 0

        self.missing_count = 0
        self.recovery_threshold = 5

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.missing_count = 0

        while self.redball.redball_position is None:
            rclpy.spin_once(self.redball, timeout_sec=0.1)

        observation = self.redball.redball_position
        info = {}
        return observation, info

    def step(self, action):
        self.step_count += 1

        observation = self.redball.step(action)

        if observation is None:
            self.missing_count += 1
            print(f"⚠️ Ball not detected. Missing count = {self.missing_count}")
            reward = -1.0
            if self.missing_count >= self.recovery_threshold:
                self.redball.search_for_ball()
                if self.redball.redball_position is not None:
                    self.missing_count = 0  # reset if ball found during search

            observation = 320  # fallback to center
        else:
            self.missing_count = 0
            reward = -abs(observation - 320) / 320

        terminated = self.step_count >= 100
        truncated = False
        info = {}

        print(f"Step {self.step_count}: Action = {action}, Observation = {observation}, Reward = {reward}")

        return observation, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        if rclpy.ok():
            self.redball.destroy_node()
            rclpy.shutdown()
