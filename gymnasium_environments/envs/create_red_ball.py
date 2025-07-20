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
    """
    A Node to analyse red balls in images and publish the results
    """
    def __init__(self):
        super().__init__('redball')
        self.subscription = self.create_subscription(
            Image,
            'custom_ns/camera1/image_raw',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

        self.br = CvBridge()
        self.target_publisher = self.create_publisher(Image, 'target_redball', 10)
        self.twist_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.create3_is_stopped = True
        self.redball_position = None  # None means no detection yet

        self.create_subscription(
            StopStatus,
            '/stop_status',
            self.stop_status_callback,
            10
        )

    def step(self, action_pixel):
        self.create3_is_stopped = False

        angle = (action_pixel - 320) / 320 * (np.pi / 2)
        twist = Twist()
        twist.angular.z = angle

        self.twist_publisher.publish(twist)

        while not self.create3_is_stopped:
            rclpy.spin_once(self, timeout_sec=0.1)

        return self.redball_position

    def listener_callback(self, msg):
        frame = self.br.imgmsg_to_cv2(msg)
        # convert image to BGR format (red ball becomes blue)
        hsv_conv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        bright_red_lower_bounds = (110, 100, 100)
        bright_red_upper_bounds = (130, 255, 255)
        bright_red_mask = cv2.inRange(hsv_conv_img, bright_red_lower_bounds, bright_red_upper_bounds)

        blurred_mask = cv2.GaussianBlur(bright_red_mask,(9,9),3,3)
    # some morphological operations (closing) to remove small blobs
        erode_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilate_element = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
        eroded_mask = cv2.erode(blurred_mask,erode_element)
        dilated_mask = cv2.dilate(eroded_mask,dilate_element)

        # on the color-masked, blurred and morphed image I apply the cv2.HoughCircles-method to detect circle-shaped objects
        detected_circles = cv2.HoughCircles(dilated_mask, cv2.HOUGH_GRADIENT, 1, 150, param1=100, param2=20, minRadius=2, maxRadius=2000)
        if detected_circles is not None:
            for circle in detected_circles[0, :]:
                circled_orig = cv2.circle(frame, (int(circle[0]), int(circle[1])), int(circle[2]), (0,255,0),thickness=3)
                self.redball_position = (int(circle[0]))
            self.get_logger().info(f'✅ Red ball detected at x={self.redball_position}')            
            self.target_publisher.publish(self.br.cv2_to_imgmsg(circled_orig))
        else:
            self.redball_position = None
            self.get_logger().info('❌ No ball detected')

    def stop_status_callback(self, msg):
        self.create3_is_stopped = msg.is_stopped

class CreateRedBallEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode=None):
        super().__init__()
        rclpy.init()
        self.observation_space = spaces.Discrete(641)  # 0 to 640 inclusive
        self.action_space = spaces.Discrete(641)       # same
        self.render_mode = render_mode
        self.redball = RedBall()
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        # Spin until we get a valid red ball observation
        while self.redball.redball_position is None:
            rclpy.spin_once(self.redball, timeout_sec=0.1)

        observation = self.redball.redball_position
        info = {}
        return observation, info

    def step(self, action):
        self.step_count += 1

        # Delegate to RedBall step method
        observation = self.redball.step(action)
        if observation is None:
            observation = 320  # fallback to center

        reward = -abs(observation - 320) / 320
        terminated = self.step_count >= 100
        truncated = False
        info = {}

        print(f"Step {self.step_count}: Action = {action}, Observation = {observation}, Reward = {reward}")
        
        return observation, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        self.redball.destroy_node()
        rclpy.shutdown()
