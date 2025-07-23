#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import os
from datetime import datetime

class RoadFollowerNode(Node):
    def __init__(self):
        super().__init__('road_follower_node')

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Load YOLOv11 segmentation model
        self.model = YOLO('best.pt')  # Ensure this is a segmentation model

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/down/image', self.image_callback, 10)
        self.pose_sub = self.create_subscription(
            Odometry, '/mavros/odometry/in', self.pose_callback, 10)

        # Publishers
        self.cmd_vel_pub = self.create_publisher(
            Twist, '/mavros/setpoint_velocity/cmd_vel_unstamped', 10)
        self.debug_pub = self.create_publisher(Image, '/road_debug', 10)

        # Control parameters
        self.Kp = 0.01  # Proportional gain
        self.target_altitude = 10.0  # meters
        self.image_save_dir = 'road_images/'
        self.current_pose = None

        # Create image directory
        os.makedirs(self.image_save_dir, exist_ok=True)

        self.get_logger().info("Road follower node initialized!")

    def pose_callback(self, msg):
        self.current_pose = msg.pose.pose

    def compute_centroid(self, mask):
        M = cv2.moments(mask)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)

    def fit_polynomial(self, mask, order=2):
        # Find nonzero points
        y, x = np.where(mask == 255)
        if len(x) < 10:
            return None

        # Fit polynomial (x = Ay² + By + C)
        coeffs = np.polyfit(y, x, order)
        ploty = np.linspace(0, mask.shape[0]-1, mask.shape[0])
        try:
            fitx = np.polyval(coeffs, ploty)
        except:
            return None

        return np.stack([fitx, ploty], axis=1).astype(int)

    def image_callback(self, msg):
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

            # Run segmentation
            results = self.model.predict(source=cv_image, verbose=False, task='segment')
            masks = results[0].masks

            if masks is None or masks.data is None:
                self.get_logger().warn("No masks detected")
                return

            # Combine all masks into a single binary mask
            mask_array = masks.data.cpu().numpy()  # shape: (num_masks, H, W)
            combined_mask = np.any(mask_array, axis=0).astype(np.uint8) * 255  # shape: (H, W)
            gray_mask = combined_mask

            # Path extraction
            centroid = self.compute_centroid(gray_mask)
            poly_path = self.fit_polynomial(gray_mask)

            # Generate control command
            twist = Twist()
            twist.linear.x = 1.0  # Forward speed

            if poly_path is not None:
                # Follow polynomial path (use point 30% down the path)
                idx = int(0.3 * len(poly_path))
                target_x = poly_path[idx][0]
                error = target_x - cv_image.shape[1] // 2
            elif centroid:
                # Fallback to centroid
                error = centroid[0] - cv_image.shape[1] // 2
            else:
                error = 0

            twist.angular.z = -self.Kp * error

            # Publish command
            self.cmd_vel_pub.publish(twist)

            # Save image with metadata
            self.save_image(cv_image)

            # Publish debug visualization
            debug_img = self.create_debug_image(cv_image, gray_mask, centroid, poly_path)
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug_img, 'bgr8'))

        except Exception as e:
            self.get_logger().error(f"Image processing failed: {str(e)}")

    def create_debug_image(self, image, mask, centroid, poly_path):
    # Resize mask to match image dimensions
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Convert mask to BGR
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # Blend images
        debug_img = cv2.addWeighted(image, 0.7, mask_bgr, 0.3, 0)
        
        # Draw centroid
        if centroid:
            cv2.circle(debug_img, centroid, 10, (0, 255, 0), -1)
            cv2.line(debug_img, (image.shape[1]//2, image.shape[0]), 
                    (centroid[0], centroid[1]), (0, 255, 0), 2)
        
        # Draw polynomial path
        if poly_path is not None:
            for pt in poly_path:
                x, y = pt.astype(int)
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    cv2.circle(debug_img, (x, y), 2, (0, 0, 255), -1)
    
        return debug_img


    def save_image(self, image):
        if self.current_pose is None:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{self.image_save_dir}/{timestamp}.png"
        cv2.imwrite(filename, image)

        # Optional: Save pose data to CSV
        # with open(f"{self.image_save_dir}/metadata.csv", 'a') as f:
        #     f.write(f"{timestamp},{self.current_pose.position.x},"
        #             f"{self.current_pose.position.y},"
        #             f"{self.current_pose.position.z}\n")

def main(args=None):
    rclpy.init(args=args)
    node = RoadFollowerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
