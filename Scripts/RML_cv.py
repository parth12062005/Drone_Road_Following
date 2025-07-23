import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class RoadSegmentationNode(Node):
    def __init__(self):
        super().__init__('road_segmentation_node')
        self.bridge = CvBridge()

        # Subscribe to the down-facing camera topic
        self.create_subscription(
            Image,
            '/camera/down_left/image',
            self.image_callback,
            10
        )

        # Publisher for mask + overlay
        self.mask_pub = self.create_publisher(Image, '/road_mask', 10)
        # Publisher for center-line only
        self.line_pub = self.create_publisher(Image, '/road_line', 10)

        self.get_logger().info("RoadSegmentationNode started, waiting for images...")

    def get_road_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Given a BGR frame, returns a binary 8‐bit mask (255=road, 0=background)
        using HSV‐threshold + contour‐filling on the largest gray region.
        """
        # 1) Convert BGR → HSV, threshold for gray/asphalt
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # ─── Tweak these values if your lighting or road color shifts ───
        lower_gray = np.array([0,   0,  80])   # V_min =  80 (darker roads remain, lighter sidewalks drop)
        upper_gray = np.array([180, 60, 200])   # S_max =  60 (allow some texture on road)
        mask_color = cv2.inRange(hsv, lower_gray, upper_gray)

        # 2) Close small gaps so the road region is one connected blob
        #    (bigger kernel ensures small noise is removed, and the road “hole” is filled)
        kernel_close = np.ones((15, 15), np.uint8)
        mask_closed = cv2.morphologyEx(mask_color, cv2.MORPH_CLOSE, kernel_close)

        # 3) Find all contours in that closed mask
        gray_mask = mask_closed.copy()
        contours, _ = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 4) Pick the largest contour by area (assumed to be the road)
        road_mask = np.zeros_like(gray_mask)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            cv2.drawContours(road_mask, [largest], -1, 255, thickness=cv2.FILLED)

        return road_mask

    def image_callback(self, msg: Image) -> None:
        try:
            # 1) Convert ROS Image → OpenCV BGR
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            H, W = cv_image.shape[:2]

            # 2) Compute binary road mask
            road_mask = self.get_road_mask(cv_image)  # 0/255 binary

            # 3) Create a blue “mask overlay” on top of the original
            blue_mask = np.zeros_like(cv_image)
            blue_mask[:, :, 0] = road_mask           # put mask in blue channel
            mask_overlay = cv2.addWeighted(cv_image, 0.7, blue_mask, 0.3, 0)

            # 4) Compute center points for each row where mask is white
            centers = []
            ys = np.arange(H)
            for y in ys:
                xs = np.where(road_mask[y] > 0)[0]
                if xs.size > 0:
                    cx = int((int(xs[0]) + int(xs[-1])) / 2)
                    centers.append((cx, y))

            # 5) Draw center‐line on a blank image (red line)
            line_img = np.zeros_like(cv_image)
            if len(centers) > 10:
                # Fit a first‐degree polynomial x = a*y + b
                ys_arr = np.array([pt[1] for pt in centers], dtype=np.float32)
                xs_arr = np.array([pt[0] for pt in centers], dtype=np.float32)
                coeffs = np.polyfit(ys_arr, xs_arr, 1)
                poly = np.poly1d(coeffs)

                # Evaluate line at all y
                line_ys = np.arange(H, dtype=np.int32)
                line_xs = poly(line_ys).astype(np.int32)

                # Stack into Nx1x2 points for polylines
                pts = np.vstack((line_xs, line_ys)).T.reshape(-1, 1, 2)
                cv2.polylines(mask_overlay, [pts], isClosed=False, color=(0, 0, 255), thickness=2)
                cv2.polylines(line_img,       [pts], isClosed=False, color=(0, 0, 255), thickness=2)
            else:
                self.get_logger().warn("Too few mask rows detected to fit a center line.")

            # 6) Publish the two images back to ROS topics
            mask_msg = self.bridge.cv2_to_imgmsg(mask_overlay, encoding='bgr8')
            line_msg = self.bridge.cv2_to_imgmsg(line_img,      encoding='bgr8')
            self.mask_pub.publish(mask_msg)
            self.line_pub.publish(line_msg)
            self.get_logger().info("Published /road_mask and /road_line.")

        except Exception as e:
            self.get_logger().error(f"image_callback error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = RoadSegmentationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
