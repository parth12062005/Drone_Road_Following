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

        # Publishers for overlay and center-line
        self.mask_pub = self.create_publisher(Image, '/road_mask', 10)
        self.line_pub = self.create_publisher(Image, '/road_line', 10)

        self.get_logger().info("RoadSegmentationNode (with equalization) started!")

    def equalize_histogram(self, bgr_frame: np.ndarray) -> np.ndarray:
        """
        Perform histogram equalization on the Y channel of the YUV conversion of the BGR frame.
        This enhances contrast so that gray/asphalt stands out more against sidewalks/grass.
        Returns an equalized BGR image.
        """
        # Convert BGR -> YUV color space
        yuv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2YUV)
        # Equalize the Y channel (luma)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        # Convert back to BGR
        equalized_bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        return equalized_bgr

    def get_road_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Given an equalized BGR frame, returns a binary road mask via HSV threshold + contour fill.
        """
        # 1) Convert BGR → HSV, threshold for asphalt-gray
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # ─── tweak these if lighting/texture changes ───
        lower_gray = np.array([0,   0,  80])   # V_min =  80 (dark roads remain)
        upper_gray = np.array([180, 60, 190])   # S_max =  60 (allow some texture)
        mask_color = cv2.inRange(hsv, lower_gray, upper_gray)

        # 2) Morphological close to fill small holes
        kernel_close = np.ones((15, 15), np.uint8)
        mask_closed = cv2.morphologyEx(mask_color, cv2.MORPH_CLOSE, kernel_close)

        # 3) Find contours in the closed mask
        gray_mask = mask_closed.copy()
        contours, _ = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 4) Pick largest contour (the road) and fill it
        road_mask = np.zeros_like(gray_mask)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            cv2.drawContours(road_mask, [largest], -1, 255, thickness=cv2.FILLED)

        return road_mask

    def image_callback(self, msg: Image) -> None:
        try:
            # 1) Convert ROS Image ➔ OpenCV BGR
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            H, W = cv_image.shape[:2]

            # 2) Histogram equalization (Y-channel in YUV)
            eq_image = self.equalize_histogram(cv_image)

            # 3) Compute binary road mask on equalized image
            road_mask = self.get_road_mask(eq_image)  # 0/255 mask

            # 4) Build a blue overlay on the equalized image (or original, up to you)
            blue_mask = np.zeros_like(eq_image)
            blue_mask[:, :, 0] = road_mask  # put mask into blue channel
            mask_overlay = cv2.addWeighted(eq_image, 0.7, blue_mask, 0.3, 0)

            # 5) Compute center points (x_center at each y where mask is white)
            centers = []
            for y in range(H):
                xs = np.where(road_mask[y] > 0)[0]
                if xs.size > 0:
                    cx = int((int(xs[0]) + int(xs[-1])) / 2)
                    centers.append((cx, y))

            # 6) Fit & draw center line if enough valid rows
            line_img = np.zeros_like(eq_image)
            if len(centers) > 10:
                ys_arr = np.array([pt[1] for pt in centers], dtype=np.float32)
                xs_arr = np.array([pt[0] for pt in centers], dtype=np.float32)
                coeffs = np.polyfit(ys_arr, xs_arr, 1)
                poly = np.poly1d(coeffs)

                line_ys = np.arange(H, dtype=np.int32)
                line_xs = poly(line_ys).astype(np.int32)

                pts = np.vstack((line_xs, line_ys)).T.reshape(-1, 1, 2)
                cv2.polylines(mask_overlay, [pts], isClosed=False, color=(0, 0, 255), thickness=2)
                cv2.polylines(line_img,       [pts], isClosed=False, color=(0, 0, 255), thickness=2)
            else:
                self.get_logger().warn("Too few mask rows to fit a center line.")

            # 7) Publish back to ROS
            mask_msg = self.bridge.cv2_to_imgmsg(mask_overlay, encoding='bgr8')
            line_msg = self.bridge.cv2_to_imgmsg(line_img,      encoding='bgr8')
            self.mask_pub.publish(mask_msg)
            self.line_pub.publish(line_msg)
            self.get_logger().info("Published /road_mask & /road_line (with equalization).")

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
