import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO

class RoadSegmentationNode(Node):
    def __init__(self):
        super().__init__('road_segmentation_node')
        self.bridge = CvBridge()

        # load your YOLO segmentation model
        self.model = YOLO('Models/best.pt')  # ← update if needed
        self.model.eval()

        # sub to the down‑facing camera
        self.create_subscription(
            Image,
            '/camera/down_left/image',
            self.image_callback,
            10
        )

        # pub the mask+overlay
        self.mask_pub = self.create_publisher(Image, '/road_mask', 10)
        # pub the center‑line only
        self.line_pub = self.create_publisher(Image, '/road_line', 10)

    def image_callback(self, msg):
        try:
            # ROS → OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            H, W = cv_image.shape[:2]

            # YOLO inference
            results = self.model.predict(source=cv_image, verbose=False)
            res     = results[0]

            # make sure we got a mask
            if not hasattr(res, 'masks') or res.masks is None or len(res.masks.data)==0:
                self.get_logger().warn("No mask detected!")
                return

            # raw mask at model‑input size (e.g. 480×640)
            mask_small = res.masks.data[0].cpu().numpy()

            # resize mask to full image size
            mask_full = cv2.resize(mask_small, (W, H), interpolation=cv2.INTER_LINEAR)

            # binarize
            mask_bin = (mask_full > 0.5).astype(np.uint8) * 255

            # ─── build the mask overlay ─────────────────────────────
            blue_mask = np.zeros_like(cv_image)
            blue_mask[:, :, 0] = mask_bin
            mask_overlay = cv2.addWeighted(cv_image, 0.7, blue_mask, 0.3, 0)

            # ─── compute center points ──────────────────────────────
            centers = []
            for y in range(H):
                xs = np.where(mask_bin[y] > 0)[0]
                if xs.size:
                    cx = int((xs[0] + xs[-1]) / 2)
                    centers.append((cx, y))

            # ─── prepare a blank line image ─────────────────────────
            line_img = np.zeros_like(cv_image)

            # ─── fit & draw polyline if we have enough points ────────
            if len(centers) > 10:
                ys = np.array([p[1] for p in centers])
                xs = np.array([p[0] for p in centers])
                coeffs = np.polyfit(ys, xs, 1)
                poly   = np.poly1d(coeffs)

                line_ys = np.arange(H)
                line_xs = poly(line_ys).astype(np.int32)
                pts     = np.vstack((line_xs, line_ys)).T.reshape(-1,1,2)

                # draw on both images
                cv2.polylines(mask_overlay, [pts], False, (0,0,255), 2)
                cv2.polylines(line_img,      [pts], False, (0,0,255), 2)
            else:
                self.get_logger().warn("Too few mask rows for polyfit.")

            # ─── optional: draw bbox & conf on mask_overlay ─────────
            box  = res.boxes.xyxy[0].cpu().numpy().astype(int)
            conf = float(res.boxes.conf[0].cpu().numpy())
            x1,y1,x2,y2 = box
            cv2.rectangle(mask_overlay, (x1,y1), (x2,y2), (255,255,0), 2)
            cv2.putText(mask_overlay, f"Road {conf:.2f}", (x1, y1-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

            # ─── publish both images ────────────────────────────────
            mask_msg = self.bridge.cv2_to_imgmsg(mask_overlay, encoding='bgr8')
            line_msg = self.bridge.cv2_to_imgmsg(line_img,     encoding='bgr8')
            self.mask_pub.publish(mask_msg)
            self.line_pub.publish(line_msg)
            self.get_logger().info("Published /road_mask & /road_line.")

        except Exception as e:
            self.get_logger().error(f"Callback error: {e}")

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
