#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from geometry_msgs.msg import PoseStamped, Twist, Point, Quaternion
from sensor_msgs.msg import Image
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from cv_bridge import CvBridge

import numpy as np
import math
import cv2  # For fitLine, moments, etc.


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> Quaternion:
    """
    Convert Euler angles (in radians) to a geometry_msgs/Quaternion.
    """
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    q = Quaternion()
    q.w = cr * cp * cy + sr * sp * sy
    q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy
    q.z = cr * cp * sy - sr * sp * cy
    return q


def quaternion_to_yaw(q: Quaternion) -> float:
    """
    Convert a geometry_msgs/Quaternion to a single yaw angle (radians).
    """
    # yaw (z-axis rotation) calculation from quaternion
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class MissionControllerNode(Node):
    def __init__(self):
        super().__init__('mission_controller_node')
        self.bridge = CvBridge()

        # --- QoS PROFILES ---
        state_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        pose_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        img_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        setpoint_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # --- PUBLISHERS & CLIENTS ---
        self.setpoint_pos_pub = self.create_publisher(
            PoseStamped,
            '/mavros/setpoint_position/local',
            setpoint_qos
        )
        self.setpoint_vel_pub = self.create_publisher(
            Twist,
            '/mavros/setpoint_velocity/cmd_vel_unstamped',
            setpoint_qos
        )
        self.arm_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode')

        # --- SUBSCRIBERS ---
        self.state_sub = self.create_subscription(
            State,
            '/mavros/state',
            self.state_callback,
            state_qos
        )
        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/mavros/local_position/pose',
            self.pose_callback,
            pose_qos
        )
        self.line_sub = self.create_subscription(
            Image,
            '/road_line',
            self.line_callback,
            img_qos
        )

        # --- INTERNAL STATE ---
        self.current_state = State()
        self.current_pose = PoseStamped()
        self.current_yaw = 0.0              # store yaw for body→local rotation
        self.initial_pose_captured = False
        self.takeoff_altitude_reached = False
        self.offboard_set_and_armed = False
        self.last_request_time = self.get_clock().now()
        self.road_line_img = None

        # --- ANGULAR ALIGN + EMA PARAMETERS (fitLine) ---
        self.kp_yaw = 0.2                 # P-gain for yaw correction (tweak in sim)
        self.angle_tolerance = 0.2          # radians (~6°) tolerance before moving forward
        self.filtered_angle_error = None
        self.alpha_angle = 0.2              # EMA alpha for angle_error (0 < α < 1)

        # --- CENTROID (lateral) + EMA PARAMETERS ---
        self.kp_centroid = 2              # P-gain for centroid error → lateral
        self.max_lateral_speed = 2.5        # m/s max sideways
        self.prev_lateral_vel = None
        self.alpha_vel = 0.3                # EMA alpha for lateral velocity

        # --- FLIGHT PARAMETERS ---
        self.target_altitude = 10.0         # meters
        self.altitude_tolerance = 0.3       # meters
        self.takeoff_timeout = 30.0         # seconds
        self.forward_speed = 3.0            # m/s (body-frame forward)

        # Start the 10 Hz control loop
        self.create_timer(0.1, self.control_loop)
        self.get_logger().info("Mission Controller Initialized (body-frame velocities)")

    def state_callback(self, msg: State):
        self.current_state = msg

    def pose_callback(self, msg: PoseStamped):
        self.current_pose = msg
        # extract yaw from quaternion
        self.current_yaw = quaternion_to_yaw(msg.pose.orientation)
        if not self.initial_pose_captured and msg.header.stamp.sec > 0:
            self.initial_pose_captured = True
            self.get_logger().info(
                f"Captured initial pose: x={msg.pose.position.x:.2f}, "
                f"y={msg.pose.position.y:.2f}, z={msg.pose.position.z:.2f}, "
                f"yaw={math.degrees(self.current_yaw):.1f}°"
            )

    def line_callback(self, msg: Image):
        try:
            self.road_line_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"Line img conversion failed: {e}")

    def call_service_async(self, client, req, name: str):
        if not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warning(f"{name} service not ready")
            return
        client.call_async(req)
        self.get_logger().info(f"Requested {name}")

    def request_arm(self):
        req = CommandBool.Request()
        req.value = True
        self.call_service_async(self.arm_client, req, "Arming")

    def request_set_mode(self, mode: str):
        req = SetMode.Request()
        req.custom_mode = mode
        self.call_service_async(self.set_mode_client, req, f"SetMode={mode}")

    def create_initial_setpoint(self) -> PoseStamped:
        """
        Publish a trivial setpoint so OFFBOARD mode stays valid while waiting
        for the FCU & pose topics.
        """
        sp = PoseStamped()
        sp.header.stamp = self.get_clock().now().to_msg()
        sp.header.frame_id = "map"

        if self.initial_pose_captured:
            sp.pose = self.current_pose.pose
        else:
            sp.pose.position = Point(0.0, 0.0, 0.1)
            sp.pose.orientation = euler_to_quaternion(0, 0, 0)

        return sp

    def control_loop(self):
        now = self.get_clock().now()

        # 1) WAIT FOR FCU & POSE
        if not self.current_state.connected or not self.initial_pose_captured:
            self.get_logger().info("Waiting for FCU & pose...", throttle_duration_sec=5.0)
            if self.initial_pose_captured:
                # keep sending a dummy setpoint so OFFBOARD can be engaged later
                self.setpoint_pos_pub.publish(self.create_initial_setpoint())
            return

        # 2) OFFBOARD & ARM
        if not self.offboard_set_and_armed:
            self.setpoint_pos_pub.publish(self.create_initial_setpoint())
            elapsed = (now - self.last_request_time).nanoseconds / 1e9
            if elapsed > 1.0:
                if self.current_state.mode != "OFFBOARD":
                    self.request_set_mode("OFFBOARD")
                elif not self.current_state.armed:
                    self.request_arm()
                self.last_request_time = now

            if self.current_state.mode == "OFFBOARD" and self.current_state.armed:
                self.offboard_set_and_armed = True
                self.takeoff_start = now
                self.get_logger().info("OFFBOARD & ARMED!")
            return

        # 3) TAKEOFF PHASE
        if not self.takeoff_altitude_reached:
            elapsed_tk = (now - self.takeoff_start).nanoseconds / 1e9
            if elapsed_tk > self.takeoff_timeout:
                self.get_logger().error("Takeoff timed out!")
                return

            sp = PoseStamped()
            sp.header.stamp = now.to_msg()
            sp.header.frame_id = "map"
            sp.pose.position.x = self.current_pose.pose.position.x
            sp.pose.position.y = self.current_pose.pose.position.y
            sp.pose.position.z = self.target_altitude
            sp.pose.orientation = self.current_pose.pose.orientation
            self.setpoint_pos_pub.publish(sp)

            if abs(self.current_pose.pose.position.z - self.target_altitude) < self.altitude_tolerance:
                self.takeoff_altitude_reached = True
                self.get_logger().info("Reached takeoff altitude!")
            return

        # 4) LINE FOLLOWING (fitLine + centroid + EMA, body-frame)
        if self.road_line_img is None:
            self.get_logger().warning("No /road_line detected, hovering.", throttle_duration_sec=2.0)
            self.setpoint_vel_pub.publish(Twist())
            return

        img = self.road_line_img
        H, W = img.shape[:2]
        center_x = W / 2.0

        # 4.1) Threshold red channel (BGR)
        red = img[..., 2]
        gr  = img[..., 1]
        bl  = img[..., 0]
        mask = (red > 150) & (gr < 50) & (bl < 50)

        ys, xs = np.where(mask)
        if xs.size == 0:
            # No red pixels → lost line → hover
            self.get_logger().warning("Line lost (no red pixels), hovering.", throttle_duration_sec=2.0)
            self.setpoint_vel_pub.publish(Twist())
            return

        # 4.2) Fit a line to all red-pixel coordinates
        pts = np.column_stack((xs, ys)).astype(np.float32).reshape(-1, 1, 2)
        try:
            vx_fit, vy_fit, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        except Exception as e:
            self.get_logger().error(f"cv2.fitLine failed: {e}")
            self.setpoint_vel_pub.publish(Twist())
            return

        # 4.3) Compute the line’s angle relative to +X image axis
        angle_line = math.atan2(vy_fit, vx_fit)  # in [-π, π]

        # fitLine can return either “up” or “down”; force to [-π/2, π/2]
        if angle_line > 0:
            angle_line -= math.pi

        # “Straight up” in image = vector (0, -1) → angle = -π/2
        desired_angle = -math.pi / 2.0

        # 4.4) Compute raw angle_error = angle_line - desired_angle, then normalize
        raw_angle_error = angle_line - desired_angle
        while raw_angle_error > math.pi:
            raw_angle_error -= 2 * math.pi
        while raw_angle_error < -math.pi:
            raw_angle_error += 2 * math.pi

        # 4.5) EMA filter on angle_error
        if self.filtered_angle_error is None:
            self.filtered_angle_error = raw_angle_error
        else:
            self.filtered_angle_error = (
                self.alpha_angle * raw_angle_error
                + (1.0 - self.alpha_angle) * self.filtered_angle_error
            )

        angle_error = self.filtered_angle_error

        # 4.6) Compute centroid of mask to get lateral offset
        mask_uint8 = (mask.astype(np.uint8) * 255)
        M = cv2.moments(mask_uint8)
        if M["m00"] == 0:
            centroid_x = center_x
        else:
            centroid_x = M["m10"] / M["m00"]

        # pixel error: negative = line is left, positive = line is right
        error_x = centroid_x - center_x
        # normalize to [-1, +1]
        error_x_norm = error_x / (W / 2.0)

        # 4.7) Raw body-frame lateral velocity from centroid error
        raw_lateral_vel = -self.kp_centroid * error_x_norm * self.max_lateral_speed

        # 4.8) EMA filter on lateral velocity
        if self.prev_lateral_vel is None:
            lateral_vel = raw_lateral_vel
        else:
            lateral_vel = (
                self.alpha_vel * raw_lateral_vel
                + (1.0 - self.alpha_vel) * self.prev_lateral_vel
            )
        self.prev_lateral_vel = lateral_vel

        # 4.9) Determine body-frame forward/lateral/or angular commands
        body_forward = 0.0
        body_right   = 0.0
        body_yawrate = 0.0

        if abs(angle_error) > self.angle_tolerance:
            # Rotate in place (body-frame yaw rate), no body-forward or lateral
            body_forward = 0.0
            body_right   = 0.0
            body_yawrate = -self.kp_yaw * angle_error
        else:
            # Aligned enough → move forward + strafe in body frame
            body_forward = self.forward_speed
            body_right   = lateral_vel
            body_yawrate = -self.kp_yaw * angle_error  # small correction if drifting

        # 4.10) Convert body-frame (forward, right) → local ENU (north, east)
        ψ = self.current_yaw
        # body-forward = u, body-right = v
        u = body_forward
        v = body_right
        # Forward is “along drone’s nose,” Right is “along its right wing”
        v_north =  u * math.cos(ψ) - v * math.sin(ψ)
        v_east  =  u * math.sin(ψ) + v * math.cos(ψ)

        # 4.11) Build final Twist in local ENU (MAVROS handles ENU→NED)
        cmd = Twist()
        cmd.linear.x = float(v_north)      # local “north” velocity
        cmd.linear.y = float(v_east)       # local “east” velocity
        cmd.linear.z = 0.0                 # no vertical velocity here
        cmd.angular.z = float(body_yawrate)  # body-frame yaw rate

        # 4.12) Publish the velocity command
        self.setpoint_vel_pub.publish(cmd)

        # 4.13) Debug info
        self.get_logger().debug(
            f"yaw={math.degrees(ψ):.1f}°, angle_line={angle_line:.2f}, "
            f"raw_ang_err={raw_angle_error:.2f}, filt_ang_err={angle_error:.2f}, "
            f"centroid_x={centroid_x:.1f}, err_x_norm={error_x_norm:.2f}, "
            f"raw_lat_vel={raw_lateral_vel:.2f}, filt_lat_vel={lateral_vel:.2f}, "
            f"body_fwd={body_forward:.2f}, body_lat={body_right:.2f}, "
            f"local_north={v_north:.2f}, local_east={v_east:.2f}, yawrate={body_yawrate:.2f}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = MissionControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
