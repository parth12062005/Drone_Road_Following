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

def euler_to_quaternion(roll, pitch, yaw):
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

class MissionControllerNode(Node):
    def __init__(self):
        super().__init__('mission_controller_node')
        self.bridge = CvBridge()

        # QoS for MAVROS state (RELIABLE, TRANSIENT_LOCAL)
        state_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        # QoS for pose (BEST_EFFORT to match MAVROS publisher)
        pose_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        # QoS for images (BEST_EFFORT)
        img_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        # QoS for setpoints (RELIABLE)
        setpoint_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Publishers & service clients
        self.setpoint_pos_pub = self.create_publisher(
            PoseStamped, '/mavros/setpoint_position/local', setpoint_qos)
        self.setpoint_vel_pub = self.create_publisher(
            Twist, '/mavros/setpoint_velocity/cmd_vel_unstamped', setpoint_qos)
        self.arm_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode')

        # Subscribers
        self.state_sub = self.create_subscription(
            State, '/mavros/state', self.state_callback, state_qos)
        self.pose_sub = self.create_subscription(
            PoseStamped, '/mavros/local_position/pose', self.pose_callback, pose_qos)
        self.line_sub = self.create_subscription(
            Image, '/road_line', self.line_callback, img_qos)

        # Internal state
        self.current_state = State()
        self.current_pose = PoseStamped()
        self.initial_pose_captured = False
        self.takeoff_altitude_reached = False
        self.offboard_set_and_armed = False
        self.last_request_time = self.get_clock().now()
        self.road_line_img = None

        # Params
        self.target_altitude = 10.0
        self.altitude_tolerance = 0.3
        self.takeoff_timeout = 30.0
        self.forward_speed = 2
        self.yaw_rate_gain = -0.01

        # Control loop at 10 Hz
        self.create_timer(0.1, self.control_loop)
        self.get_logger().info("Mission Controller Initialized (road_line mode)")

    def state_callback(self, msg: State):
        self.current_state = msg

    def pose_callback(self, msg: PoseStamped):
        self.current_pose = msg
        if not self.initial_pose_captured and msg.header.stamp.sec > 0:
            self.initial_pose_captured = True
            self.get_logger().info(
                f"Captured initial pose: x={msg.pose.position.x:.2f}, "
                f"y={msg.pose.position.y:.2f}, z={msg.pose.position.z:.2f}"
            )

    def line_callback(self, msg: Image):
        try:
            self.road_line_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"Line img convert failed: {e}")

    def call_service_async(self, client, req, name):
        if not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(f"{name} service not ready")
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

        # 1) Wait for FCU & pose
        if not self.current_state.connected or not self.initial_pose_captured:
            self.get_logger().info("Waiting for FCU & pose...", throttle_duration_sec=5)
            if self.initial_pose_captured:
                self.setpoint_pos_pub.publish(self.create_initial_setpoint())
            return

        # 2) OFFBOARD & ARM
        if not self.offboard_set_and_armed:
            self.setpoint_pos_pub.publish(self.create_initial_setpoint())
            if (now - self.last_request_time).nanoseconds / 1e9 > 1.0:
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

        # 3) TAKEOFF
        if not self.takeoff_altitude_reached:
            if (now - self.takeoff_start).nanoseconds / 1e9 > self.takeoff_timeout:
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

                    # 4) LINE FOLLOWING: steer *toward* forward‑point + yaw so the red line is vertical
        if self.road_line_img is None:
            self.get_logger().warn("No /road_line, hovering.", throttle_duration_sec=2)
            self.setpoint_vel_pub.publish(Twist())
            return

        img = self.road_line_img
        H, W = img.shape[:2]
        cx, cy = W/2, H/2

        # mask out red line
        red, gr, bl = img[...,2], img[...,1], img[...,0]
        mask = (red>150)&(gr<50)&(bl<50)
        ys, xs = np.where(mask)
        if xs.size == 0:
            self.get_logger().warn("Line lost, hovering.", throttle_duration_sec=2)
            self.setpoint_vel_pub.publish(Twist())
            return

        # 1) pick forward‑most pixel (y small)
        idx_fwd = np.argmin(ys)
        x_fwd, y_fwd = xs[idx_fwd], ys[idx_fwd]

        # 2) vector centre→forward
        dx = x_fwd - cx
        dy = cy   - y_fwd
        alpha = math.atan2(dy, dx)   # angle wrt image X axis

        # 3) decompose forward_speed into that direction
        speed = self.forward_speed
        vx = speed * math.cos(alpha)
        vy = speed * math.sin(alpha)

        # 4) fit line, compute yaw so that line becomes vertical
        m       = np.polyfit(xs, ys, 1)[0]    # slope = Δy/Δx
        theta   = math.atan(m)                # line angle wrt X axis
        phi     = (math.pi/2 - theta)         # rotation to make it vertical
        yaw_cmd = phi * self.yaw_rate_gain

        # 5) publish
        cmd = Twist()
        cmd.linear.x  = -vx
        cmd.linear.y  = -vy
        cmd.angular.z = -yaw_cmd
        self.setpoint_vel_pub.publish(cmd)

        self.get_logger().debug(
            f"fwd_pt=({x_fwd},{y_fwd}), α={math.degrees(alpha):.1f}°, "
            f"vel=({vx:.2f},{vy:.2f}), yaw_rate={yaw_cmd:.2f}"
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
