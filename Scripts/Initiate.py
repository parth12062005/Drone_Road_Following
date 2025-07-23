import subprocess
import signal
import sys
import os
import time

def main():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.abspath(os.path.join(script_dir, ".."))

    px4_dir = os.path.join(parent_dir, "/home/mr_robot", "PX4-Autopilot")
    rviz_config_path = os.path.join(parent_dir, "Scripts", "config.rviz")
    qgc_path = os.path.join(parent_dir, "/home/mr_robot/Desktop/Git/Drone", "QGroundControl-x86_64.AppImage")

    px4_cmd = ["make", "px4_sitl", "gz_x500_mono_cam_baylands"]

    #/world/baylands/model/x500_mono_cam_down_0/link/camera_link/sensor/imager/image

    bridge_params = [
        "ros2", "run", "ros_gz_bridge", "parameter_bridge",
        "/camera@sensor_msgs/msg/Image@gz.msgs.Image",
        "/camera_info@gz.msgs.CameraInfo",
        "--ros-args",
        "-r", "/camera:=/camera/down_left/image",
        "-r", "/camera_info:=/camera/down_left/camera_info"
    ]
    bridge_cmd_str = " ".join(bridge_params) + "; exec bash"

    mavros_params = [
        "ros2", "run", "mavros", "mavros_node",
        "--ros-args", "-p", "fcu_url:=udp://:14540@127.0.0.1:14557"
    ]
    mavros_cmd_str = " ".join(mavros_params) + "; exec bash"

    rviz_cmd_str = f"rviz2 -d {rviz_config_path}; exec bash"

    qgc_cmd_str = f"{qgc_path}; exec bash"

    print("🛫 Launching PX4 SITL in THIS terminal…")
    px4_proc = subprocess.Popen(
        px4_cmd,
        cwd=px4_dir
    )
    print(f"  • PX4 SITL PID: {px4_proc.pid}")

    time.sleep(20)

    print("🔥 Spawning Bridge, MAVROS, RViz2 & QGC in separate terminals…")
    terminals = [] 

    def spawn_terminal(cmd_str):
        proc = subprocess.Popen(
            ["gnome-terminal", "--disable-factory", "--", "bash", "-c", cmd_str],
            preexec_fn=os.setpgrp  
        )
        terminals.append(proc)
        return proc

    spawn_terminal(bridge_cmd_str)
    spawn_terminal(mavros_cmd_str)
    spawn_terminal(rviz_cmd_str)
    spawn_terminal(qgc_cmd_str)

    def shutdown(sig, frame):
        print("\n🛑 Ctrl+C detected—shutting down everything…")

        try:
            px4_proc.send_signal(signal.SIGINT)
        except Exception as e:
            print(f"  • couldn’t SIGINT PX4 (PID {px4_proc.pid}): {e}")

        for term in terminals:
            try:
                os.killpg(term.pid, signal.SIGINT)
            except Exception as e:
                print(f"  • couldn’t SIGINT PGID {term.pid}: {e}")

        time.sleep(2)

        for term in terminals:
            try:
                os.killpg(term.pid, signal.SIGKILL)
            except Exception as e:
                print(f"  • couldn’t SIGKILL PGID {term.pid}: {e}")

        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()

