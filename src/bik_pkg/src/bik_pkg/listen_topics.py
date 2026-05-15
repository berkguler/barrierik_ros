#! /usr/bin/env python3
import rospy
import os
import csv
import json
import argparse

# Import your custom message types here (adjust the package paths as needed)
from ros_tcp_endpoint.msg import ROSToUnityMessage
from ros_tcp_endpoint.msg import UnityToROSMessage
from ros_tcp_endpoint.msg import ObstacleInfos
from sensor_msgs.msg import JointState


def init_output(log_dir):
    """Initialize CSV files and return dicts of file handles and csv writers."""
    files = {}
    writers = {}

    # Unity to ROS
    unity_path = os.path.join(log_dir, 'unity_to_ros.csv')
    f_unity = open(unity_path, 'w', newline='')
    writer_unity = csv.writer(f_unity)
    # header: stamp, q_actual_positions, x_target, x_helper, robot_status
    writer_unity.writerow([
        'stamp', 'q_actual_positions',
        'x_target', 'x_helper', 'robot_status'
    ])
    files['unity_to_ros'] = f_unity
    writers['unity_to_ros'] = writer_unity

    # ROS to Unity
    ros_path = os.path.join(log_dir, 'ros_to_unity.csv')
    f_ros = open(ros_path, 'w', newline='')
    writer_ros = csv.writer(f_ros)
    writer_ros.writerow([
        'stamp', 'frame_id', 'q_target_positions', 'poses_ai_target'
    ])
    files['ros_to_unity'] = f_ros
    writers['ros_to_unity'] = writer_ros

    # Obstacle Info
    obs_path = os.path.join(log_dir, 'obstacle_info.csv')
    f_obs = open(obs_path, 'w', newline='')
    writer_obs = csv.writer(f_obs)
    writer_obs.writerow([
        'stamp', 'frame_id', 'type',
        'position', 'orientation', 'scale'
    ])
    files['obstacle_info'] = f_obs
    writers['obstacle_info'] = writer_obs

    # bik Output
    bik_path = os.path.join(log_dir, 'bik_output.csv')
    f_bik = open(bik_path, 'w', newline='')
    writer_bik = csv.writer(f_bik)
    writer_bik.writerow([
        'stamp', 'q_positions'
    ])
    files['bik_output'] = f_bik
    writers['bik_output'] = writer_bik

    return files, writers


def unity_to_ros_cb(msg, writers):
    stamp = msg.q_actual.header.stamp.to_sec()
    q_positions = list(msg.q_actual.position)
    x_t = msg.x_target
    x_h = msg.x_helper
    # pack pose as dict
    xt = {'pos': [x_t.position.x, x_t.position.y, x_t.position.z],
          'orient': [x_t.orientation.x, x_t.orientation.y, x_t.orientation.z, x_t.orientation.w]}
    xh = {'pos': [x_h.position.x, x_h.position.y, x_h.position.z],
          'orient': [x_h.orientation.x, x_h.orientation.y, x_h.orientation.z, x_h.orientation.w]}
    status = msg.robot_status.data
    writers['unity_to_ros'].writerow([
        stamp,
        json.dumps(q_positions),
        json.dumps(xt),
        json.dumps(xh),
        status
    ])

def bik_output_cb(msg, writers):
    stamp = msg.header.stamp.to_sec()
    q_positions = list(msg.position)

    writers['bik_output'].writerow([
        stamp,
        json.dumps(q_positions)
    ])


def ros_to_unity_cb(msg, writers):
    stamp = msg.q_target.header.stamp.to_sec()
    frame = msg.poses_ai_target.header.frame_id
    q_positions = list(msg.q_target.position)
    poses_list = []
    for p in msg.poses_ai_target.poses:
        poses_list.append({
            'pos': [p.position.x, p.position.y, p.position.z],
            'orient': [p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]
        })
    writers['ros_to_unity'].writerow([
        stamp,
        frame,
        json.dumps(q_positions),
        json.dumps(poses_list)
    ])


def obstacle_info_cb(msg, writers):
    # ObstacleInfos contains a header and list of obstacles
    
    for obs in msg.obstacles:
        stamp = obs.header.stamp.to_sec()
        frame = obs.header.frame_id
        t = obs.type
        pos = [obs.position.x, obs.position.y, obs.position.z]
        orient = [obs.orientation.x, obs.orientation.y, obs.orientation.z, obs.orientation.w]
        scale = [obs.scale.x, obs.scale.y, obs.scale.z]
        writers['obstacle_info'].writerow([
            stamp,
            frame,
            t,
            json.dumps(pos),
            json.dumps(orient),
            json.dumps(scale)
        ])


def main():
    rospy.init_node('topic_csv_logger', anonymous=True)
    parser = argparse.ArgumentParser(description='ROS CSV Logger for selected topics')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Base directory to store CSV logs')
    parser.add_argument('--user_id', type=str, required=True, help='User ID')
    parser.add_argument('--config_id', type=str, required=True, help='Config ID')
    args = parser.parse_args()

    # Determine log directory
    if args.log_dir:
        base = args.log_dir
    else:
        base = os.path.expanduser('~/cbf_latest/csv_logs')
    log_dir = os.path.join(base, f'user_{args.user_id}', f'config_{args.config_id}')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    else:
        rospy.logwarn(f"Directory {log_dir} exists, appending timestamp")
        suffix = str(rospy.Time.now().to_sec())
        log_dir = os.path.join(log_dir, suffix)
        os.makedirs(log_dir)

    
    rospy.loginfo(f"Logging CSVs to {log_dir}")

    files, writers = init_output(log_dir)

    # Subscribers
    rospy.Subscriber('unity_to_ros', UnityToROSMessage, unity_to_ros_cb, writers)
    rospy.Subscriber('ros_to_unity', ROSToUnityMessage, ros_to_unity_cb, writers)
    rospy.Subscriber('obstacle_info', ObstacleInfos, obstacle_info_cb, writers)
    rospy.Subscriber('bik_output',JointState, bik_output_cb, writers)

    rospy.spin()

    # Close files on shutdown
    for f in files.values():
        f.close()


if __name__ == '__main__':
    main()
