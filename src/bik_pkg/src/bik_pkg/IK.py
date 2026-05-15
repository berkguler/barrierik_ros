#! /usr/bin/env python3
import sys
import rospy
import pinocchio as pin
import numpy as np
import os
from os.path import dirname, join, abspath
from pinocchio.visualize import MeshcatVisualizer
import sys
from numpy.linalg import pinv
from matplotlib import pyplot as plt
import time
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, PoseArray
from std_msgs.msg import Header
import sys
from ros_tcp_endpoint.msg import UnityToROSMessage
from ros_tcp_endpoint.msg import ROSToUnityMessage
from ros_tcp_endpoint.msg import ObstacleInfos, ObstacleInfo
import tf.transformations as tft

import rospkg
import bik_pkg.parse_urdf as parse_urdf
import threading
import tf
from rosgraph_msgs.msg import Clock


rospack = rospkg.RosPack()
package_path = rospack.get_path('bik_pkg')
loaded_single_robot = False
while (not loaded_single_robot):
    try:
        PANDAmodel, _, _, _, _, _, _, _ , _ = parse_urdf.init(yaml_file = package_path + "/configs/franka_description.yaml", open_viewer = False, disable_pin_models = False, disable_viewer = True)
        loaded_single_robot = True
    except:
        pass



AI_POSE_TIMEOUT = 3  # seconds before considering AI poses stale

class AIMonitor:
    def __init__(self):
        self.ai_last_update = None
        self.ai_targets = PoseArray()
        self.lock = threading.Lock()
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_timeouts)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def _reset_pose(self, pose):
        pose.position.x = 0
        pose.position.y = 0
        pose.position.z = -3  # Hidden position
        pose.orientation.x = 0
        pose.orientation.y = 0
        pose.orientation.z = 0
        pose.orientation.w = 1
    
    def _monitor_timeouts(self):
        """ Monitor AI pose timeouts and reset them if necessary """
        while not rospy.is_shutdown():
            current_time = time.time()
            with self.lock:
                # Check AI poses
                if (self.ai_last_update is not None and 
                    (current_time - self.ai_last_update) > AI_POSE_TIMEOUT):
                    for pose in self.ai_targets.poses:
                        self._reset_pose(pose)
                    self.ai_last_update = None
            
            time.sleep(0.1)  # Use time.sleep instead of rospy.sleep

    def update_poses(self, poses):
        with self.lock:
            self.ai_targets = poses
            self.ai_last_update = time.time()

    def get_poses(self):
        with self.lock:
            return  self.ai_targets


# Create global AI monitor
ai_monitor = AIMonitor()

def talker():
    global q_target, q_init , q_actual, names_actual, name
    pub = rospy.Publisher('ros_to_unity', ROSToUnityMessage, queue_size=10)
    
    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        
        if not RealRobot:
            test_str = JointState()
            test_str.header = Header()
            test_str.header.stamp = rospy.Time.now()
            
            test_str.name =  name
            test_str.position = q_target
            test_str.velocity = []
            test_str.effort = []
        else:
            test_str = JointState()
            test_str.header = Header()
            test_str.header.stamp = rospy.Time.now()
            # rospy.Time.now()
            #Fill out the joint positions based on the names, if they are not exists, fill with zeros
            test_str.name = name
            test_str.position = []
            for i, n in enumerate(name):
                if n in names_actual:
                    test_str.position.append(q_actual[names_actual.index(n)])
                

            test_str.velocity = []
            test_str.effort = []

        # Get latest AI poses with timeout handling
        ai_targets = ai_monitor.get_poses()

        msg = ROSToUnityMessage()
        msg.poses_ai_target = ai_targets
        msg.q_target = test_str
        if q_init is not None:
            pub.publish(msg)

        
        
        rate.sleep()

q_target = pin.neutral(PANDAmodel)
q_ik_target = np.zeros(7)
q_actual = np.zeros(8)
q_init = None
name = []
for n in PANDAmodel.names:
    name.append(str(n))
name.pop(0)
names_actual = []


helper_init = False


def IK_callback(data):
    global q_target, q_init, q_ik_target,  helper_init
    if helper_init == False:
        rospy.loginfo('IK: Waiting')
    else:
        if q_init is None: #Read Unity
            
            q_target[0:8] = np.array(data.q_actual.position)
            q_target[7] = q_target[6]

            q_init = q_target 
        else:
            
            q_target[0:7] = q_ik_target
            q_target[7] = q_target[6]

def bik_callback(data):
    global q_ik_target, helper_init
    if not helper_init:
        helper_init = True
    q_ik_target[0:7] = data.position[0:7]


PANDA_HAND_HEIGHT = 0.127
PANDA_FINGERS_HEIGHT = 0.05
rot_180_x = tft.quaternion_from_euler(np.pi, 0, 0)
rot_90_z = tft.quaternion_from_euler(0, 0, np.pi/2)
rot_ros_to_unity = tft.quaternion_multiply(rot_90_z, rot_180_x)

# …same imports and PANDA constants…


PANDA_HAND_HEIGHT    = 0.127
PANDA_FINGERS_HEIGHT = 0.05




# tf_broadcaster = tf.TransformBroadcaster()


def propose_grasping_pose(obj_item):
    # 1) Read object orientation & scales
    q_obj   = np.array([obj_item.orientation.x,
                        obj_item.orientation.y,
                        obj_item.orientation.z,
                        obj_item.orientation.w])
    scales  = np.array([obj_item.scale.x,
                        obj_item.scale.z,
                        obj_item.scale.y])
    

    # Eliminate the grasping pose, if the object is above the table
    if obj_item.position.z > 0.5:
        return None
    R = tft.quaternion_matrix(q_obj)[:3,:3]

    # 2) Find which local axis is “up” → top face half‐height
    z_w         = np.array([0,0,1.0])
    proj_heights = [abs(R[:,i].dot(z_w)) * (scales[i]/2.0)
                    for i in range(3)]
    top_idx     = int(np.argmax(proj_heights))
    half_height = proj_heights[top_idx]

    # 3) Pick the smaller of the two in‐plane edges by their XY‐projected half lengths
    candidates = [0,1,2]
    candidates.remove(top_idx)
    def proj_len(i):
        v = R[:,i]
        return np.linalg.norm(v[:2]) * (scales[i]/2.0)
    small_axis = min(candidates, key=proj_len)

    # 4) Build b = normalized world‐XY projection of that small axis
    v = R[:, small_axis]
    b = v[:2]
    bn = np.linalg.norm(b)
    if bn < 1e-6:
        b = np.array([1.0,0.0])
    else:
        b /= bn
    b_x, b_y = b

    # 5) Solve yaw so that gripper’s LOCAL Y (jaw opening) aligns with b:
    #    (-sinθ, cosθ) == (b_x, b_y) -> θ = atan2(-b_x, b_y)
    yaw = np.arctan2(-b_x, b_y)

    # 6) Build the “fixed‐down” quaternion: roll=π, pitch=0, yaw=0
    #    this makes local Z point straight down (0,0,-1)
    q_down = tft.quaternion_from_euler(np.pi, 0.0, 0.0, axes='sxyz')

    # 7) Build the spin ABOUT that Z by yaw
    #    Since after q_down, the gripper’s Z==world‐down,
    #    a yaw about Z is simply:
    q_spin = tft.quaternion_from_euler(0.0, 0.0, -yaw, axes='sxyz')

    # 8) Compose: first spin, then flip down (order matters)
    q_final = tft.quaternion_multiply(q_down, q_spin)

    # 9) Fill in your Pose
    grasp = Pose()
    grasp.position.x = obj_item.position.x
    grasp.position.y = obj_item.position.y
    grasp.position.z = (obj_item.position.z
                        + half_height
                        + (PANDA_HAND_HEIGHT - PANDA_FINGERS_HEIGHT))
    grasp.orientation.x = q_final[0]
    grasp.orientation.y = q_final[1]
    grasp.orientation.z = q_final[2]
    grasp.orientation.w = q_final[3]

    # 10) (Optional) broadcast for visualization
    #name = obj_item.header.frame_id + "_grasp"
    #tf_broadcaster.sendTransform(
    #    (grasp.position.x, grasp.position.y, grasp.position.z),
    #    (grasp.orientation.x, grasp.orientation.y,
    #     grasp.orientation.z, grasp.orientation.w),
    #    rospy.Time.now(),
    #    name,
    #    "base"
    #)

    return grasp

def ai_callback(data):
    global ai_monitor
    #Read All data, and extract the AI target poses if they contains the object name
    PoseArray_msg = PoseArray()
    PoseArray_msg.header = Header()
    PoseArray_msg.header.stamp = rospy.Time.now()
    pose_names = []
    for obstacle in data.obstacles:
        obstacle_name = obstacle.header.frame_id
        if "object" in obstacle_name:
            Posemsg = propose_grasping_pose(obstacle)
            if Posemsg is not None:  # Only add valid poses
                PoseArray_msg.poses.append(Posemsg)
                pose_names.append(obstacle_name)
    #make string from the list
    pose_names_str = ';'.join(pose_names)
    PoseArray_msg.header.frame_id = pose_names_str


    ai_monitor.update_poses(PoseArray_msg)

def RealRobot_callback(data):
    global q_actual, names_actual, name
    q_actual = data.position
    names_actual = data.name
    if not (len(names_actual) == len(name)):
        print("Names are not equal, Lengths: ", len(names_actual), len(name))


if __name__ == '__main__':
    try:
        rospy.init_node('ROS_w_Unity')

        rospy.Subscriber("unity_to_ros", UnityToROSMessage, IK_callback)#Listens to the Unity messages contains the actual joint positions, fills the target joint positions
        rospy.Subscriber("/joint_states", JointState, RealRobot_callback) #Listens to the actual joint positions from the real robot
        rospy.Subscriber("bik_output", JointState, bik_callback) #Listens to the robot's target joint positions from the bik (published by the bik_helper.py)
        rospy.Subscriber("obstacle_info", ObstacleInfos, ai_callback) #Listens to the all possible AI target pose (published by the ropedetection/central_publish2unity.py
        RealRobot = rospy.get_param('~RealRobot', False)
        rospy.loginfo('IK: Ready, Are we reading data from Real Robot: ' + str(RealRobot))
        talker() #Runs in loop, sends the target joint positions, AI target poses, left and right robot's target joint positions to the Unity
    except rospy.ROSInterruptException:
        pass