#! /usr/bin/env python3
import sys
import os
import rospy
from bik_pkg.robot_class import Robot_bik, bik
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from ros_tcp_endpoint.msg import UnityToROSMessage
import rospkg
from ros_tcp_endpoint.msg import ObstacleInfos


#RelaxedIK original
rospack = rospkg.RosPack()
bik_package_path = rospack.get_path('bik_pkg')


import threading

class RobotMonitor:
    def __init__(self, robot):
        self.robot = robot
        self.lock = threading.Lock()
        self.reset_status = False

    def reset(self, q_actual = None):
        if q_actual is None:
            with self.lock:
                self.robot.reset()
        else:
            with self.lock:
                self.robot.reset(q_actual)

    def set_target(self, x_target_pos, x_target_quat):
        with self.lock:
            self.robot.set_target(x_target_pos, x_target_quat)

    def set_other(self, q_other):
        with self.lock:
            self.robot.set_other(q_other)

    def set_helper(self, position, orientation):
        with self.lock:
            self.robot.set_helper(position, orientation)

    def set_obstacles(self, obstacles):
        with self.lock:
            self.robot.set_obstacles(obstacles)

    def sharedautonomy_arbitration(self):
        with self.lock:
            self.robot.sharedautonomy_arbitration()

    def get_sharedautonomy_mode(self):
        with self.lock:
            return self.robot.sharedautonomy_mode
    

# Initialize RobotMonitor


robot = None #Robot_bik()
q_actual = None # robot.Q
callback_called = False
#import tf
#tf_broadcaster = tf.TransformBroadcaster()
def callback(data):
    global robot_monitor, callback_called

    if not callback_called:
        rospy.loginfo("Callback called")
        callback_called = True
    # data.q_actual.header.stamp = rospy.Time.now()
    q_actual= bik.np.array(data.q_actual.position)[0:7]
    x_target_pos = data.x_target.position
    x_target_quat = data.x_target.orientation
    x_helper_pos = data.x_helper.position
    x_helper_quat = data.x_helper.orientation
    #tf_broadcaster.sendTransform(
    #    (x_helper_pos.x, x_helper_pos.y, x_helper_pos.z),
    #    (x_helper_quat.x, x_helper_quat.y, x_helper_quat.z, x_helper_quat.w),
    #    rospy.Time.now(),
    #    "helper",
    #    "base"
    #)

    if data.robot_status.data:
        robot_monitor.reset()
        robot_monitor.reset_status = True
    else:
        robot_monitor.reset_status = False

    robot_monitor.set_target(x_target_pos,x_target_quat)
    if solver_mode == "relaxedik" or solver_mode == "relaxedik_cbf" or solver_mode == "collisionik":
        if robot_monitor.get_sharedautonomy_mode()  != "None":
            if (robot._helperpos is None or 
                robot._helperquat is None or 
                not bik.np.array_equal(x_helper_pos, robot._helperpos) or 
                not bik.np.array_equal(x_helper_quat, robot._helperquat)):
                robot_monitor.set_helper(x_helper_pos,x_helper_quat)
            
        if robot.sharedautonomy_mode == "Arbitration":
            robot_monitor.sharedautonomy_arbitration()

name = []


def talker():
    global robot_monitor, name, q_actual, callback_called
    pub = rospy.Publisher('bik_output' , JointState, queue_size=10)
    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        if callback_called:  
            Q = JointState()
            if not robot_monitor.reset_status:
                q_target = robot_monitor.robot.run(q_actual)
            else:
                robot_monitor.robot.run(q_actual)
                q_target = robot_monitor.robot.init
    
            Q.header = Header()
            Q.header.stamp = rospy.Time.now()
            # rospy.Time.now()
            Q.name = name[0:len(q_target)]
            Q.position = q_target
            Q.velocity = []
            Q.effort = []
            pub.publish(Q)
        rate.sleep()

solver_mode = "relaxedik"

if __name__ == '__main__':
    try:
        rospy.init_node('bik_robot', anonymous=True, log_level=rospy.INFO)
        sharedautonomy_mode = rospy.get_param('~sharedautonomy_mode', "None")
        solver_mode = rospy.get_param('~solver_mode', "relaxedik") 
        # "relaxedik_original" for rust version
        # "relaxedik" for jax version
        # "relaxedik_cbf" for jax version with cbf
        if solver_mode == "relaxedik" or solver_mode == "relaxedik_cbf"or solver_mode == "collisionik":
            robot = Robot_bik(sharedautonomy_mode, solver_mode)
            robot.set_mode(sharedautonomy_mode, solver_mode)
            for n in robot.PANDAmodel.names:
                name.append(str(n))
            
                
        elif solver_mode == "relaxedik_original":
            relaxedik_core_package_path = rospack.get_path('relaxed_ik_ros1')
            print("RelaxedIK Core Package Path: ", relaxedik_core_package_path)
            sys.path.append(relaxedik_core_package_path + '/scripts')
            from relaxed_ik_rust import RelaxedIK as relaxedik_original  # Import directly
            robot = relaxedik_original()
        else:
            rospy.logerr("Solver mode not recognized. Exiting. " \
            "Possible values are: relaxedik, relaxedik_original, relaxedik_cbf")
            sys.exit(1)
        robot_monitor = RobotMonitor(robot)
        if solver_mode == "relaxedik_cbf"or solver_mode == "collisionik":
            rospy.Subscriber("obstacle_info", ObstacleInfos, robot_monitor.set_obstacles)
        

        
        rospy.Subscriber("unity_to_ros", UnityToROSMessage, callback)
        rospy.loginfo('Helper : Ready!' + '-> Shared Autonomy Mode: ' + sharedautonomy_mode + ' -> Solver Mode: ' + solver_mode)
        talker()
    except rospy.ROSInterruptException:
        pass