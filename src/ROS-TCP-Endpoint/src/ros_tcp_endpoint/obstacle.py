#!/usr/bin/env python

import rospy
from ros_tcp_endpoint.msg import ObstacleInfo

def callback(data):
    rospy.loginfo("Obstacle Position: x=%f, y=%f, z=%f", data.position.x, data.position.y, data.position.z)
    rospy.loginfo("Obstacle Scale: x=%f, y=%f, z=%f", data.scale.x, data.scale.y, data.scale.z)

def listener():
    rospy.init_node('obstacle_listener', anonymous=True)
    rospy.Subscriber('obstacle_info', ObstacleInfo, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()
