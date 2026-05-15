cmake_minimum_required(VERSION 2.8.3)
project(ros_tcp_endpoint)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  sensor_msgs
  rosgraph_msgs
  message_generation
)
add_message_files(
  FILES
  ROSToUnityMessage.msg
  UnityToROSMessage.msg
  ObstacleInfo.msg
  ObstacleInfos.msg
)

catkin_python_setup()
generate_messages(
  DEPENDENCIES
  std_msgs  # Or any other dependencies your messages might have
  geometry_msgs
  sensor_msgs
  rosgraph_msgs
)
catkin_package(CATKIN_DEPENDS message_runtime geometry_msgs)
