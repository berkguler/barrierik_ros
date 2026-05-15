#! /usr/bin/env python3
"""
Robot State Publisher Node (Timer-based Collision Check)

This node manages the joint states of a dual-arm robot system.
Instead of a dedicated collision thread, collision checks are scheduled
via a timer callback. This avoids potential thread-safety issues with
Pinocchio’s collision functions and the underlying ROS communication.
"""

import sys
import rospy
import pinocchio as pin
import numpy as np
import threading
from os.path import join
from pinocchio.visualize import MeshcatVisualizer
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from ros_tcp_endpoint.msg import UnityToROSMessage
import rospkg
import bik_pkg.parse_urdf as parse_urdf
from ros_tcp_endpoint.msg import ObstacleInfos
import tf2_ros
import hppfcl
import meshcat.geometry as mg
import meshcat.transformations as mt
import time


class RobotStatePublisher:
    def __init__(self):
        # Load the robot model and initialize Pinocchio data.
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('bik_pkg')
        self.loaded_single_robot = False
        while not self.loaded_single_robot:
            try:
                (self.PANDAmodel, self.PANDAcollision_model, self.PANDAvisual_model,
                self.PANDAdata, self.PANDAcollision_data, self.PandaViewer,
                _, _, _) = parse_urdf.init(
                    yaml_file=join(package_path, "configs/franka_description.yaml"),
                    open_viewer=True,
                    disable_pin_models=False,
                    disable_viewer=False)
                self.loaded_single_robot = True
                rospy.loginfo("Robot model loaded successfully.")
            except Exception as e:
                rospy.logerr("Error loading robot model., retrying: %s", str(e))
                time.sleep(1)
            

        # Get joint names (skipping the dummy root element)
        self.name = [str(n) for n in self.PANDAmodel.names][1:]
        self.q_target = pin.neutral(self.PANDAmodel)
        self.obstacles = {}
        self.NUM_GEOM_PANDA = len(self.PANDAcollision_model.geometryObjects)
        #Remove all self collision pairs
        self.PANDAcollision_model.removeAllCollisionPairs()
        self.PANDAcollision_data = self.PANDAcollision_model.createData()
        for colres in self.PANDAcollision_data.collisionResults:
            colres.clear()
        # Remove all collision pairs
            colres.security_margin = 100

        # Lock to protect shared data.
        self.viewer_lock = threading.Lock()
        self.reset_visualization = False

        # Set up ROS subscribers & publishers.
        rospy.Subscriber("unity_to_ros", UnityToROSMessage, self.IK_callback)
        rospy.Subscriber("obstacle_info", ObstacleInfos, self.OBS_callback)
        self.pub = rospy.Publisher('joint_states_modified', JointState, queue_size=10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.ignored_collision_list = ["holder", "ground", "leg", "object"]
        rospy.loginfo('Robot State Publisher: Ready')

        # Start a Timer for running collision checking periodically.

        # Start the talker loop for publishing joint states and updating the viewer.
        self.talker()

    def OBS_callback(self, data):
        self.reset_visualization = False
        with self.viewer_lock:
            for obstacle in data.obstacles:
                self.OBS_tf_publisher(obstacle)
                obst_name = obstacle.header.frame_id

                if obst_name not in self.obstacles:
                    #rospy.loginfo("Adding obstacle: %s", obst_name)
                    #rospy.loginfo("Adding obstacle: %s", obst_name)
                    collision_id, visual_id = self.load_obstacle(obstacle)
                    self.obstacles[obst_name] = [obstacle, collision_id, visual_id]
                    self.reset_visualization = True
                else:
                    self.obstacles[obst_name][0] = obstacle
                    self.update_obstacle(*self.obstacles[obst_name])
            if self.reset_visualization:
                rospy.loginfo("Resetting visualization")
                self.PANDAcollision_data = self.PANDAcollision_model.createData()
                for colres in self.PANDAcollision_data.collisionResults:
                    colres.security_margin = 100
                self.PANDAvisual_data = self.PANDAvisual_model.createData()
                self.PandaViewer.rebuildData()
                self.PandaViewer.loadViewerModel(rootNodeName="pinocchio")
                #Print all collision pairs
                #for i in range(len(self.PANDAcollision_model.collisionPairs)):
                    #cp = self.PANDAcollision_model.collisionPairs[i]
                    #rospy.loginfo("Collision pair %d: %d, %d", i, cp.first, cp.second)

    def load_obstacle(self, obj):
        # Create a transformation from the obstacle's pose.
        R = pin.Quaternion(obj.orientation.w, obj.orientation.x, obj.orientation.y, obj.orientation.z)
        M = pin.SE3(R, np.array([obj.position.x, obj.position.y, obj.position.z]))
        scale = np.array([obj.scale.x, obj.scale.z, obj.scale.y])
        if obj.type == 0:
            geom = hppfcl.Box(scale[0], scale[1], scale[2])
        elif obj.type == 1:
            geom = hppfcl.Sphere(scale[0])
        elif obj.type == 2:
            geom = hppfcl.Capsule(scale[0]/2, scale[1]*4)
        else:
            rospy.logerr("Obstacle type %s not supported", obj.type)
            return None, None

        geom_pin = pin.GeometryObject(obj.header.frame_id, 0, M, geom)
        color = np.append(np.random.rand(3), 0.8)
        if "obstacle" in obj.header.frame_id:
            color = np.array([0, 0, 0, 0.8])
        geom_pin.meshColor = color
        geom_pin.overrideMaterial = True
        geom_pin.meshMaterial = pin.GeometryPhongMaterial()
        geom_pin.meshMaterial.meshEmissionColor = np.array([1., 0.1, 0.1, 1.])
        geom_pin.meshMaterial.meshSpecularColor = np.array([0.1, 1., 0.1, 1.])
        geom_pin.meshMaterial.meshShininess = 0.8

        collision_id = self.PANDAcollision_model.addGeometryObject(geom_pin)
        for i in range(self.NUM_GEOM_PANDA):
            #Check if the obstacle name contains any word in the ignored collision list
            if any(ignored in obj.header.frame_id for ignored in self.ignored_collision_list):
                continue
            col_pair = pin.CollisionPair(i, collision_id)
            self.PANDAcollision_model.addCollisionPair(col_pair)
            #add the safety margin
            #rospy.loginfo("Collision pair added: %d, %d", i, collision_id)
        




        visual_id = self.PANDAvisual_model.addGeometryObject(geom_pin)
        rospy.loginfo("Obstacle added: %s, Collision ID: %d, Visual ID: %d", 
                      obj.header.frame_id, collision_id, visual_id)
        return collision_id, visual_id

    def update_obstacle(self, obj, collision_id, visual_id):
        existing_visual_ref = self.PANDAvisual_model.geometryObjects[visual_id]
        existing_collision_ref = self.PANDAcollision_model.geometryObjects[collision_id]
        placement = existing_visual_ref.placement
        placement.translation = np.array([obj.position.x, obj.position.y, obj.position.z])
        placement.rotation = pin.Quaternion(obj.orientation.w, obj.orientation.x,
                                            obj.orientation.y, obj.orientation.z).matrix()
        existing_visual_ref.placement = placement
        existing_collision_ref.placement = placement


    def collision_callback(self):
        """Timer callback to perform collision checking periodically."""

        # Update geometry placements with the latest joint configuration.
        pin.computeCollisions(self.PANDAmodel, self.PANDAdata, self.PANDAcollision_model, 
                              self.PANDAcollision_data, self.q_target,
                              stop_at_first_collision=False)
        pin.computeDistances(self.PANDAcollision_model,
                             self.PANDAcollision_data)
        self.distance_matrix = np.zeros((self.NUM_GEOM_PANDA, len(self.obstacles)))
        # Print the status of collision for all collision pairs
        for k in range(len(self.PANDAcollision_model.collisionPairs)): 
            cr = self.PANDAcollision_data.collisionResults[k]
            cp = self.PANDAcollision_model.collisionPairs[k]
            #print("collision pair:",cp.first,",",cp.second,"- collision:","Yes" if cr.isCollision() else "No")
            
            if cr.isCollision():
                colres = self.PANDAcollision_data.collisionResults[k]
                contact: hppfcl.Contact = colres.getContacts()[0]
                p1 = contact.getNearestPoint1()
                p2 = contact.getNearestPoint2()
                dist = np.linalg.norm(p1 - p2)
                self.distance_matrix[cp.first, cp.second-self.NUM_GEOM_PANDA] = dist
                #print(p1, p2, p1 - p2)
                
                name = self.PANDAcollision_model.geometryObjects[cp.first].name
                name2 = self.PANDAcollision_model.geometryObjects[cp.second].name
                total_name =  name + " - "+name2
                random_color = np.random.rand(3)
                hex_color = "0x{:02x}{:02x}{:02x}".format(int(random_color[0]*255), int(random_color[1]*255), int(random_color[2]*255))
                
                #plot on the viewer
                points = np.hstack([p1.reshape(-1,1), p2.reshape(-1,1)]).astype(np.float32)
                
                self.PandaViewer.viewer[total_name].set_object(mg.Line(mg.PointsGeometry(points), mg.MeshBasicMaterial(color=hex_color, linewidth=50)))
                self.PandaViewer.viewer[name].set_object(mg.Sphere(0.01), mg.MeshLambertMaterial(color=hex_color))
                self.PandaViewer.viewer[name].set_transform(mt.translation_matrix(p1))
                self.PandaViewer.viewer[name2].set_object(mg.Sphere(0.01), mg.MeshLambertMaterial(color=hex_color))
                self.PandaViewer.viewer[name2].set_transform(mt.translation_matrix(p2))
            else:
                # Objects are not colliding - print the nearest distance
                dr = self.PANDAcollision_data.distanceResults[k]
                name = self.PANDAcollision_model.geometryObjects[cp.first].name
                name2 = self.PANDAcollision_model.geometryObjects[cp.second].name
                
                #print(f"  Distance between {name} and {name2}: {dr.min_distance:.6f}")
                
                # Try to get and visualize nearest points using the direct methods
                try:
                    p1 = dr.getNearestPoint1()
                    p2 = dr.getNearestPoint2()
                    dist = np.linalg.norm(p1 - p2)
                    self.distance_matrix[cp.first, cp.second-self.NUM_GEOM_PANDA] = dist
                    
                    # Only visualize if the points are valid
                    if p1 is not None and p2 is not None:
                        #print(f"  Nearest points: {p1}, {p2}")
                        
                        # Visualize with a green color for non-colliding objects
                        total_name = name + " - " + name2
                        safe_color = "0x00ff00"  # Green color for safe distances
                        
                        points = np.hstack([p1.reshape(-1,1), p2.reshape(-1,1)]).astype(np.float32)
                        self.PandaViewer.viewer[total_name].set_object(mg.Line(mg.PointsGeometry(points), 
                                                                            mg.MeshBasicMaterial(color=safe_color, linewidth=10)))
                except Exception as e:
                    # If the getNearestPoint methods fail, log the error but continue
                    rospy.logwarn(f"Could not get nearest points for {name} and {name2}: {e}")
        # Print the distance matrix
        #print("Distance matrix:")
        #print(self.distance_matrix)
    def OBS_tf_publisher(self, obstacle):
        # Publish obstacle transform for tf and RViz.
        obs_name = obstacle.header.frame_id
        obs_tf = tf2_ros.TransformStamped()
        obs_tf.header.stamp = rospy.Time.now()
        obs_tf.header.frame_id = "base"
        obs_tf.child_frame_id = obs_name
        obs_tf.transform.translation.x = obstacle.position.x
        obs_tf.transform.translation.y = obstacle.position.y
        obs_tf.transform.translation.z = obstacle.position.z
        obs_tf.transform.rotation.x = obstacle.orientation.x
        obs_tf.transform.rotation.y = obstacle.orientation.y
        obs_tf.transform.rotation.z = obstacle.orientation.z
        obs_tf.transform.rotation.w = obstacle.orientation.w
        self.tf_broadcaster.sendTransform(obs_tf)

    def IK_callback(self, data):
        self.q_target[0:8] = np.array(data.q_actual.position)
        self.q_target[8] = self.q_target[7]

    def talker(self):
        rate = rospy.Rate(10)
        self.PandaViewer.initializeFrames()
        while not rospy.is_shutdown():
            js = JointState()
            js.header = Header(stamp=rospy.Time.now())
            js.name = self.name
            js.position = self.q_target.copy()
            js.velocity = []
            js.effort = []
            self.pub.publish(js)
            with self.viewer_lock:
                try:
                    self.PandaViewer.updatePlacements(pin.GeometryType.COLLISION)
                    self.PandaViewer.display(js.position)
                except Exception as e:
                    rospy.logerr("Error in visualization: %s", str(e))

                #try:
                #start = time.time()
                self.collision_callback()
                #end = time.time()
                #print("Collision check time: ", end - start)
                #except Exception as e:
                #    rospy.logerr("Error in collision callback: %s", str(e))

            rate.sleep()


if __name__ == '__main__':
    try:
        rospy.init_node('RobotJointStatePublisher', anonymous=True)
        robot_state_publisher = RobotStatePublisher()
        rospy.loginfo('Robot State Publisher Node: Running')
    except rospy.ROSInterruptException:
        pass
