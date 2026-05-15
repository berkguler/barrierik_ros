#! /usr/bin/env python3
import sys
import rospy
import pinocchio as pin
import numpy as np
import threading
from os.path import join
from pinocchio.visualize import MeshcatVisualizer
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from ros_tcp_endpoint.msg import UnityToROSMessage, ObstacleInfos
import rospkg
import bik_pkg.parse_urdf as parse_urdf
from bik_pkg.parse_urdf import jax, jacfwd, config, grad, jit, vmap, jnp, rbda, pin, js
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import bik_pkg.bik_collision as bik_collision
import time
import jaxlie
import matplotlib
matplotlib.use('TkAgg')

class RealTimeRenderCollision:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('capsule_visualizer', anonymous=True)
        
        # Get package path
        rospack = rospkg.RosPack()
        self.package_path = rospack.get_path('bik_pkg')
        
        # Load robot model
        self.PANDAmodel, _, _, self.PANDAdata,  _, _, self.robotmodel, self.robotdata, self.list_panda_capsules = parse_urdf.init(
            yaml_file = self.package_path + "/configs/franka_description.yaml", 
            open_viewer = False, 
            disable_pin_models = False, 
            disable_viewer = True)
        
        print(f"Model name: {self.robotmodel.name()}")
        print(f"Number of links: {self.robotmodel.number_of_links()}")
        print(f"Number of joints: {self.robotmodel.number_of_joints()}")

        # print()
        print(f"Links:\n{self.robotmodel.link_names()}")

        # print()
        print(f"Joints:\n{self.robotmodel.joint_names()}")

        print()
        print(f"Frames:\n{self.robotmodel.frame_names()}")
        print(self.robotdata.joint_positions())


        # Set up forward kinematics functions
        self.forward_kinematics_all = jit(rbda.forward_kinematics_model)
        self.forward_kinematix_all = jax.jit(lambda Q, Pos: self.forward_kinematics_all(self.robotmodel, base_position=Pos,base_quaternion=jnp.array([1.0,0.0,0.0,0.0]), joint_positions=Q))

        # Extract capsule data
        self.N_panda = len(self.list_panda_capsules)
        a_list, b_list, C_list, L_list, R_list, T_list = zip(*self.list_panda_capsules)
        self.a_panda = jnp.stack(a_list)
        self.b_panda = jnp.stack(b_list)
        self.C_panda = jnp.stack(C_list)
        self.L_panda = jnp.stack(L_list)
        self.R_panda = jnp.stack(R_list)
        self.T_panda = jnp.stack(T_list)
        
        # Current joint positions and obstacles
        self.joint_positions = np.zeros(9)  # 9 DOF (7 robot + 2 gripper)
        self.obstacles = {}
        self.obstacle_data = ObstacleInfos()
        self.lock = threading.Lock()
        
        # Set up matplotlib figure
        plt.ion()  # Turn on interactive mode
        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Set axis limits and labels
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_zlim(0, 2)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1
        self.ax.set_title('Robot Capsule Visualization')
        
        # List to store visualization objects
        self.capsule_artists = []
        self.obstacle_artists = []
        
        # Set up ROS subscribers
        rospy.Subscriber("unity_to_ros", UnityToROSMessage, self.unity_callback)
        rospy.Subscriber("obstacle_info", ObstacleInfos, self.set_obstacles)
        
        # Update rate
        self.update_rate = 10  # Hz
        self.rate = rospy.Rate(self.update_rate)
        
        # Flag to indicate visualization is running
        self.running = False

    def render_capsules(self):
        """Update and render all capsules (robot and obstacles)"""
        # Clear the current axes for fresh rendering
        self.ax.clear()
        
        # Configure the axes
        self.ax.set_xlim(-.5, 1.5)
        self.ax.set_ylim(-1, 1)
        self.ax.set_zlim(0, 2)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_box_aspect([1, 1, 1])
        self.ax.set_title('Robot Capsule Visualization')
        
        with self.lock:
            # Update robot capsules with current joint positions
            current_joint_pos = np.copy(self.joint_positions[:7])
        
        # Get updated robot capsules
        try:
            updated_capsules = self.update_robot_capsules(current_joint_pos)
            
            # Render robot capsules
            for a, b, C, L, R, T in updated_capsules:
                bik_collision.render_capsule(self.ax, T, L, R, color='blue')
                
            # Render line between capsule endpoints for clarity
            for a, b, C, L, R, T in updated_capsules:
                self.ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], 'k--', linewidth=0.5)
            
            # Render obstacles
            with self.lock:
                for name, (a, b, C, L, R, T) in self.obstacles.items():
                    bik_collision.render_capsule(self.ax, T, L, R, color='red')
                    # Add text label for the obstacle
                    self.ax.text(C[0], C[1], C[2], name, fontsize=8)
        
        except Exception as e:
            rospy.logerr(f"Error in render_capsules: {e}")
            
        # Redraw the canvas
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        # Pause to allow for rendering
        plt.pause(0.001)
        
    def unity_callback(self, data):
        with self.lock:
            # Extract joint positions if provided
            if hasattr(data, 'q_actual') and data.q_actual.position:
                self.joint_positions[:7] = np.array(data.q_actual.position)[:7]
                
    def set_obstacles(self,data):
        if data == self.obstacle_data:
            return
        with self.lock:
            for obstacle in data.obstacles:
                obst_name = obstacle.header.frame_id
                if "obstacle" in obst_name:
                    
                    if obst_name not in self.obstacles:
                        print("Obstacle Name: ", obst_name)
                        obstacle_parameters = self.load_obstacle(obstacle)
                        if obstacle_parameters:
                            self.obstacles[obst_name] = obstacle_parameters
                    else:
                        obstacle_parameters = self.obstacles[obst_name]
                        new_obstacle_parameters = self.update_obstacle(obstacle, obstacle_parameters)
                        self.obstacles[obst_name] = new_obstacle_parameters
            self.obstacle_data = data

    def load_obstacle(self, obj):
        R = jaxlie.SO3.from_quaternion_xyzw(jnp.array([obj.orientation.x, obj.orientation.y, obj.orientation.z, obj.orientation.w]))
        M = jaxlie.SE3.from_rotation_and_translation(R, jnp.array([obj.position.x, obj.position.y, obj.position.z]))
        scale = jnp.array([obj.scale.x, obj.scale.z, obj.scale.y])
        if obj.type == 0:
            scale = jnp.array([obj.scale.x, obj.scale.z, obj.scale.y])
            capsule_representation = bik_collision.box2capsule(center = jnp.array([obj.position.x, obj.position.y, obj.position.z]), dimensions = scale, keep_dimensions= False)
            print(obj.header.frame_id, capsule_representation)
        elif obj.type == 2:
            scale = jnp.array([obj.scale.x, obj.scale.z, obj.scale.y*2])
            capsule_representation = bik_collision.box2capsule(center = jnp.array([obj.position.x, obj.position.y, obj.position.z]), dimensions = scale, keep_dimensions= False)
            print(obj.header.frame_id, capsule_representation)
        #else:
        #    rospy.logerr("Obstacle type %s not supported", obj.type)
        #    return None
        #Rotate the capsule to the correct orientation
        T_obs = M.as_matrix()
        a_obs, b_obs = bik_collision.capsule_ab_from_T(T_obs, capsule_representation[3])
        C_obs = (a_obs + b_obs) / 2
        L_obs = np.linalg.norm(a_obs - b_obs)
        R_obs = capsule_representation[4]
        capsule_representation = (a_obs, b_obs, C_obs, L_obs, R_obs, T_obs)
        # Add the capsule to the obstacles dictionary
        return capsule_representation
        
    def update_obstacle(self, obj, obstacle_parameters):
        
        a_obs, b_obs, C_obs, L_obs, R_obs, T_obs = obstacle_parameters
        R = jaxlie.SO3.from_quaternion_xyzw(jnp.array([obj.orientation.x, obj.orientation.y, obj.orientation.z, obj.orientation.w]))
        M = jaxlie.SE3.from_rotation_and_translation(R, jnp.array([obj.position.x, obj.position.y, obj.position.z])).as_matrix()
        T_obs = M
        a_obs, b_obs = bik_collision.capsule_ab_from_T(T_obs, L_obs)
        C_obs = (a_obs + b_obs) / 2
        L_obs = np.linalg.norm(a_obs - b_obs)
        print("Upd. obs. %s", obj.header.frame_id, C_obs, L_obs, R_obs)
        return (a_obs, b_obs, C_obs, L_obs, R_obs, T_obs)
    
    def update_robot_capsules(self, joint_pos):
        # Ensure proper padding for the joint positions vector
        padded_joint_pos = np.zeros(8)
        padded_joint_pos[:7] = joint_pos
        
        # Convert to JAX array
        jax_joint_pos = jnp.array(padded_joint_pos)
        
        # Compute forward kinematics for all links
        FK_all = self.forward_kinematix_all(jax_joint_pos, jnp.array([0.0, 0.0, 0.0]))
        #print("FK_all: ", len(FK_all), self.N_panda, len(self.list_panda_capsules))
        #for i, name in enumerate(self.robotmodel.link_names()):
            #print(f"Link {i} FK: ", name, FK_all[i])
        pin.forwardKinematics(self.PANDAmodel, self.PANDAdata, np.append(padded_joint_pos, [0.0]))
        pin.updateFramePlacements(self.PANDAmodel, self.PANDAdata)
        pin_FK_all = self.PANDAdata.oMi
        #for i in range(len(pin_FK_all)):
            #print(f"Link {i} FK: ", pin_FK_all[i])
#            print(f"Link {i} FK: ", pin_FK_all[i])
        # Create a list to store updated capsule parameters
        new_capsule_params = []
        
        for i in range(self.N_panda):
            # Get the original capsule parameters
            obstacle_parameters = self.list_panda_capsules[i]
            a, b, C, L, R, T = obstacle_parameters
            
            # Apply the transformation from forward kinematics
            # Note: We need to check if the FK transform is available for this link
            if i < len(FK_all):
                T_fk = FK_all[i]
                new_T = T_fk @ T
                a_new, b_new = bik_collision.capsule_ab_from_T(new_T, L)
                C_new = (a_new + b_new) / 2
                L_new = jnp.linalg.norm(a_new - b_new)
                new_capsule_param = (a_new, b_new, C_new, L_new, R, new_T)
                new_capsule_params.append(new_capsule_param)
            else:
                # If FK is not available, keep the original parameters
                new_capsule_params.append(obstacle_parameters)
        
        return new_capsule_params

    def run(self):
        """Main loop to update visualization"""
        rospy.loginfo("Starting capsule visualization...")
        self.running = True
        
        try:
            while not rospy.is_shutdown() and self.running:
                try:
                    self.render_capsules()
                    self.rate.sleep()
                except Exception as e:
                    rospy.logerr(f"Error in visualization loop: {e}")
        
        except KeyboardInterrupt:
            rospy.loginfo("Visualization terminated by user")
        finally:
            self.running = False
            plt.close('all')

if __name__ == "__main__":
    try:
        visualizer = RealTimeRenderCollision()
        visualizer.run()
    except rospy.ROSInterruptException:
        pass
