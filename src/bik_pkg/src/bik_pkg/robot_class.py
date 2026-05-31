#! /usr/bin/env python3
import sys
import traceback
import rospy
import bik_pkg.bik_core as bik
from ros_tcp_endpoint.msg import ObstacleInfo, ObstacleInfos
from collections import deque 
import random
import math
from scipy.spatial.transform import Rotation as R
import threading
import time

SUPPORTED_SOLVER_MODES = {
    "relaxedik",
    "barrierik",
    "barrierik_moving",
    "collisionik",
}

class Robot_bik:
    def __init__(self, sharedautonomy_mode = "None", solver_mode = "relaxedik", robot_tag = "",  XTOL = 1e-6, MAXEVAL = 40):
        self.target = bik._Mtarget_SE3
        self.init = bik._theta_0
        self.sharedautonomy_mode = "None" #"None" or "Arbitration"
        self.solver_mode = "relaxedik" #"relaxedik", "barrierik", "barrierik_moving", "collisionik", "other"
        self.helper = bik._Mhelper_SE3
        self._helperpos = None
        self._helperquat = None
        self.init = bik.np.array(bik.q_median_config[0:7])
        self.Q =  bik.np.array(bik.q_median_config[0:7])
        self.Q_history = deque(maxlen=10)
        self.tp = bik._theta_poses
        self.tv = bik._theta_vels
        self.ta = bik._theta_accels
        self.XTOL = XTOL
        self.MAXEVAL = MAXEVAL
        self.value_threshold = 200
        self.last_step_success = True
        self.simulation_gap = False
        self.robot_tag = robot_tag
        bik.side = robot_tag
        self.alpha = 1
        self.alpha_pos = 1
        self.alpha_orient = 1
        self.Cposes = bik.Cposes
        self.lower_limit = bik.q_lower_limit[0:7]
        self.upper_limit = bik.q_upper_limit[0:7]
        self.joint_centers = bik.q_median_config[0:7]
        self.lower_limit_safe = self.lower_limit + 0.01
        self.upper_limit_safe = self.upper_limit - 0.01
        self.joint_limit_flag = 0
        self.rot_z = bik.np.array([[-1,0,0],[0,-1,0],[0,0,1]])  #Rotation matrix for 180 degrees around the Z-axis
        rot_z90 = bik.np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]) #Rotation matrix for 90 degrees around the Z-axis
        self.rotation_around_z90 = bik.pin.SE3(rot_z90, bik.np.array([0, 0, 0]))  # SE3 is for pose (rotation + translation)
        self.rotation_around_z  = bik.pin.SE3(self.rot_z, bik.np.array([0,0,0]))
        self.set_mode(sharedautonomy_mode, solver_mode)
        if self.solver_mode == "relaxedik":
            Q, tp, tv,ta, Cposes, self.opt_result = bik.run(self.Q, self.target, self.tp, self.tv, self.ta, self.Cposes, XTOL = self.XTOL, MAXEVAL = self.MAXEVAL)
        elif self.solver_mode == "barrierik":
            Q, tp, tv,ta, Cposes, self.opt_result = bik.run_with_cbf(self.Q, self.target, self.tp, self.tv, self.ta, self.Cposes, XTOL = self.XTOL, MAXEVAL = self.MAXEVAL)
        elif self.solver_mode == "barrierik_moving":
            Q, tp, tv,ta, Cposes, self.opt_result = bik.run_with_cbf_moving(self.Q, self.target, self.tp, self.tv, self.ta, self.Cposes, 
                                                                            XTOL = self.XTOL, MAXEVAL = self.MAXEVAL,
                                                                            obstacles_dict = {}, obstacles_vel_dict = {}, dt = 0.01)
        elif self.solver_mode == "collisionik":
            Q, tp, tv, ta, Cposes, self.opt_result  = bik.run_with_collisionik(self.Q, self.target, self.tp, self.tv, self.ta, self.Cposes,XTOL = self.XTOL, MAXEVAL = self.MAXEVAL)

        print("Robot Class initialized")
        self.prev_a = 1
        self.init_sharedautonomy = False
        self.helper_unknown = True
        self.x_helper = bik.pin.SE3(self.rot_z, bik.np.array([0,0,0]))
        self.x_helper_rotated = bik.pin.SE3(self.rot_z, bik.np.array([0,0,0]))
        self.x_helper_rotated_shifted = bik.pin.SE3(self.rot_z, bik.np.array([0,0,0]))
        self.prev_x_helper_pos =  bik.np.array([0,0,0])
        self.prev_x_helper_orient = bik.np.array([0,0,0,1])
        self.prev_x_original_helper_orient = bik.np.array([0,0,0,1])
        self.sa_target = bik.pin.SE3(bik.np.eye(3), bik.np.array([0,0,0]))
        self.target = bik.jaxlie.SE3.from_matrix(bik.pin.SE3(bik.np.eye(3), bik.np.array([0,0,0])).homogeneous)
        self.helper_is_new = True
        self.x_start = bik.pin.SE3(bik.np.eye(3), bik.np.array([0,0,1]))
        self.obstacle_pos = None
        self.obstacle_radius = None
        self.last_time = time.time()

        
        #Get Robot Models from bik_core.py
        self.PANDAmodel = bik.PANDAmodel
        #OBSTACLES
        self.obstacle_data = ObstacleInfos()
        self.obstacles = {}
        
        # Obstacle velocity tracking for moving CBF
        self.obstacle_position_history = {}  # Store position history for velocity calculation
        self.obstacle_velocities = {}        # Computed velocities
        self.max_history_length = 5         # Number of positions to keep for velocity estimation

        
    def set_mode(self, sa_mode, solver_mode):
        if not (sa_mode == "None" or sa_mode == "Arbitration"):
            raise ValueError("Invalid mode")
        self.sharedautonomy_mode = sa_mode
        if solver_mode not in SUPPORTED_SOLVER_MODES:
            raise ValueError("Invalid solver mode")
        self.solver_mode = solver_mode
        print(self.solver_mode, self.sharedautonomy_mode)
    
    def Rquat(self,x, y, z, w): #Converts quaternion to rotation matrix and quaternion coefficient
        q = bik.pin.Quaternion(x, y, z, w)
        q.normalize()
        return q.matrix(), q.coeffs(), q
    
    def RotMat2Euler(self, rot): #Converts rotation matrix to euler angles
        r = R.from_matrix(rot)
        return r, r.as_euler('zyx', degrees=False)

    def track_obstacle_position(self, obstacle_id, position):
        """Track obstacle position for velocity calculation."""
        current_time = time.time()
        
        # Initialize history if not exists
        if obstacle_id not in self.obstacle_position_history:
            self.obstacle_position_history[obstacle_id] = []
            self.obstacle_velocities[obstacle_id] = bik.np.zeros(3)
        
        # Add current position to history
        self.obstacle_position_history[obstacle_id].append({
            'position': position.copy(),
            'timestamp': current_time
        })
        
        # Limit history length
        if len(self.obstacle_position_history[obstacle_id]) > self.max_history_length:
            self.obstacle_position_history[obstacle_id].pop(0)
        
        # Calculate velocity if we have enough history
        history = self.obstacle_position_history[obstacle_id]
        if len(history) >= 2:
            # Use finite difference with the last two positions
            pos_curr = history[-1]['position']
            pos_prev = history[-2]['position']
            time_curr = history[-1]['timestamp']
            time_prev = history[-2]['timestamp']
            
            dt_actual = time_curr - time_prev
            if dt_actual > 1e-6:  # Avoid division by very small numbers
                velocity = (pos_curr - pos_prev) / dt_actual
                self.obstacle_velocities[obstacle_id] = velocity
            else:
                self.obstacle_velocities[obstacle_id] = bik.np.zeros(3)
        else:
            # Not enough history, assume zero velocity
            self.obstacle_velocities[obstacle_id] = bik.np.zeros(3)

    def update_obstacle_velocities(self, dt):
        """Update obstacle velocities for moving CBF - this is now handled in track_obstacle_position."""
        # Velocities are now calculated in real-time during obstacle updates
        # This method is kept for compatibility but doesn't need to do anything
        pass
                
    
    def create_velocity_obstacles_dict(self):
        """Create properly formatted velocity dictionary for CBF moving obstacles."""
        vel_obstacles_dict = {}
        for obstacle_id, obstacle in self.obstacles.items():
            if obstacle_id in self.obstacle_velocities:
                velocity = self.obstacle_velocities[obstacle_id]
                # Format similar to obstacles but with velocity instead of position
                # (a_vel, b_vel, _, _, radius_dummy, _)
                vel_obstacles_dict[obstacle_id] = (
                    velocity,  # a_vel (start point velocity)
                    velocity,  # b_vel (end point velocity - same for sphere obstacles)
                    bik.np.zeros(3),  # dummy
                    bik.np.zeros(3),  # dummy  
                    0.0,  # dummy radius
                    0.0   # dummy
                )
        return vel_obstacles_dict
    
    def alpha_composer(self, arg, mode="Position"):
       # Constants for the computation
       if mode == "Position":
           c = 5.5
           r = 0.2
           h = 1
       if mode == "Orientation":
           c = 5.5
           r = 0.1
           h = 3.6
       if mode == "Shifted":
           c = 4.9
           r = 0.1
           h = 1.4

       # Compute the current value of 'a'
       a = 1 / (1 + bik.np.exp(-c * (arg / r - h)))
       # Save the current 'a' value for the next iteration
       self.prev_a = a

       # Ensure 'a' is within the [0, 1] range
       a = min(1, max(0, a))
       
       return a


        
    def set_target(self, _targetpos, _targetquat):
        self._targetpos = _targetpos
        self._targetquat = _targetquat
        self.targetrotmat, self._targetquatcoeffs, self._targetquat = self.Rquat(_targetquat.x, _targetquat.y, _targetquat.z, _targetquat.w) #Pinocchio takes w x y z, but the coming data from the Unity, requires x y z w
        self.x_target = bik.pin.SE3(self.targetrotmat, bik.np.array([_targetpos.x, _targetpos.y, _targetpos.z]))
        if self.sharedautonomy_mode == "None" or self.sharedautonomy_mode == "Objective":
            self.target = bik.jaxlie.SE3.from_matrix(self.x_target.homogeneous)
        if self.helper_is_new:
            self.x_start = self.x_target
            self.helper_is_new = False

    def set_other(self, _q_other):
        self.q_other = self.q_other.at[0:7].set(_q_other)


    def set_helper(self, _helperpos, _helperquat):  
        if self.init_sharedautonomy == False:
            self.prev_x_helper_pos =  _helperpos
            self.prev_x_helper_orient = _helperquat
            self.init_sharedautonomy = True
            self.helper_is_new = True
        
        if bik.np.array_equal(self.prev_x_helper_pos, _helperpos) and bik.np.array_equal(self.prev_x_original_helper_orient, _helperquat):
            self.helper_is_new = False
        else:
            self.helper_is_new = True

        #Check if helper is at the default position
        if _helperpos.z == -3:
            self.helper_unknown = True
            print("Helper is at the default position")

        else:
            self.helper_unknown = False
            self._helperpos = _helperpos
            self._helperquat = _helperquat
            self._original_helperquat = _helperquat
            self.helperrotmat, _,_ = self.Rquat(_helperquat.w, _helperquat.x, _helperquat.y, _helperquat.z) #This directly comes from ROS
            self.x_helper_rotated = bik.pin.SE3(self.helperrotmat, bik.np.array([_helperpos.x, _helperpos.y, _helperpos.z]))
            self.x_helper_rotated = self.rotate_by_closest_axis(self.x_target, self.x_helper_rotated) #Find closest angle for z-axis
            self._helperquat = bik.pin.Quaternion(self.x_helper_rotated.rotation)
            #self.x_helper_rotated_shifted = self.x_helper_rotated * self.helper_shifting_matrix
            if self.sharedautonomy_mode == "Objective":
                self.helper = bik.jaxlie.SE3.from_matrix(self.x_helper_rotated.homogeneous)
            #NEW
            self.prev_x_helper_pos =  self._helperpos
            self.prev_x_helper_orient = self._helperquat
            self.prev_x_original_helper_orient = self._original_helperquat
            
    def rotate_by_closest_axis(self, target, helper, threshold=0.1, hysteresis=0.05):
        """Align helper rotation with target, allowing only 180° Z-axis rotation"""
        target_quat = bik.pin.Quaternion(target.rotation)
        helper_quat = bik.pin.Quaternion(helper.rotation)
        # Calculate current orientation difference
        current_distance = self.quaternion_distance(target_quat, helper_quat)
        # Try 180° rotated helper
        rotated_helper = bik.pin.SE3(
            helper.rotation @ self.rot_z,
            helper.translation
        )
        rotated_quat = bik.pin.Quaternion(rotated_helper.rotation)

        rotated_distance = self.quaternion_distance(target_quat, rotated_quat)
        
        # Keep current orientation if:
        # 1. Current distance is good enough (within threshold)
        # 2. Difference between current and rotated is small (within hysteresis)
        if current_distance < threshold or abs(current_distance - rotated_distance) < hysteresis:
            #print(self.robot_tag, "Returning Original Helper due to the threshold")
            return helper
        
        # Use rotated version only if significantly better
        if rotated_distance < current_distance:
            #print(self.robot_tag, "Rotating Rotated Helper")
            return rotated_helper
        else:
            #print(self.robot_tag, "Returning Orginal Helper")
            return helper


    def quaternion_distance(self, q1, q2):
        """Calculate angular distance between two quaternions"""
        dot_product = abs(q1.x * q2.x + q1.y * q2.y + q1.z * q2.z + q1.w * q2.w)
        return 2 * bik.np.arccos(min(1.0, max(-1.0,dot_product)))
    # ======================================== ARBITRATION ========================================
    def regular_mechanism(self):
        if self.helper_unknown: #If helper is at the default position, then no arbitration
           #print("Helper is at the default position")
           if (self.alpha  < 0.95): # and self.traj_blending:  #Which means arbitration happened before:
               #print("Current Alpha: ", self.alpha)
               #TODO: Smooth out the alpha value, do not compute new one, interpolate the previous target to current_target
               self.alpha_pos = self.alpha_pos + 0.01
               self.alpha_orient = self.alpha_orient + 0.01
               self.alpha_pos = min(1, max(0, self.alpha_pos))
               self.alpha_orient = min(1, max(0, self.alpha_orient))
               self.alpha = self.alpha_pos #For IK solver
               new_pos = self.alpha_pos * self.x_target.translation + (1-self.alpha_pos) * bik.np.array([self.prev_x_helper_pos.x, self.prev_x_helper_pos.y, self.prev_x_helper_pos.z])
               new_quat = bik.pin.Quaternion.slerp(self.prev_x_helper_orient, self.alpha_orient, self._targetquat)
               self.sa_target = bik.pin.SE3(new_quat.matrix(), new_pos) #New helper position and rotation
               self.target = bik.jaxlie.SE3.from_matrix(self.sa_target.homogeneous) #New Target
           else: 
               #print("No arbitration")
               self.target = bik.jaxlie.SE3.from_matrix(self.x_target.homogeneous)
        else:
           #print("Arbitration")
           #print(self.robot_tag, ":", self.helper_mode)
           self.x_helper = self.x_helper_rotated

           #print("Helper Position: ", self.x_helper.translation)
           #print("Target Position: ", self.x_target.translation)
           diff_original = bik.pin.log(self.x_target.inverse() * self.x_helper_rotated).vector #Difference between target and helper
           norm_diff_pos_original = bik.np.linalg.norm(diff_original[0:3])

           
           alpha_pos_desired = self.alpha_composer(norm_diff_pos_original, mode="Position")
           #alpha_pos_desired = self.new_alpha_composer(self.x_helper_rotated.translation, self.x_start.translation, self.x_target.translation)
           self.alpha_pos = alpha_pos_desired
           #print("Alpha Pos: ", self.alpha_pos)
           #alpha_orient_desired = self.alpha_composer(norm_diff_pos_original, mode="Orientation")
           alpha_orient_desired = alpha_pos_desired
           self.alpha_orient = alpha_orient_desired
           self.alpha = self.alpha_pos #For IK solver

           # +++++++++++ BLENDING +++++++++++++
           new_pos = self.alpha_pos * self.x_target.translation + (1-self.alpha_pos) * self.x_helper.translation #New position of the helper
           if self._helperquat is not None:
               new_quat = bik.pin.Quaternion.slerp(self._helperquat, self.alpha_orient, self._targetquat)
           else:
               new_quat = self._targetquat
           self.sa_target = bik.pin.SE3(new_quat.matrix(), new_pos) #New helper position and rotation
           self.target = bik.jaxlie.SE3.from_matrix(self.sa_target.homogeneous) #New Target
    
            
    def sharedautonomy_arbitration(self):
        '''Arbitration based distance between the target and helper'''
        self.regular_mechanism()

    def set_obstacles(self,data):
        if data == self.obstacle_data:
            return
        for obstacle in data.obstacles:
            obst_name = obstacle.header.frame_id
            if "obstacle" in obst_name:
                
                if obst_name not in self.obstacles:
                    print("Obstacle Name: ", obst_name)
                    obstacle_parameters = self.load_obstacle(obstacle)
                    self.obstacles[obst_name] = obstacle_parameters
                else:
                    obstacle_parameters = self.obstacles[obst_name]
                    new_obstacle_parameters = self.update_obstacle(obstacle, obstacle_parameters)
                    self.obstacles[obst_name] = new_obstacle_parameters
        self.obstacle_data = data

    def load_obstacle(self, obj):
        R = bik.jaxlie.SO3.from_quaternion_xyzw(bik.jnp.array([obj.orientation.x, obj.orientation.y, obj.orientation.z,obj.orientation.w]))
        M = bik.jaxlie.SE3.from_rotation_and_translation(R, bik.jnp.array([obj.position.x, obj.position.y, obj.position.z]))
        scale = bik.jnp.array([obj.scale.x, obj.scale.z, obj.scale.y])
        max_scale_ind = bik.jnp.argmax(scale)
        if max_scale_ind != 2:
            message_str = "Obstacle scale is not in the correct order" + obj.header.frame_id
            rospy.logerr(message_str)
        if obj.type == 0: #Box
            scale = bik.jnp.array([obj.scale.x, obj.scale.z, obj.scale.y])
            capsule_representation = bik.bik_collision.box2capsule(center = bik.jnp.array([obj.position.x, obj.position.y, obj.position.z]), dimensions = scale)
        elif obj.type == 2: #Capsule
            scale = bik.jnp.array([obj.scale.x, obj.scale.z, obj.scale.y*2])  #2* because the capsule is represented by a cylinder with height 2*scale.y
            capsule_representation = bik.bik_collision.box2capsule(center = bik.jnp.array([obj.position.x, obj.position.y, obj.position.z]), dimensions = scale)
        else:
            rospy.logerr("Obstacle type %s not supported", obj.type)
            return None

        #Rotate the capsule to the correct orientation
        T_obs = M.as_matrix()
        a_obs, b_obs = bik.bik_collision.capsule_ab_from_T(T_obs, capsule_representation[3])
        C_obs = (a_obs + b_obs) / 2
        L_obs = bik.np.linalg.norm(a_obs - b_obs)
        R_obs = capsule_representation[4]
        capsule_representation = (a_obs, b_obs, C_obs, L_obs, R_obs, T_obs)
        
        # Track obstacle position for velocity calculation (also for new obstacles)
        self.track_obstacle_position(obj.header.frame_id, bik.np.array([obj.position.x, obj.position.y, obj.position.z]))
        
        return capsule_representation

        
    def update_obstacle(self, obj, obstacle_parameters):
        a_obs, b_obs, C_obs, L_obs, R_obs, T_obs = obstacle_parameters
        R = bik.jaxlie.SO3.from_quaternion_xyzw(bik.jnp.array([obj.orientation.x, obj.orientation.y, obj.orientation.z, obj.orientation.w]))
        M = bik.jaxlie.SE3.from_rotation_and_translation(R, bik.jnp.array([obj.position.x, obj.position.y, obj.position.z])).as_matrix()
        T_obs = M
        a_obs, b_obs = bik.bik_collision.capsule_ab_from_T(T_obs, L_obs)
        C_obs = (a_obs + b_obs) / 2
        L_obs = bik.np.linalg.norm(a_obs - b_obs)
        
        # Track obstacle position for velocity calculation
        if self.solver_mode == "barrierik_moving":
            self.track_obstacle_position(obj.header.frame_id, bik.np.array([obj.position.x, obj.position.y, obj.position.z]))
        
        return (a_obs, b_obs, C_obs, L_obs, R_obs, T_obs)



    def reset(self, q_actual = None):
        if q_actual is not None:
            self.Q = q_actual
        else:
            self.Q = self.init
        self.tp, self.tv, self.ta = bik.reset_derivators(self.Q)
        self.Cposes = bik.calc_Cposes(self.Q)
        #Reset obstacle velocities
        self.obstacle_position_history = {}
        self.obstacle_velocities = {}
        

    def run(self, q_actual = None):
        Q_guess = self.Q
        if self.last_step_success == False:
            #Perturb the guess, little bit
            Q_guess = self.Q + bik.np.random.uniform(-0.01, 0.01, size=(7,))
            Q_guess = bik.np.clip(Q_guess, self.lower_limit_safe, self.upper_limit_safe)
        try: 
            current_time = time.time()
            dt = current_time - self.last_time

            if self.solver_mode == "relaxedik":
                Q_new, self.tp, self.tv, self.ta, self.Cposes, self.opt_result  = bik.run(Q_guess, self.target, self.tp, self.tv, self.ta, self.Cposes,
                                                                                                        XTOL = self.XTOL, MAXEVAL = self.MAXEVAL)
            elif self.solver_mode == "barrierik":
                Q_new, self.tp, self.tv, self.ta, self.Cposes, self.opt_result  = bik.run_with_cbf(Q_guess, self.target, self.tp, self.tv, self.ta, self.Cposes,
                                                                                                            XTOL = self.XTOL, MAXEVAL = self.MAXEVAL,
                                                                                                        obstacles_dict = self.obstacles, dt = dt)
            elif self.solver_mode == "barrierik_moving":
                # Update obstacle velocities from position history
                self.update_obstacle_velocities(dt)
                # Create properly formatted velocity dictionary
                vel_obstacles_dict = self.create_velocity_obstacles_dict()
                Q_new, self.tp, self.tv, self.ta, self.Cposes, self.opt_result  = bik.run_with_cbf_moving(Q_guess, self.target, self.tp, self.tv, self.ta, self.Cposes,
                                                                                                            XTOL = self.XTOL, MAXEVAL = self.MAXEVAL,
                                                                                                        obstacles_dict = self.obstacles, 
                                                                                                        obstacles_vel_dict = vel_obstacles_dict, dt = dt)
            elif self.solver_mode == "collisionik":
                Q_new, self.tp, self.tv, self.ta, self.Cposes, self.opt_result  = bik.run_with_collisionik(Q_guess, self.target, self.tp, self.tv, self.ta, self.Cposes,
                    XTOL = self.XTOL, MAXEVAL = self.MAXEVAL,
                    obstacles_dict = self.obstacles)
            #Joint Velocity Limit Check
            q_dot = Q_new - self.Q
            q_dot = bik.np.clip(q_dot, -5*dt, 5*dt) #Joint velocity limits
            Q_new = self.Q + q_dot
            self.Q = Q_new
            self.last_time = current_time
            self.last_step_success = True
        except Exception as e:
            self.last_step_success = False
            print("Error in bik", e)
        return self.Q
    
