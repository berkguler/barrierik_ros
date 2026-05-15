#! /usr/bin/env python3
import numpy as np
from os.path import dirname, join, abspath
import sys
import math
import yaml
import os

from scipy.spatial.transform import Rotation as R
import time
import bik_pkg.parse_urdf as parse_urdf
from bik_pkg.parse_urdf import jax, jacfwd, config, grad, jit, vmap, jnp, rbda, pin,  js
import bik_pkg.bik_collision as bik_collision


from functools import partial
from flax import linen as nn
import pickle
import jaxlie
import nlopt
import rospkg
from jax import debug

import dpax
from dpax.endpoints import proximity



rospack = rospkg.RosPack()
package_path = rospack.get_path('bik_pkg')
# %%
PANDAmodel, _, _, _, _, _, robotmodel, robotdata, list_panda_capsules = parse_urdf.init(yaml_file = package_path + "/configs/franka_description.yaml", open_viewer = False, disable_pin_models = False, disable_viewer = True)
# Print model quantities.


# %%
forward_kinematics = jit(rbda.forward_kinematics)
forward_kinematics_all = jit(rbda.forward_kinematics_model)
jacobian = jit(rbda.jacobian)
forward_kinematix = jax.jit(lambda x: forward_kinematics(robotmodel, link_index=8, base_position=jnp.array([0.0,0.0,0.0]),base_quaternion=jnp.array([1.0,0.0,0.0,0.0]), joint_positions=x))
forward_kinematix_all = jax.jit(lambda Q, Pos: forward_kinematics_all(robotmodel, base_position=Pos,base_quaternion=jnp.array([1.0,0.0,0.0,0.0]), joint_positions=Q))
jacobix = jax.jit(lambda x: jacobian(robotmodel, link_index=8, joint_positions=x)[:,6:6+7])
# %%
q_lower_limit = PANDAmodel.lowerPositionLimit
q_upper_limit = PANDAmodel.upperPositionLimit
q_median_config = (q_lower_limit + q_upper_limit) / 2.0
alpha = 100.0
beta=100.0

list_panda_capsules = list_panda_capsules
##remove first_2 capsules
#list_panda_capsules = list_panda_capsules[2:]
N_panda = len(list_panda_capsules)
a_list, b_list, C_list, L_list, R_list, T_list = zip(*list_panda_capsules)
a_panda = jnp.stack(a_list)
b_panda = jnp.stack(b_list)
C_panda = jnp.stack(C_list)
L_panda = jnp.stack(L_list)
R_panda = jnp.stack(R_list)
T_panda = jnp.stack(T_list)


class FivePointStencil2: #reset -> shift -> derive
    def reset(self, input) -> jnp.ndarray:
        return jnp.ones(5) * input
    @partial(jax.jit, static_argnums=(0,))
    def derive(self, x: jnp.ndarray) -> float:
        #return (x[0] - 8*x[1] + 8*x[3] - x[4]) / 12
        #return x[4] - x[3]
        return (x[4] - x[3])
    @partial(jax.jit, static_argnums=(0,))
    def shift(self, x: jnp.ndarray, input: float) -> jnp.ndarray:
        return jnp.array([x[1], x[2], x[3], x[4], input])

Stencil = FivePointStencil2()

@jax.jit
def reset_derivators(_init_poses: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    theta_poses = jnp.zeros((7,5))
    theta_vels = jnp.zeros((7,5))
    theta_accels = jnp.zeros((7,5))
    for i in range(7):
        thetapos = Stencil.reset(_init_poses[i])
        thetavel = Stencil.reset(thetapos)
        thetaaccel = Stencil.reset(thetavel)
        theta_poses = theta_poses.at[i].set(thetapos)
        theta_vels = theta_vels.at[i].set(thetavel)
        theta_accels = theta_accels.at[i].set(thetaaccel)
    return (theta_poses, theta_vels, theta_accels)

@jax.jit
def calc_Cposes(_init_poses: jnp.ndarray) -> jnp.ndarray:
    Cposes = jnp.zeros(3)
    theta = jnp.append(_init_poses, jnp.zeros(1))
    FK = forward_kinematix(theta)
    for i in range(3):
        Cposes = Cposes.at[i].set(FK[i,3])
    return Cposes


@jax.jit
def call_derivators(theta_poses: jnp.ndarray, theta_vels: jnp.ndarray, theta_accels: jnp.ndarray, _input: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    theta_dots = jnp.zeros(7)
    theta_ddots = jnp.zeros(7)
    theta_dddots = jnp.zeros(7)
    theta_newposes = jnp.zeros((7,5))
    theta_newvels = jnp.zeros((7,5))
    theta_newaccels = jnp.zeros((7,5))
    for i in range(7):
        theta_newposes = Stencil.shift(theta_poses[i], _input[i])
        theta_dot = Stencil.derive(theta_newposes)
        theta_newvels = Stencil.shift(theta_vels[i], theta_dot)
        theta_ddot = Stencil.derive(theta_newvels)
        theta_newaccels = Stencil.shift(theta_accels[i], theta_ddot)
        theta_dddot = Stencil.derive(theta_newaccels)

        theta_dots = theta_dots.at[i].set(theta_dot)
        theta_ddots = theta_ddots.at[i].set(theta_ddot)
        theta_dddots = theta_dddots.at[i].set(theta_dddot)
        theta_poses = theta_poses.at[i].set(theta_newposes)
        theta_vels = theta_vels.at[i].set(theta_newvels)
        theta_accels = theta_accels.at[i].set(theta_newaccels)
    return (theta_dots, theta_ddots, theta_dddots), (theta_poses, theta_vels, theta_accels)


@jax.jit
def call_derivator_for_pose(theta_poses: jnp.ndarray, _input: jnp.ndarray) -> jnp.ndarray:
    theta_newposes = jnp.zeros((7,5))
    for i in range(7):
        theta_newposes = Stencil.shift(theta_poses[i], _input[i])
        theta_dot = Stencil.derive(theta_newposes)
    return theta_dot

@jax.custom_jvp
def mean(x, axis=None, keepdims=False):
    return jnp.mean(x, axis=axis, keepdims=keepdims)


@jax.jit
def call_derivator_for_1pose(theta_poses: jnp.ndarray, _input: float) -> float:
    theta_newposes = Stencil.shift(theta_poses, _input)
    theta_dot = Stencil.derive(theta_newposes)
    return theta_dot


# ## Objectives
@jax.jit
def x_p(Mtool_pos,Mtarget_pos): # End-Effector Position Matching Error
    pos_error = jnp.linalg.norm(Mtool_pos - Mtarget_pos)
    return pos_error

@jax.jit
def x_o(Mtool_rot,Mtarget_rot): # End-Effector Orientation Matching Error
    actual_quat = Mtool_rot
    target_quat = Mtarget_rot
    target2_quat = -target_quat
    ori_error = jnp.linalg.norm(target_quat - actual_quat)
    ori_error2 = jnp.linalg.norm(target2_quat - actual_quat)
    return jnp.minimum(ori_error, ori_error2)

@jax.jit
def x_p_h(Mtool_pos,Mhelper_pos): # End-Effector Position Matching Error
    pos_error = jnp.linalg.norm(Mtool_pos - Mhelper_pos)
    return pos_error

@jax.jit
def x_o_h(Mtool_rot,Mhelper_rot): # End-Effector Orientation Matching Error
    actual_quat = Mtool_rot
    helper_quat = Mhelper_rot
    helper2_quat = -helper_quat
    ori_error = jnp.linalg.norm(helper_quat - actual_quat)
    ori_error2 = jnp.linalg.norm(helper2_quat - actual_quat)
    return jnp.minimum(ori_error, ori_error2)


@jax.jit
def x_v(theta_dot): # Smoothness of Joint Velocities
    return derivative_okay_norm(theta_dot)

@jax.jit
def x_a(theta_ddot): # Smoothness of Joint Accelerations
    return derivative_okay_norm(theta_ddot)

@jax.jit
def x_j(theta_dddot): # Smoothness of Joint Jerks
    return derivative_okay_norm(theta_dddot)

@jax.jit
def x_l(theta): #Divergence from the joint medians
    theta = theta[0:7]
    div_median = theta - q_median_config[0:7]
    result = jnp.zeros(7)
    for i in range(7):
        result = result.at[i].set((q_median_config[i] - theta[i]) / (q_upper_limit[i] - q_lower_limit[i])**2)
    return derivative_okay_norm(result)
    

@jax.jit
def derivative_okay_norm(X):
    #https://github.com/google/jax/issues/3058
    is_zero = jnp.allclose(X, 0.)
    d = jnp.where(is_zero, jnp.ones_like(X), X)  # replace d with ones if is_zero
    l = jnp.linalg.norm(d)
    l = jnp.where(is_zero, 0., l)  # replace norm with zero if is_zero
    l = jnp.max(jnp.array([l, 0.00001]))
    return l

@jax.jit
def x_e(Mtoolposes, Cposes): # Cartesian Velocity
    V = Mtoolposes - Cposes
    return derivative_okay_norm(V)

class Deep(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(60)(x)
        x = nn.relu(x)
        x = nn.Dense(100)(x)
        x = nn.relu(x)
        x = nn.Dense(120)(x)
        x = nn.relu(x)
        x = nn.Dense(100)(x)
        x = nn.relu(x)
        x = nn.Dense(60)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        x = nn.relu(x)
        return x
        
model = Deep()
with open(package_path + "/collision_params.pkl", "rb") as f:
    params = pickle.load(f)
    
@jax.jit
def x_c(theta):
     theta = jnp.append(theta, jnp.zeros(1))
     return model.apply(params, theta)[0]


@jax.jit
def fk_to_pos_rot(value_fk):
    SE3 = jaxlie.SE3.from_matrix(value_fk)
    pos = SE3.translation()
    rot = SE3.rotation().wxyz
    return pos, rot

@partial(jax.jit, static_argnums=(0,)) #Groove Loss
def make_groove_objective_function(func, n,s,c,r, g,args): 
    x_val = func(*args)
    #print(x_val)
    return (-1)**n * jnp.exp((-(x_val-s)**2)/(2*c**2)) + r * jnp.power(x_val-s,g)

@partial(jax.jit, static_argnums=(0,)) #Swamp Loss
def make_swamp_objective_function(func, n,s,c,r, args): 
    x_val = func(*args)
    #print(x_val)
    return swamp_loss(x_val, s-c, s+c, 0.0, r, 4)

base_pos = jnp.zeros(3)

@partial(jax.jit, static_argnums=(1,))
def make_cost_function(theta_inp, FK_jax, Mtarget, theta_poses, theta_vels, theta_accels, Cposes):
    #Set last 3 values to 0
    theta = jnp.append(theta_inp, jnp.zeros(1))
    wp = 100
    wo = 80
    wv = 0.02
    wa = 0.2
    wj = 0.4
    we = 1.0
    wc = 1.0

    FK = FK_jax(theta, base_pos)
    FK_ee = FK[8]
    Mtool_pos, Mtool_rot = fk_to_pos_rot(FK_ee)
    Mtarget_pos = Mtarget.translation()
    Mtarget_rot = Mtarget.rotation().wxyz
    (theta_dots, theta_ddots, theta_dddots), (theta_new_poses, theta_new_vels, theta_new_accels) = call_derivators(theta_poses, theta_vels, theta_accels, theta)
    
    f_p = wp*make_groove_objective_function(func=x_p, n=1, s=0, r=5,c = 0.2, g = 1, args=(Mtool_pos,Mtarget_pos))
    f_o = wo*make_groove_objective_function(func=x_o, n=1, s=0, r=5,c = 0.2, g = 1, args=(Mtool_rot,Mtarget_rot))
    f_v = wv*make_groove_objective_function(func=x_v, n=1, s=0, r=5,c = 0.2, g = 1, args=(theta_dots,))
    f_a = wa*make_groove_objective_function(func=x_a, n=1, s=0, r=5,c = 0.2, g = 1, args=(theta_ddots,))
    f_j = wj*make_groove_objective_function(func=x_j, n=1, s=0, r=5,c = 0.2, g = 1, args=(theta_dddots,))
    f_e = we*make_groove_objective_function(func=x_e, n=1, s=0, r=5,c = 0.2, g = 1, args=(Mtool_pos, Cposes))
    f_c = wc*make_groove_objective_function(func=x_c, n=0, s=0, r=0.002,c = 2.1, g = 2, args=(theta,))

    f = f_p
    f += f_o
    f += f_v
    f += f_a
    f += f_j
    f += f_e
    f += f_c
    return f


@jax.jit
def swamp_loss(x_val: float, l_bound: float, u_bound: float, f1: float, f2: float, p1: int) -> float:
    x = (2.0 * x_val - l_bound - u_bound) / (u_bound - l_bound)
    b = jnp.power(-1.0 / jnp.log(0.05),1.0 / p1)
    return (f1 + f2 * jnp.power(x,2)) *  (1.0 - jnp.exp((- (x/b)**p1))) - 1.0




@jax.jit
def c_ks(theta_inp): #Manipulability since nlopt expects inequality constraints to be of the form h(x) <= 0.
    theta = jnp.append(theta_inp, jnp.zeros(1))
    J =  jacobix(theta)
    
    _, S, _ = jnp.linalg.svd(J, full_matrices=False)
    #c_mean = 0.053086915331318645
    #c_std = 0.03594815648137724
    #c = S[-1]/S[0]
    #b = 1.4140
    #return (c_mean - b*c_std) - c
    ratio = jnp.divide(S[-1], jnp.maximum(S[-1], 1e-10))
    eigen_threshold = 0.01
    #smallest_eigenvalue S[-1] > eigen_threshold
    return  ratio - eigen_threshold


@jax.jit
def class_K_function(h, gamma=100.0):
    return gamma * h

@jax.jit
def FK_for_collisions_v2(joint_pos, T_arr, L_arr, R_arr):
    FK_all = forward_kinematix_all(joint_pos, jnp.zeros(3))  # Shape (N,4,4)
    #ignore the last 2 rows

    # Batched matmul: FK_all[i] @ T_arr[i]
    new_T = jax.vmap(jnp.matmul)(FK_all, T_arr)  # Shape (N,4,4)
    # Batched computation of a, b
    new_a, new_b = jax.vmap(bik_collision.capsule_ab_from_T)(new_T, L_arr)
    new_c = (new_a + new_b) / 2.0
    
    return new_a, new_b, new_c, L_arr, R_arr, new_T

#@jax.jit
#def collision_check(a,b,C,L,R,T, a_obs,b_obs,C_obs,L_obs,R_obs, joint_pos): #7e-5
#    # Compute the forward kinematics for the given joint positions
#    #print(a.shape)
#    new_a, new_b, new_c, new_L, new_R, new_T = FK_for_collisions_v2(
#        joint_pos, T, L, R
#    )
#    # Compute the collision check using the batch proximity function
#    #print("new_a", new_a, "new_b", new_b, "new_R", new_R)
#    #print("R_obs", R_obs, "a_obs", a_obs, "b_obs", b_obs)
#    phi = batch_proximity(new_R, new_a, new_b, R_obs, a_obs, b_obs)
#    # Check for collisions
#    #collision = jnp.any(phi < 0)
#    # Return the collision status and the updated positions
#    return phi
#
#collision_check_dot = jax.jit(jax.jacfwd(collision_check, argnums=0))
safety_margin = 0.010

#@jax.jit
#def c_cbf_obstacle(theta_inp: jnp.ndarray, a_obs, b_obs, C_obs, L_obs, R_obs):
#    theta_augmented = jnp.append(theta_inp, jnp.zeros(1))
#    # Unpack the obstacle parameters
#    #a_obs, b_obs, C_obs, L_obs, R_obs, T_obs = obstacle_parameters
#    # Compute the FK for the given joint positions
#    phi = collision_check(
#        a_panda, b_panda, C_panda, L_panda, R_panda, T_panda,
#        a_obs, b_obs, C_obs, L_obs, R_obs, 
#        theta_augmented)
#    h = phi - safety_margin # 1xN  grad_h robot_cp - obs_cp / h
#    #print("phi", phi)
#    grad_h = collision_check_dot(a_panda, b_panda, C_panda, L_panda, R_panda, T_panda,
#        a_obs, b_obs, C_obs, L_obs, R_obs, 
#        theta_augmented) # 1xN
#    J = jacobix(theta_augmented)[:3, :]  # Must be JAX-compatible 3xN
#    h_dot = jnp.dot(grad_h, jnp.dot(J, theta_inp)) # 1xN
#    cbf_value = -(h_dot + class_K_function(h))
#    #print("cbf_value", cbf_value)
#    max_cbf_value = jnp.max(cbf_value)
#    #print("max_cbf_value", max_cbf_value)
#    return max_cbf_value 

batch_proximity = jax.vmap(proximity, in_axes = (0,0,0,0,0,0))

@jax.jit
def collision_proximity_check_v2(a_panda, b_panda, R_panda, a_obstacle, b_obstacle, R_obstacle):
    result = batch_proximity(R_panda,a_panda,b_panda, R_obstacle,a_obstacle,b_obstacle)
    return result

@jax.jit
def make_obs_compatible(_a_obs, _b_obs,   _R_obs):
    a_obs_extended = jnp.repeat(_a_obs[None, :], 9, axis=0)
    b_obs_extended = jnp.repeat(_b_obs[None, :], 9, axis=0)
    R_obs_extended = jnp.repeat(_R_obs[None], 9, axis=0)
    return a_obs_extended, b_obs_extended,  R_obs_extended

@jax.jit
def combine_obs_compatible(obstacles_dict):
    """
    Convert a dictionary of obstacles to compatible arrays for CBF computation
    
    Args:
        obstacles_dict: Dictionary of obstacles with their parameters
        
    Returns:
        a_obstacles: Array of obstacle start points
        b_obstacles: Array of obstacle end points
        R_obstacles: Array of obstacle radii
    """
    N_obstacles = len(obstacles_dict)
    
    # Initialize arrays with proper size
    a_obstacles = jnp.zeros((N_obstacles, 3))
    b_obstacles = jnp.zeros((N_obstacles, 3))
    R_obstacles = jnp.zeros((N_obstacles))
    
    # Process obstacles one by one
    j = 0
    for _, obs_parameters in obstacles_dict.items():
        a_obs, b_obs, _, _, R_obs, _ = obs_parameters
        a_obstacles = a_obstacles.at[j].set(a_obs)
        b_obstacles = b_obstacles.at[j].set(b_obs)
        R_obstacles = R_obstacles.at[j].set(R_obs)        
        j += 1
        
    return a_obstacles, b_obstacles, R_obstacles

temperature = 10.0
@jax.jit
def cbf_extended_function(theta_inp, _a_obs,_b_obs, _R_obs):
    theta_augmented = jnp.append(theta_inp, jnp.zeros(1))
    N_obstacle = _a_obs.shape[0]
    nA, nB, nC, nL, nR,nT = FK_for_collisions_v2(theta_augmented, T_panda, L_panda, R_panda)
    J = jacobix(theta_augmented)[:3, :]  # Must be JAX-compatible
    vel = jnp.dot(J, theta_inp)
    cbf_values_min = jnp.zeros((N_obstacle,))
    for i in range(N_obstacle):
        _a_obs_extended, _b_obs_extended, _R_obs_extended = make_obs_compatible(
            _a_obs[i], _b_obs[i], _R_obs[i]
        )
        phi = collision_proximity_check_v2(
            nA, nB, nR,
            _a_obs_extended, _b_obs_extended, _R_obs_extended
        )
        h = phi - safety_margin
        min_idx = jnp.argmin(phi)
        @jax.jit
        def critical_proximity(a_p) -> float:
            # Replace just the critical robot position
            new_a_panda = nA.at[min_idx].set(a_p)
            prox =  batch_proximity(nR, new_a_panda, nB, _R_obs_extended, _a_obs_extended, _b_obs_extended)
            minimum_proximity = prox[min_idx]
            #be sure that the minimum_proximity is scalar
            return minimum_proximity
        gradient = jax.grad(critical_proximity)(nA[min_idx])
        
        # Create the sparse gradient matrix (most entries are zero)
        # Only the row corresponding to the minimum index has non-zero values
        grad_h = jnp.zeros_like(a_panda)
        grad_h = grad_h.at[min_idx].set(gradient)
        h_dot = jnp.dot(grad_h, vel) #delta_h(x)*vel >=  -K(h) , so for nlopt the inequality convention is ch(x) <= 0
        cbf_value = -(h_dot + class_K_function(h))
        cbf_values_min = cbf_values_min.at[i].set(cbf_value[min_idx])
    #max_cbf_values_min = jnp.max(cbf_values_min)
    #return max_cbf_values_min

    # Using LogSumExp instead of max
    # First find the max value for numerical stability
    max_val = jnp.max(cbf_values_min)
    # Apply logsumexp with temperature scaling (higher temperature = closer to true max)
    logsumexp_val = max_val + jnp.log(jnp.sum(jnp.exp(temperature * (cbf_values_min - max_val)))) / temperature
    
    return logsumexp_val


@jax.jit
def cbf_extended_vectorized(theta_inp, _a_obs, _b_obs, _R_obs, temperature=10.0):
    """
    Vectorized CBF computation for multiple obstacles and robot parts
    
    Args:
        theta_inp: Joint positions (7-DOF)
        _a_obs: Array of obstacle start points (N_obstacles, 3)
        _b_obs: Array of obstacle end points (N_obstacles, 3)
        _R_obs: Array of obstacle radii (N_obstacles,)
        temperature: Temperature parameter for LogSumExp
        
    Returns:
        LogSumExp of all CBF values
    """
    theta_augmented = jnp.append(theta_inp, jnp.zeros(1))
    N_obstacles = _a_obs.shape[0]
    
    # Get robot capsule positions and parameters
    robot_a, robot_b, _, _, robot_R, _ = FK_for_collisions_v2(
        theta_augmented, T_panda, L_panda, R_panda
    )
    
    # Jacobian for velocity calculation
    J = jacobix(theta_augmented)[:3, :]
    vel = jnp.dot(J, theta_inp)
    
    # Initialize array for CBF values
    cbf_values = jnp.zeros(N_obstacles)
    
    # Define a function to process a single obstacle
    def process_obstacle(i, cbf_values):
        # Extend obstacle parameters to match robot parts
        a_obs_ext, b_obs_ext, R_obs_ext = make_obs_compatible(
            _a_obs[i], _b_obs[i], _R_obs[i]
        )
        
        # Calculate proximity for all robot parts
        phi = collision_proximity_check_v2(
            robot_a, robot_b, robot_R, 
            a_obs_ext, b_obs_ext, R_obs_ext
        )
        
        # Find minimum proximity (most critical robot part)
        h = phi - safety_margin
        min_idx = jnp.argmin(phi)
        
        # Calculate gradient at the most critical point
        def critical_proximity(a_p):
            new_a = robot_a.at[min_idx].set(a_p)
            prox = batch_proximity(robot_R, new_a, robot_b, 
                                  R_obs_ext, a_obs_ext, b_obs_ext)
            return prox[min_idx]
            
        gradient = jax.grad(critical_proximity)(robot_a[min_idx])
        
        # Create sparse gradient (only non-zero at critical point)
        grad_h = jnp.zeros_like(robot_a)
        grad_h = grad_h.at[min_idx].set(gradient)
        
        # CBF condition: ḣ + class_K(h) ≥ 0
        h_dot = jnp.dot(grad_h[min_idx], vel)
        cbf_value = -(h_dot + class_K_function(h[min_idx]))
        
        # Store result for this obstacle
        return cbf_values.at[i].set(cbf_value)
    
    # Process all obstacles using a scan (more efficient than a loop)
    cbf_values = jax.lax.fori_loop(0, N_obstacles, process_obstacle, cbf_values)
    
    # Apply LogSumExp to combine all CBF values
    max_val = jnp.max(cbf_values)
    logsumexp_val = max_val + jnp.log(jnp.sum(
        jnp.exp(temperature * (cbf_values - max_val))
    )) / temperature
    
    return logsumexp_val

jac_c_cbf = jax.jit(jax.jacfwd(cbf_extended_vectorized, argnums=0))


vel_threshold = 0.03
@jax.jit
def velocity_penalty_term(theta, theta_poses):  # Default dt=0.01s (100Hz)
    # Maximum joint velocity in rad/s (from robot specs)
    #max_joint_vel_rad_per_sec = 3.0
    
    # Compute threshold based on timestep
    ##vel_threshold = max_joint_vel_rad_per_sec * dt
    
    penalty = 0.0
    scale = 100.0
    def vel_penalty_func(theta_poses, theta):
        vel = call_derivator_for_1pose(theta_poses, theta)
        excess = jnp.maximum(0.0, jnp.abs(vel) - vel_threshold)
        return scale * jnp.square(excess)
    
    # Compute penalty for each joint
    # Note: theta_poses is a 7x5 matrix, where each row corresponds to a joint
    # and each column corresponds to a time step
    # We need to compute the penalty for each joint separately
    # and sum them up and use vmap to vectorize the operation
    penalty = jax.vmap(vel_penalty_func, in_axes=(0, 0))(theta_poses, theta)
    result = jnp.sum(penalty)

    #for i in range(7):
    #    vel = call_derivator_for_1pose(theta_poses[i], theta[i])
    #    excess = jnp.maximum(0.0, jnp.abs(vel) - vel_threshold)
    #    penalty += 100.0 * jnp.square(excess)  # Quadratic penalty
    return result

@partial(jax.jit, static_argnums=(1,))
def make_cost_function_with_soft_constraints(theta_inp :jnp.ndarray, FK_jax: callable, 
                                             Mtarget: jnp.ndarray, theta_poses: jnp.ndarray, theta_vels: jnp.ndarray, theta_accels: jnp.ndarray, 
                                             Cposes: jnp.ndarray) -> float:
    base_cost = make_cost_function(theta_inp, FK_jax, Mtarget, theta_poses, theta_vels, theta_accels, Cposes)
    vel_penalty = velocity_penalty_term(theta_inp, theta_poses)
    return base_cost + vel_penalty

# ### Optimization
jac_nlopt = jax.jit(jax.jacfwd(make_cost_function), static_argnums=(1))

jac_c_ks = jax.jit(jax.jacfwd(c_ks))
this_nlopt = partial(make_cost_function, FK_jax=forward_kinematix_all)#, Mtarget=Mtarget_SE3, theta_poses = theta_poses, theta_vels = theta_vels, theta_accels = theta_accels)

@jax.jit
def nlopt_cost_function(theta_inp, Mtarget_SE3, theta_poses, theta_vels, theta_accels, Cposes):
    return this_nlopt(theta_inp, Mtarget=Mtarget_SE3, theta_poses = theta_poses, theta_vels = theta_vels, theta_accels = theta_accels, Cposes = Cposes)

def nlopt_obj(theta, grad, forward_kinematix_all, Mtarget_SE3, theta_poses, theta_vels, theta_accels, Cposes):
    if grad.size > 0:
        _grad = jac_nlopt(theta, forward_kinematix_all, Mtarget_SE3, theta_poses, theta_vels, theta_accels, Cposes)
        grad[:] = np.array(_grad)
    result_theta = nlopt_cost_function(theta, Mtarget_SE3, theta_poses, theta_vels, theta_accels, Cposes)
    result_theta.block_until_ready()
    return np.array(result_theta)*1.0

def nlopt_c_ks(theta_inp, grad):
    if grad.size > 0:
        _grad = jac_c_ks(theta_inp)
        grad[:] = np.array(_grad)
    return np.array(c_ks(theta_inp))*1.0

def nlopt_c_cbf(theta_inp, grad, _a_obs,_b_obs, _R_obs):
    #def cbf_extended_function(_initial_joint_positions, _T_panda, _L_panda,_R_panda,
    #              _a_obs,_b_obs, _R_obs):
    if grad.size > 0:
        _grad = jac_c_cbf(theta_inp, _a_obs,_b_obs, _R_obs)
        grad[:] = np.array(_grad)
    return np.array(cbf_extended_vectorized(theta_inp, _a_obs,_b_obs, _R_obs))*1.0


#### INIT ####

rot2 = np.array(R.from_euler('x', 180, degrees=True).as_matrix())
Mtarget = pin.SE3(rot2, np.array([0.088, 0., 0.926]))
_Mtarget_SE3 = jaxlie.SE3.from_matrix(Mtarget.homogeneous)
Mhelper = pin.SE3(rot2, np.array([0.0, 0.0, 0.0]))
_Mhelper_SE3 = jaxlie.SE3.from_matrix(Mhelper.homogeneous)
_theta_0 = robotdata.joint_positions()
_theta_0 = _theta_0.at[3].set(-0.1)
(_theta_poses, _theta_vels, _theta_accels) = reset_derivators(_theta_0)
Cposes = calc_Cposes(_theta_0[0:7])

def run(theta, Mtarget_human_SE3, theta_poses, theta_vels, theta_accels,Cposes, XTOL=1e-6, MAXEVAL = 100, dt = 0.01):
    OBJ_wrapper = lambda x, grad: nlopt_obj(x, grad, forward_kinematix_all, Mtarget_human_SE3, theta_poses, theta_vels, theta_accels, Cposes)

    CKS_wrapper = lambda x, grad: nlopt_c_ks(x, grad)
    opt = nlopt.opt(nlopt.LD_SLSQP, 7)
    opt.set_min_objective(OBJ_wrapper)
    opt.set_lower_bounds(q_lower_limit[0:7])
    opt.set_upper_bounds(q_upper_limit[0:7])

    opt.add_inequality_constraint(CKS_wrapper,1e-4)
    opt.set_xtol_rel(XTOL)
    opt.set_maxeval(MAXEVAL)
    xt = opt.optimize(theta)
    result = opt.last_optimize_result()
    (theta_dots, theta_ddots, theta_dddots), (theta_new_poses, theta_new_vels, theta_new_accels) = call_derivators(theta_poses, theta_vels, theta_accels, xt)
    Cposes = calc_Cposes(xt)
    return xt, theta_new_poses, theta_new_vels, theta_new_accels,Cposes, result


def run_with_cbf(theta, Mtarget_human_SE3, theta_poses, theta_vels, theta_accels, Cposes, 
                XTOL=1e-6, MAXEVAL=100, obstacles_dict=None, dt=0.01):
    """Run IK with Control Barrier Function constraints for obstacle avoidance
    
    Args:
        theta: Initial joint positions
        Mtarget_human_SE3: Target pose
        theta_poses, theta_vels, theta_accels: State history
        Cposes: Current end-effector position
        XTOL: Tolerance for optimization
        MAXEVAL: Maximum evaluations
        obstacles_dict: Dictionary of obstacles
        distances_prev: Previous distance matrix for velocity estimation
        
    Returns:
        Updated joint positions and states
    """
    if obstacles_dict is None or len(obstacles_dict) == 0:
        # If no obstacles, use regular IK
        return run(theta, Mtarget_human_SE3, theta_poses, theta_vels, theta_accels, Cposes, XTOL, MAXEVAL)
    
    # Objective function wrapper
    OBJ_wrapper = lambda x, grad: nlopt_obj(x, grad, forward_kinematix_all, 
                                           Mtarget_human_SE3, theta_poses, 
                                           theta_vels, theta_accels, Cposes)
    
    # Manipulability constraint wrapper
    CKS_wrapper = lambda x, grad: nlopt_c_ks(x, grad)
    
    # Setup NLopt optimizer
    opt = nlopt.opt(nlopt.LD_SLSQP, 7)
    opt.set_min_objective(OBJ_wrapper)
    opt.set_lower_bounds(q_lower_limit[0:7])
    opt.set_upper_bounds(q_upper_limit[0:7])
    
    # Add manipulability constraint
    opt.add_inequality_constraint(CKS_wrapper, 1e-4)
    
    # Add CBF constraints for each relevant link-obstacle pair
    # We don't need to add constraints for every link-obstacle pair
    # Just the ones that could potentially collide
    copied_obstacles = obstacles_dict.copy()
    a_obstacles, b_obstacles, R_obstacles = combine_obs_compatible(copied_obstacles)
    cbf_wrapper = lambda x, grad: nlopt_c_cbf(x, grad, a_obstacles.copy(), b_obstacles.copy(), R_obstacles.copy())
    # Add CBF constraints
    opt.add_inequality_constraint(cbf_wrapper, 1e-2)
    
    # Run optimization
    opt.set_xtol_rel(XTOL)
    opt.set_maxeval(MAXEVAL)
    xt = opt.optimize(theta)
    result = opt.last_optimize_result()
    
    # Update states
    (theta_dots, theta_ddots, theta_dddots), (theta_new_poses, theta_new_vels, theta_new_accels) = \
        call_derivators(theta_poses, theta_vels, theta_accels, xt)
    Cposes = calc_Cposes(xt)
    
    # Return updated state
    return xt, theta_new_poses, theta_new_vels, theta_new_accels, Cposes, result
'''
### Lets go with augmented lagrangian
from jaxopt import ScipyBoundedMinimize

def penalty_term(theta, theta_poses):
    pen = 0.0
    # List of velocity constraints; each returns h(theta, theta_poses) <= 0
    for con in [c_v_0, c_v_1, c_v_2, c_v_3, c_v_4, c_v_5, c_v_6]:
        pen += jnp.square(jnp.maximum(0.0, con(theta, theta_poses)))
    pen += jnp.square(jnp.maximum(0.0, c_ks(theta)))  # manipulability constraint
    return pen

@jax.jit
def cost_fn(theta, theta_poses, theta_vels, theta_accels, Cposes, Mtarget_SE3):
    return make_cost_function(theta, FK_jax=forward_kinematix_all,
                              Mtarget=Mtarget_SE3,
                              theta_poses=theta_poses,
                              theta_vels=theta_vels,
                              theta_accels=theta_accels,
                              Cposes=Cposes)
@jax.jit
def cost_with_penalty(theta,args):
    theta_poses, theta_vels, theta_accels, Cposes, Mtarget_SE3 = args
    penalty_weight = 1e-2
    base_cost = cost_fn(theta, theta_poses, theta_vels, theta_accels, Cposes, Mtarget_SE3)
    pen = penalty_term(theta, theta_poses)
    return base_cost + penalty_weight * pen

lower_bounds = q_lower_limit[0:7]
upper_bounds = q_upper_limit[0:7]

solver = ScipyBoundedMinimize(
    fun=cost_with_penalty,
    method="SLSQP", #Options: SLSQP, L-BFGS-B (fast but not that robust), TNC (slow). trust-constr (slow)
    maxiter=100,
    jit=True,
    tol=1e-6,)

rot2 = np.array(R.from_euler('x', 180, degrees=True).as_matrix())
Mtarget = pin.SE3(rot2, np.array([0.088, 0., 0.926]))
_Mtarget_SE3 = jaxlie.SE3.from_matrix(Mtarget.homogeneous)
Mhelper = pin.SE3(rot2, np.array([0.0, 0.0, 0.0]))
_Mhelper_SE3 = jaxlie.SE3.from_matrix(Mhelper.homogeneous)
_theta_0 = robotdata.joint_positions()
_theta_0 = _theta_0.at[3].set(-0.1)
(_theta_poses, _theta_vels, _theta_accels) = reset_derivators(_theta_0)
Cposes = calc_Cposes(_theta_0[0:7])

def run(theta, Mtarget_human_SE3, theta_poses, theta_vels, theta_accels,Cposes, XTOL=1e-6, MAXEVAL = 100):
    sol = solver.run(
        init_params=theta,
        args = (theta_poses, theta_vels, theta_accels, Cposes, Mtarget_human_SE3),
        bounds=(lower_bounds,upper_bounds)
    )
    xt = sol.params
    (theta_dots, theta_ddots, theta_dddots), (theta_new_poses, theta_new_vels, theta_new_accels) = call_derivators(theta_poses, theta_vels, theta_accels, xt)
    Cposes = calc_Cposes(xt)

    return xt, theta_new_poses, theta_new_vels, theta_new_accels,Cposes, sol.state
'''
#################################################test collisionik###################################################################
@jax.jit
def x_cik(theta, _a_obs,_b_obs, _R_obs):
    theta_aug = jnp.append(theta, jnp.zeros((1,)))
    new_a, new_b, _, _, R_robot, _ = FK_for_collisions_v2(theta_aug, T_panda, L_panda, R_panda)

    

    
    phi = collision_proximity_check_v2(new_a, new_b,R_robot, _a_obs, _b_obs,_R_obs)
    epsilon = 0.010
    safe_weight = (5 * epsilon)**2
    chi = safe_weight / (phi**2 + 1e-6)
    
    return jnp.sum(chi)



@partial(jax.jit, static_argnums=(1,))
def make_cost_function_collisionik(theta_inp: jnp.ndarray, FK_jax: callable,
                                   Mtarget: jnp.ndarray, theta_poses: jnp.ndarray,
                                   theta_vels: jnp.ndarray, theta_accels: jnp.ndarray,
                                   Cposes: jnp.ndarray, obstacles_dict: dict) -> float:
    theta = jnp.append(theta_inp, jnp.zeros(1))
    wp = 100
    wo = 80
    wv = 0.02
    wa = 0.2
    wj = 0.4
    we = 1.0
    wc = 1.0 
    wcik = 80.0
    FK = FK_jax(theta, base_pos)
    FK_ee = FK[8]
    Mtool_pos, Mtool_rot = fk_to_pos_rot(FK_ee)
    Mtarget_pos = Mtarget.translation()
    Mtarget_rot = Mtarget.rotation().wxyz
    (theta_dots, theta_ddots, theta_dddots), (theta_new_poses, theta_new_vels, theta_new_accels) = call_derivators(theta_poses, theta_vels, theta_accels, theta)
    
    f_p = wp*make_groove_objective_function(func=x_p, n=1, s=0, r=5,c = 0.2, g = 1, args=(Mtool_pos,Mtarget_pos))
    f_o = wo*make_groove_objective_function(func=x_o, n=1, s=0, r=5,c = 0.2, g = 1, args=(Mtool_rot,Mtarget_rot))
    f_v = wv*make_groove_objective_function(func=x_v, n=1, s=0, r=5,c = 0.2, g = 1, args=(theta_dots,))
    f_a = wa*make_groove_objective_function(func=x_a, n=1, s=0, r=5,c = 0.2, g = 1, args=(theta_ddots,))
    f_j = wj*make_groove_objective_function(func=x_j, n=1, s=0, r=5,c = 0.2, g = 1, args=(theta_dddots,))
    f_e = we*make_groove_objective_function(func=x_e, n=1, s=0, r=5,c = 0.2, g = 1, args=(Mtool_pos, Cposes))
    f_c = wc*make_groove_objective_function(func=x_c, n=0, s=0, r=0.002,c = 2.1, g = 2, args=(theta,))
    f_cik = 0.0
    if obstacles_dict is not None:
        copied_obstacles = obstacles_dict.copy()
        


        for obst_name, obs_parameters in copied_obstacles.items():
            
            a_obs, b_obs, C_obs, L_obs, R_obs, T_obs = obs_parameters
            
            
            a_obs_ext, b_obs_ext, R_obs_ext = make_obs_compatible(a_obs, b_obs, R_obs)
            
            # soft collision
            f_cik += wcik * make_groove_objective_function(
                func=x_cik,
                n=1, 
                s=0.005, 
                c=0.05, 
                r=0.004, 
                g=1,
                args=(theta_inp, a_obs_ext, b_obs_ext, R_obs_ext))

    f = f_p
    f += f_o
    f += f_v
    f += f_a
    f += f_j
    f += f_e
    f += f_c
    f += f_cik

    
    return f


@partial(jax.jit, static_argnums=(1,))
def make_cost_function_collisionik_with_soft_constraints(theta_inp: jnp.ndarray, FK_jax: callable,
                                   Mtarget: jnp.ndarray, theta_poses: jnp.ndarray,
                                   theta_vels: jnp.ndarray, theta_accels: jnp.ndarray,
                                   Cposes: jnp.ndarray, obstacles_dict: dict) -> float:
    # Get the base collision IK cost
    base_cost = make_cost_function_collisionik(
        theta_inp, FK_jax, Mtarget, theta_poses, theta_vels, theta_accels, Cposes, obstacles_dict
    )
    
    # Add velocity penalty term (same as in the regular IK)
    vel_penalty = velocity_penalty_term(theta_inp, theta_poses)
    
    return base_cost + vel_penalty

jac_nlopt_collisionik = jax.jit(jax.jacfwd(make_cost_function_collisionik), static_argnums=(1,))
this_nlopt_collisionik = partial(make_cost_function_collisionik, FK_jax=forward_kinematix_all)


@jax.jit
def nlopt_cost_function_collisionik(theta_inp, Mtarget_SE3, theta_poses, theta_vels, theta_accels, Cposes, obstacles_dict):
    return this_nlopt_collisionik(
        theta_inp, Mtarget=Mtarget_SE3,
        theta_poses=theta_poses, theta_vels=theta_vels, theta_accels=theta_accels,
        Cposes=Cposes,obstacles_dict= obstacles_dict)

def nlopt_obj_collisionik(theta, grad, forward_kinematix_all, Mtarget_SE3, theta_poses, theta_vels, theta_accels, Cposes, obstacles_dict):
    if grad.size > 0:
        _grad = jac_nlopt_collisionik(
            theta, forward_kinematix_all, Mtarget_SE3,
            theta_poses, theta_vels, theta_accels, Cposes, obstacles_dict
        )
        grad[:] = np.array(_grad)
    return float(nlopt_cost_function_collisionik(
        theta, Mtarget_SE3, theta_poses, theta_vels, theta_accels, Cposes, obstacles_dict
    ))

def run_with_collisionik(theta, Mtarget_human_SE3, theta_poses, theta_vels, theta_accels, Cposes, 
                         XTOL=1e-6, MAXEVAL=100, obstacles_dict=None, dt=0.01):
    """
    CollisionIK solver:
    - Soft capsule-based collision cost via Groove Loss (x_cik)
    - No CBF hard constraints
    """

    OBJ_wrapper = lambda x, grad: nlopt_obj_collisionik(
        x, grad, forward_kinematix_all, 
        Mtarget_human_SE3, theta_poses, theta_vels, theta_accels, Cposes, obstacles_dict
    )

    # manipulability constraint 
    CKS_wrapper = lambda x, grad: nlopt_c_ks(x, grad)

    
    opt = nlopt.opt(nlopt.LD_SLSQP, 7)
    opt.set_min_objective(OBJ_wrapper)
    opt.set_lower_bounds(q_lower_limit[0:7])
    opt.set_upper_bounds(q_upper_limit[0:7])
    opt.add_inequality_constraint(CKS_wrapper, 1e-4)
    opt.set_xtol_rel(XTOL)
    opt.set_maxeval(MAXEVAL)

    
    xt = opt.optimize(theta)
    result = opt.last_optimize_result()

    
    (theta_dots, theta_ddots, theta_dddots), (theta_new_poses, theta_new_vels, theta_new_accels) = \
        call_derivators(theta_poses, theta_vels, theta_accels, xt)
    Cposes = calc_Cposes(xt)

    return xt, theta_new_poses, theta_new_vels, theta_new_accels, Cposes, result