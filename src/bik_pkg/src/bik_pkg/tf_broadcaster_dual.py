#! /usr/bin/env python3


import jax
from jax import jacfwd, config, grad,jit,vmap
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')
config.update("jax_debug_nans", False) #Warning
import xml.etree.ElementTree as ET
import os
import sys
import jaxsim
import jaxlie
from jaxsim import logging
logging.set_logging_level(logging.LoggingLevel.INFO)
logging.info(f"Running on {jax.devices()}")
import jaxsim.api as js
import jax.numpy as jnp
from jaxsim import rbda
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import os
import json
import rospy
import tf2_ros
import pandas as pd
from geometry_msgs.msg import TransformStamped
import numpy as np
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation as R
import xml.etree.ElementTree as ET
import rospkg
import pinocchio as pin
import bik_collision
import hppfcl
from pinocchio.visualize import MeshcatVisualizer
import meshcat.geometry as mg
import meshcat.transformations as mt
package_path = rospkg.RosPack().get_path('bik_pkg')
franka_path = rospkg.RosPack().get_path('franka_description')
urdf_path = os.path.join(franka_path, "urdfs/fer_franka_hand_absolute_wsl.urdf")
mesh_path = os.path.join(franka_path, "meshes")


def fix_compatibility_for_sdf(urdf_file):
    tree = ET.parse(urdf_file)
    root = tree.getroot()
    # Find and add dummy inertia for fixed links
    fixed_joint_names = []
    for link in root.findall(".//link"):
        name = link.attrib.get("name", "")
        if link.find("inertial") is None:
            inertial = ET.Element("inertial")
            mass = ET.Element("mass")
            mass.attrib["value"] = "0.00001"
            origin = ET.Element("origin")
            origin.attrib["xyz"] = "0 0 0"
            origin.attrib["rpy"] = "0 0 0"
            inertial.append(mass)
            inertial.append(origin)
            inertia = ET.Element("inertia")
            inertia.attrib["ixx"] = "0.00001"
            inertia.attrib["ixy"] = "0"
            inertia.attrib["ixz"] = "0"
            inertia.attrib["iyy"] = "0.00001"
            inertia.attrib["iyz"] = "0"
            inertia.attrib["izz"] = "0.00001"
            inertial.append(inertia)
            link.append(inertial)
    joints = []
    for joint in root.findall(".//joint"):
        name = joint.attrib.get("name", "")
        joints.append(name)
        if joint.attrib.get("type", "") == "fixed":
            fixed_joint_names.append(name)
    #add gazebo tag with preserveFixedJoint flag
    for fixed_joint_name in fixed_joint_names:
        gazebo = ET.Element("gazebo")
        gazebo.attrib["reference"] = fixed_joint_name
        FixedJoint = ET.Element("disableFixedJointLumping")
        FixedJoint.text = "true"
        gazebo.append(FixedJoint)
        root.append(gazebo)

    tree.write(os.path.join(package_path, "assets/urdf_for_sdf.urdf"))
    #print("Fixed URDF file saved at assets/urdf_for_sdf.urdf")
    return os.path.join(package_path, "assets/urdf_for_sdf.urdf"), joints

def fix_urdf(urdf_file):
    with open(urdf_file, 'r') as file:
        filedata = file.read()
    new_resource_path = os.path.join(package_path, "resources")
    filedata = filedata.replace('resources',new_resource_path)
    with open(os.path.join(package_path, "assets/urdf_file.urdf"), 'w') as file:
        file.write(filedata)
    #print("Fixed URDF file saved at assets/urdf_file.urdf")
    return os.path.join(package_path, "assets/urdf_file.urdf")
def urdf_to_sdf(urdf_file):
    # Run the command to convert URDF to SDF
    if sys.platform == "darwin":
        print("Running on MacOS")
        os.system("export DYLD_LIBRARY_PATH=/opt/homebrew/Cellar/ogre1.9/1.9-20160714-108ab0bcc69603dba32c0ffd4bbbc39051f421c9_10/lib:$DYLD_LIBRARY_PATH\n" +
                  "gz sdf -p " + urdf_file + " > " + os.path.join(package_path,  "assets/sdf_file.sdf"))
    else:
        print("Running on Linux")
        os.system("gz sdf -p " + urdf_file + " > " + os.path.join(package_path,  "assets/sdf_file.sdf"))
    #print("SDF file saved at assets/sdf_file")
    tree = ET.parse( os.path.join(package_path,  "assets/sdf_file.sdf"))
    root = tree.getroot()
    # Change the root tags version to 1.9 it should be 1.09
    current_version =  convert_format(root.attrib["version"])
    print("Current SDF version: ", current_version)
    
    if current_version < 1.09:
        print("Changing version to 1.9")
        root.attrib["version"] = "1.9"

    tree.write( os.path.join(package_path,  "assets/sdf_file.sdf"))
    return  os.path.join(package_path,  "assets/sdf_file.sdf")
def loadRobot(modelPath, meshPath, _fix_urdf=False, basic_collision_model=True, default_shape = "capsule"):
    if _fix_urdf:
        modelPath = fix_urdf(modelPath)
    model, collision_model, visual_model = pin.buildModelsFromUrdf(modelPath, meshPath)
    data = model.createData()
    if basic_collision_model:
        # Copy the collision model to preserve original references
        cmodel = collision_model.copy()
        list_replaced_geometries_pin = []
        list_replaced_geometries_jax = []
        # Process the original geometries and create replacement spheres
        for geom_object in cmodel.geometryObjects:
            geometry = geom_object.geometry
            base_name = geom_object.name.split("_")[0] if "_" in geom_object.name else geom_object.name
            if "finger" in geom_object.name:
                collision_model.removeGeometryObject(geom_object.name)
                continue
            parent_joint = geom_object.parentJoint
            parent_frame = geom_object.parentFrame
            placement = geom_object.placement
            placement_jaxlie = jaxlie.SE3.from_matrix(placement.homogeneous)
            if isinstance(geometry, hppfcl.Box):
                # For Box: compute radius as half the diagonal length
                half_side = geometry.halfSide
                radius = np.linalg.norm(half_side)
                sphere_geom = hppfcl.Sphere(radius)
                
                sphere_obj = pin.GeometryObject(
                    name=f"{geom_object.name}_sphere",
                    parent_frame=parent_frame,
                    parent_joint=parent_joint,
                    collision_geometry=sphere_geom,
                    placement=placement
                )
                sphere_obj.meshColor = np.array([0.7, 0.7, 1.0, 0.7])  # Blue-ish transparent
                collision_model.addGeometryObject(sphere_obj)
                list_replaced_geometries_pin.append(geom_object.name)
                # JAX VERSION
                center = placement.translation
                dimensions = np.array[radius*2, radius*2, radius*2]
                capsule_representation = bik_collision.box2capsule(center, dimensions)
                list_replaced_geometries_jax.append(capsule_representation)
            elif isinstance(geometry, (hppfcl.BVHModelOBBRSS, hppfcl.BVHModelBase)):
                if default_shape == "sphere":
                    raise ValueError("Default shape is set to sphere, but only capsule is supported for now.")
                else:
                    try:
                        # Compute the local AABB
                        geometry.computeLocalAABB()
                        aabb = geometry.aabb_local
                        center = (aabb.min_ + aabb.max_) / 2.0
                        extents = aabb.max_ - aabb.min_
                        
                        # Find the principal axis (longest dimension)
                        max_dim = np.argmax(extents)
                        length = extents[max_dim]
                        
                        # Use the average of the other two dimensions for the radius
                        other_dims = np.delete(extents, max_dim)
                        radius = np.mean(other_dims) / 2.0
                        
                        # Create a rotation matrix to align the capsule with the principal axis
                        R = pin.SE3.Identity().rotation
                        if max_dim == 0:  # X is longest
                            R = pin.Quaternion(np.array([0, 1, 0, 1]) / np.sqrt(2)).matrix()
                        elif max_dim == 1:  # Y is longest
                            R = pin.Quaternion(np.array([1, 0, 0, 1]) / np.sqrt(2)).matrix()
                        # For Z being longest, no rotation needed
                        
                        # Adjust placement to position capsule at AABB center and with proper orientation
                        center_placement = pin.SE3.Identity()
                        center_placement.translation = center
                        center_placement.rotation = R
                        #print(geom_object.name, center_placement, radius, length)
                        if "link5" in geom_object.name:
                            radius += 0.02
                        if "link6" in geom_object.name:
                            radius += 0.01
                        if "hand" in geom_object.name:
                            print("Hand detected")
                            placement_jaxlie = jaxlie.SE3.identity()
                            radius += 0.01
                        adjusted_placement = placement * center_placement
                        
                        # Create capsule
                        capsule_geom = hppfcl.Capsule(radius, length)
                        
                        capsule_obj = pin.GeometryObject(
                            name=f"{geom_object.name}_capsule",
                            parent_frame=parent_frame,
                            parent_joint=parent_joint,
                            collision_geometry=capsule_geom,
                            placement=adjusted_placement
                        )
                        capsule_obj.meshColor = np.array([1.0, 0.7, 0.7, 0.7])  # Red-ish transparent
                        collision_model.addGeometryObject(capsule_obj)
                        list_replaced_geometries_pin.append(geom_object.name)

                        # JAX VERSION
                        Rot =jaxlie.SO3.identity()
                        if max_dim == 0: #X is the longest
                            Rot = jaxlie.SO3.from_quaternion_xyzw(jnp.divide(jnp.array([0,1,0,1]), jnp.sqrt(2)))
                        elif max_dim == 1:
                            Rot = jaxlie.SO3.from_quaternion_xyzw(jnp.divide(jnp.array([1,0,0,1]), jnp.sqrt(2)))
                        translation = center
                        center_placement = jaxlie.SE3.from_rotation_and_translation(Rot, translation)
                        adjusted_placement  = placement_jaxlie @ center_placement
                        C = center
                        local_T_matrix = adjusted_placement.as_matrix()
                        a, b = bik_collision.capsule_ab_from_T(local_T_matrix, length)
                        T,L,R = bik_collision.capsule_dpax_comp(a, b, radius)
                        list_replaced_geometries_jax.append((a,b,C,L,R,T))
                    
                    except Exception as e:
                        print(f"Could not create bounding capsule for {geom_object.name}: {str(e)}")
        
        # Remove original geometries that were replaced
        for name in list_replaced_geometries_pin:
            collision_model.removeGeometryObject(name)
        
        # Handle cylinders
        list_names_capsules = []
        for geom_object in cmodel.geometryObjects:
            geometry = geom_object.geometry
            base_name = "_".join(geom_object.name.split("_")[:-1]) if "_" in geom_object.name else geom_object.name
            
            if isinstance(geometry, hppfcl.Cylinder):
                name = f"{base_name}_capsule_{len(list_names_capsules)}"
                list_names_capsules.append(name)
                capsule = pin.GeometryObject(
                    name=name,
                    parent_frame=int(geom_object.parentFrame),
                    parent_joint=int(geom_object.parentJoint),
                    collision_geometry=hppfcl.Capsule(
                        geometry.radius, geometry.halfLength * 2
                    ),
                    placement=geom_object.placement,
                )
                capsule.meshColor = np.array([249, 136, 126, 125]) / 255  # Red color
                collision_model.addGeometryObject(capsule)
                collision_model.removeGeometryObject(geom_object.name)
    
    # Create collision data and set margin
    collision_data = collision_model.createData()
    collision_model.addAllCollisionPairs()
    for req in collision_data.collisionRequests:
        req.security_margin = 1e-3
    
    return model, collision_model, visual_model, data, collision_data, list_replaced_geometries_jax
def createViewer(model, collision_model, visual_model, _openURL = False):
    viewer = MeshcatVisualizer(model, collision_model, visual_model)
    viewer.initViewer(open=_openURL, loadModel=True)
    q_neutral = pin.neutral(model)
    viewer.display(q_neutral)
    return viewer

def convert_format(num_str):
    parts = num_str.split(".")
    if len(parts) == 2 and len(parts[1]) == 1:  # If single decimal digit
        return float(f"{parts[0]}.0{parts[1]}")
    return float(num_str)  # Return as is if already correct
modelPath, joints = fix_compatibility_for_sdf(urdf_path)
modelSDFPath = urdf_to_sdf(modelPath)
dt = 0.01
full_model = js.model.JaxSimModel.build_from_model_description(
    model_description=modelSDFPath, is_urdf=False)
model = js.model.reduce(model=full_model,
                        considered_joints=tuple(
                            j
                            for j in full_model.joint_names()
                            if "base" not in j
                            and "finger" not in j
                            and "hand_tcp" not in j
                            and "8" not in j
                        ))
data = js.data.JaxSimModelData.zero(model)

forward_kinematics = jit(rbda.forward_kinematics)
forward_kinematics_all = jit(rbda.forward_kinematics_model)
jacobian = jit(rbda.jacobian)
collision = jit(rbda.collidable_points.collidable_points_pos_vel)
forward_kinematix = jax.jit(lambda x: forward_kinematics(model, link_index=8, base_position=jnp.array([0.0,0.0,0.0]),base_quaternion=jnp.array([1.0,0.0,0.0,0.0]), joint_positions=x))
forward_kinematix_all = jax.jit(lambda Q, Pos: forward_kinematics_all(model, base_position=Pos,base_quaternion=jnp.array([1.0,0.0,0.0,0.0]), joint_positions=Q))
jacobix = jax.jit(lambda x: jacobian(model, link_index=8, joint_positions=x)[:,6:6+7])
collision_points = jax.jit(lambda x: collision(model, base_position=jnp.array([0.0,0.0,0.0]),base_quaternion=jnp.array([1.0,0.0,0.0,0.0]), joint_positions=x,
                                               base_linear_velocity=jnp.array([0.0, 0.0, 0.0]),base_angular_velocity=jnp.array([0.0, 0.0, 0.0]), joint_velocities=jnp.zeros_like(x)))

initial_joint_positions = jnp.array(
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0]
)  # Replace with your actual initial joint positions
FK_all = forward_kinematix_all(initial_joint_positions, jnp.array([0.0,0.0,0.0]))

collision_result = collision_points(initial_joint_positions)[0]
PANDAmodel, PANDAcollision_model, PANDAvisual_model, PANDAdata, PANDAcollision_data, list_panda_capsules_jax = loadRobot(urdf_path,mesh_path, _fix_urdf = True)
PandaViewer = createViewer(PANDAmodel, PANDAcollision_model, PANDAvisual_model, _openURL = True)




def load_ros_csv(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    if "q_target_positions" in df.columns:
        try:
            df["q_target_positions"] = df["q_target_positions"].apply(json.loads)
        except Exception as e:
            rospy.logwarn(f"JSON parsing failed in 'q_target_positions': {e}")
    else:
        rospy.logerr("Expected column 'q_target_positions' not found in CSV.")
    return df

def load_unity_csv(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    if "x_target" in df.columns:
        try:
            df["x_target"] = df["x_target"].apply(json.loads)
        except Exception as e:
            rospy.logwarn(f"JSON parsing failed in 'x_target': {e}")
    else:
        rospy.logerr("Expected column 'x_target' not found in CSV.")
    return df

import ast

def parse_joint_poses(array_of_list, isunity: bool = False) -> np.ndarray:
    """
    Parse an array of strings representing lists into a NumPy array of floats.

    Args:
        array_of_list: List of strings, where each string represents a list of floats.

    Returns:
        A NumPy array of shape (n, m), where n is the number of lists and m is the length of each list.
    """
    array = np.array([
        np.array(sublist if isinstance(sublist, list) else ast.literal_eval(sublist))
        for sublist in array_of_list
    ])
    if isunity:
        array[:, 1] = array[:, 1]
        array[:, 3] = array[:, 3]
        array[:, 5] = array[:, 5]
    return array

def parse_cartesian_pose(joint_pos):
    HTM = np.zeros((joint_pos.shape[0], 4, 4))
    pos = np.zeros((joint_pos.shape[0], 3))
    orientation = np.zeros((joint_pos.shape[0], 4))
    for i in range(joint_pos.shape[0]):
        HTM[i] = forward_kinematix(joint_pos[i, 0:8])
        pos[i] = HTM[i][0:3, 3]
        orientation[i] = jaxlie.SO3.from_matrix(HTM[i][0:3, 0:3]).as_quaternion_xyzw()
    return HTM, pos, orientation

def load_obstacle_csv(file_path: str, byname = 'position') -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    
    # Load CSV with error handling for malformed rows
    df = pd.read_csv(file_path, on_bad_lines='warn')

    for col in df.columns:
        first_row = df[col].iloc[0] if not df[col].empty else None
        
        if isinstance(first_row, str) and first_row.strip().startswith('{'):
            print(f"Column '{col}' appears to contain JSON data.")
            try:
                # Attempt to parse JSON
                df[col] = df[col].apply(json.loads)
            except json.JSONDecodeError:
                print(f"Column '{col}' is not valid JSON.")

     # Trim obstacle dataframes based on the last timestamp in unity_df
    obstacle_time = df['stamp'].values
    obstacle_time = obstacle_time -  obstacle_time[0]
    obstacle_time_diff = np.diff(obstacle_time)
    #find a indices where the time difference is larger than 10 seconds
    reset_indices = np.where(obstacle_time_diff< -10)[0]
    if len(reset_indices) > 0:
        rospy.logwarn("Obstacle data reset detected at index: %d", reset_indices[0])
        #trim the dataframe up to the this point
        df = df.iloc[:reset_indices[0]-11]


    wide = df.pivot(index='stamp', columns='frame_id', values=byname)

    wide_parsed = wide.applymap(parse_list_str)

    return wide_parsed

ignored_collision_list = ["holder", "ground", "leg", "object"]
NUM_GEOM_PANDA = len(PANDAcollision_model.geometryObjects)
PANDAcollision_model.removeAllCollisionPairs()
PANDAcollision_data = PANDAcollision_model.createData()
for colres in PANDAcollision_data.collisionResults:
    colres.clear()
# Remove all collision pairs
    colres.security_margin = 100

def load_obstacle(position,orientation, scale, type, name):
    # Create a transformation from the obstacle's pose.
    # orientation obj.orientation.w, obj.orientation.x, obj.orientation.y, obj.orientation.z
    # position obj.position.x, obj.position.y, obj.position.z
    # scale obj.scale.x, obj.scale.z, obj.scale.y
    R = pin.Quaternion(orientation[0], orientation[1], orientation[2], orientation[3])
    M = pin.SE3(R, np.array(position))
    scale = np.array(scale)
    if type == 0:
        geom = hppfcl.Box(scale[0], scale[1], scale[2])
    elif type == 1:
        geom = hppfcl.Sphere(scale[0])
    elif type == 2:
        geom = hppfcl.Capsule(scale[0]/2, scale[1]*4)
    else:
        print("Obstacle type %s not supported", type)
        return None, None
    geom_pin = pin.GeometryObject(name, 0, M, geom)
    color = np.append(np.random.rand(3), 0.8)
    if "obstacle" in name:
        color = np.array([0, 0, 0, 0.8])
    geom_pin.meshColor = color
    geom_pin.overrideMaterial = True
    geom_pin.meshMaterial = pin.GeometryPhongMaterial()
    geom_pin.meshMaterial.meshEmissionColor = np.array([1., 0.1, 0.1, 1.])
    geom_pin.meshMaterial.meshSpecularColor = np.array([0.1, 1., 0.1, 1.])
    geom_pin.meshMaterial.meshShininess = 0.8
    collision_id = PANDAcollision_model.addGeometryObject(geom_pin)
    for i in range(NUM_GEOM_PANDA):
        #Check if the obstacle name contains any word in the ignored collision list:
        if any(ignored in name for ignored in ignored_collision_list):
            continue
        col_pair = pin.CollisionPair(i, collision_id)
        PANDAcollision_model.addCollisionPair(col_pair)
        #add the safety margin
        #rospy.loginfo("Collision pair added: %d, %d", i, collision_id)
    
    visual_id = PANDAvisual_model.addGeometryObject(geom_pin)
    #print("Obstacle added:", name, 'Collision ID:',collision_id, 'Visual ID:',visual_id)
    return collision_id, visual_id

unity_to_ros_rotation = jaxlie.SO3.from_rpy_radians(
    roll=0.0, pitch=0.0, yaw=0.0)  # No rotation

def parse_X_poses(data):
    positions = np.zeros((data.shape[0], 3))
    quaternions = np.zeros((data.shape[0], 4))
    rotations = np.zeros((data.shape[0], 4))
    for i in range(data.shape[0]):
        item = data.iloc[i]  
        position = item['pos']
        quaternion = item['orient']
        targetrotmat, _targetquatcoeffs, _targetquat = Rquat(quaternion[0], quaternion[1], quaternion[2], quaternion[3]) #Pinocchio takes w x y z, but the coming data from the Unity, requires x y z w
        x_target = pin.SE3(targetrotmat, np.array(position))
        target = jaxlie.SE3.from_matrix(x_target.homogeneous)
        positions[i] = np.array(np.array(position))
        quaternions[i] = np.array(quaternion)
        quat = jaxlie.SO3.from_matrix(x_target.homogeneous[0:3,0:3]).as_quaternion_xyzw()
        quaternions[i] = np.array(quat)
        
        rotations[i] = quaternions[i] # np.array(corrected.as_quaternion_xyzw())  # now in ROS frame
    return positions, quaternions, rotations


from itertools import permutations

def quaternion_angle_error(q1, q2):
    dot = np.clip(np.dot(q1, q2), -1.0, 1.0)
    angle = 2 * np.arccos(abs(dot))
    return angle

import ast
def parse_list_str(s):
    # safely evaluate the string into a list
    if isinstance(s, str):
        return np.array(ast.literal_eval(s))
    if s is None:
        return np.array([0.0,0.0,0.0])

def test_all_axis_permutations_no_sign_flip(pos_unity, quats_unity, positions_fk, orientations_fk):
    indices = [0, 1, 2, 3]  # x, y, z, w
    best_combination = None
    min_total_error = float('inf')

    for perm in permutations(indices):     
        errors = []

        for i in range(len(quats_unity)):
            q_unity = quats_unity[i].copy()

            new_q = np.array([
                q_unity[perm[0]],
                q_unity[perm[1]],
                q_unity[perm[2]],
                q_unity[perm[3]]
            ])

            new_q /= np.linalg.norm(new_q)

            q_ros = orientations_fk[i]

            err = quaternion_angle_error(new_q, q_ros)
            errors.append(err)

        total_error = np.sum(errors)

        print(f"Permutation: {perm}, total_error: {total_error}")

        if total_error < min_total_error:
            min_total_error = total_error
            best_combination = perm

    print("\n best combi：", best_combination)
    print("total error：", min_total_error)



import select
import sys
import termios
import tty

def is_key_pressed():
    """Check if a key was pressed"""
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

def get_key():
    """Get the pressed key"""
    settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setraw(sys.stdin.fileno())
        key = sys.stdin.read(1)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

def Rquat(x, y, z, w): #Converts quaternion to rotation matrix and quaternion coefficient
    q = pin.Quaternion(x, y, z, w)
    q.normalize()
    return q.matrix(), q.coeffs(), q

def get_only_experiments(interped_time_stamps, interped_robot_status):
    """
    Get the start and end indices of periods where the robot_status is False for more than 5 seconds.
    
    Args:
        interped_time_stamps: Array of time stamps.
        interped_robot_status: Array of boolean values indicating robot status.
        
    Returns:
        List of tuples containing start and end indices of periods where robot_status is False for more than 5 seconds.
    """
    time_indices = []
    start_idx = None
    
    for i in range(len(interped_robot_status)):
        # If we find a False value and we're not already tracking an interval
        if not interped_robot_status[i] and start_idx is None:
            start_idx = i
        # If we find a True value and we were tracking an interval
        elif interped_robot_status[i] and start_idx is not None:
            # Check if the interval was longer than 5 seconds
            if interped_time_stamps[i-1] - interped_time_stamps[start_idx] > 5.0:
                time_indices.append((start_idx, i-1))
            start_idx = None
    
    # Check if there's an ongoing interval at the end of the list
    if start_idx is not None:
        if interped_time_stamps[-1] - interped_time_stamps[start_idx] > 5.0:
            time_indices.append((int(start_idx), len(interped_robot_status) - 1))
    
    return time_indices




def collision_callback(q_target, PANDAmodel, PANDAdata, PANDAcollision_model, 
                          PANDAcollision_data, visualize = True):
    """Timer callback to perform collision checking periodically."""    
    q_target = np.array(q_target)
    q_target = np.append(q_target, np.zeros((NUM_GEOM_PANDA-len(q_target),)))
    # Update geometry placements with the latest joint configuration.
    pin.computeCollisions(PANDAmodel, PANDAdata, PANDAcollision_model, 
                          PANDAcollision_data, q_target,
                          stop_at_first_collision=False)
    pin.computeDistances(PANDAcollision_model,
                         PANDAcollision_data)
    # Print the status of collision for all collision pairs
    collisions = []
    for k in range(len(PANDAcollision_model.collisionPairs)): 
        cr = PANDAcollision_data.collisionResults[k]
        cp = PANDAcollision_model.collisionPairs[k]
        #print("collision pair:",cp.first,",",cp.second,"- collision:","Yes" if cr.isCollision() else "No")

        if cr.isCollision():
            colres = PANDAcollision_data.collisionResults[k]
            contact: hppfcl.Contact = colres.getContacts()[0]

            p1 = contact.getNearestPoint1()
            p2 = contact.getNearestPoint2()
            dist = np.linalg.norm(p1 - p2)
            #print(p1, p2, p1 - p2)

            name = PANDAcollision_model.geometryObjects[cp.first].name
            name2 = PANDAcollision_model.geometryObjects[cp.second].name
            collisions.append((name, name2))
            if visualize:
                total_name =  name + " - "+name2
                random_color = np.random.rand(3)
                hex_color = "0x{:02x}{:02x}{:02x}".format(int(random_color[0]*255), int(random_color[1]*255), int(random_color[2]*255))

                #plot on the viewer
                points = np.hstack([p1.reshape(-1,1), p2.reshape(-1,1)]).astype(np.float32)

                PandaViewer.viewer[total_name].set_object(mg.Line(mg.PointsGeometry(points), mg.MeshBasicMaterial(color=hex_color, linewidth=50)))
                PandaViewer.viewer[name].set_object(mg.Sphere(0.01), mg.MeshLambertMaterial(color=hex_color))
                PandaViewer.viewer[name].set_transform(mt.translation_matrix(p1))
                PandaViewer.viewer[name2].set_object(mg.Sphere(0.01), mg.MeshLambertMaterial(color=hex_color))
                PandaViewer.viewer[name2].set_transform(mt.translation_matrix(p2))
        else:
            if visualize:
                # Objects are not colliding - print the nearest distance
                dr = PANDAcollision_data.distanceResults[k]
                name = PANDAcollision_model.geometryObjects[cp.first].name
                name2 = PANDAcollision_model.geometryObjects[cp.second].name

                #print(f"  Distance between {name} and {name2}: {dr.min_distance:.6f}")

                # Try to get and visualize nearest points using the direct methods
                try:
                    p1 = dr.getNearestPoint1()
                    p2 = dr.getNearestPoint2()
                    dist = np.linalg.norm(p1 - p2)
                    # Only visualize if the points are valid
                    if p1 is not None and p2 is not None:
                        #print(f"  Nearest points: {p1}, {p2}")

                        # Visualize with a green color for non-colliding objects
                        total_name = name + " - " + name2
                        safe_color = "0x00ff00"  # Green color for safe distances

                        points = np.hstack([p1.reshape(-1,1), p2.reshape(-1,1)]).astype(np.float32)
                        PandaViewer.viewer[total_name].set_object(mg.Line(mg.PointsGeometry(points), 
                                                                            mg.MeshBasicMaterial(color=safe_color, linewidth=10)))
                except Exception as e:
                    # If the getNearestPoint methods fail, log the error but continue
                    print(f"Could not get nearest points for {name} and {name2}: {e}")  
    return collisions
def play_tf_dual(user: str, config: str, data_dir: str = "/home/sun/CBF_results", rate_hz: float = 100.0):
    rospy.init_node("csv_tf_broadcaster_dual")
    br = tf2_ros.TransformBroadcaster()
    rate = rospy.Rate(rate_hz)
    trial_idx_path = os.path.join(data_dir, user, config, "trial_idx.txt")
    trial_idx = []
    if os.path.exists(trial_idx_path):
        os.remove(trial_idx_path)
    os.makedirs(os.path.dirname(trial_idx_path), exist_ok=True)
    ros_path = os.path.join(data_dir, user, config, "ros_to_unity.csv")
    unity_path = os.path.join(data_dir, user, config, "unity_to_ros.csv")
    

    if not os.path.exists(ros_path) or not os.path.exists(unity_path):
        rospy.logerr("Missing CSV files.")
        return
    
    ros_df = load_ros_csv(ros_path)
    unity_df = load_unity_csv(unity_path)
    obstacle_df = load_obstacle_csv(os.path.join(data_dir, user, config, "obstacle_info.csv"), byname='position')
    obstacle_orient_df = load_obstacle_csv(os.path.join(data_dir, user, config, "obstacle_info.csv"), byname='orientation')
    obstacle_scale_df = load_obstacle_csv(os.path.join(data_dir, user, config, "obstacle_info.csv"), byname='scale')
    rospy.loginfo(f"ros_to_unity.csv length: {len(ros_df)}")
    rospy.loginfo(f"unity_to_ros.csv length: {len(unity_df)}")
    rospy.loginfo(f"obstacle_info.csv length: {len(obstacle_df)}")
    obstacle_time = obstacle_df.index.values
    unity_time = unity_df["stamp"].values
    ros_time = ros_df["stamp"].values

    # Check for sudden drops in unity_time before normalization
    time_diffs = np.diff(unity_time)
    # A negative time difference larger than 10 seconds is considered a reset
    reset_threshold = -10
    reset_indices = np.where(time_diffs < reset_threshold)[0]

    if len(reset_indices) > 0:
        # Found at least one time reset
        first_reset_idx = reset_indices[0]
        rospy.logwarn(f"Detected time reset at index {first_reset_idx}. Data trimmed.")
        
        # Trim unity_df
        unity_df = unity_df.iloc[:first_reset_idx+1]
        unity_time = unity_df["stamp"].values - unity_df["stamp"].values[0]  # Normalize to start from 0
        
        # Get the last timestamp in unity_df
        last_unity_stamp = unity_time[-1]
        
        # Trim ros_df based on the last timestamp in unity_df
       
        ros_time = ros_df["stamp"].values   
        ros_time = ros_time - ros_time[0]  # Normalize to start from 0
        ros_time = ros_time[ros_time <= last_unity_stamp]
        ros_df = ros_df[ros_df["stamp"].values - ros_df["stamp"].values[0] <= last_unity_stamp]
        
       
        
        # Safety check: ensure we have data after trimming
        if len(unity_time) == 0 or len(ros_time) == 0 or len(obstacle_time) == 0:
            if len(unity_time) == 0:
                rospy.logerr("No data left in unity_df after trimming. Check data for inconsistencies.")
            if len(ros_time) == 0:
                rospy.logerr("No data left in ros_df after trimming. Check data for inconsistencies.")
            if len(obstacle_time) == 0:
                rospy.logerr("No data left in obstacle_df after trimming. Check data for inconsistencies.")
            rospy.logerr("No data left after trimming. Check data for inconsistencies.")
            rospy.signal_shutdown("No data left after trimming")

    # Now normalize all timestamps by subtracting their first value
    unity_time = unity_time - unity_time[0]  # 90Hz
    ros_time = ros_time - ros_time[0]  # 100Hz
    obstacle_time = obstacle_time - obstacle_time[0]  # 10Hz
    print("Unity time: ", unity_time[-1])
    print("ROS time: ", ros_time[-1])
    print("Obstacle time: ", obstacle_time[-1])
    # Parse joint positions
    q_targets_np = parse_joint_poses(ros_df["q_target_positions"].values)
    a_actual_np = parse_joint_poses(unity_df["q_actual_positions"].values, isunity=True)
    pos_unity, quats_unity, rot_unity = parse_X_poses(unity_df["x_target"])

    # Interpolate the q_actual_np to match the q_targets_np
    interpolated_actual_np = np.zeros_like(q_targets_np)
    for joint_idx in range(a_actual_np.shape[1]):  # Iterate over each joint
        interpolated_actual_np[:, joint_idx] = np.interp(
            ros_time, unity_time, a_actual_np[:, joint_idx]
        )
    a_actual_np = interpolated_actual_np

    # Interpolate pos_unity to match ros_time
    interpolated_pos_unity = np.zeros((ros_time.shape[0], pos_unity.shape[1]))
    for dim in range(pos_unity.shape[1]):  # Iterate over each dimension (x, y, z)
        interpolated_pos_unity[:, dim] = np.interp(
            ros_time, unity_time, pos_unity[:, dim]
        )
    pos_unity = interpolated_pos_unity

    # Interpolate quats_unity to match ros_time using Slerp
    slerp_quats = Slerp(unity_time, R.from_quat(quats_unity))
    clipped_ros_time = np.clip(ros_time, unity_time[0], unity_time[-1])
    interpolated_quats_unity = slerp_quats(clipped_ros_time).as_quat()
    quats_unity = interpolated_quats_unity

    # Interpolate the unity_df['robot_status'] to match the ros_time using nearest neighbor
    # Convert boolean values to integers for interpolation
    robot_status_int = unity_df['robot_status'].astype(int).values
    interpolated_robot_status = np.zeros(len(ros_time), dtype=bool)
    for i, t in enumerate(ros_time):
        # Find nearest neighbor index
        nearest_idx = np.abs(unity_time - t).argmin()
        interpolated_robot_status[i] = bool(robot_status_int[nearest_idx])
    time_indices = get_only_experiments(ros_time, interpolated_robot_status)
    time_durations = [ros_time[end] - ros_time[start] for start, end in time_indices]
    print("Time indices: ", time_durations)

    obs0_obj_pos = np.array(obstacle_df["ground"].to_list())
    obj1_obj_pos = np.array(obstacle_df["object_1"].to_list())
    obj2_obj_pos = np.array(obstacle_df["object_2"].to_list())
    obj3_obj_pos = np.array(obstacle_df["object_3"].to_list())
    obs1_obj_pos = np.array(obstacle_df["obstacle_1"].to_list())
    obs2_obj_pos = np.array(obstacle_df["obstacle_2"].to_list())
    obs3_obj_pos = np.array(obstacle_df["obstacle_3"].to_list())
    obs4_obj_pos = np.array(obstacle_df["obstacle_4"].to_list())
    obs5_obj_pos = np.array(obstacle_df["obstacle_5"].to_list())
    obs6_obj_pos = np.array(obstacle_df["obstacle_6"].to_list())
    obs7_obj_pos = np.array(obstacle_df["obstacle_table"].to_list())
    obs8_obj_pos = np.array(obstacle_df["robot_holder"].to_list())

    obs1_obj_orient = np.array(obstacle_orient_df["obstacle_1"].to_list())
    obs2_obj_orient = np.array(obstacle_orient_df["obstacle_2"].to_list())
    obs3_obj_orient = np.array(obstacle_orient_df["obstacle_3"].to_list())
    obs4_obj_orient = np.array(obstacle_orient_df["obstacle_4"].to_list())
    obs5_obj_orient = np.array(obstacle_orient_df["obstacle_5"].to_list())
    obs6_obj_orient = np.array(obstacle_orient_df["obstacle_6"].to_list())
    obs7_obj_orient = np.array(obstacle_orient_df["obstacle_table"].to_list())

    obs1_obj_scale = np.array(obstacle_scale_df["obstacle_1"].to_list())
    obs2_obj_scale = np.array(obstacle_scale_df["obstacle_2"].to_list())
    obs3_obj_scale = np.array(obstacle_scale_df["obstacle_3"].to_list())
    obs4_obj_scale = np.array(obstacle_scale_df["obstacle_4"].to_list())
    obs5_obj_scale = np.array(obstacle_scale_df["obstacle_5"].to_list())
    obs6_obj_scale = np.array(obstacle_scale_df["obstacle_6"].to_list())
    obs7_obj_scale = np.array(obstacle_scale_df["obstacle_table"].to_list())



    #Load obstacles from their first frame
    load_obstacle([obs1_obj_pos[0,0], obs1_obj_pos[0,1], obs1_obj_pos[0,2]], 
                                            [obs1_obj_orient[0,3], obs1_obj_orient[0,0], obs1_obj_orient[0,1], obs1_obj_orient[0,2]], 
                                            [obs1_obj_scale[0,0], obs1_obj_scale[0,2], obs1_obj_scale[0,1]], 2, "obstacle_1")
    load_obstacle([obs2_obj_pos[0,0], obs2_obj_pos[0,1], obs2_obj_pos[0,2]], 
                                            [obs2_obj_orient[0,3], obs2_obj_orient[0,0], obs2_obj_orient[0,1], obs2_obj_orient[0,2]], 
                                            [obs2_obj_scale[0,0], obs2_obj_scale[0,2], obs2_obj_scale[0,1]], 2, "obstacle_2")                                        
    load_obstacle([obs3_obj_pos[0,0], obs3_obj_pos[0,1], obs3_obj_pos[0,2]], 
                                            [obs3_obj_orient[0,3], obs3_obj_orient[0,0], obs3_obj_orient[0,1], obs3_obj_orient[0,2]], 
                                            [obs3_obj_scale[0,0], obs3_obj_scale[0,2], obs3_obj_scale[0,1]], 2, "obstacle_3")
    load_obstacle([obs4_obj_pos[0,0], obs4_obj_pos[0,1], obs4_obj_pos[0,2]], 
                                            [obs4_obj_orient[0,3], obs4_obj_orient[0,0], obs4_obj_orient[0,1], obs4_obj_orient[0,2]], 
                                            [obs4_obj_scale[0,0], obs4_obj_scale[0,2], obs4_obj_scale[0,1]], 2, "obstacle_4")
    load_obstacle([obs5_obj_pos[0,0], obs5_obj_pos[0,1], obs5_obj_pos[0,2]], 
                                            [obs5_obj_orient[0,3], obs5_obj_orient[0,0], obs5_obj_orient[0,1], obs5_obj_orient[0,2]], 
                                            [obs5_obj_scale[0,0], obs5_obj_scale[0,2], obs5_obj_scale[0,1]], 2, "obstacle_5")
    load_obstacle([obs6_obj_pos[0,0], obs6_obj_pos[0,1], obs6_obj_pos[0,2]], 
                                            [obs6_obj_orient[0,3], obs6_obj_orient[0,0], obs6_obj_orient[0,1], obs6_obj_orient[0,2]], 
                                            [obs6_obj_scale[0,0], obs6_obj_scale[0,2], obs6_obj_scale[0,1]], 2, "obstacle_6")
    load_obstacle([obs7_obj_pos[0,0], obs7_obj_pos[0,1], obs7_obj_pos[0,2]], 
                                            [obs7_obj_orient[0,3], obs7_obj_orient[0,0], obs7_obj_orient[0,1], obs7_obj_orient[0,2]], 
                                            [obs7_obj_scale[0,0], obs7_obj_scale[0,2], obs7_obj_scale[0,1]], 2, "obstacle_7")
    

    PANDAcollision_data = PANDAcollision_model.createData()
    for colres in PANDAcollision_data.collisionResults:
        colres.clear()
        colres.security_margin = 100
    PANDAvisual_data = PANDAvisual_model.createData()
    PandaViewer.rebuildData()
    PandaViewer.loadViewerModel(rootNodeName="pinocchio")

    obs0_obj_pos_interpolated = np.zeros((ros_time.shape[0], obs0_obj_pos.shape[1]))
    for dim in range(obs0_obj_pos.shape[1]):  # Iterate over each dimension (x, y, z)
        obs0_obj_pos_interpolated[:, dim] = np.interp(
            ros_time, obstacle_time, obs0_obj_pos[:, dim]
        )
    obs0_obj_pos = obs0_obj_pos_interpolated
    obj1_obj_pos_interpolated = np.zeros((ros_time.shape[0], obj1_obj_pos.shape[1]))
    for dim in range(obj1_obj_pos.shape[1]):  # Iterate over each dimension (x, y, z)
        obj1_obj_pos_interpolated[:, dim] = np.interp(
            ros_time, obstacle_time, obj1_obj_pos[:, dim]
        )
    obj1_obj_pos = obj1_obj_pos_interpolated
    obj2_obj_pos_interpolated = np.zeros((ros_time.shape[0], obj2_obj_pos.shape[1]))
    for dim in range(obj2_obj_pos.shape[1]):  # Iterate over each dimension (x, y, z)
        obj2_obj_pos_interpolated[:, dim] = np.interp(
            ros_time, obstacle_time, obj2_obj_pos[:, dim]
        )
    obj2_obj_pos = obj2_obj_pos_interpolated
    obj3_obj_pos_interpolated = np.zeros((ros_time.shape[0], obj3_obj_pos.shape[1]))
    for dim in range(obj3_obj_pos.shape[1]):  # Iterate over each dimension (x, y, z)
        obj3_obj_pos_interpolated[:, dim] = np.interp(
            ros_time, obstacle_time, obj3_obj_pos[:, dim]
        )
    obj3_obj_pos = obj3_obj_pos_interpolated
    obs1_obj_pos_interpolated = np.zeros((ros_time.shape[0], obs1_obj_pos.shape[1]))
    for dim in range(obs1_obj_pos.shape[1]):  # Iterate over each dimension (x, y, z)
        obs1_obj_pos_interpolated[:, dim] = np.interp(
            ros_time, obstacle_time, obs1_obj_pos[:, dim]
        )
    obs1_obj_pos = obs1_obj_pos_interpolated
    obs2_obj_pos_interpolated = np.zeros((ros_time.shape[0], obs2_obj_pos.shape[1]))
    for dim in range(obs2_obj_pos.shape[1]):  # Iterate over each dimension (x, y, z)
        obs2_obj_pos_interpolated[:, dim] = np.interp(
            ros_time, obstacle_time, obs2_obj_pos[:, dim]
        )
    obs2_obj_pos = obs2_obj_pos_interpolated
    obs3_obj_pos_interpolated = np.zeros((ros_time.shape[0], obs3_obj_pos.shape[1]))
    for dim in range(obs3_obj_pos.shape[1]):  # Iterate over each dimension (x, y, z)
        obs3_obj_pos_interpolated[:, dim] = np.interp(
            ros_time, obstacle_time, obs3_obj_pos[:, dim]
        )
    obs3_obj_pos = obs3_obj_pos_interpolated
    obs4_obj_pos_interpolated = np.zeros((ros_time.shape[0], obs4_obj_pos.shape[1]))
    for dim in range(obs4_obj_pos.shape[1]):  # Iterate over each dimension (x, y, z)
        obs4_obj_pos_interpolated[:, dim] = np.interp(
            ros_time, obstacle_time, obs4_obj_pos[:, dim]
        )
    obs4_obj_pos = obs4_obj_pos_interpolated
    obs5_obj_pos_interpolated = np.zeros((ros_time.shape[0], obs5_obj_pos.shape[1]))
    for dim in range(obs5_obj_pos.shape[1]):  # Iterate over each dimension (x, y, z)
        obs5_obj_pos_interpolated[:, dim] = np.interp(
            ros_time, obstacle_time, obs5_obj_pos[:, dim]
        )
    obs5_obj_pos = obs5_obj_pos_interpolated
    obs6_obj_pos_interpolated = np.zeros((ros_time.shape[0], obs6_obj_pos.shape[1]))
    for dim in range(obs6_obj_pos.shape[1]):  # Iterate over each dimension (x, y, z)
        obs6_obj_pos_interpolated[:, dim] = np.interp(
            ros_time, obstacle_time, obs6_obj_pos[:, dim]
        )
    obs6_obj_pos = obs6_obj_pos_interpolated
    obs7_obj_pos_interpolated = np.zeros((ros_time.shape[0], obs7_obj_pos.shape[1]))
    for dim in range(obs7_obj_pos.shape[1]):  # Iterate over each dimension (x, y, z)
        obs7_obj_pos_interpolated[:, dim] = np.interp(
            ros_time, obstacle_time, obs7_obj_pos[:, dim]
        )
    obs7_obj_pos = obs7_obj_pos_interpolated
    obs8_obj_pos_interpolated = np.zeros((ros_time.shape[0], obs8_obj_pos.shape[1]))
    for dim in range(obs8_obj_pos.shape[1]):  # Iterate over each dimension (x, y, z)
        obs8_obj_pos_interpolated[:, dim] = np.interp(
            ros_time, obstacle_time, obs8_obj_pos[:, dim]
        )
    obs8_obj_pos = obs8_obj_pos_interpolated



    _, positions_fk, orientations_fk = parse_cartesian_pose(q_targets_np)
    _, positions_fk_a, orientations_fk_a = parse_cartesian_pose(a_actual_np)



    duration = len(ros_time)
    j = len(time_indices) - 1
    i = time_indices[j][0]
    paused = False
    rospy.loginfo("Playback started. Press 'p' to pause, 'r' to resume, 's' to skip")
    new_trial = True
    visualize = False
    collision_lists = []
    last_collision_time = {}  # Dictionary to track when each collision pair was last recorded
    active_collisions = {}  # Dictionary to track active collisions and their timestamps
    collision_lists = []  # List to store unique collision events
    collision_markers = {}  # Dictionary to track visualizations for active collisions
    while not rospy.is_shutdown() and i < duration:
        # Check for keyboard input
        paused = False
        if is_key_pressed():
            key = get_key()
            if key == 'p':
                paused = True
                rospy.loginfo("Playback paused. Press 'r' to resume, press s to skip to next trial")
            elif key == 'r':
                paused = False
                rospy.loginfo("Playback resumed. Press 'p' to pause, press s to skip to next trial")
            elif key == 'm':
                print("New stopped indices: ", i+1 , "instead of ", time_indices[j][1])
                time_indices[j] = (time_indices[j][0], i+1)
            elif key == 's':
                print("Skipping to next trial")
                i = time_indices[j][1]
                paused = True
                new_trial = False

        finished = False
        if new_trial:
            collision_lists = []
            new_trial = False
            print(f"This experiment took {ros_time[time_indices[j][1]] - ros_time[time_indices[j][0]]} seconds")
            # Get the cartesian velocity of human target

            human_target_pos = positions_fk[i:time_indices[j][1],:]
            human_target_pos = np.array(human_target_pos)
            human_target_pos_amplitude = np.linalg.norm(human_target_pos, axis=1)
            human_target_vel_amplitude = np.diff(human_target_pos_amplitude)
            human_target_vel_amplitude = np.append(human_target_vel_amplitude, 0)
            
            # Calculate human target velocity directly
            human_vel = np.zeros_like(human_target_pos)
            human_vel[1:] = human_target_pos[1:] - human_target_pos[:-1]
            human_vel_magnitude = np.linalg.norm(human_vel, axis=1)
            human_vel_magnitude = np.append(human_vel_magnitude, 0)

            
            # Find the interval where the human_target_vel_amplitude is > 0
            start_indices = np.where(human_target_vel_amplitude > 0)[0]
            
            # Check if all objects are above the height threshold (0.6m) at the same time
            height_threshold = 0.6
            velocity_threshold = 0.005  # Threshold for considering an object stationary
            time_range = range(i, time_indices[j][1]+1)
            
            # Create masks for each object being above threshold
            obj1_above = obj1_obj_pos[time_range, 2] > height_threshold
            obj2_above = obj2_obj_pos[time_range, 2] > height_threshold
            obj3_above = obj3_obj_pos[time_range, 2] > height_threshold
            
            # Calculate velocities for objects
            obj1_vel = np.zeros_like(obj1_obj_pos)
            obj2_vel = np.zeros_like(obj2_obj_pos)
            obj3_vel = np.zeros_like(obj3_obj_pos)
            
            obj1_vel[1:] = obj1_obj_pos[1:] - obj1_obj_pos[:-1]
            obj2_vel[1:] = obj2_obj_pos[1:] - obj2_obj_pos[:-1]
            obj3_vel[1:] = obj3_obj_pos[1:] - obj3_obj_pos[:-1]
            
            obj1_vel_mag = np.linalg.norm(obj1_vel, axis=1)
            obj2_vel_mag = np.linalg.norm(obj2_vel, axis=1)
            obj3_vel_mag = np.linalg.norm(obj3_vel, axis=1)
            
            # Check if objects are stationary
            obj1_stationary = obj1_vel_mag[time_range] < velocity_threshold
            obj2_stationary = obj2_vel_mag[time_range] < velocity_threshold
            obj3_stationary = obj3_vel_mag[time_range] < velocity_threshold
            human_stationary = human_vel_magnitude[range(0, len(time_range))] < 1e-4
            
            # Find where ALL objects are above threshold AND all objects plus human are stationary
            all_objs_above = np.logical_and(obj1_above, np.logical_and(obj2_above, obj3_above))
            all_stationary = np.logical_and(np.logical_and(obj1_stationary, obj2_stationary), 
                                           np.logical_and(obj3_stationary, human_stationary))
            
            # Combined condition: all objects above threshold AND everything stationary
            #final_condition = np.logical_and(all_objs_above, all_stationary)
            
            # Find continuous periods meeting all conditions
            first_all_above = time_indices[j][1]  # Default fallback
            """if np.any(final_condition):
                # Get indices where all conditions are met
                end_indices = np.where(final_condition)[0]
                
                # Find continuous sequences
                sequences = []
                if len(end_indices) > 0:  # Check that we have at least one valid index
                    current_seq = [end_indices[0]]
                    
                    for idx in range(1, len(end_indices)):
                        if end_indices[idx] == end_indices[idx-1] + 1:
                            # This is part of the current continuous sequence
                            current_seq.append(end_indices[idx])
                        else:
                            # This starts a new sequence
                            sequences.append(current_seq)
                            current_seq = [end_indices[idx]]
                    
                    # Add the last sequence if it exists
                    if current_seq:
                        sequences.append(current_seq)
                    
                    # Use the first continuous sequence
                    if sequences:
                        first_sequence = sequences[0]
                        first_all_above = first_sequence[0] + i
                        last_all_above = first_sequence[-1] + i
                        if len(sequences) > 100:
                            first_all_above = last_all_above
                            print(f"All objects above {height_threshold}m AND all objects + human stationary from index {first_all_above} to {last_all_above}")
                        else:
                            first_all_above = time_indices[j][1]  # Default fallback
                else:
                    print("Found valid condition but couldn't extract sequences")
                    first_all_above = time_indices[j][1]  # Default fallback
            """
            if (len(start_indices) < 1):
                print("No movement detected, please press i to ignore")
                paused = True
                i = time_indices[j][1]
            else:
                print("Starting index normally was", i, "now is", start_indices[0] + i)
                print("Ending index normally was", time_indices[j][1], "now is", first_all_above)
                i = start_indices[0] + i if len(start_indices) > 0 else i
                time_indices[j] = (i, first_all_above)
        # Skip frame update if paused
        if not paused:
            i += 1
            
        
        print(f"{i}/{duration}", end='\r')
        time_now = rospy.Time.now()

        t_fk = TransformStamped()
        t_fk.header.stamp = time_now
        t_fk.header.frame_id = "world"
        t_fk.child_frame_id = f"FK_target"
        t_fk.transform.translation.x = positions_fk[i][0]
        t_fk.transform.translation.y = positions_fk[i][1]
        t_fk.transform.translation.z = positions_fk[i][2]
        t_fk.transform.rotation.x = orientations_fk[i][0]
        t_fk.transform.rotation.y = orientations_fk[i][1]
        t_fk.transform.rotation.z = orientations_fk[i][2]
        t_fk.transform.rotation.w = orientations_fk[i][3]
        br.sendTransform(t_fk)

        t_fk_a = TransformStamped()
        t_fk_a.header.stamp = time_now
        t_fk_a.header.frame_id = "world"
        t_fk_a.child_frame_id = f"FK_actual"
        t_fk_a.transform.translation.x = positions_fk_a[i][0]
        t_fk_a.transform.translation.y = positions_fk_a[i][1]
        t_fk_a.transform.translation.z = positions_fk_a[i][2]
        t_fk_a.transform.rotation.x = orientations_fk_a[i][0]
        t_fk_a.transform.rotation.y = orientations_fk_a[i][1]
        t_fk_a.transform.rotation.z = orientations_fk_a[i][2]
        t_fk_a.transform.rotation.w = orientations_fk_a[i][3]
        br.sendTransform(t_fk_a)

        t_ros = TransformStamped()
        t_ros.header.stamp = time_now
        t_ros.header.frame_id = "world"
        t_ros.child_frame_id = f"human_target"
        t_ros.transform.translation.x = pos_unity[i][0]
        t_ros.transform.translation.y = pos_unity[i][1]
        t_ros.transform.translation.z = pos_unity[i][2]
        t_ros.transform.rotation.x = quats_unity[i][0]
        t_ros.transform.rotation.y = quats_unity[i][1]
        t_ros.transform.rotation.z = quats_unity[i][2]
        t_ros.transform.rotation.w = quats_unity[i][3]
        br.sendTransform(t_ros)

        #publish object positions
        t_obj1 = TransformStamped()
        t_obj1.header.stamp = time_now
        t_obj1.header.frame_id = "world"
        t_obj1.child_frame_id = f"obj_1"
        t_obj1.transform.translation.x = obj1_obj_pos[i][0]
        t_obj1.transform.translation.y = obj1_obj_pos[i][1]
        t_obj1.transform.translation.z = obj1_obj_pos[i][2]
        t_obj1.transform.rotation.x = 0.0
        t_obj1.transform.rotation.y = 0.0
        t_obj1.transform.rotation.z = 0.0
        t_obj1.transform.rotation.w = 1.0
        br.sendTransform(t_obj1)

        t_obj2 = TransformStamped()
        t_obj2.header.stamp = time_now
        t_obj2.header.frame_id = "world"
        t_obj2.child_frame_id = f"obj_2"
        t_obj2.transform.translation.x = obj2_obj_pos[i][0]
        t_obj2.transform.translation.y = obj2_obj_pos[i][1]
        t_obj2.transform.translation.z = obj2_obj_pos[i][2]
        t_obj2.transform.rotation.x = 0.0
        t_obj2.transform.rotation.y = 0.0
        t_obj2.transform.rotation.z = 0.0
        t_obj2.transform.rotation.w = 1.0
        br.sendTransform(t_obj2)

        t_obj3 = TransformStamped()
        t_obj3.header.stamp = time_now
        t_obj3.header.frame_id = "world"
        t_obj3.child_frame_id = f"obj_3"
        t_obj3.transform.translation.x = obj3_obj_pos[i][0]
        t_obj3.transform.translation.y = obj3_obj_pos[i][1]
        t_obj3.transform.translation.z = obj3_obj_pos[i][2]
        t_obj3.transform.rotation.x = 0.0
        t_obj3.transform.rotation.y = 0.0
        t_obj3.transform.rotation.z = 0.0
        t_obj3.transform.rotation.w = 1.0
        br.sendTransform(t_obj3)
        if visualize:
            PandaViewer.updatePlacements(pin.GeometryType.COLLISION)
            PandaViewer.display(a_actual_np[i,:])
        collisions= collision_callback(a_actual_np[i,:], PANDAmodel, PANDAdata, PANDAcollision_model, 
                          PANDAcollision_data, visualize)
        if collisions:
            current_time = ros_time[i]
            new_collisions_this_frame = []
            
            # Check for new collisions and update timestamps for existing ones
            for collision_pair in collisions:
                # Sort collision pair names for consistent dictionary keys
                sorted_pair = tuple(sorted(collision_pair))
                
                if sorted_pair not in active_collisions:
                    # This is a new collision - register and handle it
                    active_collisions[sorted_pair] = current_time
                    new_collisions_this_frame.append(collision_pair)
                else:
                    # This is an ongoing collision - just update the timestamp
                    active_collisions[sorted_pair] = current_time
            
            # Only add new collisions to the collision_lists
            if new_collisions_this_frame:
                collision_lists.append(new_collisions_this_frame)
            
            # Check for expired collisions (no longer active)
            expired_collisions = []
            for pair, last_time in active_collisions.items():
                # If a collision wasn't updated this frame and hasn't been updated in a while
                if abs(last_time - current_time) > 0.1:  # Small threshold to detect when collision ends
                    expired_collisions.append(pair)
            
            # Remove expired collisions
            for pair in expired_collisions:
                del active_collisions[pair]

        if i == time_indices[j][1]:
            _duration = ros_time[time_indices[j][1]] - ros_time[time_indices[j][0]]
            print("The experiment took ", _duration, " seconds", "with the number of collisions: ", len(collision_lists))
            print("Collision list: ", collision_lists)
            user_input_passed = False
            while not user_input_passed:
                res = input("Press 'c' for add experiment, 'i' for remove experiment" \
                ", 'q' to quit: ")
                print("User input: ", res)
                if res == 'q':
                    user_input_passed = True
                    print("Playback stopped")
                    return
                if res == 'c' or res == 'i':
                    user_input_passed = True
                    if res == 'c':
                        isgameover = input("Is game over? (y/n): ")
                        if isgameover == 'y':
                            print("Game over")
                            time_indices[j] = (time_indices[j][0], time_indices[j][1], "GAMEOVER")
                            trial_idx.append(time_indices[j])
                            print("Interval added: ", time_indices[j])
                        else:
                            print("Not game over")
                            time_indices[j] = (time_indices[j][0], time_indices[j][1], "OKAY")
                            trial_idx.append(time_indices[j])
                            print("Interval added: ", time_indices[j])
                    j -= 1
                    
                    print("Next experiment: ", time_indices[j], "will take ", (time_indices[j][1] - time_indices[j][0])/90, "seconds. j:", j)
                    print(f"Completed Experiments {len(trial_idx)}")
                    if j == -1:
                        print("All experiments are done")
                        new_trial = False
                        finished = True
                        break
                    else:
                        i = time_indices[j][0]
                        print("Continue with experiment. J", j , "I", i)
                        new_trial = True

            if finished:
                break            
        

        rate.sleep()

    with open(trial_idx_path, "a") as f:
        for j_idx, (trial_start, trial_end, status) in enumerate(trial_idx):
            f.write(f"{j_idx},{trial_start},{trial_end},{status}\n")
            print(f"Trial {j_idx}: {trial_start} to {trial_end} with {status}")
    
    rospy.loginfo("Playback complete")

if __name__ == "__main__":
    # Set stdin to non-blocking mode
    sys.stdin = open('/dev/tty')
    os.system('stty -echo')  # Disable terminal echo
    
    try:
        rospy.init_node("csv_tf_broadcaster_dual")

        user = rospy.get_param("~user", "user_42")   # user_11
        config = rospy.get_param("~config", "config_5")  # config_1
        data_dir = rospy.get_param("~data_dir", "/home/sun/CBF_results")
        rate_hz = rospy.get_param("~rate", 100.0)
        print("user: ", user, "config: ", "data_dir: ", data_dir, "rate_hz: ", rate_hz)
        play_tf_dual(user, config, data_dir=data_dir, rate_hz=rate_hz)
    finally:
        os.system('stty echo')  # Re-enable terminal echo