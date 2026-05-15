#! /usr/bin/env python3

import sys
import pinocchio as pin
import hppfcl

import yaml
import xml.etree.ElementTree as ET
import os


import jax
import jaxlie

from jax import jacfwd, config, grad,jit,vmap
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')
config.update("jax_debug_nans", False) #Warning

import jaxsim

from jaxsim import logging
logging.set_logging_level(logging.LoggingLevel.ERROR)
logging.info(f"Running on {jax.devices()}")
import jaxsim.api as js

from jaxsim import rbda
from pinocchio.visualize import MeshcatVisualizer
import jax.numpy as jnp
import numpy as np
import rospkg

import bik_pkg.bik_collision as bik_collision

rospack = rospkg.RosPack()
package_path = rospack.get_path('bik_pkg')

def fix_urdf(urdf_file):
    with open(urdf_file, 'r') as file:
        filedata = file.read()
    new_resource_path = os.path.join(package_path, "resources")
    filedata = filedata.replace('resources',new_resource_path)
    with open(os.path.join(package_path, "assets/urdf_file.urdf"), 'w') as file:
        file.write(filedata)
    #print("Fixed URDF file saved at assets/urdf_file.urdf")
    return os.path.join(package_path, "assets/urdf_file.urdf")

def remove_link_joints(urdf_file, word_to_remove = "finger"):
    tree = ET.parse(urdf_file)
    root = tree.getroot()
    # Find and remove links containing "finger" in their name
    for link in root.findall(".//link"):
        name = link.attrib.get("name", "")
        if word_to_remove in name:
            print("Removing link: ", name)
            root.remove(link)

    # Find and remove joints containing "finger" in their name
    joints = []
    for joint in root.findall(".//joint"):
        name = joint.attrib.get("name", "")
        if word_to_remove in name:
            root.remove(joint)
            print("Removing joint: ", name)
                
    # Save the modified URDF
    tree.write(os.path.join(package_path, "assets/urdf_file_no_" + word_to_remove + ".urdf"))
    #print("Fixed URDF file saved at assets/urdf_file_no_" + word_to_remove + ".urdf")
    return os.path.join(package_path, "assets/urdf_file_no_" + word_to_remove + ".urdf"), joints

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

def convert_format(num_str):
    parts = num_str.split(".")
    if len(parts) == 2 and len(parts[1]) == 1:  # If single decimal digit
        return float(f"{parts[0]}.0{parts[1]}")
    return float(num_str)  # Return as is if already correct
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


def generate_capsule_name(self, base_name: str, existing_names: list) -> str:
        """Generates a unique capsule name for a geometry object.

        Args:
            base_name (str): The base name of the geometry object.
            existing_names (list): List of names already assigned to capsules.

        Returns:
            str: Unique capsule name.
        """
        i = 0
        while f"{base_name}_capsule_{i}" in existing_names:
            i += 1
        return f"{base_name}_capsule_{i}"


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
                    try:
                        # Compute the local AABB
                        geometry.computeLocalAABB()
                        aabb = geometry.aabb_local
                        center = (aabb.min_ + aabb.max_) / 2.0
                        radius = np.linalg.norm(aabb.max_ - aabb.min_) / 2.0
                        
                        # Adjust placement to position sphere at AABB center
                        center_placement = pin.SE3.Identity()
                        center_placement.translation = center
                        adjusted_placement = placement * center_placement
                        
                        sphere_geom = hppfcl.Sphere(radius)
                        
                        sphere_obj = pin.GeometryObject(
                            name=f"{geom_object.name}_sphere",
                            parent_frame=parent_frame,
                            parent_joint=parent_joint,
                            collision_geometry=sphere_geom,
                            placement=adjusted_placement
                        )
                        sphere_obj.meshColor = np.array([1.0, 0.7, 0.7, 0.7])  # Red-ish transparent
                        collision_model.addGeometryObject(sphere_obj)
                        list_replaced_geometries.append(geom_object.name)
                        
                    except Exception as e:
                        print(f"Could not create bounding sphere for {geom_object.name}: {str(e)}")
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

def load_urdf_jax(modelPath,_fix_urdf = False):
    if _fix_urdf:
        modelPath = fix_urdf(modelPath)
    modelPath, joints = fix_compatibility_for_sdf(modelPath)
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

    return model, data #, integrator, integrator_state

def createViewer(model, collision_model, visual_model, _openURL = False):
    viewer = MeshcatVisualizer(model, collision_model, visual_model)
    viewer.initViewer(open=_openURL, loadModel=True)
    q_neutral = pin.neutral(model)
    viewer.display(q_neutral)
    return viewer

def init(yaml_file = "franka_description.yaml", open_viewer = True, disable_pin_models = False, disable_viewer = False):
    with open(yaml_file, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    package_path = rospack.get_path(data_loaded["PACKAGE_NAME"])
    PANDA_URDF_PATH = os.path.join(package_path, data_loaded["URDF_PATH"])
    PANDA_MESH_DIR = os.path.join(package_path, data_loaded["MESH_DIR"])
    if not os.path.exists(PANDA_URDF_PATH):
        raise FileNotFoundError(f"File {PANDA_URDF_PATH} does not exist")
    if not os.path.exists(PANDA_MESH_DIR):
        raise FileNotFoundError(f"Directory {PANDA_MESH_DIR} does not exist")


    if not disable_pin_models:
        PANDAmodel, PANDAcollision_model, PANDAvisual_model, PANDAdata, PANDAcollision_data, list_panda_capsules_jax = loadRobot(PANDA_URDF_PATH, PANDA_MESH_DIR, _fix_urdf = True)
    else: 
        PANDAmodel = None
        PANDAcollision_model = None
        PANDAvisual_model = None
        PANDAdata = None
        PANDAcollision_data = None

    if not disable_viewer:
        PandaViewer = createViewer(PANDAmodel, PANDAcollision_model, PANDAvisual_model, _openURL = open_viewer)
    else:
        PandaViewer = None
    robotmodel, robotdata = load_urdf_jax(PANDA_URDF_PATH,_fix_urdf = True)
    
    
    return PANDAmodel, PANDAcollision_model, PANDAvisual_model, PANDAdata, PANDAcollision_data, PandaViewer, robotmodel, robotdata, list_panda_capsules_jax

