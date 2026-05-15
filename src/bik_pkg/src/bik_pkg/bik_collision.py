#! /usr/bin/env python3
from typing import Tuple
from jax import jacfwd, config, grad,jit,vmap
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')
config.update("jax_debug_nans", False) #Warning
import jax
import jax.numpy as jnp
from jax import jit
from dpax.endpoints import proximity
import matplotlib.pyplot as plt

import numpy as np
def capsule_dpax_comp(a:jnp.ndarray,b:jnp.ndarray,R:float):
    """
            /\
           /  \     Z
          /    \    ^
         |   b  |   |
         |      |   |
         |   .  |   ---> X
         |      |
         |   a  |
          \    /
           \  /
            \/ 
    
        |<--R-->|
    """

    ab = b - a
    length = jnp.linalg.norm(ab)

    if length < 1e-6:
        direction = jnp.array([0., 0., 1.])
    else:
        direction = ab / length

    z_axis = jnp.array([0., 0., 1.])
    v = jnp.cross(z_axis, direction)
    c = jnp.dot(z_axis, direction)
    s = jnp.linalg.norm(v)

    if s < 1e-6:
        R_mat = jnp.eye(3) if c > 0 else jnp.diag(jnp.array([1., -1., -1.]))
    else:
        vx = jnp.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
        R_mat = jnp.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))

    center = (a + b) / 2
    T = jnp.eye(4)
    T = T.at[:3, :3].set(R_mat)
    T = T.at[:3, 3].set(center)

    return T, length, R

def capsule_ab_from_T(T: jnp.ndarray, length: float):
    """
    Given a transformation matrix T and length, reconstruct the capsule endpoints a and b.
    
    Args:
        T (jnp.ndarray): 4x4 homogeneous transform
        length (float): Length between the two hemispheres (not including them)

    Returns:
        a (jnp.ndarray): Start point (3,)
        b (jnp.ndarray): End point (3,)
    """
    z_axis = T[:3, 2]  # Z-axis of the rotation
    center = T[:3, 3]  # Center point

    half = (length / 2.0)
    a = center - half * z_axis
    b = center + half * z_axis

    return a, b

def capsule_ab_from_CL(C:jnp.array, L:jnp.ndarray):
    """
    Given a center point C, length L, and radius R, reconstruct the capsule endpoints a and b.
    
    Args:
        C (jnp.ndarray): Center point (3,)
        L (float): Length between the two hemispheres (not including them)
        R (float): Radius of the capsule

    Returns:
        a (jnp.ndarray): Start point (3,)
        b (jnp.ndarray): End point (3,)
    """
    z_axis = jnp.array([0., 0., 1.])  # Assuming Z-axis is the direction of the capsule
    half = (L / 2.0) * z_axis
    a = C - half
    b = C + half

    return a, b


def box2capsule(center:jnp.ndarray, dimensions:jnp.ndarray, keep_dimensions = False) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float, float, jnp.ndarray]:
    """ Converts a box defined by its center and dimensions into a capsule representation.
    Args:
        center (jnp.ndarray): Center of the box (3,)
        dimensions (jnp.ndarray): Dimensions of the box (3,)
    Returns:
        a (jnp.ndarray): Start point of the capsule (3,)
        b (jnp.ndarray): End point of the capsule (3,)
        C (jnp.ndarray): Center point of the capsule (3,)
        L (float): Length of the capsule
        R (float): Radius of the capsule
    """

    if not keep_dimensions:
        max_ind_dim= jnp.argmax(dimensions)
        length = dimensions[max_ind_dim]
        other_dims = jnp.delete(dimensions, max_ind_dim)
        radius = jnp.mean(other_dims) / 2

        a,b = capsule_ab_from_CL(center, length)
        T,L,R = capsule_dpax_comp(a, b, radius)
        C = (a + b) / 2
        obstacle = (a,b,C,L,R,T)

    else:
        max_ind_dim = -1 #Assuming the last dimension is the length
        length = dimensions[max_ind_dim]
        other_dims = jnp.delete(dimensions, max_ind_dim)
        radius = jnp.mean(other_dims) / 2
        a, b = capsule_ab_from_CL(center, length)
        T, L, R = capsule_dpax_comp(a, b, radius)
        C = (a + b) / 2
        obstacle = (a, b, C, L, R, T)
    return a, b, C, L, R, T

    
    return obstacle
        
def render_capsule(ax, T, height, radius, resolution=30, color='blue'):
    """
    Renders a capsule in 3D using matplotlib.
    """
    # Create a cylinder along Z
    z = np.linspace(-height / 2, height / 2, 50)
    theta = np.linspace(0, 2 * np.pi, resolution)
    theta_grid, z_grid = np.meshgrid(theta, z)

    x_grid = radius * np.cos(theta_grid)
    y_grid = radius * np.sin(theta_grid)

    # Stack into 3D points and rotate + translate
    points = np.stack([x_grid, y_grid, z_grid], axis=-1)
    points_flat = points.reshape(-1, 3).T
    points_transformed = T[:3, :3] @ points_flat + T[:3, 3:4]
    x, y, z = points_transformed.reshape(3, *x_grid.shape)

    ax.plot_surface(x, y, z, color=color, alpha=0.5, linewidth=0)

    # Hemispheres
    u = np.linspace(0, np.pi, resolution)
    v = np.linspace(0, 2 * np.pi, resolution)
    u_grid, v_grid = np.meshgrid(u, v)

    for sign in [-1, 1]:
        xh = radius * np.sin(u_grid) * np.cos(v_grid)
        yh = radius * np.sin(u_grid) * np.sin(v_grid)
        zh = radius * np.cos(u_grid)

        zh = zh + sign * height / 2
        pts = np.stack([xh, yh, zh], axis=-1)
        pts_flat = pts.reshape(-1, 3).T
        pts_trans = T[:3, :3] @ pts_flat + T[:3, 3:4]
        xh, yh, zh = pts_trans.reshape(3, *xh.shape)

        ax.plot_surface(xh, yh, zh, color=color, alpha=0.5, linewidth=0)