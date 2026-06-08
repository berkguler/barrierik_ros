# barrierik_ros

## Quick Start

1. Start the Docker environment:

```bash
cd barrierik_ros
./docker/run_docker.sh
```

2. Inside the container, source the workspace:

```bash
source devel/setup.bash
```

3. Start ROS TCP Endpoint:

```bash
roslaunch ros_tcp_endpoint endpoint.launch
```

4. In a second terminal inside the same container, start IK:

```bash
roslaunch bik_pkg ik.launch
```

5. Run the [Unity Project](https://github.com/berkguler/barrierik_unity). on the host machine. 

Notes:
- Use ROS-TCP endpoint port `10000` in Unity.
- If you need to rebuild/replace the container:

```bash
./docker/run_docker.sh --build --replace
```

## `ik.launch` Arguments

The launch file is [`src/bik_pkg/launch/ik.launch`](/barrierik_ros/src/bik_pkg/launch/ik.launch).

| Argument | Default | Description | Allowed values |
|---|---|---|---|
| `helper_start_delay` | `5.0` | Reserved helper delay argument (currently not consumed by a launch-prefix in this file). | float (seconds) |
| `main_start_delay` | `10.0` | Delay before starting `IK.py`. | float (seconds) |
| `model` | `$(find franka_description)/urdfs/fer_franka_hand_relative.urdf` | URDF loaded into `my_robot_description`. | URDF file path |
| `sharedautonomy_mode` | `None` | Shared autonomy blending mode used by `Robot_bik`. | `None`, `Arbitration` |
| `solver_mode` | `relaxedik` | IK solver backend/constraint mode. | `relaxedik`, `barrierik`, `barrierik_moving`, `collisionik`, `relaxedik_original` (legacy) |

`sharedautonomy_mode` behavior:
- `None`: robot follows the user target directly.
- `Arbitration`: blends user target and helper target in `robot_class.py`.

`solver_mode` behavior:
- `relaxedik`: baseline JAX IK, based on [RelaxedIK](https://pages.graphics.cs.wisc.edu/relaxed_ik_core/).
- `barrierik`: JAX IK + static-obstacle CBF constraints.
- `barrierik_moving`: JAX IK + moving-obstacle CBF constraints.
- `collisionik`: collision-aware IK solver path, based on [CollisionIK](https://arxiv.org/abs/2102.13187).
- `relaxedik_original`: legacy Rust-backed RelaxedIK path via [`uwgraphics/relaxed_ik_core`](https://github.com/uwgraphics/relaxed_ik_core).

Important note:
- `barrierik_moving` is currently not fully tested. Obstacle velocities are approximated online from position history (finite differences), so behavior may vary with noisy or low-rate obstacle updates.

Legacy note:
- `relaxedik_original` is still handled in `bik_helper.py` (Rust `relaxed_ik_core` path), but it does not run through the current `Robot_bik` CBF/shared-autonomy pipeline above.

Examples:

```bash
# Baseline IK
roslaunch bik_pkg ik.launch solver_mode:=relaxedik sharedautonomy_mode:=None

# Static-obstacle CBF
roslaunch bik_pkg ik.launch solver_mode:=barrierik sharedautonomy_mode:=None

# Moving-obstacle CBF with arbitration
roslaunch bik_pkg ik.launch solver_mode:=barrierik_moving sharedautonomy_mode:=Arbitration
```

## Changelog

### 2026-05-31
- Updated capsule-based CBF and manipulability formulations in `bik_core.py` and runtime wiring in `robot_class.py` / `bik_helper.py`.

**Manipulability Constraint (implemented)**

For the translational Jacobian \(J(q)\) and singular values \(\sigma_{\min}, \sigma_{\max}\):

$$
c_{\mathrm{ks}}(q) =
\max\!\Big(
\sigma_{\text{th}} - \sigma_{\min}(J(q)),
\frac{\sigma_{\max}(J(q))}{\max(\sigma_{\min}(J(q)), \varepsilon)} - \kappa_{\max}
\Big) \le 0
$$

with \(\sigma_{\text{th}} = 10^{-5}\), \(\kappa_{\max}=10^4\), \(\varepsilon=10^{-10}\).

**Static-Obstacle CBF (implemented)**

For obstacle \(i\) and nearest robot capsule/contact point:

$$
h_i(q) = \phi_i(q) - d_{\text{safe}}
$$

$$
\dot{h}_i(q,\dot{q}) =
\nabla \phi_i(q)^\top J_{\ell_i}(q)\,\dot{q}
$$

Class-\(\mathcal{K}\) function:

$$
\alpha(h) = \gamma h + \beta h^3
$$

with \(\gamma=90,\ \beta=220\), and CBF inequality written for NLopt as:

$$
c_i(q,\dot{q}) = -\big(\dot{h}_i + \alpha(h_i)\big) \le 0
$$

Obstacle aggregation uses LogSumExp (\(\tau = 500\)):

$$
c_{\mathrm{cbf}} =
\max_i c_i + \frac{1}{\tau}\log\!\sum_i
\exp\!\left(\tau(c_i-\max_i c_i)\right)
$$

**Moving-Obstacle CBF (implemented via `barrierik_moving`)**

For obstacle velocity \(\dot{p}_{\text{obs},i}\):

$$
\dot{h}_i(q,\dot{q}) =
\nabla \phi_i(q)^\top
\left(J_{\ell_i}(q)\dot{q} - \dot{p}_{\text{obs},i}\right)
$$

$$
\alpha_{\text{mov}}(h) = \gamma_{\text{mov}} h + \beta_{\text{mov}} h^3
$$

with \(\gamma_{\text{mov}}=200,\ \beta_{\text{mov}}=50\), and:

$$
c_i^{\text{mov}}(q,\dot{q}) =
-\big(\dot{h}_i + \alpha_{\text{mov}}(h_i)\big) \le 0
$$

## Citation

If you use this repository in research, please cite:

```bibtex
@misc{guler2026safetyawaresharedautonomyframework,
      title={A Safety-Aware Shared Autonomy Framework with BarrierIK Using Control Barrier Functions}, 
      author={Berk Guler and Kay Pompetzki and Yuanzheng Sun and Simon Manschitz and Jan Peters},
      year={2026},
      eprint={2603.01705},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2603.01705}, 
}
```
