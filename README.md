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

5. Run the Unity project on the host machine.

Notes:
- Use ROS-TCP endpoint port `10000` in Unity.
- If you need to rebuild/replace the container:

```bash
./docker/run_docker.sh --build --replace
```
