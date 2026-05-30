# Barrierik ROS Noetic Docker

This folder contains the ROS Noetic Docker setup for this repository.

Default environment names:
- image: `barrierik_docker`
- container: `barrierik_docker`
- workspace inside container: `/root/barrierik_docker`

## Build

```bash
./run_docker.sh --build
```

Rebuild and replace a running container:

```bash
./run_docker.sh --build --replace
```

## Run

```bash
./run_docker.sh
```

If the container is already running, attach with:

```bash
docker exec -it barrierik_docker bash
```

## ARM64 (Mac Apple Silicon / Linux ARM64)

This project is intended to run as `linux/amd64`. On ARM64 hosts, the runner automatically uses `--platform linux/amd64`.

One-time emulator setup (if needed):

```bash
docker run --privileged --rm tonistiigi/binfmt --install amd64
```

Or let the script do it:

```bash
./run_docker.sh --setup-binfmt --build
```

Explicit platform example:

```bash
./run_docker.sh --build --platform linux/amd64
```

## Unity Host Communication (macOS)

On macOS, the runner uses bridge networking and publishes ports to the host by default:

- `10000:10000`
- `30000:30000`

So Unity can connect to `127.0.0.1` using whichever ROS-TCP port you launch.

If you launch ROS-TCP-Endpoint on a different port, override mapping:

```bash
NETWORK_MODE=bridge PORT_MAPS=40000:40000 ./run_docker.sh
```

Then launch endpoint in the container with matching port:

```bash
roslaunch ros_tcp_endpoint endpoint.launch tcp_ip:=0.0.0.0 tcp_port:=40000
```

## Run A Command Directly

```bash
./run_docker.sh -- roslaunch ros_tcp_endpoint endpoint.launch
```

## Inside The Container

```bash
source /opt/ros/noetic/setup.bash
source /root/barrierik_docker/devel/setup.bash
export ROS_PACKAGE_PATH=/root/barrierik_docker/src:${ROS_PACKAGE_PATH:-}
```

## Typical Workflow

Terminal 1:

```bash
roscore
```

Terminal 2:

```bash
roslaunch ros_tcp_endpoint endpoint.launch
```

Terminal 3:

```bash
roslaunch bik_pkg ik.launch solver_mode:=relaxedik
```

## GUI Notes

The runner enables local X11 with:

```bash
xhost +local:root
```

If GUI windows still do not open, run that command on the host and restart the container.
