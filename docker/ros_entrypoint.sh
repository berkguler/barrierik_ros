#!/bin/bash
set -e

if [ -n "${VIRTUAL_ENV}" ] && [ -f "${VIRTUAL_ENV}/bin/activate" ]; then
    source "${VIRTUAL_ENV}/bin/activate"
fi

source "/opt/ros/${ROS_DISTRO:-noetic}/setup.bash"

if [ -n "${CATKIN_WS}" ] && [ -f "${CATKIN_WS}/devel/setup.bash" ]; then
    source "${CATKIN_WS}/devel/setup.bash"
fi

if [ -n "${CATKIN_WS}" ] && [ -d "${CATKIN_WS}/src" ]; then
    export ROS_PACKAGE_PATH="${CATKIN_WS}/src:${ROS_PACKAGE_PATH:-}"
fi

exec "$@"
