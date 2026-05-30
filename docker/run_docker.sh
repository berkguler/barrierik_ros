#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-barrierik_docker}"
CONTAINER_NAME="${CONTAINER_NAME:-barrierik_docker}"
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_IN_CONTAINER="${WORKSPACE_IN_CONTAINER:-/root/barrierik_docker}"
PLATFORM="${PLATFORM:-}"
NETWORK_MODE="${NETWORK_MODE:-auto}"
PORT_MAPS="${PORT_MAPS:-}"
DOCKER="${DOCKER:-docker}"

usage() {
    echo "Usage: $0 [--build] [--replace] [--setup-binfmt] [--platform PLATFORM] [--name NAME] [--image IMAGE] [-- CMD...]"
    echo
    echo "Env vars:"
    echo "  NETWORK_MODE=auto|host|bridge   (default: auto)"
    echo "  PORT_MAPS=HOST:CONT[,HOST:CONT] (used in bridge mode)"
    echo
    echo "Examples:"
    echo "  $0"
    echo "  $0 --build"
    echo "  $0 --build --platform linux/amd64"
    echo "  $0 --setup-binfmt --build"
    echo "  NETWORK_MODE=bridge PORT_MAPS=30000:30000 $0"
    echo "  $0 --replace"
    echo "  $0 -- roslaunch ros_tcp_endpoint endpoint.launch"
}

BUILD_IMAGE=0
REPLACE_CONTAINER=0
SETUP_BINFMT=0
CMD_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build)
            BUILD_IMAGE=1
            shift
            ;;
        --replace)
            REPLACE_CONTAINER=1
            shift
            ;;
        --setup-binfmt)
            SETUP_BINFMT=1
            shift
            ;;
        --platform)
            PLATFORM="$2"
            shift 2
            ;;
        --name)
            CONTAINER_NAME="$2"
            shift 2
            ;;
        --image)
            IMAGE_NAME="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        --)
            shift
            CMD_ARGS=("$@")
            break
            ;;
        *)
            CMD_ARGS=("$@")
            break
            ;;
    esac
done

# On ARM hosts (e.g. Apple Silicon), default to AMD64 containers to match this
# ROS Noetic setup and avoid architecture-specific package mismatches.
if [[ -z "${PLATFORM}" ]]; then
    HOST_ARCH="$(uname -m)"
    if [[ "${HOST_ARCH}" == "arm64" || "${HOST_ARCH}" == "aarch64" ]]; then
        PLATFORM="linux/amd64"
    fi
fi

HOST_OS="$(uname -s)"
if [[ "${NETWORK_MODE}" == "auto" ]]; then
    if [[ "${HOST_OS}" == "Darwin" ]]; then
        # Docker Desktop on macOS runs containers in a VM. Using explicit TCP
        # publishing is more reliable for host<->container Unity communication.
        NETWORK_MODE="bridge"
    else
        NETWORK_MODE="host"
    fi
fi

if [[ "${SETUP_BINFMT}" -eq 1 ]]; then
    BINFMT_TARGET="all"
    if [[ "${PLATFORM}" == "linux/amd64" ]]; then
        BINFMT_TARGET="amd64"
    fi
    "${DOCKER}" run --privileged --rm tonistiigi/binfmt --install "${BINFMT_TARGET}"
fi

if [[ "${BUILD_IMAGE}" -eq 1 ]]; then
    if [[ -n "${PLATFORM}" ]] && "${DOCKER}" buildx version >/dev/null 2>&1; then
        "${DOCKER}" buildx build \
            --load \
            --platform "${PLATFORM}" \
            -f "${WORKSPACE_DIR}/docker/Dockerfile.noetic" \
            -t "${IMAGE_NAME}" \
            "${WORKSPACE_DIR}"
    else
        BUILD_CMD=("${DOCKER}" build)
        if [[ -n "${PLATFORM}" ]]; then
            BUILD_CMD+=(--platform "${PLATFORM}")
        fi
        BUILD_CMD+=(-f "${WORKSPACE_DIR}/docker/Dockerfile.noetic" -t "${IMAGE_NAME}" "${WORKSPACE_DIR}")
        "${BUILD_CMD[@]}"
    fi
fi

if ! "${DOCKER}" info >/dev/null 2>&1; then
    echo "Cannot access the Docker daemon as user '${USER}'."
    echo "Ask an admin to add this user to the docker group, or run this script from an account that can run Docker."
    exit 1
fi

if ! "${DOCKER}" image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
    echo "Docker image '${IMAGE_NAME}' was not found."
    echo "If it was built from another Linux user with rootless Docker, this user cannot see that user's image store."
    echo "Build it first with: $0 --build"
    exit 1
fi

EXISTING_CONTAINER="$("${DOCKER}" ps -aq -f "name=^/${CONTAINER_NAME}$")"
if [[ -n "${EXISTING_CONTAINER}" ]]; then
    if "${DOCKER}" ps -q -f "name=^/${CONTAINER_NAME}$" | grep -q .; then
        if [[ "${REPLACE_CONTAINER}" -eq 1 ]]; then
            "${DOCKER}" rm -f "${CONTAINER_NAME}" >/dev/null
        else
            echo "Container '${CONTAINER_NAME}' is already running."
            echo "Attach with: ${DOCKER} exec -it ${CONTAINER_NAME} bash"
            echo "Or replace it with: $0 --replace"
            exit 1
        fi
    else
        "${DOCKER}" rm "${CONTAINER_NAME}" >/dev/null
    fi
fi

DISPLAY_VALUE="${DISPLAY:-:0}"
XAUTH_ARGS=()
if [[ -f "${HOME}/.Xauthority" ]]; then
    XAUTH_ARGS=(-e XAUTHORITY=/root/.Xauthority -v "${HOME}/.Xauthority:/root/.Xauthority:ro")
fi

if command -v xhost >/dev/null 2>&1; then
    xhost +local:root >/dev/null
fi

DOCKER_ARGS=(
    --rm
    -it
    --name "${CONTAINER_NAME}"
    -e DISPLAY="${DISPLAY_VALUE}"
    -e QT_X11_NO_MITSHM=1
    -e LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw
    -v "${WORKSPACE_DIR}/csv_logs:${WORKSPACE_IN_CONTAINER}/csv_logs:rw"
)

case "${NETWORK_MODE}" in
    host)
        DOCKER_ARGS+=(--network host)
        ;;
    bridge)
        DOCKER_ARGS+=(--network bridge)
        if [[ -z "${PORT_MAPS}" ]]; then
            # Keep both common ROS-TCP endpoint ports available for Unity.
            PORT_MAPS="10000:10000,30000:30000"
        fi
        IFS=',' read -r -a PORT_MAP_ARRAY <<< "${PORT_MAPS}"
        for mapping in "${PORT_MAP_ARRAY[@]}"; do
            if [[ -n "${mapping}" ]]; then
                DOCKER_ARGS+=(-p "${mapping}")
            fi
        done
        ;;
    *)
        echo "Invalid NETWORK_MODE: ${NETWORK_MODE} (expected: auto|host|bridge)"
        exit 1
        ;;
esac

if [[ -n "${PLATFORM}" ]]; then
    DOCKER_ARGS+=(--platform "${PLATFORM}")
fi

if [[ -d /dev/dri ]]; then
    DOCKER_ARGS+=(--device /dev/dri)
fi

RUN_CMD=("${DOCKER}" run "${DOCKER_ARGS[@]}" "${XAUTH_ARGS[@]}" "${IMAGE_NAME}")
if [[ ${#CMD_ARGS[@]} -gt 0 ]]; then
    RUN_CMD+=("${CMD_ARGS[@]}")
fi

"${RUN_CMD[@]}"
