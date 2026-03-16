#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yml"
SERVICE="trajectory-testbench"
CONTAINER_WORKDIR="/workspace/husky-assembly-teleop/external/husky_assembly_tamp"
VIRTUAL_DESKTOP_SCRIPT="/workspace/husky-assembly-teleop/external/husky_assembly_tamp/docker/trajectory_testbench/start_virtual_desktop.sh"
VIRTUAL_DISPLAY="${VIRTUAL_DISPLAY:-:99}"
NOVNC_URL="http://localhost:6080/vnc_lite.html?autoconnect=1&resize=remote&host=localhost&port=6080&path=websockify"
HOST_PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
HOST_XAUTH_FILE="${HOST_XAUTH_FILE:-/tmp/husky-trajectory-testbench.xauth}"
HOST_OS="$(uname -s)"
HEADLESS="${HUSKY_DOCKER_HEADLESS:-0}"

compose_files() {
    case "$HOST_OS" in
        Darwin)
            printf '%s\n' "$BASE_COMPOSE_FILE" "$SCRIPT_DIR/docker-compose.mac.yml"
            ;;
        Linux)
            printf '%s\n' "$BASE_COMPOSE_FILE" "$SCRIPT_DIR/docker-compose.linux.yml"
            ;;
        *)
            echo "Unsupported host OS: $HOST_OS" >&2
            exit 1
            ;;
    esac
}

prepare_linux_env() {
    if [ "$HEADLESS" = "1" ]; then
        return
    fi

    export DISPLAY="${DISPLAY:-:0}"
    export HOST_XAUTH_FILE

    mkdir -p "$(dirname "$HOST_XAUTH_FILE")"
    touch "$HOST_XAUTH_FILE"

    if [ ! -d /tmp/.X11-unix ]; then
        echo "Warning: /tmp/.X11-unix is missing on the host. PyBullet GUI may not open." >&2
    fi

    if command -v xauth >/dev/null 2>&1; then
        xauth nlist "$DISPLAY" 2>/dev/null | sed -e 's/^..../ffff/' | xauth -f "$HOST_XAUTH_FILE" nmerge - >/dev/null 2>&1 || true
    fi
}

prepare_mac_env() {
    if [ "$HEADLESS" = "1" ]; then
        return
    fi

    # On macOS the container must talk to XQuartz over TCP via the host alias.
    # Do not inherit a host shell DISPLAY like :0 or a local launchd socket path.
    export DISPLAY="${HUSKY_DOCKER_DISPLAY:-host.docker.internal:0}"

    if [ ! -d /Applications/Utilities/XQuartz.app ] && [ ! -d /Applications/XQuartz.app ]; then
        echo "XQuartz is not installed." >&2
        echo "Install it on the host with: brew install --cask xquartz" >&2
        echo "Homebrew will prompt for your macOS admin password because the pkg installer writes system files." >&2
        exit 1
    fi

    if ! pgrep -x XQuartz >/dev/null 2>&1; then
        echo "Warning: XQuartz is not running. Start it with: open -a XQuartz" >&2
    fi
}

prepare_env() {
    export HOST_GID="$(id -g)"
    export HOST_PROJECT_ROOT
    export HOST_UID="$(id -u)"

    case "$HOST_OS" in
        Darwin)
            prepare_mac_env
            ;;
        Linux)
            prepare_linux_env
            ;;
        *)
            echo "Unsupported host OS: $HOST_OS" >&2
            exit 1
            ;;
    esac
}

compose() {
    local files=()
    while IFS= read -r file; do
        files+=(-f "$file")
    done < <(compose_files)
    docker compose "${files[@]}" "$@"
}

ensure_up() {
    local mode="${1:-host-gui}"
    local saved_headless="$HEADLESS"
    if [ "$mode" = "headless" ]; then
        HEADLESS=1
    fi
    prepare_env
    HEADLESS="$saved_headless"
    compose up -d --build
}

ensure_virtual_desktop() {
    ensure_up headless
    compose exec -w "$CONTAINER_WORKDIR" "$SERVICE" \
        bash "$VIRTUAL_DESKTOP_SCRIPT" start
    echo "noVNC desktop: $NOVNC_URL"
}

ACTION="${1:-up}"
if [ "$#" -gt 0 ]; then
    shift
fi
if [ "${1:-}" = "--" ]; then
    shift
fi

case "$ACTION" in
    up)
        ensure_up
        echo "Container is running."
        echo "Shell: $0 shell"
        echo "Run testbench: $0 testbench -- --stage 3"
        echo "Run Stage 1: $0 stage1"
        echo "Start browser desktop: $0 desktop-up"
        echo "Run Stage 1 in noVNC desktop: $0 stage1-vnc"
        echo "Debug on localhost:5678: $0 debug -- --stage 3"
        echo "Debug Stage 1 on localhost:5678: $0 debug-stage1"
        echo "Headless example: HUSKY_DOCKER_HEADLESS=1 $0 stage1 -- --no-gui"
        ;;
    down)
        prepare_env
        compose down
        ;;
    shell)
        ensure_up
        compose exec -w "$CONTAINER_WORKDIR" "$SERVICE" bash
        ;;
    desktop-up)
        ensure_virtual_desktop
        ;;
    desktop-down)
        ensure_up headless
        compose exec -w "$CONTAINER_WORKDIR" "$SERVICE" \
            bash "$VIRTUAL_DESKTOP_SCRIPT" stop
        ;;
    desktop-status)
        ensure_up headless
        compose exec -w "$CONTAINER_WORKDIR" "$SERVICE" \
            bash "$VIRTUAL_DESKTOP_SCRIPT" status
        ;;
    testbench)
        ensure_up
        compose exec -w "$CONTAINER_WORKDIR" "$SERVICE" \
            python -B -m husky_assembly_tamp.motion_planner.trajectory_testbench "$@"
        ;;
    testbench-vnc)
        ensure_virtual_desktop
        compose exec -w "$CONTAINER_WORKDIR" "$SERVICE" \
            env DISPLAY="$VIRTUAL_DISPLAY" \
            python -B -m husky_assembly_tamp.motion_planner.trajectory_testbench "$@"
        ;;
    stage1)
        ensure_up
        compose exec -w "$CONTAINER_WORKDIR" "$SERVICE" \
            python -B -m husky_assembly_tamp.motion_planner.stage1.minimal_rrt "$@"
        ;;
    stage1-vnc)
        ensure_virtual_desktop
        compose exec -w "$CONTAINER_WORKDIR" "$SERVICE" \
            env DISPLAY="$VIRTUAL_DISPLAY" \
            python -B -m husky_assembly_tamp.motion_planner.stage1.minimal_rrt "$@"
        ;;
    debug)
        ensure_up
        echo "Waiting for a debugger on localhost:5678 ..."
        compose exec -w "$CONTAINER_WORKDIR" "$SERVICE" \
            python -B -m debugpy --listen 0.0.0.0:5678 --wait-for-client \
            -m husky_assembly_tamp.motion_planner.trajectory_testbench "$@"
        ;;
    debug-stage1)
        ensure_up
        echo "Waiting for a debugger on localhost:5678 ..."
        compose exec -w "$CONTAINER_WORKDIR" "$SERVICE" \
            python -B -m debugpy --listen 0.0.0.0:5678 --wait-for-client \
            -m husky_assembly_tamp.motion_planner.stage1.minimal_rrt "$@"
        ;;
    logs)
        prepare_env
        compose logs -f "$SERVICE"
        ;;
    rebuild)
        prepare_env
        compose build --no-cache
        compose up -d
        ;;
    *)
        echo "Usage: $0 [up|down|shell|desktop-up|desktop-down|desktop-status|testbench|testbench-vnc|stage1|stage1-vnc|debug|debug-stage1|logs|rebuild] [-- <module args>]" >&2
        exit 1
        ;;
esac
