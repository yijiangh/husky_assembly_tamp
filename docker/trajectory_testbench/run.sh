#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yml"
SERVICE="trajectory-testbench"
CONTAINER_WORKDIR="/workspace/husky-assembly-teleop/external/husky_assembly_tamp"
HOST_PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
HOST_XAUTH_FILE="${HOST_XAUTH_FILE:-/tmp/husky-trajectory-testbench.xauth}"
HOST_OS="$(uname -s)"

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
    export DISPLAY="${DISPLAY:-host.docker.internal:0}"

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
    prepare_env
    compose up -d --build
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
        echo "Debug on localhost:5678: $0 debug -- --stage 3"
        ;;
    down)
        prepare_env
        compose down
        ;;
    shell)
        ensure_up
        compose exec -w "$CONTAINER_WORKDIR" "$SERVICE" bash
        ;;
    testbench)
        ensure_up
        compose exec -w "$CONTAINER_WORKDIR" "$SERVICE" \
            python -m husky_assembly_tamp.motion_planner.trajectory_testbench "$@"
        ;;
    debug)
        ensure_up
        echo "Waiting for a debugger on localhost:5678 ..."
        compose exec -w "$CONTAINER_WORKDIR" "$SERVICE" \
            python -m debugpy --listen 0.0.0.0:5678 --wait-for-client \
            -m husky_assembly_tamp.motion_planner.trajectory_testbench "$@"
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
        echo "Usage: $0 [up|down|shell|testbench|debug|logs|rebuild] [-- <trajectory_testbench args>]" >&2
        exit 1
        ;;
esac
