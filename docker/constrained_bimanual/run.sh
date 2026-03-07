#!/usr/bin/env bash
# Start the constrained bimanual planner Docker container.
# Usage: ./run.sh [up|down|jupyter]
#   up      - Start container in background (default)
#   down    - Stop container
#   jupyter - Start container and launch Jupyter notebook

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

ACTION="${1:-up}"

case "$ACTION" in
    up)
        echo "Starting constrained bimanual planner container..."
        docker compose up -d
        echo "Container running. Use 'docker exec -it constrained-bimanual-planner bash' to enter."
        echo "Or run: ./run.sh jupyter"
        ;;
    down)
        echo "Stopping container..."
        docker compose down
        ;;
    jupyter)
        echo "Starting container with Jupyter..."
        docker compose up -d
        docker exec -it constrained-bimanual-planner bash -c \
            "cd /opt/proj/notebooks && jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root"
        ;;
    *)
        echo "Usage: $0 [up|down|jupyter]"
        exit 1
        ;;
esac
