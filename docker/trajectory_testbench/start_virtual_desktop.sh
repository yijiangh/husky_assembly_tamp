#!/usr/bin/env bash

set -euo pipefail

ACTION="${1:-start}"
DISPLAY_NUM="${DISPLAY_NUM:-:99}"
XVFB_GEOMETRY="${XVFB_GEOMETRY:-1600x1000x24}"
NOVNC_PORT="${NOVNC_PORT:-6080}"
VNC_PORT="${VNC_PORT:-5900}"
RUNTIME_DIR="${HOME:-/tmp/husky-home}/virtual-desktop"
NOVNC_URL="http://localhost:${NOVNC_PORT}/vnc_lite.html?autoconnect=1&resize=remote&host=localhost&port=${NOVNC_PORT}&path=websockify"

mkdir -p "$RUNTIME_DIR"

pid_file() {
    printf '%s/%s.pid\n' "$RUNTIME_DIR" "$1"
}

log_file() {
    printf '%s/%s.log\n' "$RUNTIME_DIR" "$1"
}

is_running() {
    local pidfile="$1"
    if [ ! -f "$pidfile" ]; then
        return 1
    fi
    local pid
    pid="$(cat "$pidfile")"
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

start_process() {
    local name="$1"
    shift
    local pidfile
    pidfile="$(pid_file "$name")"
    local logfile
    logfile="$(log_file "$name")"

    if is_running "$pidfile"; then
        echo "$name is already running (pid $(cat "$pidfile"))."
        return 0
    fi

    nohup "$@" >"$logfile" 2>&1 &
    local pid=$!
    echo "$pid" >"$pidfile"
    sleep 1
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "Failed to start $name. Log: $logfile" >&2
        tail -n 50 "$logfile" >&2 || true
        exit 1
    fi
    echo "Started $name (pid $pid)."
}

stop_process() {
    local name="$1"
    local pidfile
    pidfile="$(pid_file "$name")"

    if ! is_running "$pidfile"; then
        rm -f "$pidfile"
        return 0
    fi

    local pid
    pid="$(cat "$pidfile")"
    kill "$pid" 2>/dev/null || true
    sleep 1
    if kill -0 "$pid" 2>/dev/null; then
        kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$pidfile"
    echo "Stopped $name."
}

print_status() {
    local name
    for name in xvfb fluxbox x11vnc novnc; do
        local pidfile
        pidfile="$(pid_file "$name")"
        if is_running "$pidfile"; then
            echo "$name: running (pid $(cat "$pidfile"))"
        else
            echo "$name: stopped"
        fi
    done
    echo "noVNC URL: ${NOVNC_URL}"
}

start_all() {
    start_process xvfb \
        Xvfb "$DISPLAY_NUM" -screen 0 "$XVFB_GEOMETRY" -ac +extension GLX +render -noreset
    start_process fluxbox \
        env DISPLAY="$DISPLAY_NUM" fluxbox
    start_process x11vnc \
        env DISPLAY="$DISPLAY_NUM" x11vnc \
        -forever -shared -nopw -listen 0.0.0.0 -rfbport "$VNC_PORT" -xkb
    start_process novnc \
        websockify --web=/usr/share/novnc/ "$NOVNC_PORT" "localhost:${VNC_PORT}"
    print_status
}

case "$ACTION" in
    start)
        start_all
        ;;
    stop)
        stop_process novnc
        stop_process x11vnc
        stop_process fluxbox
        stop_process xvfb
        ;;
    restart)
        "$0" stop
        "$0" start
        ;;
    status)
        print_status
        ;;
    *)
        echo "Usage: $0 [start|stop|restart|status]" >&2
        exit 1
        ;;
esac
