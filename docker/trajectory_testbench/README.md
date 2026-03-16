# Trajectory Testbench Container

This container is dedicated to local PyBullet planning/debug runs.

Supported entry points:
- `husky_assembly_tamp/motion_planner/trajectory_testbench.py`
- `husky_assembly_tamp/motion_planner/stage1/minimal_rrt.py`

- Base image: Ubuntu 22.04
- Live code: the whole `husky-assembly-teleop` tree is bind-mounted into the container
- GUI:
  - Linux hosts can use the X11 socket directly
  - macOS hosts can still try XQuartz over `host.docker.internal:0`
  - preferred cross-platform option: in-container virtual desktop over noVNC at `http://localhost:6080/vnc.html`
- Debugging: `debugpy` listens on `localhost:5678`

## Usage

```bash
cd external/husky_assembly_tamp/docker/trajectory_testbench

./run.sh up
./run.sh shell
./run.sh testbench -- --stage 3
./run.sh stage1
./run.sh desktop-up
./run.sh stage1-vnc
./run.sh debug -- --stage 3
./run.sh debug-stage1
HUSKY_DOCKER_HEADLESS=1 ./run.sh stage1 -- --no-gui
./run.sh down
```

Pass any normal module arguments after `--`.

## Notes

- Linux: this uses `/tmp/.X11-unix` and Xauthority forwarding.
- macOS: install XQuartz first with `brew install --cask xquartz`, then start it with `open -a XQuartz`.
- On macOS, the XQuartz package install is host-level and Homebrew will ask for your admin password.
- For headless runs, set `HUSKY_DOCKER_HEADLESS=1` to skip the host GUI checks.
- For browser-based GUI runs, use `./run.sh desktop-up` and open `http://localhost:6080/vnc.html`.
- Host edits are visible immediately because the source tree is not copied into the image.
- `LIBGL_ALWAYS_SOFTWARE=1` is enabled by default for stability inside Docker. Override it if you want to try hardware OpenGL:

```bash
LIBGL_ALWAYS_SOFTWARE=0 ./run.sh up
```

## noVNC Setup

This is now the recommended GUI path on macOS because it avoids the XQuartz/OpenGL issues that PyBullet often hits from a Linux container.

Start the in-container desktop:

```bash
./run.sh desktop-up
```

Then open this in your browser:

```text
http://localhost:6080/vnc_lite.html?autoconnect=1&resize=remote&host=localhost&port=6080&path=websockify
```

Run Stage 1 inside that virtual desktop:

```bash
./run.sh stage1-vnc
```

Run the legacy testbench inside that virtual desktop:

```bash
./run.sh testbench-vnc -- --stage 1
```

Useful desktop commands:

```bash
./run.sh desktop-status
./run.sh desktop-down
```

Notes:

- The virtual desktop uses `DISPLAY=:99` inside the container.
- `x11vnc` listens on port `5900`.
- noVNC listens on port `6080`.
- You do not need to set `DISPLAY` manually on the host for noVNC runs.
- You do not need XQuartz for noVNC runs.

## GUI Setup

### macOS

For GUI runs on macOS, do this once before starting the planner:

```bash
brew install --cask xquartz
open -a XQuartz
```

Then in XQuartz:

1. Open `Preferences -> Security`
2. Enable `Allow connections from network clients`
3. Restart XQuartz

Then allow the local X11 connection on the host:

```bash
xhost + 127.0.0.1
```

If that is still too restrictive for Docker Desktop on macOS, use this temporary diagnostic fallback instead:

```bash
xhost +
```

That disables X access control for the current XQuartz session. It is less secure, but it is the fastest way to tell whether the remaining issue is authorization versus display routing. You can revoke it later with:

```bash
xhost -
```

Now run the GUI version:

```bash
./run.sh stage1
```

or

```bash
./run.sh testbench -- --stage 1
```

Do not set `HUSKY_DOCKER_HEADLESS=1` for GUI runs.
Also do not manually export `DISPLAY` before running `./run.sh` on macOS. The script now forces the correct Docker-side display value: `host.docker.internal:0`.

### Linux

Linux GUI runs use the host X11 socket directly:

```bash
./run.sh stage1
```
