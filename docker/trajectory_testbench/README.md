# Trajectory Testbench Container

This container is dedicated to local PyBullet planning/debug runs.

Supported entry points:
- `husky_assembly_tamp/motion_planner/trajectory_testbench.py`
- `husky_assembly_tamp/motion_planner/stage1/minimal_rrt.py`

- Base image: Ubuntu 22.04
- Live code: the whole `husky-assembly-teleop` tree is bind-mounted into the container
- GUI: Linux hosts use the X11 socket directly; macOS hosts use XQuartz over `host.docker.internal:0`
- Debugging: `debugpy` listens on `localhost:5678`

## Usage

```bash
cd external/husky_assembly_tamp/docker/trajectory_testbench

./run.sh up
./run.sh shell
./run.sh testbench -- --stage 3
./run.sh stage1
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
- Host edits are visible immediately because the source tree is not copied into the image.
- `LIBGL_ALWAYS_SOFTWARE=1` is enabled by default for stability inside Docker. Override it if you want to try hardware OpenGL:

```bash
LIBGL_ALWAYS_SOFTWARE=0 ./run.sh up
```

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

Now run the GUI version:

```bash
./run.sh stage1
```

or

```bash
./run.sh testbench -- --stage 1
```

Do not set `HUSKY_DOCKER_HEADLESS=1` for GUI runs.

### Linux

Linux GUI runs use the host X11 socket directly:

```bash
./run.sh stage1
```
