# Trajectory Testbench Container

This container is dedicated to `husky_assembly_tamp/motion_planner/trajectory_testbench.py`.

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
./run.sh debug -- --stage 3
./run.sh down
```

Pass any normal `trajectory_testbench.py` arguments after `--`.

## Notes

- Linux: this uses `/tmp/.X11-unix` and Xauthority forwarding.
- macOS: install XQuartz first with `brew install --cask xquartz`, then start it with `open -a XQuartz`.
- On macOS, the XQuartz package install is host-level and Homebrew will ask for your admin password.
- Host edits are visible immediately because the source tree is not copied into the image.
- `LIBGL_ALWAYS_SOFTWARE=1` is enabled by default for stability inside Docker. Override it if you want to try hardware OpenGL:

```bash
LIBGL_ALWAYS_SOFTWARE=0 ./run.sh up
```
