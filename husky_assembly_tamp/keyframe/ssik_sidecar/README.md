# ssik analytical-IK sidecar — one-time setup

Rhino runs CPython 3.9; [ssik](https://github.com/personalrobotics/ssik) needs
Python 3.11+ and ships compiled extensions, so it runs in a **separate 3.11 venv**
that the caller talks to over stdin/stdout (`serve.py` here; `keyframe/ssik_client.py`
on the caller side). ssik is the default IK backend
(`keyframe.config.IK_BACKEND == "ssik"`, overridable with `HUSKY_IK_BACKEND`).

**Most users only need step 1.** The per-arm solver modules are already built and
committed in the design repo under `asset/ssik/`, so a fresh clone just needs the
venv — the minimal version of this is in the design repo README ("Analytical IK
solver (ssik)"). Steps 2–3 below are the full record of how those modules were
generated; redo them only if the calibrated URDF changes.

**Where the venv and artifacts live.** The code here resolves both at spawn time
(`keyframe.config`): the env vars `HUSKY_SSIK_VENV_DIR` / `HUSKY_SSIK_ARTIFACT_DIR`
win when set; otherwise, when this package is vendored inside a host repo (the
design repo has it at `external/husky_assembly_tamp`), the defaults are
`<host>/external/ssik_env` and `<host>/asset/ssik`. If neither resolves, the spawn
raises with these instructions instead of guessing. Paths in the examples below
are relative to the design repo root.

## 1. Create the venv and install ssik

The venv needs **Python 3.11**, but not as your default interpreter. The design
repo README ("Analytical IK solver (ssik)") covers the
[uv](https://docs.astral.sh/uv/) route that provisions 3.11 for you from any
Python. Same idea here, just add the `[urdf]` extra so `ssik build` (step 2) has
its URDF parser:

```bash
uv venv --python 3.11 external/ssik_env
uv pip install --python external/ssik_env "ssik[urdf]"
```

`[urdf]` pulls in the URDF parser used by `ssik build`; the runtime solve path
itself does not need it (run-only setups use plain `ssik`, as the design repo
README shows). If you already have a real Python 3.11, stdlib venv works too:
`<your-3.11-python> -m venv external/ssik_env`, then
`external/ssik_env/Scripts/pip install "ssik[urdf]"` (Windows) or
`external/ssik_env/bin/pip install "ssik[urdf]"` (macOS/Linux).

## 2. Build one solver per arm from the calibrated URDF

The URDF is the single source of truth — calibration is baked into the generated
solver, so there is nothing to tune. Build both arms into `asset/ssik/`.

On Windows, set UTF-8 first (ssik prints Unicode; the default cp1252 console
otherwise crashes with `UnicodeEncodeError`). In PowerShell:

```powershell
$env:PYTHONUTF8 = "1"
$urdf = "asset\husky_urdf\mt_husky_dual_ur5_e_moveit_config\urdf\husky_dual_ur5_e_no_base_joint_All_Calibrated.urdf"

external\ssik_env\Scripts\ssik build $urdf `
  --base left_ur_arm_base_link  --ee left_ur_arm_tool0  --out asset\ssik\left_ur_arm_ik.py

external\ssik_env\Scripts\ssik build $urdf `
  --base right_ur_arm_base_link --ee right_ur_arm_tool0 --out asset\ssik\right_ur_arm_ik.py
```

The calibrated UR5e classifies as a **non-Pieper 6R** chain (ssik tier-2,
`ikgeo.general_6r`), which is exactly what ssik handles; each build validates on
100 random poses (max FK error ~1e-11) and finishes in a few seconds. The
`--base`/`--ee` links and the output filenames must match
`keyframe.config.SSIK_ARM_BUILD`.

## 3. Smoke-test the sidecar (optional but recommended)

Confirm the server answers before wiring it in. Send one identity-ish target and
check the reply has solutions with tiny `fk_residual` (run from the design repo
root; `serve.py` now lives inside the tamp package):

```bash
printf '{"ready-check":1}\n{"arm":"left","module":"left_ur_arm_ik","T":[1,0,0,0.4,0,1,0,0.1,0,0,1,0.3,0,0,0,1],"max_solutions":8}\n{"cmd":"quit"}\n' \
  | external/ssik_env/Scripts/python external/husky_assembly_tamp/husky_assembly_tamp/keyframe/ssik_sidecar/serve.py --artifact-dir asset/ssik
```

You should see `{"ready": true}` then a `{"ok": true, "solutions": [...]}` line.

## Then, in Rhino (or headless)

`keyframe.config.IK_BACKEND` already defaults to `"ssik"`, so in Rhino just run
`RSPBStart` and solve keyframes as usual (`RSIKKeyframe`); headless, run
`headless_bar_action_planner.py --solve-keyframes`. The sidecar starts on the
first solve and is shut down by `RSPBStop` (or when the headless run exits). Set
`HUSKY_IK_BACKEND=gradient` only to benchmark against the original PyBullet solver.
