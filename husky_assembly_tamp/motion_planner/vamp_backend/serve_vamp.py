"""VAMP constrained-planner sidecar server (runs on Linux, inside WSL2/Docker).

Speaks newline-delimited JSON over stdin/stdout, mirroring the ssik sidecar
protocol (``keyframe/ssik_sidecar/serve.py``): one ``{"ready": true}`` handshake
line at startup, then one JSON reply line per JSON request line.

This file has NO dependency on the rest of husky_assembly_tamp -- it is copied
or invoked inside the Linux environment where the forked vamp (with the
``husky_dual_ur5e`` robot module) is installed, e.g.::

    ~/mcvamp/venv/bin/python serve_vamp.py

Request schema (all geometry in the ROBOT ROOT frame, meters / quaternions):

    {"cmd": "plan",
     "robot": "husky_dual_ur5e",
     "start": [12 floats], "goal": [12 floats],
     "rTl": [qw, qx, qy, qz, tx, ty, tz],   # right tool0 pose in left tool0 frame
     "tolerance": [6 floats],                # +/- band on the 6D constraint error
     "cuboids": [{"center": [3], "euler_xyz": [3], "half_extents": [3]}, ...],
     "spheres": [{"xyz": [3], "r": float}, ...],
     "pointcloud": {"points": [[3], ...], "point_radius": float},
     "attachments": [{"eef": 0 | 1,
                      "tf_ee_from_att": [16 floats row-major],
                      "spheres": [{"xyz": [3], "r": float}, ...]}, ...],
     "settings": {"range": 1.0, "num_projection_iterations": 15,
                  "std_dev_scaling_factor": 0.1, "simplify": true}}

Reply:

    {"ok": true, "solved": true,
     "path": [[12 floats], ...],             # raw C-RRTC waypoints
     "simplified_path": [[12 floats], ...],  # after constrained shortcutting
     "iterations": int, "plan_ms": float, "simplify_ms": float,
     "worst_constraint_residual": float}

Other commands: {"cmd": "validate", "config": [...], ...scene...} -> per-config
collision answer, and {"cmd": "quit"}.
"""

import json
import sys
import time

import numpy as np

# * stray prints from vamp must not corrupt the JSON channel: keep stdout clean
_REAL_STDOUT = sys.stdout
sys.stdout = sys.stderr

import vamp  # noqa: E402  (import after the stdout redirect on purpose)


def _reply(payload: dict):
    """Write one JSON line to the real stdout and flush."""
    _REAL_STDOUT.write(json.dumps(payload) + "\n")
    _REAL_STDOUT.flush()


def build_environment(req: dict) -> "vamp.Environment":
    """Construct a vamp collision environment from a plan/validate request.

    Args:
        req (dict): request payload carrying cuboids/spheres/pointcloud/
            attachments as documented in the module docstring.

    Returns:
        vamp.Environment: environment with all obstacles and attachments added.
    """
    env = vamp.Environment()
    for c in req.get("cuboids", []):
        env.add_cuboid(vamp.Cuboid(c["center"], c["euler_xyz"], c["half_extents"]))
    for s in req.get("spheres", []):
        env.add_sphere(vamp.Sphere(s["xyz"], s["r"]))
    pc = req.get("pointcloud")
    if pc and pc.get("points"):
        # CAPT needs the robot's min/max sphere radii; fetched per robot module
        module = getattr(vamp, req.get("robot", "husky_dual_ur5e"))
        r_min, r_max = module.min_max_radii()
        env.add_pointcloud(pc["points"], r_min, r_max, pc["point_radius"])
    for att in req.get("attachments", []):
        tf = np.array(att["tf_ee_from_att"], dtype=float).reshape(4, 4)
        a = vamp.Attachment(tf.astype(np.float32))
        a.add_spheres([vamp.Sphere(s["xyz"], s["r"]) for s in att["spheres"]])
        env.attach(a, int(att.get("eef", 0)))
    return env


# original joint limits per robot module, captured before any runtime override
_ORIG_BOUNDS: dict = {}


def _apply_sampling_envelope(module, robot: str, req: dict) -> dict:
    """Set the module's SAMPLING bounds for this request (or restore originals).

    With ``envelope_pad`` in the request, samples are drawn from
    hull(start, goal) +/- pad, intersected with the robot's original joint
    limits -- collision checking and the endpoints themselves are untouched,
    so any returned path remains valid under the original limits.

    Args:
        module: the vamp robot module.
        robot (str): robot name (keys the original-bounds cache).
        req (dict): the plan request.

    Returns:
        dict: {"lows": [...], "highs": [...]} actually applied.
    """
    if robot not in _ORIG_BOUNDS:
        _ORIG_BOUNDS[robot] = (
            [float(v) for v in np.asarray(module.lower_bounds()).ravel()],
            [float(v) for v in np.asarray(module.upper_bounds()).ravel()],
        )
    orig_lo, orig_hi = _ORIG_BOUNDS[robot]
    if not (hasattr(module, "set_lows") and hasattr(module, "set_highs")):
        return {"lows": orig_lo, "highs": orig_hi}

    pad = req.get("envelope_pad")
    if pad is None:
        lo, hi = list(orig_lo), list(orig_hi)
    else:
        s = np.asarray(req["start"], dtype=float)
        g = np.asarray(req["goal"], dtype=float)
        lo = np.maximum(np.minimum(s, g) - float(pad), orig_lo).tolist()
        hi = np.minimum(np.maximum(s, g) + float(pad), orig_hi).tolist()
    module.set_lows(lo)
    module.set_highs(hi)
    return {"lows": lo, "highs": hi}


def handle_plan(req: dict) -> dict:
    """Run C-RRTC + constrained simplification for one request.

    Args:
        req (dict): full plan request (see module docstring).

    Returns:
        dict: reply payload (see module docstring).
    """
    robot = req.get("robot", "husky_dual_ur5e")
    module, planner_func, plan_settings, simp_settings = (
        vamp.configure_robot_and_planner_with_kwargs(robot, "crrtc")
    )
    envelope = _apply_sampling_envelope(module, robot, req)
    st = req.get("settings", {})
    plan_settings.rrtc_settings.range = float(st.get("range", 1.0))
    plan_settings.rrtc_settings.dynamic_domain = bool(st.get("dynamic_domain", False))
    plan_settings.constraint_settings.num_projection_iterations = int(
        st.get("num_projection_iterations", 15))
    plan_settings.constraint_settings.std_dev_scaling_factor = float(
        st.get("std_dev_scaling_factor", 0.1))

    tol = req.get("tolerance", [1e-3] * 6)
    constraint = module.BimanualTaskSpaceConstraint(
        req["rTl"], [-abs(t) for t in tol], [abs(t) for t in tol])
    constraints = module.Composable_BimanualTaskSpaceConstraint(constraint)

    env = build_environment(req)

    if "max_iterations" in st:
        plan_settings.rrtc_settings.max_iterations = int(st["max_iterations"])
        plan_settings.rrtc_settings.max_samples = int(st["max_iterations"])

    # * multi-attempt retries: alternate sampler families and skip ahead in the
    # * sequence per attempt so each restart explores differently
    attempts = max(1, int(st.get("attempts", 1)))
    base_skip = int(req.get("rng_skip", 0))  # caller-driven stream variation
    t0 = time.perf_counter_ns()
    result, used_attempts = None, 0
    for i in range(attempts):
        used_attempts = i + 1
        sampler = module.xorshift() if i % 2 else module.halton()
        skip = base_skip + (i // 2) * 25000
        if skip:
            # skip ahead so repeat visits to a sampler family see a fresh stream
            sampler.skip(skip)
        result = planner_func(req["start"], req["goal"], env, plan_settings,
                              constraints, sampler)
        if result.solved:
            break
    plan_ms = (time.perf_counter_ns() - t0) / 1e6

    reply = {"ok": True, "solved": bool(result.solved),
             "iterations": int(result.iterations), "plan_ms": plan_ms,
             "envelope": envelope, "attempts_used": used_attempts}
    if not result.solved:
        # help the caller distinguish "endpoints invalid" from "search failed"
        reply["start_valid"] = bool(module.validate(req["start"], env))
        reply["goal_valid"] = bool(module.validate(req["goal"], env))
        return reply

    path = result.path.numpy()
    reply["path"] = [[float(v) for v in q] for q in path]

    simplify_ms = 0.0
    spath = path
    # ! vamp's constrained simplifier segfaults on trivial 2-waypoint paths
    if st.get("simplify", True) and len(path) > 2:
        t1 = time.perf_counter_ns()
        simple = module.simplify_with_constraints(result.path, env, constraints,
                                                  simp_settings)
        simplify_ms = (time.perf_counter_ns() - t1) / 1e6
        spath = simple.path.numpy()
    reply["simplified_path"] = [[float(v) for v in q] for q in spath]
    reply["simplify_ms"] = simplify_ms

    def _worst_residual(waypoints) -> float:
        worst = 0.0
        for q in waypoints:
            # fresh numpy copy from plain floats: rows of result.path.numpy()
            # are views into planner-owned memory that nanobind rejects
            q32 = np.array(q, dtype=np.float32)
            worst = max(worst, float(constraint.distanceToConstraint(q32)))
        return worst

    # ! the constrained simplifier has produced paths violating the manifold
    # ! (residual 0.12 observed) -- report both so the caller can fall back
    reply["worst_constraint_residual"] = _worst_residual(reply["simplified_path"])
    reply["worst_constraint_residual_raw"] = _worst_residual(reply["path"])
    return reply


def handle_validate(req: dict) -> dict:
    """Collision-validate a single configuration in the request's scene."""
    robot = req.get("robot", "husky_dual_ur5e")
    module = getattr(vamp, robot)
    env = build_environment(req)
    return {"ok": True, "valid": bool(module.validate(req["config"], env))}


def main():
    """Serve newline-JSON requests until quit/EOF."""
    # handshake carries module provenance so a stale install is visible at once
    diag = {"vamp_file": getattr(vamp, "__file__", "?")}
    if hasattr(vamp, "husky_dual_ur5e"):
        diag["husky_n_spheres"] = int(vamp.husky_dual_ur5e.n_spheres())
    _reply({"ready": True, "robots": [m for m in dir(vamp)
                                      if m in ("husky_dual_ur5e", "bimanual_panda")],
            **diag})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            cmd = req.get("cmd")
            if cmd == "quit":
                _reply({"ok": True, "bye": True})
                return
            if cmd == "plan":
                _reply(handle_plan(req))
            elif cmd == "validate":
                _reply(handle_validate(req))
            else:
                _reply({"ok": False, "error": f"unknown cmd: {cmd}"})
        except Exception as exc:  # noqa: BLE001 -- sidecar must never die silently
            import traceback
            _reply({"ok": False, "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()})


if __name__ == "__main__":
    main()
