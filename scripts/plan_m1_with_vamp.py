"""Plan a real M1 movement with the VAMP C-RRTC sidecar backend.

Pipeline (one bar action):
  1. load RobotCell + BarAction exactly like headless_bar_action_planner,
  2. derive the M1 endpoints (goal IK + tracked start) with the repo's own
     machinery -- identical inputs to what plan_constrained_dual_arm uses,
  3. export the collision scene into the robot-root frame as a pointcloud
     (obstacle rigid bodies surface-sampled) + bar attachment spheres,
  4. plan with the WSL vamp sidecar (Constrained RRT-Connect),
  5. post-validate every densified waypoint with the compas_fab checker --
     the single source of truth for collisions stays compas_fab.

Usage:
  python scripts/plan_m1_with_vamp.py <data_root> --problem <name> --bar-action B6.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pybullet_planning as pp
from compas.data import json_load
from compas.geometry import Transformation

# * reuse the headless planner's loaders/helpers (safe to import: main-guarded)
sys.path.insert(0, str(Path(__file__).parent))
import headless_bar_action_planner as hb  # noqa: E402

from husky_assembly_tamp.motion_planner.api import (  # noqa: E402
    TOOL_LINK_LEFT, TOOL_LINK_RIGHT,
    _build_cfab_collision_fn, _collect_obstacle_puids, _bar_body_id,
    _derive_constrained_start_for_plan,
)
from husky_assembly_tamp.motion_planner.vamp_backend import client  # noqa: E402


# ! all VAMP geometry is expressed in the robot URDF-root frame (the compiled
# ! robot sits at the origin); world data is transformed by inv(world_from_root)
def pose_to_matrix(pose) -> np.ndarray:
    """4x4 matrix from a pybullet (pos, quat_xyzw) pose."""
    pos, quat = pose
    m = np.eye(4)
    m[:3, :3] = np.array(pp.matrix_from_quat(quat)).reshape(3, 3)
    m[:3, 3] = pos
    return m


def sample_mesh_surface(vertices: np.ndarray, faces: np.ndarray,
                        spacing: float) -> np.ndarray:
    """Area-weighted surface sampling of a triangle mesh.

    Args:
        vertices (np.ndarray): (n, 3) vertex positions.
        faces (np.ndarray): (m, 3) triangle vertex indices.
        spacing (float): target distance between samples, meters.

    Returns:
        np.ndarray: (k, 3) sampled points (includes the vertices themselves).
    """
    v0, v1, v2 = (vertices[faces[:, i]] for i in range(3))
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    # ! scan meshes carry garbage topology: a few degenerate faces spanning
    # ! meters inflate the nominal area 1000x (one joint scan sampled to 47M
    # ! points before this guard). Clamp per-face area so junk faces cannot
    # ! dominate the sampling budget, and cap the total sample count.
    areas = np.minimum(areas, (10.0 * spacing) ** 2)
    total = float(areas.sum())
    if total <= 0.0:
        return vertices.copy()
    n_samples = min(300_000, max(len(faces), int(total / (spacing * spacing))))
    rng = np.random.default_rng(1234)
    tri = rng.choice(len(faces), size=n_samples, p=areas / total)
    r1, r2 = rng.random(n_samples), rng.random(n_samples)
    s = np.sqrt(r1)
    pts = (1 - s)[:, None] * v0[tri] + (s * (1 - r2))[:, None] * v1[tri] \
        + (s * r2)[:, None] * v2[tri]
    pts = np.vstack([vertices, pts])
    # reject junk-geometry fliers (scan meshes carry stray vertices/faces far
    # from the real part): keep only points near the dense vertex cluster.
    # median +/- MAD is robust even when junk outnumbers the percentile tails.
    med = np.median(vertices, axis=0)
    mad = np.median(np.abs(vertices - med), axis=0)
    half = np.maximum(6.0 * mad, 0.15)
    keep = (np.abs(pts - med) <= half).all(axis=1)
    return pts[keep]


def mesh_from_rigid_body(rcell, name: str):
    """(vertices, faces) of a rigid body's first collision mesh, local frame."""
    model = rcell.rigid_body_models[name]
    mesh = model.collision_meshes[0]
    verts, faces = mesh.to_vertices_and_faces(triangulated=True)
    return np.array(verts, dtype=float), np.array(faces, dtype=int)


def surface_sphere_shell(verts: np.ndarray, faces: np.ndarray,
                         radius: float = 0.012, pitch: float = 0.014) -> list:
    """Cover a mesh's surface with a shell of small spheres (local frame).

    A shell is conservative for collisions with EXTERNAL geometry: nothing can
    reach the interior without crossing the covered surface. Keeping spheres
    small avoids over-approximating tight-fitting parts (joint connectors).

    Args:
        verts (np.ndarray): mesh vertices.
        faces (np.ndarray): triangle indices.
        radius (float): sphere radius, meters.
        pitch (float): voxel pitch for sphere placement, meters.

    Returns:
        list[dict]: sphere specs {"xyz": [3], "r": float}.
    """
    pts = sample_mesh_surface(verts, faces, spacing=pitch * 0.7)
    centers = np.unique(np.round(pts / pitch).astype(np.int64), axis=0) * pitch
    return [{"xyz": [float(v) for v in c], "r": float(radius)} for c in centers]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", nargs="?", default=None)
    parser.add_argument("--problem", default=hb.DEFAULT_PROBLEM)
    parser.add_argument("--bar-action", default=hb.DEFAULT_BAR_ACTION)
    parser.add_argument("--spacing", type=float, default=0.015,
                        help="obstacle surface sampling spacing (m)")
    parser.add_argument("--point-radius", type=float, default=0.01)
    parser.add_argument("--case-out", default=None,
                        help="optional path to dump the exported case JSON")
    parser.add_argument("--save-motion", action="store_true",
                        help="on success, write the VAMP path into the "
                             "<bar>.solved_motion.json sidecar (replayable with "
                             "scripts/replay_bar_action_plan.py --load solved_motion)")
    parser.add_argument("--envelope-pad", type=float, default=1.0,
                        help="sampling-envelope padding around hull(start, goal) "
                             "in rad; the server intersects with the true joint "
                             "limits. 0 disables the envelope (full limits).")
    parser.add_argument("--max-points", type=int, default=2_200_000,
                        help="pointcloud cap; above it the cloud is re-voxeled "
                             "coarser WITH the point radius grown to cover every "
                             "removed point (strictly conservative)")
    parser.add_argument("--export-only", action="store_true",
                        help="derive + export the case JSON (--case-out) and exit "
                             "without planning")
    parser.add_argument("--attempts", type=int, default=4,
                        help="C-RRTC restarts inside the sidecar, alternating "
                             "halton/xorshift samplers")
    parser.add_argument("--max-iterations", type=int, default=200_000,
                        help="planner iteration cap PER attempt")
    args = parser.parse_args()

    # ------------------------------------------------------------------ load
    data_root = Path(hb.resolve_data_root(args.data_root))
    problem_dir = data_root / args.problem
    print(f"[load] RobotCell <- {problem_dir / 'RobotCell.json'}")
    rcell = json_load(str(problem_dir / "RobotCell.json"))
    _client, planner = hb.start_planner(rcell, use_gui=False)
    action = json_load(str(problem_dir / "BarActions" / args.bar_action))
    groups = hb.resolve_arm_groups(rcell)
    joint_names_12 = hb.arm_joint_names_12(rcell, groups)
    home12 = hb.home_conf12_from_action(action, joint_names_12)
    active_bar = action.active_bar_id
    # rigid-body name: the export uses either "<id>" or "bar_<id>"
    bar_rb_name = active_bar if active_bar in rcell.rigid_body_models \
        else f"bar_{active_bar}"

    selected = hb.select_movement(action, "M1")
    state = selected.start_state
    if state.robot_configuration is None:
        hb.fill_missing_config(state, rcell, groups, home12[:6], home12[6:])
    planner.set_robot_cell_state(state)

    robot_puid = planner.client.robot_puid
    arm_joints = pp.joints_from_names(robot_puid, joint_names_12)
    tool_l = pp.link_from_name(robot_puid, TOOL_LINK_LEFT)
    tool_r = pp.link_from_name(robot_puid, TOOL_LINK_RIGHT)
    bar_body = _bar_body_id(planner, bar_rb_name)

    # ! bodies attached to the robot (the bar AND its pre-mounted connector
    # ! parts riding the grippers) move with the arms: they must become vamp
    # ! ATTACHMENTS, never static obstacles
    carried = {
        name: rbs for name, rbs in (state.rigid_body_states or {}).items()
        if getattr(rbs, "attached_to_link", None)
    }
    print(f"[scene] carried bodies: {sorted(carried)}")
    obstacles = _collect_obstacle_puids(planner, exclude=set(carried))

    # -------------------------------------------------- derive M1 endpoints
    (start_conf, world_from_bar_start, world_from_bar_goal, goal_conf_arr,
     grasp_l, grasp_r, info) = _derive_constrained_start_for_plan(
        planner, state,
        active_bar_id=bar_rb_name, bar_body=bar_body, obstacles=obstacles,
        robot_puid=robot_puid, arm_joints=arm_joints,
        tool_link_left=tool_l, tool_link_right=tool_r,
        joint_names_12=joint_names_12,
        goal_conf=None, goal_ee_frames=selected.target_ee_frames,
        random_seed=None, max_ik_attempts=20, bar_sweep_box=None)
    if start_conf is None:
        raise SystemExit(f"endpoint derivation failed: {info.get('failure_reason')}")
    print(f"[derive] max |start-goal| = "
          f"{np.max(np.abs(np.array(start_conf) - np.array(goal_conf_arr))):.3f} rad")

    # ------------------------------------------------- root frame + rTl
    world_from_root = pose_to_matrix(pp.get_pose(robot_puid))
    root_from_world = np.linalg.inv(world_from_root)

    pp.set_joint_positions(robot_puid, arm_joints, start_conf)
    t_l = pose_to_matrix(pp.get_link_pose(robot_puid, tool_l))
    t_r = pose_to_matrix(pp.get_link_pose(robot_puid, tool_r))
    rTl_m = np.linalg.inv(t_l) @ t_r  # right tool0 in left tool0 frame
    quat = pp.quat_from_matrix(rTl_m[:3, :3].tolist())  # xyzw
    rTl = [float(quat[3]), float(quat[0]), float(quat[1]), float(quat[2]),
           float(rTl_m[0, 3]), float(rTl_m[1, 3]), float(rTl_m[2, 3])]

    # ------------------------------------------------- obstacle pointcloud
    # rigid_bodies_puids maps name -> list of pybullet ids (possibly several)
    puid_to_name = {}
    for name, puids in planner.client.rigid_bodies_puids.items():
        for puid in (puids if isinstance(puids, (list, tuple)) else [puids]):
            puid_to_name[puid] = name
    points = []
    for puid in obstacles:
        name = puid_to_name.get(puid)
        if name is None or name not in rcell.rigid_body_models:
            continue
        verts, faces = mesh_from_rigid_body(rcell, name)
        pts_local = sample_mesh_surface(verts, faces, args.spacing)
        world_from_body = pose_to_matrix(pp.get_pose(puid))
        pts_root = (root_from_world @ world_from_body
                    @ np.hstack([pts_local, np.ones((len(pts_local), 1))]).T).T[:, :3]
        # keep only points the arms could ever reach (2 m box around the root)
        keep = (np.abs(pts_root[:, :2]) < 2.0).all(axis=1) & (pts_root[:, 2] < 2.5)
        pts_root = pts_root[keep]
        # per-mesh voxel dedupe right away: the joint scan meshes carry 70k+
        # raw vertices each and would otherwise dominate export time
        voxel = max(args.spacing, 1e-4)
        pts_root = np.unique(np.round(pts_root / voxel).astype(np.int64), axis=0) * voxel
        points.append(pts_root)
        print(f"[scene] {name}: {len(pts_root)} points")
    cloud = np.vstack(points) if points else np.zeros((0, 3))
    voxel = max(args.spacing, 1e-4)
    cloud = np.unique(np.round(cloud / voxel).astype(np.int64), axis=0) * voxel
    # radius must at least span the voxel grid so dedupe cannot open holes,
    # but over-fattening closes real corridors: 2.3 cm blocked 45/211 motions
    # of a mesh-verified-valid path, 1.0-1.3 cm left it 90% open (measured)
    point_radius = max(args.point_radius, voxel * 0.6)
    while len(cloud) > args.max_points:
        voxel *= 2.0
        cloud = np.unique(np.round(cloud / voxel).astype(np.int64), axis=0) * voxel
        # on cap escalation the grid coarsens beyond the sampling density, so
        # here the full sqrt(3)/2 cover bound is required for conservativeness
        point_radius = args.point_radius + voxel * 0.87
        print(f"[scene] cap exceeded -> re-voxel at {voxel * 100:.1f} cm, "
              f"point radius {point_radius * 100:.1f} cm")
    print(f"[scene] total pointcloud: {len(cloud)} points, "
          f"point radius {point_radius * 100:.1f} cm")

    # ------------------------- attachments: bar + carried connector parts
    # one vamp Attachment per end effector, aggregating every body attached to
    # that tool; frames come from the authored attachment_frame (body pose in
    # the tool frame), the authoritative grasp data -- no FK round trip
    eef_of_link = {"left_ur_arm_tool0": 0, "right_ur_arm_tool0": 1}
    per_eef_spheres = {0: [], 1: []}
    for name, rbs in carried.items():
        link = rbs.attached_to_link
        eef = eef_of_link.get(link)
        if eef is None:
            raise SystemExit(f"carried body {name} on unexpected link {link}")
        tool_from_body = np.asarray(
            Transformation.from_frame(rbs.attachment_frame), dtype=float
        ).reshape(4, 4)
        verts, faces = mesh_from_rigid_body(rcell, name)
        shell = surface_sphere_shell(verts, faces)
        # move each sphere into the TOOL frame so one attachment per EE works
        for s in shell:
            p = tool_from_body @ np.array([*s["xyz"], 1.0])
            per_eef_spheres[eef].append({"xyz": [float(v) for v in p[:3]],
                                         "r": s["r"]})
        print(f"[scene] carried {name} -> eef {eef}: {len(shell)} spheres")

    attachments = [
        {"eef": eef, "tf_ee_from_att": [float(v) for v in np.eye(4).reshape(-1)],
         "spheres": spheres}
        for eef, spheres in per_eef_spheres.items() if spheres
    ]

    # ------------------------------------------------------------ plan call
    case = {
        "cmd": "plan", "robot": "husky_dual_ur5e",
        "start": [float(v) for v in start_conf],
        "goal": [float(v) for v in goal_conf_arr],
        "rTl": rTl, "tolerance": [1e-3] * 6,
        "pointcloud": {"points": cloud.tolist(),
                       "point_radius": point_radius},
        "attachments": attachments,
        "settings": {"range": 1.0, "num_projection_iterations": 15,
                     "std_dev_scaling_factor": 0.1, "simplify": True,
                     "attempts": args.attempts,
                     "max_iterations": args.max_iterations},
    }
    if args.envelope_pad > 0.0:
        case["envelope_pad"] = args.envelope_pad
    if args.case_out:
        Path(args.case_out).write_text(json.dumps(case))
        print(f"[export] case -> {args.case_out}")
    if args.export_only:
        print("EXPORT_ONLY_DONE")
        return

    collide = None
    path, bad = None, -1
    for round_i in range(3):
        case["rng_skip"] = round_i * 173000
        t0 = time.perf_counter()
        reply = client.request(case)
        roundtrip_ms = (time.perf_counter() - t0) * 1e3
        print(f"[vamp] round {round_i}: solved={reply.get('solved')} "
              f"plan={reply.get('plan_ms', 0):.1f} ms "
              f"simplify={reply.get('simplify_ms', 0):.1f} ms "
              f"roundtrip={roundtrip_ms:.0f} ms")
        if not reply.get("solved"):
            print(f"[vamp] start_valid={reply.get('start_valid')} "
                  f"goal_valid={reply.get('goal_valid')}")
            client.shutdown()
            raise SystemExit("VAMP_M1_FAILED")

        res_simp = reply.get("worst_constraint_residual", float("inf"))
        res_raw = reply.get("worst_constraint_residual_raw", float("inf"))
        print(f"[vamp] waypoints raw {len(reply['path'])} "
              f"(residual {res_raw:.2e}) -> simplified "
              f"{len(reply['simplified_path'])} (residual {res_simp:.2e})")
        # the simplifier can break the closed-chain constraint; keep it only
        # when it stays within a small multiple of the projection tolerance
        if res_simp <= 5e-3:
            path = reply["simplified_path"]
        else:
            path = reply["path"]
            print("[vamp] simplified violates the constraint -> using RAW path")

        # --------------------------------- compas_fab post-validation gate
        planner.set_robot_cell_state(state)
        if collide is None:
            collide = _build_cfab_collision_fn(planner, state, joint_names_12)
        dense, bad, bad_qs = 0, 0, []
        for a, b in zip(path[:-1], path[1:]):
            a, b = np.array(a), np.array(b)
            steps = max(2, int(np.ceil(np.max(np.abs(b - a)) / 0.05)))
            for t in np.linspace(0.0, 1.0, steps, endpoint=False):
                q = (1 - t) * a + t * b
                dense += 1
                if collide(q.tolist()):
                    bad += 1
                    bad_qs.append(q)
        if collide(path[-1]):
            bad += 1
        dense += 1
        print(f"[cfab] round {round_i}: {dense} dense waypoints, {bad} in collision")
        if bad == 0:
            break
        # diagnose the offending states so model gaps stay visible
        for q in bad_qs[:3]:
            st_bad = state.copy()
            hb.apply_conf12(st_bad, rcell, joint_names_12, q)
            try:
                planner.check_collision(st_bad, options={"full_report": True,
                                                         "verbose": False})
            except Exception as exc:  # CollisionCheckError carries the pairs
                pairs = getattr(exc, "collision_pairs", None)
                print(f"[cfab]   colliding pairs: {pairs}")
        print("[cfab] retrying with a shifted sampler stream")
    client.shutdown()

    # ------------------------------------------- save replayable sidecar
    if bad == 0 and args.save_motion:
        # write the VAMP path onto M1 exactly where the replay script looks:
        # mv.trajectory (12-vec waypoints) + a concrete start configuration
        selected.trajectory = [[float(v) for v in wp] for wp in path]
        hb.apply_conf12(selected.start_state, rcell, joint_names_12, start_conf)
        out_path = hb.solved_action_path(
            str(problem_dir / "BarActions" / args.bar_action), "motion")
        hb.save_solved_action(action, out_path)
        # ! read-back verification: the Insync/Google-Drive share has rolled
        # ! back writes before -- prove the file on disk holds THIS path
        check = json_load(out_path)
        n_disk = len(hb.select_movement(check, "M1").trajectory or [])
        marker = "OK" if n_disk == len(path) else "MISMATCH -- sync rolled back?"
        print(f"[save] VAMP M1 trajectory ({len(path)} wp) -> {out_path}")
        print(f"[save] read-back: {n_disk} wp on disk [{marker}]")

    print("VAMP_M1_OK" if bad == 0 else "VAMP_M1_POST_VALIDATION_FAILED")


if __name__ == "__main__":
    main()
