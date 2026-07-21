"""Dual-arm IK solvers (ssik analytical backend + PyBullet gradient backend).

Extracted from the design front-end's ``scripts/core/robot_cell.py`` so ONE
implementation serves both callers:

- the Rhino command ``RSIKKeyframe`` (quick interactive IK checks), and
- the headless planner ``scripts/headless_bar_action_planner.py`` (offline solve).

Everything here is Rhino-free and cache-free: the solvers read the RobotCell
from the planner they are handed (``planner.client.robot_cell``) and take the
design-specific inputs (planning groups, home configuration) as explicit
arguments -- the Rhino session cache and the exported JSON are both just ways
for a CALLER to obtain those inputs.

compas_fab consumes SI units (meters, radians). Public 4x4 transforms here are
mm (the project convention); unit conversion happens at the compas boundary
inside this module.
"""

from __future__ import annotations

import numpy as np

from husky_assembly_tamp.keyframe import config


# ---------------------------------------------------------------------------
# Deferred imports
# ---------------------------------------------------------------------------
# compas, compas_fab, compas_robots, and pybullet are heavy dependencies. We
# import them lazily so that importing this module does not pay that cost on
# every Rhino script run (the Rhino front-end imports this module at startup).


def import_compas_stack():
    """Lazy-import the compas / compas_fab / pybullet dependency stack.

    Returned dict contains the concrete classes/modules used by this module and
    by the design front-end (``core.robot_cell`` / ``core.ik_viz`` re-import this
    function so there is one definition project-wide). Reuse across the project
    to avoid scattered top-level imports (which Rhino's ScriptEditor pays the
    cost for on first run).
    """
    import compas  # noqa: F401 - consumer below
    from compas.geometry import Frame, Transformation  # noqa: F401
    from compas.scene import Scene  # noqa: F401
    from compas.datastructures import Mesh  # noqa: F401
    from compas_robots import Configuration, RobotModel, ToolModel  # noqa: F401
    from compas_robots.resources import LocalPackageMeshLoader  # noqa: F401
    from compas_fab.robots import RobotCell, RobotSemantics  # noqa: F401
    from compas_fab.robots import RigidBody, RigidBodyState  # noqa: F401
    from compas_fab.robots import ToolState  # noqa: F401
    from compas_fab.backends import PyBulletClient, PyBulletPlanner  # noqa: F401
    from compas_fab.robots import FrameTarget, TargetMode  # noqa: F401
    import pybullet_planning as pp  # noqa: F401

    return {
        "compas": compas,
        "Frame": Frame,
        "Transformation": Transformation,
        "Scene": Scene,
        "Mesh": Mesh,
        "Configuration": Configuration,
        "RobotModel": RobotModel,
        "ToolModel": ToolModel,
        "LocalPackageMeshLoader": LocalPackageMeshLoader,
        "RobotCell": RobotCell,
        "RobotSemantics": RobotSemantics,
        "RigidBody": RigidBody,
        "RigidBodyState": RigidBodyState,
        "ToolState": ToolState,
        "PyBulletClient": PyBulletClient,
        "PyBulletPlanner": PyBulletPlanner,
        "FrameTarget": FrameTarget,
        "TargetMode": TargetMode,
        "pp": pp,
    }


# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------


def mm_matrix_to_m_frame(Frame, matrix_mm: np.ndarray):
    """Convert a 4x4 matrix with mm translation to a compas Frame in meters."""
    matrix = np.asarray(matrix_mm, dtype=float)
    origin = matrix[:3, 3] / 1000.0
    x_axis = matrix[:3, 0]
    y_axis = matrix[:3, 1]
    return Frame(list(map(float, origin)), list(map(float, x_axis)), list(map(float, y_axis)))


def frame_to_m_matrix(frame) -> np.ndarray:
    """Convert a compas Frame (meters) to a 4x4 numpy transform in meters."""
    matrix = np.eye(4, dtype=float)
    matrix[:3, 0] = np.asarray(frame.xaxis, dtype=float)
    matrix[:3, 1] = np.asarray(frame.yaxis, dtype=float)
    matrix[:3, 2] = np.asarray(frame.zaxis, dtype=float)
    matrix[:3, 3] = np.asarray(frame.point, dtype=float)
    return matrix


def apply_base_frame_mm(state, base_frame_world_mm: np.ndarray):
    """Set a cell state's robot base frame from a 4x4 mm world transform."""
    deps = import_compas_stack()
    state.robot_base_frame = mm_matrix_to_m_frame(deps["Frame"], base_frame_world_mm)


# ---------------------------------------------------------------------------
# Cell + group resolution (the cache-free seam)
# ---------------------------------------------------------------------------


def _require_loaded_cell(planner):
    """The planner's already-loaded RobotCell; hard error if none / bare.

    The solvers never load a cell themselves -- the caller must have pushed a
    fully-populated cell (tools + rigid bodies registered) onto the planner
    first. Refusing to cold-load here avoids the old failure mode where a bare,
    tool-less cell silently clobbered the planner and IK later died with a
    cryptic "tools do not match" mismatch.

    Args:
        planner (PyBulletPlanner): the active planner.

    Returns:
        RobotCell: ``planner.client.robot_cell``.

    Raises:
        RuntimeError: when the planner has no cell, or a cell with no tools.
    """
    rcell = getattr(planner.client, "robot_cell", None)
    if rcell is None or not getattr(rcell, "tool_models", None):
        raise RuntimeError(
            "keyframe.dual_arm_ik: the planner has no RobotCell with tools loaded, so "
            "IK cannot run. Push the fully-built cell first:\n"
            "  - In Rhino: run RSPBStart then RSRebuildRobotCell.\n"
            "  - Headless: json_load RobotCell.json and call planner.set_robot_cell(rcell) "
            "before solving."
        )
    return rcell


def resolve_arm_groups(robot_cell) -> tuple:
    """Derive the (left, right) planning-group names from the cell's semantics.

    The exported ``RobotCell.json`` embeds the SRDF semantics, so the group names
    never need to be hardcoded by a consumer. The dual-arm SRDF defines several
    groups per side (e.g. ``"Left arm"`` and ``"base_left_arm_manipulator"``); the
    solvers want the full base->tool0 MANIPULATOR chain, so among the groups whose
    name contains the side word we prefer the one that also says "manipulator".

    Args:
        robot_cell (RobotCell): the loaded cell (its ``robot_semantics`` is read).

    Returns:
        tuple[str, str]: ``(left_group, right_group)``.

    Raises:
        RuntimeError: when the rule does not isolate exactly one group per side --
            the message lists every group name so the caller can see why.
    """
    names = list(robot_cell.robot_semantics.group_names)

    def _pick(side: str) -> str:
        matches = [g for g in names if side in g.lower()]
        if len(matches) > 1:
            # Several side groups (arm-only + manipulator): keep the manipulator chain.
            narrowed = [g for g in matches if "manipulator" in g.lower()]
            if narrowed:
                matches = narrowed
        if len(matches) != 1:
            raise RuntimeError(
                f"keyframe.dual_arm_ik.resolve_arm_groups: cannot derive the {side!r} "
                f"planning group from the cell semantics groups {names}; expected exactly "
                f"one group whose name contains {side!r} (preferring 'manipulator'), "
                f"got {matches}. Pass groups=(left, right) explicitly to override."
            )
        return matches[0]

    return _pick("left"), _pick("right")


def _resolve_groups_arg(planner, groups):
    """Normalize the ``groups`` argument: explicit pair, or derived from the cell."""
    if groups is not None:
        left_group, right_group = groups
        return str(left_group), str(right_group)
    return resolve_arm_groups(_require_loaded_cell(planner))


# ---------------------------------------------------------------------------
# Small solver helpers
# ---------------------------------------------------------------------------


def _is_cell_state_mismatch(reason: str) -> bool:
    """True if an IK error is a RobotCell <-> RobotCellState mismatch (setup error).

    compas_fab's ``assert_cell_state_match`` raises a ``ValueError`` when the tools
    or workpieces named in the cell STATE don't match those registered in the
    robot CELL, e.g. "The tools in the cell state do not match the tools in the
    robot cell. Mismatch: {...}" (and the equivalent workpieces message). These
    are structural configuration errors, not reachability/collision failures, so
    every random restart would hit the same wall -- ``solve_dual_arm_ik`` aborts
    immediately on them instead of retrying.

    Args:
        reason (str): the stringified IK exception.

    Returns:
        bool: True for a cell/state tools-or-workpieces mismatch.
    """
    text = (reason or "").lower()
    return "do not match" in text and "robot cell" in text


def _set_random_dual_arm_config(planner, state, groups):
    """Warm-start both arms from a fresh random configuration (for IK restarts).

    Mutates ``state.robot_configuration`` in place with random joint values for the
    left and right arm groups, using the planner's robot-cell sampler (the same one
    compas_fab's IK uses for its internal random restarts).

    Args:
        planner (PyBulletPlanner): active planner (its ``client.robot_cell`` samples).
        state (RobotCellState): state whose configuration is randomized in place.
        groups (tuple[str, str]): the ``(left_group, right_group)`` names.

    Returns:
        None.
    """
    rcell = planner.client.robot_cell
    for group in groups:
        state.robot_configuration.merge(rcell.random_configuration(group))


def _link_world_frame_m(planner, link_name: str):
    """Return a URDF link's current world pose as a compas Frame (meters).

    Reads the pose straight from PyBullet so it reflects the robot's CURRENT base
    placement and joint configuration -- exactly what the ssik solver needs for its
    per-arm base frame. Callers must push the desired base frame / configuration
    into PyBullet (``planner.set_robot_cell_state(...)``) first.

    Args:
        planner (PyBulletPlanner): active planner (its client owns the robot body).
        link_name (str): URDF link name, e.g. ``"left_ur_arm_base_link"``.

    Returns:
        compas.geometry.Frame: the link frame in world coordinates, meters.

    Raises:
        KeyError: if ``link_name`` is not a link of the robot body.
    """
    import pybullet

    deps = import_compas_stack()
    Frame = deps["Frame"]
    cid = planner.client.client_id
    uid = planner.client.robot_puid

    # The base link is PyBullet index -1; every other link is the CHILD link of
    # joint j, whose name is jointInfo[12].
    base_name = pybullet.getBodyInfo(uid, physicsClientId=cid)[0].decode("utf-8")
    if link_name == base_name:
        pos, orn = pybullet.getBasePositionAndOrientation(uid, physicsClientId=cid)
    else:
        index = None
        for j in range(pybullet.getNumJoints(uid, physicsClientId=cid)):
            child = pybullet.getJointInfo(uid, j, physicsClientId=cid)[12].decode("utf-8")
            if child == link_name:
                index = j
                break
        if index is None:
            raise KeyError(f"link {link_name!r} not found on robot body {uid}.")
        state = pybullet.getLinkState(
            uid, index, computeForwardKinematics=True, physicsClientId=cid
        )
        pos, orn = state[4], state[5]  # worldLinkFramePosition / Orientation

    # PyBullet quaternion is xyzw; turn it into the frame's x/y axes.
    rot = pybullet.getMatrixFromQuaternion(orn)  # row-major 3x3 as 9 floats
    x_axis = (rot[0], rot[3], rot[6])
    y_axis = (rot[1], rot[4], rot[7])
    return Frame(list(map(float, pos)), list(map(float, x_axis)), list(map(float, y_axis)))


def _group_arm_values(configuration, joint_names):
    """Pull a group's joint values out of a Configuration, in ``joint_names`` order.

    Args:
        configuration (Configuration): a config holding at least these joints.
        joint_names (list[str]): the group's 6 joint names, in the order the ssik
            branch configs use (the group's ``zero_configuration`` order).

    Returns:
        list[float]: the joint values ordered to match ``joint_names``.
    """
    by_name = dict(zip(configuration.joint_names, configuration.joint_values))
    return [by_name[name] for name in joint_names]


def _nearest_cfg(cfgs, seed_values):
    """Return the branch config whose joint values are closest (L2) to a seed.

    Both ``cfgs[i].joint_values`` and ``seed_values`` are in the same group joint
    order, so a plain per-joint distance picks the "same branch" as the seed.

    Args:
        cfgs (list[Configuration]): candidate branch configs for one arm.
        seed_values (list[float]): the arm's start-configuration joint values.

    Returns:
        Configuration: the nearest branch (or ``None`` if ``cfgs`` is empty).
    """
    seed = np.asarray(seed_values, dtype=float)
    best, best_d = None, None
    for cfg in cfgs:
        d = float(np.linalg.norm(np.asarray(cfg.joint_values, dtype=float) - seed))
        if best_d is None or d < best_d:
            best, best_d = cfg, d
    return best


# ---------------------------------------------------------------------------
# ssik analytical IK backend (config.IK_BACKEND == "ssik")
# ---------------------------------------------------------------------------
# ssik solves the CALIBRATED URDF directly (single source of truth, no tuning) in
# a Python 3.11 sidecar; `keyframe.ssik_client` bridges to it. The helpers below
# feed it (arm-base world frame) and consume its per-arm branch solutions. See
# `solve_dual_arm_ik_ssik`.


def _ssik_branch_data(
    planner,
    seed_state,
    base_frame_world_mm: np.ndarray,
    tool0_left_world_mm: np.ndarray,
    tool0_right_world_mm: np.ndarray,
    tolerance_position: float,
    tolerance_orientation: float,
    groups,
):
    """Query the ssik sidecar for every per-arm branch at a given base frame.

    Pushes ``seed_state`` (already posed at ``base_frame_world_mm``) into PyBullet
    so each arm's base-link world pose reflects that base, asks the ssik sidecar for
    all analytical branches of each arm's tool0 target, and packages them as
    URDF-space ``Configuration``s plus the ``FrameTarget``s used by the gradient
    polish. Written once here so the production solver
    (:func:`solve_dual_arm_ik_ssik`) and the debug enumerator
    (:func:`enumerate_ssik_candidate_pairs`) share exactly the same ssik query.

    Args:
        planner (PyBulletPlanner): active planner with the dual-arm cell loaded.
        seed_state (RobotCellState): state posed at the target base; pushed into
            PyBullet here so ``_link_world_frame_m`` reads the right base pose.
        base_frame_world_mm (np.ndarray): 4x4 mm base frame (accepted for parity /
            documentation; the base is already baked into ``seed_state``).
        tool0_left_world_mm, tool0_right_world_mm (np.ndarray): 4x4 mm tool0 targets.
        tolerance_position, tolerance_orientation (float): FrameTarget tolerances.
        groups (tuple[str, str]): the ``(left_group, right_group)`` names.

    Returns:
        tuple: ``(branch_cfgs, group_joint_names, targets)`` where
        ``branch_cfgs[side]`` is a list of URDF-space ``Configuration``s (empty if
        that arm is unreachable), ``group_joint_names[side]`` the 6 joint names in
        group order, and ``targets[side]`` the ``FrameTarget`` for the polish.
    """
    # Imported here (not at module top) because ssik_client shells out to the
    # Python 3.11 sidecar and should only load when the ssik backend is actually used.
    from husky_assembly_tamp.keyframe import ssik_client

    deps = import_compas_stack()
    Frame = deps["Frame"]
    FrameTarget = deps["FrameTarget"]
    TargetMode = deps["TargetMode"]
    rcell = planner.client.robot_cell
    left_group, right_group = groups

    # Push the chosen base into PyBullet so each arm's base-link world pose (read
    # below for the ssik target) reflects THIS base placement.
    planner.set_robot_cell_state(seed_state)

    sides = (
        ("left", left_group, tool0_left_world_mm),
        ("right", right_group, tool0_right_world_mm),
    )

    branch_cfgs = {}         # side -> list[Configuration] (URDF joint space)
    group_joint_names = {}   # side -> [6 joint names, group order]
    targets = {}             # side -> FrameTarget (for the gradient polish)
    for side, group, tool0_mm in sides:
        base_link = config.SSIK_ARM_BUILD[side][0]
        # Both frames in meters: arm base in world, and the tool0 target in world.
        base_mat = frame_to_m_matrix(_link_world_frame_m(planner, base_link))
        tool0_mat = frame_to_m_matrix(mm_matrix_to_m_frame(Frame, tool0_mm))
        # ssik wants the target in the arm's base-link frame.
        target_in_base = np.linalg.inv(base_mat) @ tool0_mat

        solutions = ssik_client.solve(side, target_in_base, max_solutions=8)

        # ssik returns the 6 arm joints in kinematic (base->tool0) order, which is
        # exactly the group's zero_configuration order -- so we drop the values
        # straight in and get a URDF-space Configuration.
        group_joint_names[side] = list(rcell.zero_configuration(group).joint_names)
        cfgs = []
        for q in solutions:
            cfg = rcell.zero_configuration(group)
            cfg.joint_values = list(q)
            cfgs.append(cfg)
        branch_cfgs[side] = cfgs

        targets[side] = FrameTarget(
            mm_matrix_to_m_frame(Frame, tool0_mm),
            TargetMode.ROBOT,
            tolerance_position=tolerance_position,
            tolerance_orientation=tolerance_orientation,
        )
    return branch_cfgs, group_joint_names, targets


def solve_dual_arm_ik_ssik(
    planner,
    template_state,
    base_frame_world_mm: np.ndarray,
    tool0_left_world_mm: np.ndarray,
    tool0_right_world_mm: np.ndarray,
    *,
    check_collision: bool = True,
    max_restart_iter: int = None,
    max_descend_iterations: int = None,
    tolerance_position: float = None,
    tolerance_orientation: float = None,
    verbose_pairs: bool = False,
    groups=None,
    home_conf_12=None,
):
    """ssik dual-arm IK: analytical samples -> pick pair -> collision -> polish.

    Drop-in alternative to :func:`solve_dual_arm_ik` (same signature and
    ``RobotCellState``-in / ``RobotCellState``-out contract), selected by
    ``config.IK_BACKEND == "ssik"``. ssik solves the CALIBRATED URDF directly, so
    the branch configs are already in URDF joint space -- no tuning constants.

    Pipeline:
      1. Ask the ssik sidecar for every branch of EACH arm independently (<=8),
         with the tool0 target expressed in that arm's base-link frame.
      2. WARM solve (``max_restart_iter == 1`` -- M2/M3 chained off a real previous
         config): keep the branch nearest that config per arm (1 pair). COLD solve
         (M1, no meaningful seed): take the full Cartesian product (<=64 pairs).
      3. Collision-check the candidate pairs; drop the ones in collision.
      4. Polish a surviving pair (cold: the pair nearest ``home_conf_12`` first)
         with the SAME per-arm ``planner.inverse_kinematics`` descent, then one
         final combined-pose collision check.

    Args:
        planner (PyBulletPlanner): active planner with the dual-arm cell loaded.
        template_state (RobotCellState): seed state; its ``robot_configuration``
            selects the branch on a warm solve and warm-starts the polish.
        base_frame_world_mm (np.ndarray): 4x4 mm robot base frame.
        tool0_left_world_mm, tool0_right_world_mm (np.ndarray): 4x4 mm tool0 targets.
        check_collision (bool): collision-check candidate pairs and the final pose.
        max_restart_iter (int): the cold/warm signal from ``solve_keyframe_chain`` --
            ``1`` means a real warm start (reduce to the nearest branch); ``None`` or
            larger means a cold solve (try every branch pair). Its actual value does
            not scale ssik (it enumerates all branches once, not random restarts).
        max_descend_iterations, tolerance_position, tolerance_orientation: gradient
            polish tuning; ``None`` -> the ``config.IK_*`` defaults.
        verbose_pairs (bool): print the collision-pair summary on success.
        groups (tuple[str, str] | None): the ``(left_group, right_group)`` planning
            group names; ``None`` -> derived from the cell semantics
            (:func:`resolve_arm_groups`).
        home_conf_12 (Sequence[float] | None): the 12 home joint values (left 6
            then right 6) used to order candidate pairs on a COLD solve. Required
            for cold solves -- Rhino passes its config constant, the headless
            planner passes M4's ``target_configuration`` from the BarAction JSON.

    Returns:
        RobotCellState | None: the solved state copy, or ``None`` if no collision-
        free branch pair polishes onto the targets (or an arm is unreachable).

    Raises:
        RuntimeError: on a cold solve with no ``home_conf_12`` (no silent
            arbitrary ordering), or when the planner has no loaded cell.
    """
    rcell = _require_loaded_cell(planner)
    left_group, right_group = _resolve_groups_arg(planner, groups)

    max_descend_iterations = (
        max_descend_iterations
        if max_descend_iterations is not None
        else config.IK_MAX_DESCEND_ITERATIONS
    )
    tolerance_position = (
        tolerance_position if tolerance_position is not None else config.IK_TOLERANCE_POSITION
    )
    tolerance_orientation = (
        tolerance_orientation
        if tolerance_orientation is not None
        else config.IK_TOLERANCE_ORIENTATION
    )

    # WARM vs COLD comes from max_restart_iter (the same signal solve_keyframe_chain
    # uses): 1 == chained off a real previous keyframe -> pick the nearest branch;
    # anything else (M1's None/cold budget) -> try every branch pair. Note we do NOT
    # key this off "robot_configuration is not None", because M1's start_state is
    # pre-filled with HOME as a placeholder, which is not a real warm start.
    warm = (
        max_restart_iter is not None
        and int(max_restart_iter) == 1
        and template_state.robot_configuration is not None
    )

    # Cold solves sort the surviving branch pairs by distance to home so the
    # returned pose is stable/repeatable. That home must come from the caller --
    # no silent zero-vector or config-file fallback here.
    if home_conf_12 is not None:
        home_conf = np.asarray(home_conf_12, dtype=float)
    elif warm:
        home_conf = None  # single candidate pair; the home-distance sort is moot
    else:
        raise RuntimeError(
            "keyframe.dual_arm_ik.solve_dual_arm_ik_ssik: a COLD solve needs "
            "home_conf_12 (12 joint values, left arm then right) to order candidate "
            "branch pairs deterministically.\n"
            "  - In Rhino: pass config.HUSKY_DUAL_ARM_HOME_CONF_12.\n"
            "  - Headless: pass the home read from M4's target_configuration in the "
            "BarAction JSON."
        )

    seed_state = template_state.copy()
    apply_base_frame_mm(seed_state, base_frame_world_mm)
    if seed_state.robot_configuration is None:
        seed_state.robot_configuration = rcell.default_cell_state().robot_configuration

    # Ask the ssik sidecar for every per-arm branch at this base (shared helper).
    branch_cfgs, group_joint_names, targets = _ssik_branch_data(
        planner, seed_state, base_frame_world_mm,
        tool0_left_world_mm, tool0_right_world_mm,
        tolerance_position, tolerance_orientation,
        (left_group, right_group),
    )

    sides = (
        ("left", left_group, tool0_left_world_mm),
        ("right", right_group, tool0_right_world_mm),
    )
    branch_counts = {side: len(branch_cfgs[side]) for side in ("left", "right")}

    # If an arm has no branch, the pose is unreachable -- no silent fallback. Print
    # the per-arm branch counts so a modeling problem (target out of reach) is obvious.
    unreachable = [s for s in ("left", "right") if branch_counts[s] == 0]
    if unreachable:
        print(
            "keyframe.dual_arm_ik.solve_dual_arm_ik_ssik: "
            + " and ".join(f"{s} arm" for s in unreachable)
            + " unreachable (ssik returned 0 analytical branches)."
        )
        _print_ssik_failure_breakdown(branch_counts, [], stage="reachability")
        return None

    # --- closest-conf selection (warm start reduces to 1 branch per arm) ----
    if warm:
        options_per_side = {}
        for side, group, _ in sides:
            seed_vals = _group_arm_values(
                template_state.robot_configuration, group_joint_names[side]
            )
            options_per_side[side] = [_nearest_cfg(branch_cfgs[side], seed_vals)]
        left_options, right_options = options_per_side["left"], options_per_side["right"]
    else:
        left_options, right_options = branch_cfgs["left"], branch_cfgs["right"]

    # --- collision-filter the candidate dual-arm pairs ----------------------
    candidates = []  # list of (trial_state, distance_to_home)
    pair_failures = []  # (candidate_index, collision_reason) for the breakdown
    pair_index = 0
    for lcfg in left_options:
        for rcfg in right_options:
            trial = seed_state.copy()
            trial.robot_configuration.merge(lcfg)
            trial.robot_configuration.merge(rcfg)
            if check_collision:
                try:
                    planner.check_collision(
                        trial, options={"full_report": False, "verbose": False}
                    )
                except Exception as exc:
                    # Record WHY this pair was rejected (the CC.x collision line) so an
                    # all-pairs-collide failure explains itself instead of printing a
                    # bare "no collision-free branch pair".
                    pair_failures.append((pair_index, str(exc) or type(exc).__name__))
                    pair_index += 1
                    continue  # pair in collision -> skip
            if home_conf is None:
                home_d = 0.0  # warm: one pair, ordering irrelevant
            else:
                lvals = _group_arm_values(trial.robot_configuration, group_joint_names["left"])
                rvals = _group_arm_values(trial.robot_configuration, group_joint_names["right"])
                home_d = float(np.linalg.norm(np.asarray(lvals + rvals, dtype=float) - home_conf))
            candidates.append((trial, home_d))
            pair_index += 1

    if not candidates:
        print(
            f"keyframe.dual_arm_ik.solve_dual_arm_ik_ssik: no collision-free branch pair "
            f"among {pair_index} candidate(s)."
        )
        _print_ssik_failure_breakdown(branch_counts, pair_failures, stage="collision")
        return None

    # Cold solve: try the home-nearest pair first (stable, repeatable choice).
    candidates.sort(key=lambda item: item[1])

    # --- gradient fine-tuning on the calibrated URDF ------------------------
    # ssik is already calibration-exact, so this descent converges immediately; it
    # stays so the returned joints come from the same PyBullet FK as the gradient
    # backend (uniform IK_TOLERANCE_* guarantee).
    grp_options = {
        "max_results": 1,
        "check_collision": check_collision,
        "max_descend_iterations": max_descend_iterations,
        "verbose": False,
    }
    polish_failures = []  # (surviving_pair_index, reason) for the breakdown
    for surv_index, (trial, _) in enumerate(candidates):
        polished = trial.copy()
        reason = None
        for side, group, _ in sides:
            try:
                cfg = planner.inverse_kinematics(
                    targets[side], polished, group=group, options=grp_options
                )
            except Exception as exc:
                reason = f"group '{group}' polish: {str(exc) or type(exc).__name__}"
                break
            if cfg is None:
                reason = f"group '{group}' polish: no inverse kinematics solution found"
                break
            polished.robot_configuration.merge(cfg)

        if reason is None and check_collision:
            try:
                planner.check_collision(
                    polished, options={"full_report": False, "verbose": False}
                )
            except Exception as exc:
                reason = f"polished-pose collision: {str(exc) or type(exc).__name__}"

        if reason is not None:
            polish_failures.append((surv_index, reason))
            continue

        if verbose_pairs:
            print(summarize_check_collision(planner, polished))
        return polished

    print(
        f"keyframe.dual_arm_ik.solve_dual_arm_ik_ssik: no candidate polished cleanly "
        f"among {len(candidates)} collision-free pair(s)."
    )
    _print_ssik_failure_breakdown(branch_counts, polish_failures, stage="polish")
    return None


def _resolve_collision_pairs(collision_pairs, rcell):
    """Resolve compas_fab collision-pair objects to readable ``(kind, name)``s.

    Each item in ``collision_pairs`` is a ``(a, b)`` tuple whose participants are a
    compas ``Link``, a ``ToolModel``, or a ``RigidBody`` model. Only ``Link`` carries
    its own ``.name``; ``ToolModel`` / ``RigidBody`` do NOT -- their name is the KEY
    in the cell's ``tool_models`` / ``rigid_body_models`` registry -- so they are
    reverse-mapped by object identity (``id``). This is why a naive
    ``getattr(item, "name")`` printed a useless "RigidBody" before.

    Args:
        collision_pairs (list): ``[(a, b), ...]`` offender object pairs from a
            ``full_report`` ``CollisionCheckError``.
        rcell (RobotCell): the SAME cell the pairs reference (so identity matches);
            supplies the ``rigid_body_models`` / ``tool_models`` registries.

    Returns:
        tuple: ``(summary, offenders, num_pairs)`` where ``summary`` is a readable
        ``"nameA <-> nameB; ..."`` string, ``offenders`` a deduped
        ``[(kind, name), ...]`` list (``kind`` in ``{"link", "tool",
        "rigid_body"}``) for highlighting, and ``num_pairs`` the pair count.
    """
    # Object -> registry-name maps (name is the dict key, not an attribute).
    rb_by_id = {id(obj): nm for nm, obj in (getattr(rcell, "rigid_body_models", None) or {}).items()}
    tool_by_id = {id(obj): nm for nm, obj in (getattr(rcell, "tool_models", None) or {}).items()}

    def _kind_name(item):
        cls = type(item).__name__
        if cls == "Link":
            return "link", getattr(item, "name", None) or "link"
        if cls == "ToolModel":
            return "tool", tool_by_id.get(id(item)) or getattr(item, "name", None) or "tool"
        # RigidBody (no .name) or anything else -> reverse-map against the registry.
        return "rigid_body", rb_by_id.get(id(item)) or getattr(item, "name", None) or cls

    pair_strs = []
    offenders = []
    seen = set()
    for a, b in collision_pairs:
        ka, na = _kind_name(a)
        kb, nb = _kind_name(b)
        pair_strs.append(f"{na} <-> {nb}")
        for kind_name in ((ka, na), (kb, nb)):
            if kind_name not in seen:
                seen.add(kind_name)
                offenders.append(kind_name)
    summary = "; ".join(pair_strs) if pair_strs else "clear (no collision)"
    return summary, offenders, len(collision_pairs)


def enumerate_ssik_candidate_pairs(
    planner,
    template_state,
    base_frame_world_mm: np.ndarray,
    tool0_left_world_mm: np.ndarray,
    tool0_right_world_mm: np.ndarray,
    *,
    tolerance_position: float = None,
    tolerance_orientation: float = None,
    groups=None,
):
    """Enumerate EVERY ssik branch pair for a pose, collision status included.

    Debug companion to :func:`solve_dual_arm_ik_ssik`: builds the full Cartesian
    product of per-arm ssik branches (NO warm-start reduction, NO collision
    filtering) so a caller can show the user every candidate -- including the
    colliding ones -- when the normal solve reports "no collision-free branch
    pair". Each pair is collision-checked with a FULL report so the offending body
    pairs are available for red-highlighting. Even the branches the production
    solver skips in warm mode are returned here, which can reveal a clean branch
    the nearest-branch selection missed.

    Args:
        planner (PyBulletPlanner): active planner with the dual-arm cell loaded.
        template_state (RobotCellState): the failing movement's start_state (carries
            the per-keyframe attachments + allowed-touch policy).
        base_frame_world_mm (np.ndarray): 4x4 mm robot base frame.
        tool0_left_world_mm, tool0_right_world_mm (np.ndarray): 4x4 mm tool0 targets.
        tolerance_position, tolerance_orientation (float | None): FrameTarget
            tolerances; ``None`` -> the ``config.IK_*`` defaults.
        groups (tuple[str, str] | None): the ``(left_group, right_group)`` names;
            ``None`` -> derived from the cell semantics.

    Returns:
        dict: ``{"branch_counts": {"left": n, "right": m}, "candidates": [...]}``.
        Each candidate is ``{"index", "state", "in_collision", "summary",
        "offenders", "num_offending_pairs"}`` -- ``summary`` is a readable pair
        list (``"left_forearm_link <-> bar_B3; ..."``) and ``offenders`` a deduped
        ``[(kind, name), ...]`` list (``kind`` in ``{"link", "tool",
        "rigid_body"}``) for highlighting. ``candidates`` is sorted collision-free
        first, then fewest offending pairs.
    """
    from compas_fab.backends import CollisionCheckError

    tolerance_position = (
        tolerance_position if tolerance_position is not None else config.IK_TOLERANCE_POSITION
    )
    tolerance_orientation = (
        tolerance_orientation
        if tolerance_orientation is not None
        else config.IK_TOLERANCE_ORIENTATION
    )

    rcell = _require_loaded_cell(planner)
    left_group, right_group = _resolve_groups_arg(planner, groups)

    seed_state = template_state.copy()
    apply_base_frame_mm(seed_state, base_frame_world_mm)
    if seed_state.robot_configuration is None:
        seed_state.robot_configuration = rcell.default_cell_state().robot_configuration

    branch_cfgs, _group_joint_names, _targets = _ssik_branch_data(
        planner, seed_state, base_frame_world_mm,
        tool0_left_world_mm, tool0_right_world_mm,
        tolerance_position, tolerance_orientation,
        (left_group, right_group),
    )
    branch_counts = {side: len(branch_cfgs[side]) for side in ("left", "right")}

    candidates = []
    index = 0
    for lcfg in branch_cfgs["left"]:
        for rcfg in branch_cfgs["right"]:
            trial = seed_state.copy()
            trial.robot_configuration.merge(lcfg)
            trial.robot_configuration.merge(rcfg)
            in_collision = False
            summary = "clear (no collision)"
            offenders = []
            num_pairs = 0
            try:
                planner.check_collision(
                    trial, options={"full_report": True, "verbose": False}
                )
            except CollisionCheckError as exc:
                in_collision = True
                pairs = list(getattr(exc, "collision_pairs", []) or [])
                summary, offenders, num_pairs = _resolve_collision_pairs(pairs, rcell)
                if not pairs:
                    summary = str(exc).splitlines()[0] if str(exc) else "collision"
            except Exception as exc:
                # A non-collision failure (e.g. a cell/state mismatch): still show it.
                in_collision = True
                summary = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
            candidates.append({
                "index": index,
                "state": trial,
                "in_collision": in_collision,
                "summary": summary,
                "offenders": offenders,          # [(kind, name), ...] for highlighting
                "num_offending_pairs": num_pairs,
            })
            index += 1

    # Collision-free first, then fewest offending pairs -- so near-misses lead.
    candidates.sort(key=lambda c: (c["in_collision"], c["num_offending_pairs"]))
    return {"branch_counts": branch_counts, "candidates": candidates}


def _print_ssik_failure_breakdown(branch_counts, failures, *, stage: str) -> None:
    """Print a grouped ssik IK-failure report (mirrors the gradient breakdown).

    Two views, matching :func:`_print_ik_failure_breakdown`'s style so both backends
    read the same:
      1. Per-arm ssik branch availability -- a ``0`` here is the strongest modeling
         signal (the analytical solver can't even reach that arm's target).
      2. The distinct failure *reasons* (which collision, or which polish failure),
         collapsed to one line per reason with a count + the candidate indices.

    Args:
        branch_counts (dict): ``{"left": n, "right": m}`` ssik branch counts per arm.
        failures (list[tuple[int, str]]): ``(candidate_index, reason)`` for every
            branch pair rejected at ``stage`` (empty for a pure reachability report).
        stage (str): ``"reachability"`` (an arm has no branch), ``"collision"``
            (every branch pair collided), or ``"polish"`` (survivors failed polish).

    Returns:
        None. Prints to stdout only.
    """
    branch_summary = ", ".join(
        f"{side}={branch_counts.get(side, 0)}" for side in ("left", "right")
    )
    print(f"  ssik branches by arm: {branch_summary}")

    if stage == "reachability":
        print(
            "  -> An arm with 0 branches means its tool0 target is outside that arm's "
            "analytical reach. Usually a modeling problem: check the placed tool block "
            "pose and the robot base frame for this bar."
        )
        return

    if not failures:
        return

    # Collapse identical reasons to one line with a count + the candidate indices.
    reason_indices = {}  # reason -> [indices]
    for index, reason in failures:
        reason_indices.setdefault(reason, []).append(index)
    label = "candidate-pair collisions" if stage == "collision" else "polish failures"
    print(f"  ssik {label} by reason:")
    # Most frequent reason first (the dominant blocker), ties broken alphabetically.
    for reason, indices in sorted(reason_indices.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        print(f"    [{len(indices)}x] {reason} (candidates {indices})")


def solve_dual_arm_ik(
    planner,
    start_state,
    base_frame_world_mm: np.ndarray,
    tool0_left_world_mm: np.ndarray,
    tool0_right_world_mm: np.ndarray,
    *,
    check_collision: bool = True,
    max_restart_iter: int = None,
    max_descend_iterations: int = None,
    tolerance_position: float = None,
    tolerance_orientation: float = None,
    verbose_pairs: bool = False,
    groups=None,
    home_conf_12=None,
):
    """Solve IK for the left then right group, warm-started from ``start_state``.

    Each attempt descends DETERMINISTICALLY (the underlying pybullet solver is run
    with ``max_results=1``) from a warm-start configuration. The outer loop here
    owns the random restarts: attempt 0 uses the caller's seed (``start_state``'s
    configuration), and every subsequent attempt resamples a random dual-arm config
    to descend from (see ``_set_random_dual_arm_config``).

    This split lets a chained caller warm-start ``M2``/``M3`` from the previous
    keyframe with ``max_restart_iter=1`` (only the given seed, no randomization),
    while a cold solve with no good seed (``M1``) uses the full restart budget --
    random restarts are the only reliable way into a narrow collision-free basin
    (pybullet's per-group internal restarts do not couple the two arms and fail to
    find tight dual-arm poses).

    Args:
        planner (PyBulletPlanner): active planner with the dual-arm cell loaded.
        start_state (RobotCellState): seed state; its ``robot_configuration`` is
            the warm-start for attempt 0.
        base_frame_world_mm (np.ndarray): 4x4 mm robot base frame (overrides the
            state's stored base).
        tool0_left_world_mm, tool0_right_world_mm (np.ndarray): 4x4 mm tool0 targets.
        check_collision (bool): collision-check each candidate.
        max_restart_iter (int): number of warm-start attempts; ``None`` ->
            ``config.IK_MAX_RESTART_ITER`` (cold default). Pass ``1`` to use only the
            given seed with no random restarts.
        max_descend_iterations, tolerance_position, tolerance_orientation: solver
            tuning; ``None`` -> the ``config.IK_*`` defaults.
        verbose_pairs (bool): print the collision-pair summary on success.
        groups (tuple[str, str] | None): the ``(left_group, right_group)`` planning
            group names; ``None`` -> derived from the cell semantics.
        home_conf_12 (Sequence[float] | None): forwarded to the ssik backend, which
            needs it on cold solves (see :func:`solve_dual_arm_ik_ssik`); unused by
            the gradient backend.

    Returns:
        RobotCellState | None: the solved state copy on success, or ``None`` if every
        restart fails.
    """
    # Backend switch (keyframe.config.IK_BACKEND). "ssik" routes to the analytical
    # sidecar path (samples -> pick pair -> collision -> polish); anything else runs
    # the original PyBullet gradient descent below. Kept as a top-of-function
    # delegate so solve_keyframe_chain and the Rhino scripts need no change.
    if config.IK_BACKEND == "ssik":
        return solve_dual_arm_ik_ssik(
            planner,
            start_state,
            base_frame_world_mm,
            tool0_left_world_mm,
            tool0_right_world_mm,
            check_collision=check_collision,
            max_restart_iter=max_restart_iter,
            max_descend_iterations=max_descend_iterations,
            tolerance_position=tolerance_position,
            tolerance_orientation=tolerance_orientation,
            verbose_pairs=verbose_pairs,
            groups=groups,
            home_conf_12=home_conf_12,
        )

    rcell = _require_loaded_cell(planner)
    left_group, right_group = _resolve_groups_arg(planner, groups)
    deps = import_compas_stack()
    Frame = deps["Frame"]
    FrameTarget = deps["FrameTarget"]
    TargetMode = deps["TargetMode"]

    max_restart_iter = (
        config.IK_MAX_RESTART_ITER if max_restart_iter is None else max(1, int(max_restart_iter))
    )
    max_descend_iterations = (
        max_descend_iterations
        if max_descend_iterations is not None
        else config.IK_MAX_DESCEND_ITERATIONS
    )
    tolerance_position = (
        tolerance_position if tolerance_position is not None else config.IK_TOLERANCE_POSITION
    )
    tolerance_orientation = (
        tolerance_orientation
        if tolerance_orientation is not None
        else config.IK_TOLERANCE_ORIENTATION
    )

    seed_state = start_state.copy()
    apply_base_frame_mm(seed_state, base_frame_world_mm)

    # A movement can arrive with no start configuration (e.g. M1, whose start
    # config is left None to be filled by its downstream motion planner). IK still
    # needs a Configuration object to descend from and to merge each group's result
    # into, so fall back to the cell's default full configuration. Attempt 0 uses
    # it as the (arbitrary) seed and the random restarts below overwrite the arm
    # joints anyway, so this does not bias a cold solve.
    if seed_state.robot_configuration is None:
        seed_state.robot_configuration = rcell.default_cell_state().robot_configuration

    left_target = FrameTarget(
        mm_matrix_to_m_frame(Frame, tool0_left_world_mm),
        TargetMode.ROBOT,
        tolerance_position=tolerance_position,
        tolerance_orientation=tolerance_orientation,
    )
    right_target = FrameTarget(
        mm_matrix_to_m_frame(Frame, tool0_right_world_mm),
        TargetMode.ROBOT,
        tolerance_position=tolerance_position,
        tolerance_orientation=tolerance_orientation,
    )
    targets = ((left_target, left_group), (right_target, right_group))

    # max_results=1 -> deterministic descent from the warm-start; the outer loop
    # below supplies the random restarts.
    options = {
        "max_results": 1,
        "check_collision": check_collision,
        "max_descend_iterations": max_descend_iterations,
        "verbose": False,
    }
    # The two arms are solved one after the other (left, then right). The FIRST
    # arm must be solved WITHOUT collision checking: the second arm has not been
    # placed yet -- it still sits at the seed / random-restart config -- so a
    # collision caused by that not-yet-solved arm would falsely reject an
    # otherwise-valid first-arm solution (exactly the "left forearm vs right arm"
    # rejections seen in practice). The SECOND arm is solved with the caller's
    # collision setting; by then the first arm is at its real solved config, so
    # that check is meaningful. After both arms reach their targets, ONE
    # whole-robot collision check on the COMBINED pose (below) is what actually
    # guarantees the returned state is collision-free -- and it is where a bad
    # first-arm pose (solved without collision) finally gets caught. When the
    # caller disables collision entirely, all of this stays off.
    first_arm_options = dict(options)
    if check_collision:
        first_arm_options["check_collision"] = False

    print(
        f"keyframe.dual_arm_ik.solve_dual_arm_ik: check_collision={check_collision}, "
        f"max_restart_iter={max_restart_iter}"
    )
    last_failure = None
    # * Collect EVERY failure (attempt + group + reason) so we can print a full
    # * breakdown at the end instead of only the last one. This is the key signal
    # * for "why is IK hard here" -- e.g. is the RIGHT arm always unreachable, or
    # * is it collisions on random restarts?
    all_failures = []  # list of (attempt, group, reason_string)
    for attempt in range(max_restart_iter):
        trial = seed_state.copy()
        if attempt > 0:
            # No good seed (or the given one failed): warm-start from a random config.
            _set_random_dual_arm_config(planner, trial, (left_group, right_group))

        solved = True
        for i, (target, group) in enumerate(targets):
            # First arm (left): collision OFF; second arm (right): caller's setting.
            grp_options = first_arm_options if i == 0 else options
            try:
                cfg = planner.inverse_kinematics(target, trial, group=group, options=grp_options)
            except Exception as exc:  # reachability or collision (max_results=1 raises)
                reason = str(exc) or type(exc).__name__
                # A cell/state tools-or-workpieces mismatch is a SETUP error, not a
                # reachability failure -- every random restart would hit the same
                # wall. Abort immediately with actionable guidance rather than
                # churning through all max_restart_iter attempts and returning None.
                if _is_cell_state_mismatch(reason):
                    raise RuntimeError(
                        "keyframe.dual_arm_ik.solve_dual_arm_ik: the RobotCellState does not "
                        "match the RobotCell (tools/workpieces mismatch), so IK cannot "
                        "run -- retrying will not help. Fix the cell/state so they agree:\n"
                        "  - In Rhino: re-run RSRebuildRobotCell.\n"
                        "  - Headless: the loaded RobotCell.json and the movement's "
                        "start_state must reference the same tools / rigid bodies.\n"
                        f"  (planner message: {reason})"
                    ) from exc
                last_failure = f"group '{group}': {reason}"
                all_failures.append((attempt, group, reason))
                solved = False
                break

            if cfg is None:
                last_failure = f"group '{group}': no solution"
                all_failures.append((attempt, group, "no solution"))
                solved = False
                break
            trial.robot_configuration.merge(cfg)

        # Both arms reached their targets: validate the COMBINED dual-arm pose with
        # one whole-robot collision check (the first arm was solved without
        # collision, so this is where any of its collisions get caught). Uses the
        # same cell state -- hence the same allowed-collision policy -- as the IK
        # solves above. `check_collision` raises when the pose is in collision.
        if solved and check_collision:
            try:
                planner.check_collision(trial, options={"full_report": False, "verbose": False})
            except Exception as exc:
                reason = f"combined-pose collision: {str(exc) or type(exc).__name__}"
                last_failure = reason
                all_failures.append((attempt, "combined", reason))
                solved = False

        if solved:
            if attempt > 0:
                print(
                    f"keyframe.dual_arm_ik.solve_dual_arm_ik: solved on random restart "
                    f"{attempt}/{max_restart_iter - 1}."
                )
            if verbose_pairs:
                print(summarize_check_collision(planner, trial))
            return trial

    # ! No solution after all restarts -- dump the full failure breakdown.
    print(
        f"keyframe.dual_arm_ik.solve_dual_arm_ik: no solution after {max_restart_iter} "
        f"attempt(s) (last failure: {last_failure})."
    )
    _print_ik_failure_breakdown(all_failures)
    return None


def _print_ik_failure_breakdown(all_failures):
    """Summarize every IK failure so the bottleneck group / reason is obvious.

    Prints two views of the collected failures:
      1. A per-group tally of *which arm* failed and *how often* (points at a
         one-sided reachability problem -- e.g. the right arm never solves).
      2. A tally of the distinct failure *reasons* (collision vs. unreachable vs.
         no-solution), with the attempt indices that hit each one.

    Args:
        all_failures (list[tuple[int, str, str]]): ``(attempt, group, reason)``
            triples collected across every restart attempt in ``solve_dual_arm_ik``.

    Returns:
        None. Prints to stdout only.
    """
    if not all_failures:
        return

    # --- Per-group tally: which arm is the bottleneck? ---
    group_counts = {}
    for _attempt, group, _reason in all_failures:
        group_counts[group] = group_counts.get(group, 0) + 1
    group_summary = ", ".join(
        f"{group}={count}" for group, count in sorted(group_counts.items())
    )
    print(f"  IK failures by group: {group_summary}")

    # --- Per-reason tally: collision vs. unreachable vs. no-solution? ---
    # Group the attempt indices under each distinct (group, reason) so repeated
    # identical failures collapse to one line with a count.
    reason_attempts = {}  # (group, reason) -> list of attempt indices
    for attempt, group, reason in all_failures:
        reason_attempts.setdefault((group, reason), []).append(attempt)
    print("  IK failures by reason:")
    for (group, reason), attempts in sorted(reason_attempts.items()):
        print(
            f"    [{len(attempts)}x] group '{group}': {reason} "
            f"(attempts {attempts})"
        )


def extract_group_config(state, group: str, robot_cell) -> dict:
    """Return `{'joint_names': [...], 'joint_values': [...]}` for a group."""
    group_joint_names = list(robot_cell.get_configurable_joint_names(group))
    joint_values = [
        float(state.robot_configuration[name]) for name in group_joint_names
    ]
    return {"joint_names": group_joint_names, "joint_values": joint_values}


# ---------------------------------------------------------------------------
# Verbose pair-count summary
# ---------------------------------------------------------------------------


def summarize_check_collision(planner, state) -> str:
    """Run a verbose ``check_collision`` and return a formatted CC.1-CC.5 summary.

    Captures compas_fab's verbose stdout, parses each ``CC.<n> ... - <verdict>``
    line into pass/skipped(by-reason)/collision counters, and renders the
    table from the spec. Swallows the raw verbose lines (too noisy for the
    Rhino console). Moved here from the design repo's ``core.env_collision`` so
    the solvers carry their own reporting.
    """
    import contextlib
    import io
    import re

    buf = io.StringIO()
    error_msg = None
    with contextlib.redirect_stdout(buf):
        try:
            planner.check_collision(state, options={"verbose": True, "full_report": True})
        except Exception as exc:
            error_msg = str(exc)
    raw = buf.getvalue()

    line_re = re.compile(
        r"^CC\.(\d).*?-\s*(PASS|COLLISION|SKIPPED\s*\(([^)]+)\))",
        re.MULTILINE,
    )
    counters = {n: {"pass": 0, "collision": 0, "skipped": {}} for n in range(1, 6)}
    for m in line_re.finditer(raw):
        cc_n = int(m.group(1))
        verdict = m.group(2)
        if verdict == "PASS":
            counters[cc_n]["pass"] += 1
        elif verdict == "COLLISION":
            counters[cc_n]["collision"] += 1
        else:
            reason = (m.group(3) or "?").strip().lower()
            counters[cc_n]["skipped"][reason] = counters[cc_n]["skipped"].get(reason, 0) + 1

    labels = {
        1: "CC.1 link<->link",
        2: "CC.2 link<->tool",
        3: "CC.3 link<->env ",
        4: "CC.4 rb<->rb    ",
        5: "CC.5 tool<->env ",
    }
    out_lines = []
    for n in range(1, 6):
        c = counters[n]
        total = c["pass"] + c["collision"] + sum(c["skipped"].values())
        skipped_str = (
            ", ".join(f"{cnt}({reason})" for reason, cnt in c["skipped"].items())
            if c["skipped"] else "0"
        )
        out_lines.append(
            f"  {labels[n]} pairs={total:<4} "
            f"passed={c['pass']:<4} skipped={skipped_str:<24} flagged={c['collision']}"
        )
    if error_msg:
        out_lines.append(f"  (check_collision raised: {error_msg.splitlines()[0]})")
    return "\n".join(out_lines)
