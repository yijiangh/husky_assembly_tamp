"""Named planner variants for benchmark experiments.

A variant is one way of running ``plan_constrained_dual_arm`` that we want to
compare against the others on the same problems: which IK solver the search
uses, how strict its per-waypoint gates are, and whether the found path is
re-solved afterwards on the calibrated model. Every batch (``run --variant``)
carries exactly one variant name, and that name is written into the batch's
``config.json`` so a replay or a comparison always knows what was run.

The registry is plain data. Adding a variant means adding one entry here; the
runner, the summaries and the comparison page all read it by name.

Why the two "off" values look odd: the planner disables its continuity gate
when the threshold is infinite, but ``float("inf")`` cannot be written to JSON
in a way the browser's ``JSON.parse`` accepts. :func:`variant_record` is the
one place that turns it into the string ``"inf"`` for files; the planner
itself always receives the real float from :func:`resolve_variant`.
"""
from __future__ import annotations

import copy
import math
from typing import Any, Dict, List

from husky_assembly_tamp.motion_planner.dual_arm_task_space_rrt.core import (
    DEFAULT_JOINT_CONTINUITY_THRESHOLD_RAD,
)


# * The batch that is run when no variant is named: today's planner, plus the
# * strict re-solve of every found path so its outcome is recorded too.
DEFAULT_VARIANT = "v0_baseline"

# * The post-search repair every variant runs: re-solve the pose path on the
# * calibrated solver with the production continuity gate and full collision.
STRICT_REPAIR: Dict[str, Any] = {
    "joint_continuity_threshold_rad": DEFAULT_JOINT_CONTINUITY_THRESHOLD_RAD,
}

# * IK options for the search. The calibrated artifact is what production uses.
# * The nominal solver's tool pose sits off the calibrated model, so its FK
# * gate (checked against the calibrated PyBullet robot) has to be loosened
# * during the search; the repair afterwards makes the path exact.
# * Measured with benchmark/tools/nominal_gap.py over the cdfm_v1 pilot bars
# * and 3000 random configurations: at most 2.9 mm and 0.65 deg PER ARM. The
# * gate also compares the bar pose implied by the left tool with the one
# * implied by the right tool, where the two arms' errors can add up, so the
# * tolerances are set to about 2.5x the per-arm maximum: 8 mm and 0.03 rad.
CALIBRATED_IK: Dict[str, Any] = {"artifact_set": "calibrated"}
NOMINAL_IK: Dict[str, Any] = {
    "artifact_set": "nominal",
    "fk_pos_tolerance": 8e-3,
    "fk_ori_tolerance": 3e-2,
}

VARIANTS: Dict[str, Dict[str, Any]] = {
    "v0_baseline": {
        "description": "today's planner: calibrated solver, 10 deg continuity gate, "
                       "then the strict repair (recorded, so a failure is a finding)",
        "artifact_sets": ["calibrated"],
        "planner_kwargs": {
            "ik_options": dict(CALIBRATED_IK),
            "repair": dict(STRICT_REPAIR),
        },
    },
    "e0_diag": {
        "description": "v0 plus a diagnosis of every continuity stop (why the "
                       "arm solution jumped, and what the nominal solver would "
                       "have answered), for the E0 experiment",
        "artifact_sets": ["calibrated", "nominal"],
        "planner_kwargs": {
            "ik_options": {
                **CALIBRATED_IK,
                "diagnose_continuity": True,
                "diagnose_nominal_set": "nominal",
            },
            "repair": dict(STRICT_REPAIR),
        },
    },
    "v2_nocont": {
        "description": "continuity gate OFF during the search (limits and "
                       "collision on); the strict repair must undo any branch flips",
        "artifact_sets": ["calibrated"],
        "planner_kwargs": {
            "ik_options": dict(CALIBRATED_IK),
            "joint_continuity_threshold_rad": math.inf,
            "repair": dict(STRICT_REPAIR),
        },
    },
    "v1_nominal": {
        "description": "search with the closed-form nominal UR5e solver and a "
                       "loosened FK gate; the repair re-solves on the calibrated model",
        "artifact_sets": ["nominal", "calibrated"],
        "planner_kwargs": {
            "ik_options": dict(NOMINAL_IK),
            "repair": dict(STRICT_REPAIR),
        },
    },
    "v3_nominal_nocont": {
        "description": "v1 and v2 together: nominal solver, continuity gate off, "
                       "strict calibrated repair",
        "artifact_sets": ["nominal", "calibrated"],
        "planner_kwargs": {
            "ik_options": dict(NOMINAL_IK),
            "joint_continuity_threshold_rad": math.inf,
            "repair": dict(STRICT_REPAIR),
        },
    },
    # ? Added after the first smoke batches: 99 % of the coarse pass's
    # ? continuity stops happen on the first waypoint with a median jump of
    # ? 33 deg, and the nominal solver jumps just as much there. A 10 cm /
    # ? 0.1 rad bar step legitimately needs more than 10 deg on a UR5e joint,
    # ? so the coarse pass is rejecting honest steps. This variant scales the
    # ? gate with the step: 30 deg for the coarse pass, the usual 10 deg for
    # ? the fine pass (whose 2 mm steps the 10 deg gate was made for).
    "v4_coarse_gate30": {
        "description": "baseline solver, but the coarse pass's continuity gate "
                       "(search and repair) widened to 30 deg to match its 10 cm step",
        "artifact_sets": ["calibrated"],
        "planner_kwargs": {
            "ik_options": dict(CALIBRATED_IK),
            "repair": dict(STRICT_REPAIR),
        },
        "pass_kwargs": {
            "coarse": {
                "joint_continuity_threshold_rad": math.radians(30.0),
                "repair": {"joint_continuity_threshold_rad": math.radians(30.0)},
            },
        },
    },
}


def resolve_variant(name: str | None, pass_name: str | None = None) -> Dict[str, Any]:
    """Return one variant's settings, ready to hand to the planner.

    Args:
        name (str | None): a key of :data:`VARIANTS`; ``None`` means
            :data:`DEFAULT_VARIANT`.
        pass_name (str | None): when given ("coarse"/"fine"), the entry's
            ``pass_kwargs`` for that pass are merged over ``planner_kwargs``,
            so a variant can treat the two resolutions differently.

    Returns:
        Dict[str, Any]: a deep copy of the entry, with ``name`` added, so a
        caller may edit it without touching the registry.

    Raises:
        ValueError: for a name that is not in the registry. A typo must not
            quietly run the baseline.
    """
    key = name or DEFAULT_VARIANT
    if key not in VARIANTS:
        raise ValueError(
            f"Unknown planner variant {key!r}; known variants: {', '.join(sorted(VARIANTS))}"
        )
    entry = copy.deepcopy(VARIANTS[key])
    entry["name"] = key
    if pass_name is not None:
        entry["planner_kwargs"].update(entry.get("pass_kwargs", {}).get(pass_name, {}))
    return entry


def json_safe(value: Any) -> Any:
    """Return a copy of ``value`` that ``json.dump`` and the browser both accept.

    The only problem value is an infinite float (used to switch the continuity
    gate off); it becomes the string ``"inf"``. Everything else passes through.

    Args:
        value (Any): nested dicts, lists and scalars.

    Returns:
        Any: the converted copy.
    """
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and math.isinf(value):
        return "inf"
    return value


def variant_record(name: str | None) -> Dict[str, Any]:
    """Return the variant as it should be written to ``config.json``.

    Args:
        name (str | None): the variant name (``None`` = default).

    Returns:
        Dict[str, Any]: the resolved entry with every value JSON-safe.
    """
    return json_safe(resolve_variant(name))


def variant_names() -> List[str]:
    """Return the registered variant names, for the command line's choices."""
    return sorted(VARIANTS)
