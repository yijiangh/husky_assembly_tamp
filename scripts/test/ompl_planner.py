from __future__ import annotations

import os
import sys
import numpy as np
from types import SimpleNamespace
from typing import Callable, List, Tuple, Optional

# Ensure parent 'scripts' directory (where ConstrainedPlanningCommon.py resides) is on PYTHONPATH
HERE = os.path.abspath(os.path.dirname(__file__))
SCRIPTS_ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
if SCRIPTS_ROOT not in sys.path:
    sys.path.append(SCRIPTS_ROOT)

# OMPL imports with fallback to local py-bindings if necessary
try:
    from ompl import base as ob
    from ompl import geometric as og
    from ompl import util as ou
except ImportError:
    from os.path import abspath, dirname, join

    # Try to locate the *py-bindings* directory one level up (as done in other scripts)
    sys.path.insert(0, join(dirname(dirname(dirname(abspath(__file__)))), "py-bindings"))
    from ompl import base as ob  # type: ignore
    from ompl import geometric as og  # type: ignore
    from ompl import util as ou  # type: ignore

from ConstrainedPlanningCommon import ConstrainedProblem, list2vec  # noqa: E402
from utils.util import interpolate  # noqa: E402


class _CallableConstraint(ob.Constraint):
    """Wrap user-supplied *function* and *jacobian* callables as an ``ompl.base.Constraint``.

    The provided callables must follow the OMPL signature::

        def function(x: np.ndarray, out: np.ndarray) -> None: ...
        def jacobian(x: np.ndarray, out: np.ndarray) -> None: ...

    where *x* is a 1-D array of length *ambient_dim* and *out* is pre-allocated.
    """

    def __init__(
        self,
        ambient_dim: int,
        codim: int,
        func: Callable[[np.ndarray, np.ndarray], None],
        jac: Callable[[np.ndarray, np.ndarray], None],
    ) -> None:
        super().__init__(ambient_dim, codim)
        self._func = func
        self._jac = jac

    # pylint: disable=arguments-differ
    def function(self, x, out):  # type: ignore[override]
        self._func(np.asarray(x), out)

    # pylint: disable=arguments-differ
    def jacobian(self, x, out):  # type: ignore[override]
        self._jac(np.asarray(x), out)


class OMPLConstrainedPlanner:
    """Generic constrained motion planner based on OMPL.

    Parameters
    ----------
    function : Callable
        Constraint *function* with OMPL signature (see :class:`_CallableConstraint`).
    jacobian : Callable
        Constraint Jacobian with OMPL signature (see :class:`_CallableConstraint`).
    ambient_dim : int
        Dimension of the ambient space (e.g. number of joints).
    codim : int
        Co-dimension of the constraint (ambient_dim - manifold_dim).
    bounds : Optional[List[Tuple[float, float]]]
        List of (low, high) tuples for each dimension. If *None*, no bounds are set.
    is_valid : Optional[Callable[[ob.State], bool]]
        Optional state validity checker. Defaults to a checker that always returns *True*.
    space_type : str, default "PJ"
        Constrained space type - one of {"PJ", "AT", "TB"}.
    planner_name : str, default "RRT"
        Name of the OMPL planner to use (must exist in *ompl.geometric*).
    interpolate_points : int, default 50
        Number of points for trajectory resampling.
    max_planning_time : float, default 60.0
        Allowed planning time in seconds.
    """

    def __init__(
        self,
        function: Callable[[np.ndarray, np.ndarray], None],
        jacobian: Callable[[np.ndarray, np.ndarray], None],
        ambient_dim: int,
        codim: int,
        bounds: Optional[List[Tuple[float, float]]] = None,
        is_valid: Optional[Callable[[ob.State], bool]] = None,
        *,
        space_type: str = "PJ",
        planner_name: str = "RRT",
        interpolate_points: int = 50,
        max_planning_time: float = 60.0,
    ) -> None:
        self._ambient_dim = ambient_dim
        self._planner_name = planner_name
        self._interpolate_points = interpolate_points

        # Create OMPL constraint wrapper
        self._constraint = _CallableConstraint(ambient_dim, codim, function, jacobian)

        # Construct state space with bounds if provided
        space = ob.RealVectorStateSpace(ambient_dim)
        if bounds is not None:
            if len(bounds) != ambient_dim:
                raise ValueError("`bounds` length must match `ambient_dim`.")
            ob_bounds = ob.RealVectorBounds(ambient_dim)
            for i, (low, high) in enumerate(bounds):
                ob_bounds.setLow(i, float(low))
                ob_bounds.setHigh(i, float(high))
            space.setBounds(ob_bounds)

        # Build *options* namespace expected by ConstrainedProblem
        self._options = SimpleNamespace(
            # Fundamental options
            tolerance=ob.CONSTRAINT_PROJECTION_TOLERANCE,
            tries=ob.CONSTRAINT_PROJECTION_MAX_ITERATIONS,
            delta=ob.CONSTRAINED_STATE_SPACE_DELTA,
            lambda_=ob.CONSTRAINED_STATE_SPACE_LAMBDA,
            exploration=ob.ATLAS_STATE_SPACE_EXPLORATION,
            epsilon=ob.ATLAS_STATE_SPACE_EPSILON,
            rho=ob.CONSTRAINED_STATE_SPACE_DELTA * ob.ATLAS_STATE_SPACE_RHO_MULTIPLIER,
            alpha=ob.ATLAS_STATE_SPACE_ALPHA,
            bias=False,
            no_separate=False,
            charts=ob.ATLAS_STATE_SPACE_MAX_CHARTS_PER_EXTENSION,
            time=float(max_planning_time),
            range=0.0,
            # Placeholder attributes used by ConstrainedProblem helpers
            space=space_type,
            planner=planner_name,
            bench=False,
        )

        # Instantiate *ConstrainedProblem*
        self._cp = ConstrainedProblem(space_type, space, self._constraint, self._options)

        # State validity checker – default allows everything
        self._is_valid = is_valid if is_valid is not None else (lambda state: True)

    # ---------------------------------------------------------------------
    # Public interface
    # ---------------------------------------------------------------------

    def plan(
        self,
        start: List[float] | np.ndarray,
        goal: List[float] | np.ndarray,
        *,
        interpolate_points: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        """Plan a path subject to the constraint.

        Parameters
        ----------
        start, goal : array-like
            Start and goal configurations in *ambient_dim*.
        interpolate_points : int, optional
            Desired number of points in the returned trajectory. Defaults to
            the value provided at construction.

        Returns
        -------
        np.ndarray or *None*
            Interpolated trajectory with shape (N, ambient_dim) on success, or
            *None* if planning failed.
        """
        start_arr = np.asarray(start, dtype=float)
        goal_arr = np.asarray(goal, dtype=float)

        if start_arr.shape != (self._ambient_dim,) or goal_arr.shape != (self._ambient_dim,):
            raise ValueError("`start` and `goal` must be of shape (ambient_dim,)")

        # Construct OMPL states
        sstart = ob.State(self._cp.css)
        sgoal = ob.State(self._cp.css)
        for i in range(self._ambient_dim):
            sstart[i] = start_arr[i]
            sgoal[i] = goal_arr[i]

        # Configure problem
        self._cp.setStartAndGoalStates(sstart, sgoal)
        self._cp.ss.setStateValidityChecker(ob.StateValidityCheckerFn(self._is_valid))
        self._cp.setPlanner(self._planner_name)

        # Solve
        stat = self._cp.solveOnce()
        if not stat:
            return None

        # Retrieve solution path
        path = self._cp.ss.getSolutionPath()
        state_count = path.getStateCount()
        traj = np.zeros((state_count, self._ambient_dim))
        for i in range(state_count):
            state = path.getState(i)
            traj[i] = [state[j] for j in range(self._ambient_dim)]

        # Interpolate / resample trajectory
        interp_pts = interpolate_points or self._interpolate_points
        if state_count >= 2 and interp_pts > state_count:
            traj = interpolate(traj, interp_pts)
        return traj

    # Convenience aliases --------------------------------------------------

    @property
    def constraint(self) -> ob.Constraint:  # pragma: no cover
        """Return the underlying OMPL constraint object."""
        return self._constraint

    @property
    def css(self):  # pragma: no cover
        """Return the underlying constrained state space."""
        return self._cp.css

    @property
    def simple_setup(self):  # pragma: no cover
        """Return the OMPL :class:`ompl.geometric.SimpleSetup`."""
        return self._cp.ss
