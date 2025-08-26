import json
import os
import random
import sys
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List, Set, Tuple, Union

HERE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(HERE)

import matplotlib.pyplot as plt
import numpy as np
import pybullet_planning as pp
from termcolor import cprint


def normalize_angles(angles, low: float = -np.pi, high: float = np.pi):
    """
    Normalize an iterable of angles to the range (low, high].

    Supports tuple, list, and np.ndarray. The return type matches the input type.
    """
    span = high - low
    if not np.isfinite(span) or span <= 0:
        raise ValueError("Invalid angle range: 'high' must be greater than 'low'.")

    array_like = np.asarray(angles, dtype=float)
    shifted = np.fmod(array_like - low, span)
    normalized = np.where(shifted <= 0, shifted + high, shifted + low)

    if isinstance(angles, np.ndarray):
        return normalized
    if isinstance(angles, tuple):
        return tuple(normalized.tolist())
    if isinstance(angles, list):
        return normalized.tolist()
    # Fallback: return numpy array if an unexpected type is provided
    return normalized


def normalize_angle(angle, low: float = -np.pi, high: float = np.pi):
    """Normalize a single angle to the range (low, high]."""
    span = high - low
    if not np.isfinite(span) or span <= 0:
        raise ValueError("Invalid angle range: 'high' must be greater than 'low'.")

    shifted = np.fmod(angle - low, span)
    if shifted <= 0:
        return shifted + high
    return shifted + low


def angles_distance(angles1, angles2):
    """
    Compute the Euclidean norm of directed circular differences between two angle vectors.

    The per-joint difference is the signed minimal angle delta δ such that
    normalize_angle(angle2 + δ) == normalize_angle(angle1).
    """
    a1 = np.asarray(angles1, dtype=float)
    a2 = np.asarray(angles2, dtype=float)
    diff = a1 - a2
    diff = normalize_angles(diff)
    return np.linalg.norm(diff)


def angle_distance(angle1, angle2):
    """Compute the signed minimal circular difference δ with normalize_angle(angle2 + δ) == normalize_angle(angle1)."""
    diff = angle1 - angle2
    return normalize_angle(diff)


def interpolate(trajectory: np.ndarray, target_length: int) -> np.ndarray:
    """
    Resample a trajectory to the target length while preserving all original waypoints.

    Args:
        trajectory: Input trajectory of shape [N, D], where N is timesteps and D is dimensionality per step.
        target_length: Desired trajectory length.

    Returns:
        np.ndarray: Interpolated trajectory of shape [target_length, D].
    """
    # Original trajectory length and dimensionality
    orig_length, dims = trajectory.shape

    # Downsample if the target length is less than or equal to the original length
    if target_length <= orig_length:
        # Select evenly spaced indices
        indices = np.round(np.linspace(0, orig_length - 1, target_length)).astype(int)
        return trajectory[indices]

    # Allocate new trajectory initialized to zeros
    new_trajectory = np.zeros((target_length, dims))

    # Ensure all original waypoints are preserved first
    # Compute indices of original points in the new trajectory
    orig_indices_in_new = np.round(np.linspace(0, target_length - 1, orig_length)).astype(int)

    # Place original points into the new trajectory
    for i, idx in enumerate(orig_indices_in_new):
        new_trajectory[idx] = trajectory[i]

    # Create a mask for positions that already have values
    mask = np.zeros(target_length, dtype=bool)
    mask[orig_indices_in_new] = True

    # Interpolate positions without assigned values
    for i in range(target_length):
        if not mask[i]:
            # Find nearest known points on both sides
            left_idx = np.max(orig_indices_in_new[orig_indices_in_new < i]) if any(orig_indices_in_new < i) else 0
            right_idx = np.min(orig_indices_in_new[orig_indices_in_new > i]) if any(orig_indices_in_new > i) else target_length - 1

            # If both indices are the same, use the nearest point
            if left_idx == right_idx:
                new_trajectory[i] = new_trajectory[left_idx]
                continue

            # Compute interpolation weight
            left_orig_idx = np.where(orig_indices_in_new == left_idx)[0][0]
            right_orig_idx = np.where(orig_indices_in_new == right_idx)[0][0]

            weight = (i - left_idx) / (right_idx - left_idx)

            # Linear interpolation
            new_trajectory[i] = (1 - weight) * trajectory[left_orig_idx] + weight * trajectory[right_orig_idx]

    return new_trajectory


###########################################


class CounterValue:
    """
    A simple numeric counter associated with a CounterModule.

    Tracks an accumulated numeric value and the most recent increment applied.
    """

    def __init__(self, name, parent):
        """
        Initialize a CounterValue and register it with its parent module.

        Args:
            name: Unique name of this counter within the parent module.
            parent: The CounterModule that owns this counter.
        """
        self.name = name
        self.parent = parent
        self.value = 0
        self.last_update = 0
        parent.values[name] = self

    def increment(self, value=1):
        """Increase the counter by the specified value and record the increment size."""
        self.last_update = value
        self.value += value

    def update(self, value):
        """Set the counter to an absolute value without changing structure relationships."""
        self.value = value


class CounterModule:
    """
    A hierarchical counter manager.

    Each module can hold multiple named CounterValue instances and can create
    nested child modules ("handles"). All modules created within the same tree
    share a single registry (modules dict) anchored at the root, which allows
    aggregation, traversal, and persistence of counters.
    """

    def __init__(self, root=None, name=None, parent=None):
        """
        Create a counter module.

        Args:
            root: If None, create a new root registry. If provided, this module
                will share the registry from the given root (used internally).
            name: Name of this module; the root typically uses "root".
            parent: Optional parent module to form a hierarchy.
        """
        if root is None:
            self.modules = {}
            self.name = "root"
        else:
            self.modules = root.modules

        self.name = name
        self.parent = parent
        self.children = []
        self.values = {}

        if name is not None:
            if name not in self.modules:
                self.modules[name] = self
            if parent:
                parent.children.append(self)

    def create_handle(self, name):
        """Create a child module (handle) under this module and return it."""
        return CounterModule(root=self, name=name, parent=self)

    def add_counter_value(self, name):
        """
        Get or create a CounterValue with the given name within this module.

        Returns:
            CounterValue: Existing or newly created counter value.
        """
        if name in self.values:
            return self.values[name]
        else:
            counter_value = CounterValue(name, self)
            return counter_value

    def plot(self):
        """
        Render a side-by-side bar chart of all counter values grouped by module.

        Aggregates values across the hierarchy by handle (module) name and draws
        a grouped bar chart using matplotlib.
        """

        def collect_data(module, collected=None):
            if collected is None:
                collected = defaultdict(lambda: defaultdict(int))
            if module.name is not None:
                for value in module.values:
                    collected[module.name][value.name] += value.value
            for child in module.children:
                collect_data(child, collected)
            return collected

        data = collect_data(self)

        handles = list(data.keys())
        value_labels = list(set(vname for handle_values in data.values() for vname in handle_values.keys()))

        color_map = dict(zip(value_labels, plt.cm.viridis(np.linspace(0, 1, len(value_labels)))))

        bar_width = 0.8 / len(value_labels)
        index = np.arange(len(handles))

        plt.figure(figsize=(10, 5))

        for i, value_name in enumerate(value_labels):
            heights = [data[handle].get(value_name, 0) for handle in handles]
            bar_positions = index + i * bar_width
            plt.bar(bar_positions, heights, bar_width, color=color_map[value_name], label=value_name)

            for j, height in enumerate(heights):
                if height > 0:
                    plt.text(bar_positions[j], height + 0.1, str(height), ha="center")

        plt.xlabel("Handle Names")
        plt.ylabel("Counts")
        plt.title("Counter Module with Side-by-Side Values per Handle")
        plt.xticks(index + bar_width * (len(value_labels) - 1) / 2, handles)
        plt.legend(title="Value Names")
        plt.show()

    def reset(self):
        """Reset all counters' values in the entire registry to zero."""
        for module in self.modules.values():
            for value in module.values:
                value.value = 0

    def save(self, path, filename):
        """
        Persist all counters in the registry to a JSON file.

        The JSON structure is {module_name: {counter_name: value, ...}, ...}.

        Args:
            path: Directory to save the file into. Created if it does not exist.
            filename: Target JSON filename.
        """
        data_to_save = {name: {value.name: value.value for value in module.values.values()} for name, module in self.modules.items()}
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, filename)
        with open(file_path, "w") as file:
            json.dump(data_to_save, file)
        print(f"Counter values saved to {filename}")


class TermPrint(object):
    last_empty_line = False

    def __init__(self) -> None:
        pass

    @classmethod
    def print(cls, text: str, color: str = "white", blank_f: bool = False, blank_b: bool = False):
        if blank_f and not cls.last_empty_line:
            print("")
        cprint(text, color)
        if blank_b:
            print("")
            cls.last_empty_line = True
        else:
            cls.last_empty_line = False


class PrintManager:
    """Print control utility for consistent terminal output with indentation levels."""

    # Predefined color mapping for different message types
    COLORS = {"info": "white", "success": "green", "warning": "yellow", "error": "red", "debug": "cyan", "highlight": "magenta"}

    # Singleton to ensure a single manager instance
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(PrintManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, indent_size: int = 4, tab_char: str = " ", use_color: bool = True, default_color: str = "white"):
        """
        Initialize the print manager.

        Args:
            indent_size: Number of spaces per indentation level.
            tab_char: Character used for indentation.
            use_color: Whether to use colored output.
            default_color: Default color name.
        """
        # Avoid re-initialization for the singleton instance
        if hasattr(self, "indent_level"):
            return

        self.indent_size = indent_size
        self.tab_char = tab_char
        self.indent_level = 0
        self.use_color = use_color
        self.default_color = default_color

    def _get_indent(self, level: int = None) -> str:
        """
        Return the indentation string for the specified level.

        Args:
            level: Indentation level to use; defaults to the current level.

        Returns:
            Indentation string.
        """
        if level is None:
            level = self.indent_level
        return self.tab_char * self.indent_size * level

    def print(self, message: str, indent_level: int = None, color: str = None, end: str = "\n", flush: bool = True):
        """
        Print a message with a given indentation level and color.

        Args:
            message: The message to print.
            indent_level: Indentation level; defaults to the current level.
            color: Color name; defaults to the manager's default color.
            end: Line terminator.
            flush: Whether to flush the output stream immediately.
        """
        if indent_level is None:
            indent_level = self.indent_level

        if color is None:
            color = self.default_color

        indent_str = self._get_indent(indent_level)
        formatted_message = f"{indent_str}{message}"

        if self.use_color:
            cprint(formatted_message, color, end=end, flush=flush)
        else:
            print(formatted_message, end=end, flush=flush)

    def info(self, message: str, indent_level: int = None):
        """Print an informational message."""
        self.print(message, indent_level, self.COLORS["info"])

    def success(self, message: str, indent_level: int = None):
        """Print a success message."""
        self.print(message, indent_level, self.COLORS["success"])

    def warning(self, message: str, indent_level: int = None):
        """Print a warning message."""
        self.print(message, indent_level, self.COLORS["warning"])

    def error(self, message: str, indent_level: int = None):
        """Print an error message."""
        self.print(message, indent_level, self.COLORS["error"])

    def debug(self, message: str, indent_level: int = None):
        """Print a debug message."""
        self.print(message, indent_level, self.COLORS["debug"])

    def highlight(self, message: str, indent_level: int = None):
        """Print a highlighted message."""
        self.print(message, indent_level, self.COLORS["highlight"])

    def indent(self, levels: int = 1):
        """Increase the indentation level by the given number of levels."""
        self.indent_level += levels
        return self

    def dedent(self, levels: int = 1):
        """Decrease the indentation level by the given number of levels."""
        self.indent_level = max(0, self.indent_level - levels)
        return self

    def reset_indent(self):
        """Reset the indentation level to zero."""
        self.indent_level = 0
        return self

    def set_indent(self, level: int):
        """Set the indentation level directly to the specified value."""
        self.indent_level = max(0, level)
        return self

    @contextmanager
    def indented(self, levels: int = 1):
        """
        Context manager that temporarily increases the indentation level.

        Args:
            levels: Number of levels to increase the indentation by.

        Example:
            printer = PrintManager()
            printer.info("root level message")
            with printer.indented():
                printer.info("indented by 1 level")
                with printer.indented(2):
                    printer.info("indented by 3 levels")
            printer.info("back to root level")
        """
        self.indent(levels)
        try:
            yield self
        finally:
            self.dedent(levels)


def flatten(nested_list):
    """
    Flatten a nested structure of arbitrary depth containing list, tuple, and np.ndarray.

    Mixed nesting is supported (e.g., a list containing many np.ndarray).
    The return container type matches the outermost input type:
      - list -> list
      - tuple -> tuple
      - np.ndarray -> np.ndarray

    Args:
        nested_list: The nested container to flatten.

    Returns:
        A fully flattened 1-D container of the same outer type as the input
        containing the elements in traversal order.
    """

    def _append_flat(destination_list, item):
        # Recursively flatten lists and tuples
        if isinstance(item, (list, tuple)):
            for sub_item in item:
                _append_flat(destination_list, sub_item)
            return

        # For numpy arrays: handle both numeric arrays and object arrays
        if isinstance(item, np.ndarray):
            if item.dtype == object:
                # Iterate over elements (which may themselves be containers)
                for sub_item in item.flat:
                    _append_flat(destination_list, sub_item)
            else:
                # Fast path for numeric arrays
                destination_list.extend(item.ravel().tolist())
            return

        # Base case: non-container element
        destination_list.append(item)

    flat_list = []
    _append_flat(flat_list, nested_list)

    # Match the outermost container type
    if isinstance(nested_list, tuple):
        return tuple(flat_list)
    if isinstance(nested_list, np.ndarray):
        return np.asarray(flat_list)
    return flat_list


def SetSeeds(seed=24):
    random.seed(seed)
    np.random.seed(seed)
    pp.set_numpy_seed(seed)
    pp.set_random_seed(seed)
