"""Home-configuration constants used by the headless planner scripts here.

* Kept as a small standalone module next to the scripts so they stay importable
* without ros2 / the husky_assembly_teleop package. Values mirror
* ``husky_assembly_teleop/utils.py`` (HUSKY_DUAL_ARM_HOME_CONF_12).
"""

import numpy as np

# Dual-arm "home" configuration (12 joints: left arm 6 then right arm 6).
# Same numbers as husky_assembly_teleop.utils.HUSKY_DUAL_ARM_HOME_CONF_12.
HUSKY_DUAL_ARM_HOME_CONF_12 = np.array([
    -1.381079037103113, -0.08674286382411818, -2.8050931738052864,
    -1.7444565873683324, 0.23963370629882144, 1.4217452086745808,
     1.3946926052686688, -3.0267499888085663,  2.8043950421044888,
    -1.727003294848389, -0.40561451816348215, -1.2402309664671707,
])

# Per-arm halves of the home configuration, used to fill missing joints.
HOME_CONF_LEFT_6 = HUSKY_DUAL_ARM_HOME_CONF_12[:6]
HOME_CONF_RIGHT_6 = HUSKY_DUAL_ARM_HOME_CONF_12[6:]
