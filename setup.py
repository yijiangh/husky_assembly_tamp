from setuptools import setup, find_packages

setup(
    name="husky_assembly_tamp",
    version="0.1.0",
    description="Task and Motion Planning for Husky Assembly",
    author="Zihao Li, Yijiang Huang",
    packages=find_packages(exclude=["test", "test.*", "*.test", "*.test.*"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pybullet",
        "pybullet_planning==0.6.1",
        "compas>=2.0",
        "compas_fab @ git+https://github.com/compas-dev/compas_fab.git@wip_process",
        "compas_robots>=0.5",
        "matplotlib",
        # Shared BarAssemblyAction / Movement schema; pinned SHA matches
        # husky-assembly-teleop and bar_joint_rhino_design_workflow.
        "rs_data_structure @ git+https://github.com/yijiangh/rs_data_structure.git@ce01ca0606ebf4a3a07505919cf96b7c29009b2e",
    ],
    extras_require={
        "dev": [
            "pytest",
        ],
        "video": [
            "imageio[ffmpeg]",
        ],
        # Analytical IK backend, imported in-process. ssik 4.1+ ships a cp310
        # wheel and requires Python >= 3.10, so the ROS2 venv (Python 3.10) runs
        # it NATIVELY -- no sidecar. (The old pin `ssik>=3.0,<4` needed Python
        # 3.11+, which is why the sidecar existed.) Rhino's CPython 3.9 still
        # can't install ssik and reaches it via the sidecar instead.
        # NOTE: ssik pulls numpy>=1.26; on a ROS2 venv keep numpy on the 1.x line
        # (e.g. 1.26.4), NOT numpy 2.x, to stay ABI-compatible with rclpy/pybullet.
        "ssik": [
            "ssik>=4.1,<5; python_version >= '3.10'",
        ],
    },
    zip_safe=False,
)
