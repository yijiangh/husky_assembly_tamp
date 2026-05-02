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
    ],
    extras_require={
        "dev": [
            "pytest",
        ],
        "video": [
            "imageio[ffmpeg]",
        ],
    },
    zip_safe=False,
)
