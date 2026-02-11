import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'nmpc_cf2'

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            os.path.join("share", package_name, "launch"),
            glob(os.path.join("launch", "*launch.[pxy][yma]*")),
        ),
        (
            os.path.join("share", package_name, "config"),
            glob(os.path.join("config", "*.yaml")),
        ),
        (
            os.path.join("share", package_name, "rviz"),
            glob(os.path.join("config", "*.rviz")),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="varad",
    maintainer_email="vaidyavarad2001@gmail.com",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "singlemarker2fullpose = nmpc_cf2.singlemarker2fullpose:main",
            "nmpc_run = nmpc_cf2.nmpc_run:main",
            "cbf_run = nmpc_cf2.cbf_run:main",
            "cbf_run_sim = nmpc_cf2.cbf_run_sim:main",
        ],
    },
)
