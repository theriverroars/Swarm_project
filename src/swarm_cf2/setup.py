from setuptools import setup, find_packages
import os

setup(
    name='swarm_cf2',
    version='0.1.0',
    packages=find_packages(),
    package_data={
        'swarm_cf2.swarm_cf2': ['assets/*'],
    },
    include_package_data=True,
    install_requires=[
        'numpy',
        'pybullet',
        'scipy',
        'sympy',
        'matplotlib',
    ],
    entry_points={
        'console_scripts': [
            'swarm-emulate=swarm_cf2.swarm_cf2.emulate:emulate_main',
            'swarm-run=swarm_cf2.swarm_cf2.run:run_main',
        ],
    },
    author='Swarm Project',
    description='CBF-QP controlled Crazyflie swarm deployment package',
    python_requires='>=3.7',
)
