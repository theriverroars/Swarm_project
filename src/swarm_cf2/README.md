# Swarm CF2 Deployment Package

This package provides a deployment structure for the CBF-QP controlled Crazyflie swarm system.

## Installation

```bash
cd src/swarm_cf2
pip install -e .
```

## Usage

### Emulate Mode (Simulation with Visualization)

```bash
swarm-emulate --duration 30 --speed 0.05 --gamma 1.0 --hoop hoop_1
```

### Run Mode (Deployment)

```bash
swarm-run --duration 30 --speed 0.05 --gamma 1.0 --hoop hoop_1
```

## Configuration

Hard-coded hoop positions and parameters are defined in `swarm_cf2/config.py`.

## Package Structure

```
swarm_cf2/
├── swarm_cf2/
│   ├── __init__.py
│   ├── CBF_controller.py       # CBF-QP safety controller
│   ├── nominal_controller.py   # PID controller
│   ├── quadrotor_info.py       # Quadrotor parameters and data structures
│   ├── quadrotor_swarm_sim.py  # PyBullet simulation handler
│   ├── control_loop.py         # Main control loop function
│   ├── config.py               # Configuration (hoop positions, etc.)
│   ├── emulate.py              # Emulate mode entry point
│   └── run.py                  # Run mode entry point
├── setup.py
└── README.md
```
