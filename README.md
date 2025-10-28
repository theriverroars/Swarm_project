# Safe Quadrotor Swarm Simulation with CBF-QP

This project implements a multi-quadrotor simulation in PyBullet, featuring a safety-critical control system. A primary quadrotor is tasked with navigating through a swarm of other drones, which act as dynamic obstacles.

The control architecture is two-layered:

1. **Nominal Controller:** A standard PID controller (`nominal_controller.py`) calculates the ideal commands to reach a target position.

2. **Safety Filter:** A **Control Barrier Function (CBF)** implemented as a **Quadratic Program (QP)** (`CBF_controller.py`) takes the nominal command and modifies it *only if necessary* to guarantee collision avoidance.

This allows the drone to follow its primary objective (PID control) while formally guaranteeing safety (CBF-QP filter).

## File Structure

```

.
├── assets/
│   └── cf2x.urdf             \# URDF model for the Crazyflie 2.0 quadrotor
├── CBF\_controller.py         \# Defines the CBFQPControllerDrone class (safety filter)
├── nominal\_controller.py     \# Defines the NominalPIDControllerDrone class (primary controller)
├── quadrotor\_info.py         \# Defines the Quadrotor data class and default parameters
├── quadrotor\_swarm\_sim.py    \# Contains the PybulletHandler class to manage the simulation
└── run.py                    \# The main script to launch and run the simulation

```

## How it Works

1. **`run.py`**: This is the main executable. It initializes the PyBullet simulation, spawns multiple quadrotors, and runs the main control loop.

2. **`quadrotor_swarm_sim.py`**: The `PybulletHandler` class manages all direct interactions with PyBullet, such as stepping the simulation, applying forces, and reading sensor data.

3. **`quadrotor_info.py`**: The `Quadrotor` class is a data container that holds the physical parameters (mass, inertia), PID gains, and current state (position, velocity) for each drone.

4. **Control Loop (in `run.py`)**:

   * For the primary drone (Agent 0), the `NominalPIDControllerDrone` computes a `ref_thrusts` command to move it upwards.

   * This `ref_thrusts` is fed into the `CBFQPControllerDrone.solve_QP()` method, along with the positions of all other drones (which are treated as obstacles).

   * The `solve_QP` function determines if `ref_thrusts` violates the safety barrier. If it does, it calculates a minimal `safety_thrust` correction.

   * The final, safe command (`ref_thrusts + safety_thrust`) is applied to the drone.

   * All other "obstacle" drones simply run their own PID controllers to hover in place.

## Installation

1. **Clone the repository:**

```
git clone https://github.com/theriverroars/Swarm_project.git
cd Swarm_project-minimalist-cbf
```

2. **Install dependencies:**
This project requires `numpy`, `pybullet`, `scipy`, and `sympy`.

```
pip install numpy pybullet scipy sympy
```

## How to Run

Simply execute the main `run.py` script from your terminal:

```
python run.py
```

A PyBullet window will open, and you will see the primary drone (Agent 0) begin to move upwards, automatically navigating around the other hovering drones as it ascends.

```
