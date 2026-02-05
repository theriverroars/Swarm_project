"""
Configuration file for hoop positions and simulation parameters.
This file contains hard-coded hoop positions and other deployment parameters.
"""

import numpy as np

# Hard-coded hoop positions
HOOP_POSITIONS = {
    'hoop_1': {
        'center': np.array([0.0, 0.0, 1.5]),
        'direction': np.array([0.0, 0.0, 1.0]),  # Normal vector pointing up
        'radius': 0.5,
    },
    'hoop_2': {
        'center': np.array([0.5, 0.5, 2.5]),
        'direction': np.array([0.0, 0.0, 1.0]),
        'radius': 0.5,
    },
    'hoop_3': {
        'center': np.array([-0.5, 0.5, 3.5]),
        'direction': np.array([0.0, 0.0, 1.0]),
        'radius': 0.5,
    },
}

# Default spawn coordinates for drones
DEFAULT_SPAWN_COORDINATES = np.array([
    [0.0, 0.0, 0.2],    # Agent 0
    [0.0, 0.3, 0.4],    # Agent 1
    [0.0, -0.2, 0.6],   # Agent 2
])

# Simulation parameters
SIMULATION_CONFIG = {
    'duration': 30.0,              # Simulation duration in seconds
    'cbf_drone_speed': 0.05,       # Desired vertical speed (m/s)
    'cbf_gamma': 1.0,              # CBF aggressiveness parameter
    'enable_visualization': True,   # Show plots after simulation
    'export_csv': True,            # Export position data to CSV
}

# Priority weights for agent prioritization
PRIORITY_WEIGHTS = {
    'distance_weight': 0.5,        # Weight for distance to hoop
    'velocity_weight': 0.2,        # Weight for velocity towards hoop
    'cbf_weight': 0.3,             # Weight for CBF value (safety margin)
}

# Control parameters
CONTROL_CONFIG = {
    'priority_threshold': 0.001,   # Threshold for grouping equal priorities
}
