"""
Emulate mode entry point.
This script runs the simulation in emulation mode (PyBullet visualization).
"""

import sys
import argparse
from .control_loop import run_through_hoops_control
from .config import (
    DEFAULT_SPAWN_COORDINATES,
    HOOP_POSITIONS,
    SIMULATION_CONFIG,
    PRIORITY_WEIGHTS
)


def emulate_main():
    """
    Main entry point for emulate mode.
    Runs the CBF-QP control algorithm in PyBullet simulation with visualization.
    """
    parser = argparse.ArgumentParser(
        description='Run swarm control in emulation mode (PyBullet simulation)'
    )
    parser.add_argument(
        '--duration',
        type=float,
        default=SIMULATION_CONFIG['duration'],
        help='Simulation duration in seconds'
    )
    parser.add_argument(
        '--speed',
        type=float,
        default=SIMULATION_CONFIG['cbf_drone_speed'],
        help='Drone vertical speed (m/s)'
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=SIMULATION_CONFIG['cbf_gamma'],
        help='CBF gamma parameter (aggressiveness)'
    )
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Disable visualization plots'
    )
    parser.add_argument(
        '--no-csv',
        action='store_true',
        help='Disable CSV export'
    )
    parser.add_argument(
        '--hoop',
        type=str,
        default='hoop_1',
        choices=list(HOOP_POSITIONS.keys()),
        help='Select which hoop configuration to use'
    )
    
    args = parser.parse_args()
    
    # Get selected hoop configuration
    hoop_config = HOOP_POSITIONS[args.hoop]
    
    print("=" * 60)
    print("SWARM CF2 - Emulate Mode")
    print("=" * 60)
    print(f"Duration: {args.duration}s")
    print(f"Speed: {args.speed} m/s")
    print(f"CBF Gamma: {args.gamma}")
    print(f"Hoop: {args.hoop} at {hoop_config['center']}")
    print("=" * 60)
    
    # Run the control loop
    results = run_through_hoops_control(
        duration=args.duration,
        spawn_coordinates=DEFAULT_SPAWN_COORDINATES,
        hoop_coordinates=hoop_config['center'],
        hoop_direction=hoop_config['direction'],
        cbf_drone_speed=args.speed,
        cbf_gamma=args.gamma,
        enable_visualization=not args.no_viz,
        export_csv=not args.no_csv,
    )
    
    print("\n" + "=" * 60)
    print("Emulation complete!")
    print("=" * 60)
    
    return results


if __name__ == '__main__':
    emulate_main()
