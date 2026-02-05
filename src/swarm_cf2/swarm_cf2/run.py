"""
Run mode entry point.
This script runs the control algorithm for deployment (typically would connect to real hardware).
For this implementation, it calls the run_through_hoops_control function.
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


def run_main():
    """
    Main entry point for run mode.
    In a real deployment, this would connect to actual Crazyflie hardware.
    For now, it runs the same simulation as emulate mode.
    """
    parser = argparse.ArgumentParser(
        description='Run swarm control in deployment mode'
    )
    parser.add_argument(
        '--duration',
        type=float,
        default=SIMULATION_CONFIG['duration'],
        help='Mission duration in seconds'
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
        '--hoop',
        type=str,
        default='hoop_1',
        choices=list(HOOP_POSITIONS.keys()),
        help='Select which hoop configuration to use'
    )
    parser.add_argument(
        '--viz',
        action='store_true',
        help='Enable visualization plots'
    )
    
    args = parser.parse_args()
    
    # Get selected hoop configuration
    hoop_config = HOOP_POSITIONS[args.hoop]
    
    print("=" * 60)
    print("SWARM CF2 - Run Mode")
    print("=" * 60)
    print(f"Duration: {args.duration}s")
    print(f"Speed: {args.speed} m/s")
    print(f"CBF Gamma: {args.gamma}")
    print(f"Hoop: {args.hoop} at {hoop_config['center']}")
    print("=" * 60)
    print("\nNOTE: This is currently running in simulation mode.")
    print("For real hardware deployment, additional code would be needed")
    print("to interface with Crazyflie radio and motion capture system.")
    print("=" * 60)
    
    # Run the control loop (same as emulate, but in deployment would connect to real drones)
    results = run_through_hoops_control(
        duration=args.duration,
        spawn_coordinates=DEFAULT_SPAWN_COORDINATES,
        hoop_coordinates=hoop_config['center'],
        hoop_direction=hoop_config['direction'],
        cbf_drone_speed=args.speed,
        cbf_gamma=args.gamma,
        enable_visualization=args.viz,  # Visualization off by default in run mode
        export_csv=True,
    )
    
    print("\n" + "=" * 60)
    print("Mission complete!")
    print("=" * 60)
    
    return results


if __name__ == '__main__':
    run_main()
