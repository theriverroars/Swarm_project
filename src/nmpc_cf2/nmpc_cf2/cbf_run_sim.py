#!/usr/bin/env python
"""
CBF Multi-Agent Deployment - PyBullet Simulation Version

This version uses PyBullet simulator instead of real Crazyflies.
No ROS2/Crazyswarm required - purely for algorithm testing and development.

Usage:
    python cbf_run_sim.py --num_drones 3 --duration 30 --height 1.5
"""

import os
import sys
import argparse
from time import sleep
import numpy as np

# Import CBF controller components
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'CBF_controller'))
from controllers_hub import CBFQPControllerDrone, NominalPIDControllerDrone
from quadrotor_info import Quadrotor, quadrotor_params, GRAVITY_ACC, SIM_TIMESTEP
from CBF_controller.quadrotor_swarm_sim import PybulletHandler
from CBF_controller.utilities import spawn_agents, export_positions_to_csv, plot_priority_history


def initialize_simulation(num_drones, spawn_coords=None):
    """
    Initialize PyBullet simulation with multiple drones.
    
    Args:
        num_drones: Number of drones to simulate
        spawn_coords: Optional custom spawn coordinates (num_drones x 3)
        
    Returns:
        pybullet_handler: PybulletHandler simulation manager
        agents_list: List of Quadrotor state objects
    """
    # Initialize PyBullet
    pybullet_handler = PybulletHandler(
        acceleration_gravity=GRAVITY_ACC,
        time_step_size=SIM_TIMESTEP,
    )
    
    # Default spawn coordinates (spread along X-axis at different heights)
    if spawn_coords is None:
        spawn_coords = np.array([
            [i * 0.5, 0, 1.0] for i in range(num_drones)
        ])
    
    print(f"[SIM] Spawning {num_drones} agents in PyBullet")
    agents_list = spawn_agents(spawn_coords)
    
    # Step simulation once to initialize
    pybullet_handler.step_sim(agents_list, 0)
    
    print(f"[SIM] Simulation initialized with {num_drones} agents")
    return pybullet_handler, agents_list


def execute_trajectory_sim(pybullet_handler, agents_list, cbf_controller, pid_controller,
                          target_positions, target_velocities, target_accelerations,
                          duration=30.0, num_agents=None):
    """
    Execute trajectory with CBF-based safety filtering in PyBullet simulation.
    
    Args:
        pybullet_handler: PyBullet simulation manager
        agents_list: List of Quadrotor state objects
        cbf_controller: CBFQPControllerDrone instance
        pid_controller: NominalPIDControllerDrone instance
        target_positions: Array of target positions [num_agents, 3]
        target_velocities: Array of target velocities [num_agents, 3]
        target_accelerations: Array of target accelerations [num_agents, 3]
        duration: Total simulation duration
        num_agents: Number of agents (if None, uses all in agents_list)
    """
    
    if num_agents is None:
        num_agents = len(agents_list)
    
    print(f"[SIM] Starting trajectory execution with {num_agents} agents")
    print(f"[SIM] Duration: {duration}s, dt: {pybullet_handler.dt}s")
    
    # Logging
    position_data = []
    time_history = []
    priority_history = {i: [] for i in range(num_agents)}
    
    # Simulation loop
    iteration = 0
    max_iterations = int(duration / pybullet_handler.dt)
    
    while iteration < max_iterations:
        current_time = iteration * pybullet_handler.dt
        
        # Control each agent
        for i, agent in enumerate(agents_list[:num_agents]):
            
            # Get target references
            if len(target_positions.shape) > 1:
                target_pos = target_positions[i]
            else:
                target_pos = target_positions
            
            if len(target_velocities.shape) > 1:
                target_vel = target_velocities[i]
            else:
                target_vel = target_velocities
            
            if len(target_accelerations.shape) > 1:
                target_acc = target_accelerations[i]
            else:
                target_acc = target_accelerations
            
            # Step 1: Compute nominal PID control (returns motor RPMs)
            try:
                ref_propellers_rpm = pid_controller.compute_PID_control(
                    agent=agent,
                    target_linear_position=target_pos,
                    target_linear_velocity=target_vel,
                    target_linear_acceleration=target_acc
                )
            except Exception as e:
                print(f"[SIM] Error computing PID for agent {i}: {e}")
                ref_propellers_rpm = np.zeros(4)
            
            # Convert nominal RPM to thrusts for CBF QP input
            ref_thrusts = pid_controller.thrust_from_rpm(ref_propellers_rpm)
            
            # Step 2: Identify obstacles (other agents with higher priority)
            higher_priority_agents = [k for k in range(num_agents) if k < i]
            
            if higher_priority_agents:
                obstacle_pos = np.asarray([agents_list[k].pos_states for k in higher_priority_agents])
                obstacle_vel = np.asarray([agents_list[k].d_pos_states for k in higher_priority_agents])
            else:
                obstacle_pos = np.empty((0, 3))
                obstacle_vel = np.empty((0, 3))
            
            # Step 3: Apply CBF-QP safety filter
            try:
                safety_thrust, safety_flag = cbf_controller.solve_QP(
                    agent=agent,
                    u_ref=ref_thrusts,
                    obstacle_positions=obstacle_pos,
                    obstacle_velocities=obstacle_vel
                )
                
                if safety_flag:
                    # Safety constraint was violated; apply corrected command
                    total_thrust = ref_thrusts + safety_thrust
                    propellers_rpm = pid_controller.rpm_from_thrust(total_thrust)
                else:
                    # No constraint violation; use nominal PID command
                    propellers_rpm = ref_propellers_rpm
                    
            except Exception as e:
                print(f"[SIM] Error in CBF QP for agent {i}: {e}")
                propellers_rpm = ref_propellers_rpm
            
            # Step 4: Apply control effort to simulated drone
            pybullet_handler.apply_ctrl_inputs(agent, propellers_rpm)
        
        # Step simulation
        pybullet_handler.step_sim(agents_list[:num_agents])
        
        # Log data
        current_time = iteration * pybullet_handler.dt
        time_history.append(current_time)
        
        row_data = [current_time]
        for agent in agents_list[:num_agents]:
            row_data.extend([agent.pos_states[0], agent.pos_states[1], agent.pos_states[2]])
        position_data.append(row_data)
        
        # Print progress
        if iteration % 100 == 0:
            print(f"[SIM] Iteration {iteration}/{max_iterations} (t={current_time:.2f}s)")
        
        iteration += 1
        # Optional: uncomment to run in real-time
        # sleep(pybullet_handler.dt * 0.1)
    
    print(f"[SIM] Simulation complete. Total time: {current_time:.2f}s")
    
    # Export data
    timestamp = __import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = f"cbf_sim_positions_{timestamp}.csv"
    csv_headers = ['Time'] + [f'Agent_{i}_{c}' for i in range(num_agents) for c in ['X', 'Y', 'Z']]
    
    export_positions_to_csv(position_data, csv_headers, csv_filename)
    print(f"[SIM] Position data saved to {csv_filename}")
    
    return position_data, time_history


def main():
    """Main function for PyBullet simulation"""
    
    parser = argparse.ArgumentParser(description='Multi-agent CBF PyBullet Simulation')
    parser.add_argument('--num_drones', type=int, default=3,
                       help='Number of drones to simulate (default: 3)')
    parser.add_argument('--duration', type=float, default=30.0,
                       help='Simulation duration in seconds (default: 30.0)')
    parser.add_argument('--height', type=float, default=1.5,
                       help='Flight height in meters (default: 1.5)')
    parser.add_argument('--gui', action='store_true',
                       help='Enable PyBullet GUI visualization (slower)')
    parser.add_argument('--hoop', action='store_true',
                       help='Enable hoop passing constraint')
    
    args = parser.parse_args()
    num_drones = args.num_drones
    
    print(f"[SIM] PyBullet Multi-Agent CBF Simulation")
    print(f"[SIM] Configuration: {num_drones} drones, {args.duration}s duration, {args.height}m height")
    if args.gui:
        print(f"[SIM] GUI enabled (slower simulation)")
    
    # Initialize PyBullet simulation
    pybullet_handler, agents_list = initialize_simulation(num_drones)
    
    # Initialize controllers
    cbf_controller = CBFQPControllerDrone(
        acceleration_gravity=GRAVITY_ACC,
        CBF_gamma=1.0,  # Aggressiveness of safety filter
        hoop_center_coord=np.array([0., 1.0, 1.0]) if args.hoop else None,
        hoop_direction=np.array([0., 1., 0.]) if args.hoop else None,
    )
    
    pid_controller = NominalPIDControllerDrone(
        acceleration_gravity=GRAVITY_ACC,
        time_step_size=SIM_TIMESTEP
    )
    
    # Setup trajectory parameters
    Z = args.height
    
    # Create simple trajectory targets for each agent
    target_positions = np.zeros((num_drones, 3))
    target_velocities = np.zeros((num_drones, 3))
    target_accelerations = np.zeros((num_drones, 3))
    
    for i in range(num_drones):
        target_positions[i] = np.array([0.5 * i, 0, Z])  # Spread along X-axis
        target_velocities[i] = np.array([0.0, 0.0, 0.0])
        target_accelerations[i] = np.array([0.0, 0.0, 0.0])
    
    try:
        # Execute trajectory with CBF safety filter
        position_data, time_history = execute_trajectory_sim(
            pybullet_handler=pybullet_handler,
            agents_list=agents_list,
            cbf_controller=cbf_controller,
            pid_controller=pid_controller,
            target_positions=target_positions,
            target_velocities=target_velocities,
            target_accelerations=target_accelerations,
            duration=args.duration,
            num_agents=num_drones
        )
        
        print("[SIM] Mission complete successfully")
        
    except KeyboardInterrupt:
        print("[SIM] Simulation interrupted by user")
    except Exception as e:
        print(f"[SIM] Error during simulation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
