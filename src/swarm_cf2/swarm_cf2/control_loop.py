"""
Main control loop for running drones through hoops using CBF-QP control.
This module contains the core control algorithm for safely navigating
multiple drones through predefined hoops.
"""

import numpy as np
import pybullet as pb
from time import sleep
import matplotlib.pyplot as plt
import csv

from .quadrotor_info import Quadrotor, GRAVITY_ACC, SIM_TIMESTEP, URDF_PATH
from .CBF_controller import CBFQPControllerDrone
from .quadrotor_swarm_sim import PybulletHandler
from .nominal_controller import NominalPIDControllerDrone


def compute_priority(agent: Quadrotor, weights: tuple, pass_through_hoop: bool, 
                    hoop_coordinates: np.ndarray, agents_list: list, 
                    cbf_controller: CBFQPControllerDrone, num_agents: int):
    """
    Compute priority value for an agent based on distance to hoop, velocity, and CBF values.
    
    Args:
        agent: The quadrotor agent
        weights: Tuple of weights for (distance, velocity, CBF)
        pass_through_hoop: Boolean flag indicating if agent is moving towards hoop
        hoop_coordinates: 3D coordinates of the hoop center
        agents_list: List of all agents
        cbf_controller: CBF controller instance
        num_agents: Total number of agents
        
    Returns:
        float: Priority value for the agent
    """
    obstacle_positions = np.asarray([agents_list[k].pos_states for k in range(num_agents) if agents_list[k] != agent])
    obstacle_velocities = np.asarray([agents_list[k].d_pos_states for k in range(num_agents) if agents_list[k] != agent])

    h_values = cbf_controller.CBF_value(
        agent=agent,
        obstacle_positions=obstacle_positions,
        obstacle_velocities=obstacle_velocities,
    )

    distances = np.linalg.norm(hoop_coordinates - agent.pos_states)
    velocities_towards_hoop = np.dot((hoop_coordinates - agent.pos_states), agent.d_pos_states) / (distances + 1e-6)

    if not pass_through_hoop:
        weights = [0, 0, weights[2]]

    priority_value = weights[0] * distances - weights[1] * velocities_towards_hoop - weights[2] * np.sum(h_values)
    
    return priority_value


def run_through_hoops_control(duration: float = 30.0,
                              spawn_coordinates: np.ndarray = None,
                              hoop_coordinates: np.ndarray = None,
                              hoop_direction: np.ndarray = None,
                              cbf_drone_speed: float = 5e-2,
                              cbf_gamma: float = 1.0,
                              enable_visualization: bool = True,
                              export_csv: bool = True):
    """
    Main control loop for running drones through hoops with CBF-QP safety.
    
    Args:
        duration: Total simulation duration in seconds
        spawn_coordinates: Nx3 array of initial drone positions
        hoop_coordinates: 3D coordinates of hoop center
        hoop_direction: 3D direction vector of hoop normal
        cbf_drone_speed: Desired vertical speed for drones
        cbf_gamma: CBF aggressiveness parameter
        enable_visualization: Whether to show plots at the end
        export_csv: Whether to export position data to CSV
        
    Returns:
        dict: Dictionary containing simulation results (positions, priorities, etc.)
    """
    
    # Default spawn coordinates if not provided
    if spawn_coordinates is None:
        spawn_coordinates = np.array([
            [0., 0., 0.2],    # Agent 0
            [0., 0.3, 0.4],   # Agent 1
            [0., -0.2, 0.6],  # Agent 2
        ])
    
    # Default hoop coordinates if not provided (hard-coded)
    if hoop_coordinates is None:
        hoop_coordinates = np.array([0., 0.0, 1.5])
    
    # Default hoop direction if not provided
    if hoop_direction is None:
        hoop_direction = np.array([0., 0., 1.])
    
    # Initialize PyBullet simulation handler
    PH = PybulletHandler(
        acceleration_gravity=GRAVITY_ACC,
        time_step_size=SIM_TIMESTEP,
    )
    
    NUM_AGENTS = len(spawn_coordinates)
    HOOP_PATH = "assets/hoop.urdf"
    
    # Spawn agents
    print(f"[CONTROL] Spawning {NUM_AGENTS} agents...")
    AGENTS_LIST = []
    for coord in spawn_coordinates:
        agent_id = pb.loadURDF(URDF_PATH, coord, useFixedBase=False)
        agent_info = pb.getDynamicsInfo(bodyUniqueId=agent_id, linkIndex=-1)
        
        from .quadrotor_info import quadrotor_params
        AGENTS_LIST.append(
            Quadrotor(
                id=agent_id,
                mass=agent_info[0],
                inertia_matrix=np.diag(agent_info[2]),
                **quadrotor_params
            )
        )
    print(f"[CONTROL] Spawn complete. Agent IDs: {[agent.id for agent in AGENTS_LIST]}")
    
    # Initialize CBF controller
    CBF_ = CBFQPControllerDrone(
        acceleration_gravity=GRAVITY_ACC,
        CBF_gamma=cbf_gamma,
        hoop_center_coord=hoop_coordinates,
        hoop_direction=hoop_direction,
    )
    
    # Initialize nominal PID controller
    NC = NominalPIDControllerDrone(
        acceleration_gravity=GRAVITY_ACC,
        time_step_size=SIM_TIMESTEP,
    )
    
    # Load hoop into simulation
    pb.loadURDF(HOOP_PATH, hoop_coordinates, useFixedBase=True)
    
    # Initialize simulation
    PH.step_sim(AGENTS_LIST, 0)
    
    # Priority weights
    weights = [0.5,  # weight for distance
               0.2,  # weight for relative velocity
               0.3]  # weight for CBF value
    
    # Tracking variables
    priority_history = {j: [] for j in range(NUM_AGENTS)}
    time_history = []
    position_data = []
    csv_headers = ['Time'] + [f'Agent_{i}_{axis}' for i in range(NUM_AGENTS) for axis in ['X', 'Y', 'Z']]
    
    # Calculate initial signs for determining hoop crossing
    initial_sign = np.zeros(NUM_AGENTS)
    for k in range(NUM_AGENTS):
        initial_sign[k] = np.sign(np.dot(hoop_coordinates - AGENTS_LIST[k].pos_states, hoop_direction))
    
    priority_list = []
    
    print("[CONTROL] Starting simulation loop...")
    
    # Main simulation loop
    for i in range(int(duration / PH.dt)):
        eps = 0.001  # threshold for grouping equal priority
        
        # Process each agent
        for j, agent in enumerate(AGENTS_LIST):
            # Calculate target position
            target_vel = np.array([0., 0., cbf_drone_speed])
            target_pos = spawn_coordinates[j] + target_vel * i * PH.dt
            
            # Check if agent is moving towards the hoop
            to_hoop_vec = hoop_coordinates - agent.pos_states
            if np.sign(np.dot(to_hoop_vec, hoop_direction)) == initial_sign[j]:
                pass_through_hoop = 1
            else:
                pass_through_hoop = 0
            
            # Calculate priority
            priority_value = compute_priority(agent, weights, pass_through_hoop, 
                                             hoop_coordinates, AGENTS_LIST, CBF_, NUM_AGENTS)
            
            existing_index = next((index for index, (agent_id, _) in enumerate(priority_list) if agent_id == j), None)
            if existing_index is not None:
                priority_list[existing_index] = (j, priority_value)
            else:
                priority_list.append((j, priority_value))
            
            priority_list = sorted(priority_list, key=lambda x: x[1])
            
            # Assign equal ranks within threshold
            ranked_list = []
            current_rank = 1
            prev_value = None
            
            for idx, (agent_id, p_val) in enumerate(priority_list):
                if prev_value is None:
                    ranked_list.append((agent_id, p_val, current_rank))
                    prev_value = p_val
                    continue
                
                if abs(p_val - prev_value) <= eps:
                    ranked_list.append((agent_id, p_val, current_rank))
                else:
                    current_rank += 1
                    ranked_list.append((agent_id, p_val, current_rank))
                
                prev_value = p_val
            
            # Store priority values for plotting
            current_time = i * PH.dt
            time_history.append(current_time)
            for agent_id, priority_val, _ in ranked_list:
                priority_history[agent_id].append(priority_val)
            
            # Calculate nominal PID control command
            ref_propellers_rpm = NC.compute_PID_control(
                agent=agent,
                target_linear_position=target_pos,
                target_linear_velocity=target_vel,
                target_linear_acceleration=np.zeros(3)
            )
            
            # Convert nominal RPM to thrusts
            ref_thrusts = NC.thrust_from_rpm(ref_propellers_rpm)
            
            # Get current agent's rank
            current_rank = next(rank for agent_id, _, rank in ranked_list if agent_id == j)
            
            # Identify higher priority agents
            higher_priority_agents = [
                agent_id for agent_id, _, rank in ranked_list
                if rank <= current_rank and agent_id != j
            ]
            
            obstacle_pos = np.asarray([AGENTS_LIST[k].pos_states for k in higher_priority_agents]) if higher_priority_agents else np.empty((0, 3))
            obstacle_vel = np.asarray([AGENTS_LIST[k].d_pos_states for k in higher_priority_agents]) if higher_priority_agents else np.empty((0, 3))
            
            # Solve CBF-QP for safe correction
            safety_thrust, safety_flag = CBF_.solve_QP(
                agent=agent,
                u_ref=ref_thrusts,
                obstacle_positions=obstacle_pos,
                obstacle_velocities=obstacle_vel,
            )
            
            if safety_flag:
                total_thrust = (ref_thrusts + safety_thrust)
                propellers_rpm = NC.rpm_from_thrust(total_thrust)
            else:
                propellers_rpm = ref_propellers_rpm
            
            # Store RPM history
            NC.store_rpm_history(agent.id, ref_propellers_rpm, propellers_rpm, i * PH.dt)
            
            # Apply control inputs
            PH.apply_ctrl_inputs(agent, propellers_rpm)
        
        # Step simulation
        PH.step_sim(AGENTS_LIST)
        print("pos_states:", [agent.pos_states for agent in AGENTS_LIST])
        
        # Store position data
        current_time = i * PH.dt
        row_data = [current_time]
        for agent in AGENTS_LIST:
            row_data.extend([agent.pos_states[0], agent.pos_states[1], agent.pos_states[2]])
        position_data.append(row_data)
        
        sleep(PH.dt * 0.1)
    
    # End simulation
    PH.quit()
    print("[CONTROL] Simulation finished.")
    
    # Export position data to CSV if requested
    if export_csv:
        csv_filename = "drone_positions_deployment.csv"
        try:
            with open(csv_filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(csv_headers)
                writer.writerows(position_data)
            print(f"[CONTROL] Position data exported to {csv_filename}")
        except Exception as e:
            print(f"[CONTROL] Error exporting to CSV: {e}")
    
    # Generate plots if requested
    if enable_visualization:
        print("[CONTROL] Generating visualizations...")
        
        # Plot RPM history
        NC.plot_all_agents_rpm()
        
        # Plot priority values
        plt.figure(figsize=(12, 8))
        colors = ['b-', 'r-', 'g-', 'm-', 'c-', 'y-']
        for agent_id, priorities in priority_history.items():
            if len(priorities) > 0:
                color = colors[agent_id % len(colors)]
                plt.plot(time_history[0:len(priorities)], priorities, color, label=f'Agent {agent_id}', linewidth=2)
        plt.title('Priority Values Over Time', fontsize=16)
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Priority Value', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Plot 3D trajectories
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        for agent_id in range(NUM_AGENTS):
            xs = [row[1 + agent_id * 3] for row in position_data]
            ys = [row[2 + agent_id * 3] for row in position_data]
            zs = [row[3 + agent_id * 3] for row in position_data]
            ax.plot(xs, ys, zs, label=f"Drone {agent_id}", linewidth=2)
        
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_zlim(-3, 3)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("3D Trajectories of Drones Through Hoops")
        ax.legend()
        
        plt.show()
    
    # Return results
    return {
        'position_data': position_data,
        'priority_history': priority_history,
        'time_history': time_history,
        'agents_list': AGENTS_LIST,
    }
