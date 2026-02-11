from time import sleep

from utilities import *
from quadrotor_swarm_sim import PybulletHandler
from quadrotor_info import GRAVITY_ACC, SIM_TIMESTEP
from controllers_hub import CBFQPControllerDrone, NominalPIDControllerDrone


if __name__ == "__main__":
    
    # --- Simulation Parameters ---
    DURATION = 30.                              # Total simulation duration
    HOOP_COORDINATES = np.array([0., 1.0, 1.0])  # Hoop position and direction
    HOOP_DIRECTION = np.array([0., 1., 0.])     # Hoop direction of entry
    TARGET_VELS = np.array([0., 5e-2, 0.])      # Desired speed for the agents
    
    # Agents spawn coordinates
    SPAWN_COORDINATES=np.array([              
        [0., -0.25, 1.],      # Agent 0
        [-0.25, -0.5, 1.],    # Agent 1
        [0.25, -0.75, 1.],    # Agent 2
    ])
    # SPAWN_COORDINATES=np.array([
    #     [0., 0., 0.2],    # Agent 0
    #     [0., 0.3, 0.4],  # Agent 1
    #     [0., -0.2, 0.6],  # Agent 2
    # ])
    NUM_AGENTS = len(SPAWN_COORDINATES)

    # --- Initialize Handlers ---
    
    # PyBullet simulation manager
    PH = PybulletHandler(
        acceleration_gravity=GRAVITY_ACC,
        time_step_size=SIM_TIMESTEP,
    )    

    CBF = CBFQPControllerDrone(
        acceleration_gravity=GRAVITY_ACC,
        CBF_gamma=1.0, # Aggressiveness of the safety filter
        hoop_center_coord=HOOP_COORDINATES,
        hoop_direction=HOOP_DIRECTION,
    )    

    # Nominal PID position controller
    NC = NominalPIDControllerDrone(
        acceleration_gravity=GRAVITY_ACC,
        time_step_size=SIM_TIMESTEP,
    )  
    
    # --- Spawn Agents & Hoop ---
    
    pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 0)
    AGENTS_LIST = spawn_agents(SPAWN_COORDINATES)
    spawn_hoop(HOOP_COORDINATES, HOOP_DIRECTION)
    
    PH.step_sim(AGENTS_LIST, 0) 
    pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 1)
    
    # --- Main Simulation Loop ---
    
    print("[MAIN] Starting simulation loop...")
    
    # Priority logging & corresponding weights
    eps = 0.001
    priority_list = []
    priority_history = {j: [] for j in range(NUM_AGENTS)}
    weights = [0.5,  # weight for distance
               0.2,  # weight for relative velocity
               0.3]  # weight for CBF value
    
    # Logging position & rpm
    rpm_history = [] 
    time_history = []
    position_data = []
    csv_headers = ['Time', 'Agent_0_X', 'Agent_0_Y', 'Agent_0_Z', 
                   'Agent_1_X', 'Agent_1_Y', 'Agent_1_Z',
                   'Agent_2_X', 'Agent_2_Y', 'Agent_2_Z']

    initial_sign = np.zeros(NUM_AGENTS)
    for k in range(NUM_AGENTS): initial_sign[k] = np.sign(np.dot(HOOP_COORDINATES - AGENTS_LIST[k].pos_states, HOOP_DIRECTION))

    def compute_priority(agent:Quadrotor, weights:tuple, pass_through_hoop:bool):
        obstacle_positions = np.asarray([AGENTS_LIST[k].pos_states for k in range(NUM_AGENTS) if AGENTS_LIST[k]!= agent])
        obstacle_velocities = np.asarray([AGENTS_LIST[k].d_pos_states for k in range(NUM_AGENTS) if AGENTS_LIST[k]!= agent])

        h_values = CBF.CBF_value(
            agent=agent,
            obstacle_positions=obstacle_positions,
            obstacle_velocities=obstacle_velocities,
        )
        distances = np.linalg.norm(HOOP_COORDINATES - agent.pos_states)
        velocities_towards_hoop = np.dot((HOOP_COORDINATES - agent.pos_states), agent.d_pos_states) / (distances + 1e-6)
        if not pass_through_hoop: weights = [0, 0, weights[2]]

        priority_value = weights[0]*distances - weights[1]*velocities_towards_hoop - weights[2]*np.sum(h_values)
        # priority_value = weights[0]*distances - weights[2]*np.sum(h_values)        
        return priority_value
    
    agent : Quadrotor # Type hinting
    for i in range(int(DURATION / PH.dt)):

        # Iterating through all the agents
        for j, agent in enumerate(AGENTS_LIST):

            # Calculate the target position for the current agent
            target_pos = SPAWN_COORDINATES[j] + TARGET_VELS * i * PH.dt

            # Check if vector from current position to target posiiton is towards the hoop
            to_hoop_vec = HOOP_COORDINATES - agent.pos_states
            pass_through_hoop = 1 if np.sign(np.dot(to_hoop_vec, HOOP_DIRECTION)) == initial_sign[j] else 0

            # Calculate priority and assign
            priority_value = compute_priority(agent, weights, pass_through_hoop)
            existing_index = next((index for index, (agent_id, _) in enumerate(priority_list) if agent_id == j), None)
            if existing_index is not None: priority_list[existing_index] = (j, priority_value)
            else: priority_list.append((j, priority_value))
            priority_list = sorted(priority_list, key=lambda x: x[1])

            # Assign equal ranks within threshold
            ranked_list = []
            current_rank = 1
            prev_value = None

            for idx, (agent_id, p_val) in enumerate(priority_list):
                if prev_value is None:
                    # first element
                    ranked_list.append((agent_id, p_val, current_rank))
                    prev_value = p_val
                    continue

                # If within threshold → same rank
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

            # Calculate the nominal (unsafe) PID control command
            ref_propellers_rpm = NC.compute_PID_control(
                agent=agent,
                target_linear_position=target_pos,
                target_linear_velocity=TARGET_VELS,
                target_linear_acceleration=np.zeros(3)
            )

            # Convert nominal RPM to nominal thrusts
            ref_thrusts = NC.thrust_from_rpm(ref_propellers_rpm)

            # Get current agent's rank
            current_rank = next(rank for agent_id, _, rank in ranked_list if agent_id == j)

            # Agents with higher priority (lower rank number) other than itself            
            higher_priority_agents = [
                agent_id for agent_id, _, rank in ranked_list
                if rank <= current_rank and agent_id != j
            ]
            obstacle_pos = np.asarray([AGENTS_LIST[k].pos_states for k in higher_priority_agents]) if higher_priority_agents else np.empty((0, 3))
            obstacle_vel = np.asarray([AGENTS_LIST[k].d_pos_states for k in higher_priority_agents]) if higher_priority_agents else np.empty((0, 3))

            # Solve the CBF-QP to find a *safe correction*
            safety_thrust, safety_flag = CBF.solve_QP(
                agent=agent, 
                u_ref=ref_thrusts,
                obstacle_positions=obstacle_pos,
                obstacle_velocities=obstacle_vel,
            )
            if safety_flag: 
                # If a constraint was violated, apply the corrected command
                total_thrust = ref_thrusts + safety_thrust
                propellers_rpm = NC.rpm_from_thrust(total_thrust)
            else:
                # Otherwise, use the nominal (PID) command
                propellers_rpm = ref_propellers_rpm

            # propellers_rpm = ref_propellers_rpm
            # Apply control effort to the simulated drone
            PH.apply_ctrl_inputs(agent, propellers_rpm)
            
        # --- Step the Simulation ---
        PH.step_sim(AGENTS_LIST)
        
        # Store position data for CSV export
        current_time = i * PH.dt
        row_data = [current_time]
        for agent in AGENTS_LIST: 
            row_data.extend([agent.pos_states[0], agent.pos_states[1], agent.pos_states[2]])
        position_data.append(row_data)
        
        sleep(PH.dt * 0.1) # Optional: uncomment to run in "real-time"
        
    # --- End of Simulation ---
    
    PH.quit()
    print("[MAIN] Simulation finished.")
    
    # Export position data to CSV
    csv_filename = "drone_positions_new.csv"
    export_positions_to_csv(position_data, csv_headers, csv_filename)
    print(f"[MAIN] Position data exported to {csv_filename}")

    # Plot RPM history for each agent
    # print("[MAIN] Generating RPM plots...")
    # plot_all_agents_rpm(rpm_history)
    
    # Plot priority values 
    print("[MAIN] Generating priority plots...")
    plot_priority_history(priority_history, time_history)
    
    # Plot trajectories 
    print("[MAIN] Generating 3D trajectory plots...")
    plot_3d_trajectories(position_data, NUM_AGENTS)
    