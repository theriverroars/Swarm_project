import pybullet as pb
from time import sleep
import matplotlib.pyplot as plt
import csv

from quadrotor_info import *
from CBF_controller import CBFQPControllerDrone
from quadrotor_swarm_sim import PybulletHandler
from nominal_controller import NominalPIDControllerDrone


def spawn_agents(spawn_coordinates:np.ndarray):
    """
    Spawns quadrotor agents into the PyBullet simulation at the specified coordinates.

    Args:
        spawn_coordinates (np.ndarray): An Nx3 array where each row is an [x, y, z]
                                        coordinate for spawning a new agent.

    Returns:
        list[Quadrotor]: A list of initialized Quadrotor objects.
    """
    AGENTS_LIST = []
    print(f"[MAIN] Spawning {len(spawn_coordinates)} agents...")

    for coord in spawn_coordinates:
        # Load the URDF file into PyBullet
        agent_id = pb.loadURDF(URDF_PATH, coord, useFixedBase=False)
        
        # Get physical properties (mass, inertia) directly from the URDF
        agent_info = pb.getDynamicsInfo(bodyUniqueId=agent_id, linkIndex=-1)
        
        # Create a Quadrotor object to store state and parameters
        AGENTS_LIST.append(
            Quadrotor(
                id=agent_id,
                mass=agent_info[0], 
                inertia_matrix = np.diag(agent_info[2]), # Get diagonal inertia
                **quadrotor_params # Pass all other default params (PID gains, motor coeffs, etc.)
            )
        )
    print(f"[MAIN] Spawn complete. Agent IDs: {[agent.id for agent in AGENTS_LIST]}")
    return AGENTS_LIST


def export_positions_to_csv(position_data, headers, filename):
    """
    Export drone position data to a CSV file.
    
    Args:
        position_data (list): List of rows containing time and position data
        headers (list): Column headers for the CSV file
        filename (str): Name of the CSV file to create
    """
    try:
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write headers
            writer.writerow(headers)
            
            # Write data rows
            writer.writerows(position_data)
            
        print(f"Successfully exported {len(position_data)} rows of position data to {filename}")
        
    except Exception as e:
        print(f"Error exporting to CSV: {e}")


def plot_priority_history(priority_history, time_history):
    """
    Plot priority values over time for all agents.
    
    Args:
        priority_history (dict): Dictionary with agent_id as key and list of priority values
        time_history (list): List of time stamps
    """
    plt.figure(figsize=(12, 8))
    
    colors = ['b-', 'r-', 'g-', 'm-', 'c-', 'y-']  # Different colors for agents
    
    for agent_id, priorities in priority_history.items():
        if len(priorities) > 0:  # Only plot if we have data
            color = colors[agent_id % len(colors)]
            plt.plot(time_history[0:len(priorities)], priorities, color, label=f'Agent {agent_id}', linewidth=2)
    
    plt.title('Priority Values Over Time', fontsize=16)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Priority Value', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add some styling
    plt.tight_layout()
    
    # Show separate plot for better visibility
    # plt.figure(figsize=(15, 10))
    
    # num_agents = len(priority_history)
    # for idx, (agent_id, priorities) in enumerate(priority_history.items()):
    #     if len(priorities) > 0:
    #         plt.subplot(num_agents, 1, idx + 1)
    #         plt.plot(time_history, priorities, 'b-', linewidth=2)
    #         plt.title(f'Agent {agent_id} Priority Values', fontsize=12)
    #         plt.xlabel('Time (s)')
    #         plt.ylabel('Priority Value')
    #         plt.grid(True, alpha=0.3)
    
    # plt.suptitle('Individual Agent Priority Values', fontsize=16)
    # plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    
    # --- Simulation Parameters ---
    DURATION = 30.       # Total simulation duration (in simulation time)
    # SPAWN_COORDINATES=np.array([
    #     # [0., 0., 0.2],    # Agent 0
    #     [0., 0.3, 0.2],  # Agent 1
    #     [0., -0.3, 0.4],  # Agent 2
    # ])

    SPAWN_COORDINATES=np.array([
        [0., 0., 0.2],    # Agent 0
        [0., 0.3, 0.4],  # Agent 1
        [0., -0.2, 0.6],  # Agent 2
    ])


    HOOP_COORDINATES = np.array([0., 0.0, 1.5])
    HOOP_DIRECTION = np.array([0., 0., 1.])

    # --- Initialize Handlers ---
    # PyBullet simulation manager
    PH = PybulletHandler(
        acceleration_gravity=GRAVITY_ACC,
        time_step_size=SIM_TIMESTEP,
    )    

    NUM_AGENTS = len(SPAWN_COORDINATES)
    HOOP_PATH = join("assets", "hoop.urdf")
    AGENTS_LIST = spawn_agents(SPAWN_COORDINATES)

    # CBF = []

    # Iterating through all the agents
    # for j, agent in enumerate(AGENTS_LIST):

    #     # CBF-QP safety filter for each agent
    #     CBF.append(CBFQPControllerDrone(
    #         acceleration_gravity=GRAVITY_ACC,
    #         CBF_gamma=1.0, # Aggressiveness of the safety filter
    #         hoop_center_coord=HOOP_COORDINATES,
    #         hoop_direction=HOOP_DIRECTION,
    #     ))   

    CBF_ = CBFQPControllerDrone(
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

    pb.loadURDF(HOOP_PATH, HOOP_COORDINATES, useFixedBase=True)
    
    # Perform one simulation step to initialize all agent states
    PH.step_sim(AGENTS_LIST, 0) 
    
    # --- Main Simulation Loop ---
    CBF_DRONE_SPEED = 5e-2 # Desired speed for the agents
    print("[MAIN] Starting simulation loop...")

    agent : Quadrotor # Type hinting

    weights = [0.5,  # weight for distance
               0.2,  # weight for relative velocity
               0.3]  # weight for CBF value
    
    
    # Priority tracking for plotting
    priority_history = {j: [] for j in range(NUM_AGENTS)}
    time_history = []
    
    # Position tracking for CSV export
    position_data = []
    csv_headers = ['Time', 'Agent_0_X', 'Agent_0_Y', 'Agent_0_Z', 
                   'Agent_1_X', 'Agent_1_Y', 'Agent_1_Z',
                   'Agent_2_X', 'Agent_2_Y', 'Agent_2_Z']

    initial_sign = np.zeros(NUM_AGENTS)

    for k in range(NUM_AGENTS):
        initial_sign[k] = np.sign(np.dot(HOOP_COORDINATES - AGENTS_LIST[k].pos_states, HOOP_DIRECTION))

    def compute_priority(agent:Quadrotor, weights:tuple, pass_through_hoop:bool):
        obstacle_positions = np.asarray([AGENTS_LIST[k].pos_states for k in range(NUM_AGENTS) if AGENTS_LIST[k]!= agent])
        obstacle_velocities = np.asarray([AGENTS_LIST[k].d_pos_states for k in range(NUM_AGENTS) if AGENTS_LIST[k]!= agent])

        h_values = CBF_.CBF_value(
            agent=agent,
            obstacle_positions=obstacle_positions,
            obstacle_velocities=obstacle_velocities,
        )

        distances = np.linalg.norm(HOOP_COORDINATES - agent.pos_states)
        velocities_towards_hoop = np.dot((HOOP_COORDINATES - agent.pos_states), agent.d_pos_states) / (distances + 1e-6)

        if not pass_through_hoop:
            weights = [0, 0, weights[2]]

        priority_value = weights[0]*distances - weights[1]*velocities_towards_hoop - weights[2]*np.sum(h_values)

        # priority_value = weights[0]*distances - weights[2]*np.sum(h_values)
        
        # print(f"Agent ID: {agent.id} | Distance to Hoop: {distances:.3f} | Velocity Towards Hoop: {velocities_towards_hoop:.3f} | CBF Sum: {np.sum(h_values):.3f} | Priority Value: {priority_value:.3f}")
        
        return priority_value
    
    priority_list = []

    for i in range(int(DURATION / PH.dt)):

        eps = 0.001  # threshold for grouping equal priority

        # Iterating through all the agents
        for j, agent in enumerate(AGENTS_LIST):

            # Calculate the target position for the current agent
            target_vel = np.array([0., 0., CBF_DRONE_SPEED])
            target_pos = SPAWN_COORDINATES[j] + target_vel * i * PH.dt

            # Check if vector from current position to target posiiton is towards the hoop. Make this a flag.
            # # Method 1:
            # vector_to_hoop = HOOP_COORDINATES - agent.pos_states
            # target_vector = target_pos - agent.pos_states
            # pass_through_hoop = np.dot(vector_to_hoop, target_vector) > 0
            # # Method 2:
            to_hoop_vec = HOOP_COORDINATES - agent.pos_states
            if np.sign(np.dot(to_hoop_vec, HOOP_DIRECTION)) == initial_sign[j]:
                pass_through_hoop =  1
            else:
                pass_through_hoop = 0



            # Calculate priority and assign
            priority_value = compute_priority(agent, weights, pass_through_hoop)
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

            # print(f"[MAIN] Priority List with grouped ranks at step {i}: {ranked_list}")
            
            # Store priority values for plotting
            current_time = i * PH.dt
            time_history.append(current_time)
            for agent_id, priority_val, _ in ranked_list:
                priority_history[agent_id].append(priority_val)

            # Calculate the nominal (unsafe) PID control command
            ref_propellers_rpm = NC.compute_PID_control(
                agent=agent,
                target_linear_position=target_pos,
                target_linear_velocity=target_vel,
                target_linear_acceleration=np.zeros(3)
            )

            # print(f"[MAIN] Agent {j} | Target Pos: {target_pos} | Target Vel: {target_vel}")
            #if any of the ref_propellers_rpm is zero, print a warning
            # if np.any(ref_propellers_rpm == 0):
                # print(f"[WARNING] Agent {j} | One or more reference propeller RPMs are zero!")
                # print("Ref RPMs:", ref_propellers_rpm, "target pos:", target_pos, "target vel:", target_vel)
            
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

            # print(f"[MAIN] Agent {j} | Priority Index: {current_rank} | Higher Priority Agents: {higher_priority_agents}")
            # print("Size of obstacle pos:", obstacle_pos.shape, "Size of obstacle vel:", obstacle_vel.shape)

            # Solve the CBF-QP to find a *safe correction*
            safety_thrust, safety_flag = CBF_.solve_QP(
                agent=agent, 
                u_ref=ref_thrusts,
                obstacle_positions=obstacle_pos,
                obstacle_velocities=obstacle_vel,
            )
            
            if safety_flag: 
                # If a constraint was violated, apply the corrected command
                total_thrust = (ref_thrusts + safety_thrust)
                propellers_rpm = NC.rpm_from_thrust(total_thrust)
            else:
                # Otherwise, use the nominal (PID) command
                propellers_rpm = ref_propellers_rpm
                # print(f"[MAIN] Agent {j} | Using nominal RPM command.")

            # total_thrust = (ref_thrusts + safety_thrust)
            # propellers_rpm = NC.rpm_from_thrust(total_thrust)
            # # propellers_rpm = ref_propellers_rpm
            
            # Store RPM history for plotting
            NC.store_rpm_history(agent.id, ref_propellers_rpm, propellers_rpm, i * PH.dt)
            
            # Apply control effort to the simulated drone
            PH.apply_ctrl_inputs(agent, propellers_rpm)        

        # --- Step the Simulation ---
        PH.step_sim(AGENTS_LIST)
        print("pos_states:", [agent.pos_states for agent in AGENTS_LIST])
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
    print("[MAIN] Generating RPM plots...")
    NC.plot_all_agents_rpm()
    
    # Plot priority values
    print("[MAIN] Generating priority plots...")
    plot_priority_history(priority_history, time_history)

    # Plot 3 d position trajectories for each agent using position_data
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    for agent_id in range(NUM_AGENTS):
        xs = [row[1 + agent_id * 3] for row in position_data]
        ys = [row[2 + agent_id * 3] for row in position_data]
        zs = [row[3 + agent_id * 3] for row in position_data]
        # print(xs)
        # plt.plot(xs, ys, zs, label=f'Agent {agent_id}', linewidth=2)
        ax.plot(xs, ys, zs, label="Drone "+str(agent_id), linewidth=2)

    #Ensure equal scaling on all axes
    max_range = 0
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-3, 3)   
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Trajectories of 3 Drones")
    ax.legend()


    plt.show()  





