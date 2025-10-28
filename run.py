import pybullet as pb
from time import sleep

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


if __name__ == "__main__":
    
    # --- Simulation Parameters ---
    # CBF_DRONE_INDEX = 0     # The agent that will be actively controlled and avoid others
    DURATION = 500.           # Total simulation duration (in simulation time, not wall clock time)
    SPAWN_COORDINATES=np.array([
        [0., 0., 0.5],    # Agent 0: The CBF-controlled drone
        [0., 0.2, 1.5],   # Agent 1: Static obstacle
        [0., -0.15, 2.],  # Agent 2: Static obstacle
    ])

    # --- Initialize Handlers ---
    # PyBullet simulation manager
    PH = PybulletHandler(
        acceleration_gravity=GRAVITY_ACC,
        time_step_size=SIM_TIMESTEP,
    )    
    # CBF-QP safety filter
    CBF = CBFQPControllerDrone(
        acceleration_gravity=GRAVITY_ACC,
        CBF_gamma=1.0 # Aggressiveness of the safety filter
    )    
    # Nominal PID position controller
    NC = NominalPIDControllerDrone(
        acceleration_gravity=GRAVITY_ACC,
        time_step_size=SIM_TIMESTEP,
    )    

    # --- Spawn Agents ---
    agent : Quadrotor # Type hinting
    NUM_AGENTS = len(SPAWN_COORDINATES)
    AGENTS_LIST = spawn_agents(SPAWN_COORDINATES)
    
    # Perform one simulation step to initialize all agent states
    PH.step_sim(AGENTS_LIST, 0) 
    
    # --- Main Simulation Loop ---
    CBF_DRONE_SPEED = 5e-2 # Desired vertical speed for the main agent
    print("[MAIN] Starting simulation loop...")
    
    for i in range(int(DURATION / PH.dt)):
        # Calculate the target position for the main agent (moving up)
        step = i * CBF_DRONE_SPEED * PH.dt
        target_pos = np.array([0., 0., step + 0.5])
        target_vel = np.array([0., 0., CBF_DRONE_SPEED])
        
        # --- 1. Control the main CBF-controlled agent (Agent 0) ---
        agent = AGENTS_LIST[0]
        
        # Calculate the nominal (unsafe) PID control command
        ref_propellers_rpm = NC.compute_PID_control(
            agent=agent,
            target_linear_position=target_pos,
            target_linear_velocity=target_vel,
            target_linear_acceleration=np.zeros(3)
        )
        
        # Convert nominal RPM to nominal thrusts (which the CBF-QP understands)
        ref_thrusts = NC.thrust_from_rpm(ref_propellers_rpm)
        
        # Get obstacle positions and velocities (all other agents)
        obstacle_pos = np.asarray([AGENTS_LIST[k].pos_states for k in range(1, NUM_AGENTS)])
        obstacle_vel = np.asarray([AGENTS_LIST[k].d_pos_states for k in range(1, NUM_AGENTS)])
        
        # Solve the CBF-QP to find a *safe correction*
        safety_thrust, safety_flag = CBF.solve_QP(
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
            
        PH.apply_ctrl_inputs(agent, propellers_rpm)
        
        # --- 2. Control the "obstacle" agents (Agents 1 and 2) ---
        # These agents will just run a PID controller to hover at their spawn points
        for j in range(1, NUM_AGENTS):
            agent = AGENTS_LIST[j]
            
            # Compute PID command to hold their initial position
            propellers_rpm_static = NC.compute_PID_control(
                agent=agent,
                target_linear_position=SPAWN_COORDINATES[j],
                target_linear_velocity=np.zeros(3),
                target_linear_acceleration=np.zeros(3)
            )
            PH.apply_ctrl_inputs(agent, propellers_rpm_static)
        
        # --- 3. Step the Simulation ---
        PH.step_sim(AGENTS_LIST)
        
        # Optional: uncomment to run in "real-time" (slows down simulation)
        # sleep(PH.dt)

    # --- End of Simulation ---
    PH.quit()
    print("[MAIN] Simulation finished.")
