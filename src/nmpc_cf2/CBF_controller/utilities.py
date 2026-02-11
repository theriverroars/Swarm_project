import csv
import numpy as np
import pybullet as pb
from os.path import join
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from quadrotor_info import quadrotor_params, Quadrotor


#########################################################################################################
#########################################################################################################


def spawn_hoop(HOOP_COORDINATES, HOOP_DIRECTION, HOOP_URDF_PATH=join("assets", "hoop.urdf")):
    pb.loadURDF(
        HOOP_URDF_PATH, 
        HOOP_COORDINATES, 
        Rotation.align_vectors([HOOP_DIRECTION], np.array([[0., 0., 1.]]))[0].as_quat(),
        useFixedBase=True
    )


#########################################################################################################
#########################################################################################################


def spawn_agents(SPAWN_COORDINATES, QUADROTOR_URDF_PATH=join("assets", "cf2x.urdf")):
    """
    Spawns quadrotor agents into the PyBullet simulation at the specified coordinates.

    Args:
        SPAWN_COORDINATES (np.ndarray): An Nx3 array where each row is an [x, y, z]
                                        coordinate for spawning a new agent.

    Returns:
        list[Quadrotor]: A list of initialized Quadrotor objects.
    """
    AGENTS_LIST = []
    print(f"[MAIN] Spawning {len(SPAWN_COORDINATES)} agents...")

    for coord in SPAWN_COORDINATES:
        # Load the URDF file into PyBullet
        agent_id = pb.loadURDF(QUADROTOR_URDF_PATH, coord, useFixedBase=False)
        
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


#########################################################################################################
#########################################################################################################


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


#########################################################################################################
#########################################################################################################


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
    plt.tight_layout()
    plt.show()


#########################################################################################################
#########################################################################################################


def plot_3d_trajectories(position_data, num_agents:int):
    # Plot 3 d position trajectories for each agent using position_data
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    for agent_id in range(num_agents):
        xs = [row[1 + agent_id * 3] for row in position_data]
        ys = [row[2 + agent_id * 3] for row in position_data]
        zs = [row[3 + agent_id * 3] for row in position_data]
        ax.plot(xs, ys, zs, label="Drone "+str(agent_id), linewidth=2)

    #Ensure equal scaling on all axes
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-3, 3)   
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Trajectories of 3 Drones")
    ax.legend()
    plt.show()  


#########################################################################################################
#########################################################################################################


def plot_rpm_history(agent_id, rpm_history):
    """
    Plot RPM history for a specific agent.
    
    Args:
        agent_id (int): The agent ID to plot
    """
    if agent_id not in rpm_history:
        print(f"No RPM history found for agent {agent_id}")
        return
    
    history = rpm_history[agent_id]
    time_array = np.array(history['time'])
    
    # Create subplots for each motor
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'RPM History for Agent {agent_id}', fontsize=16)
    
    motors = ['Motor 1', 'Motor 2', 'Motor 3', 'Motor 4']
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    for i, (motor_name, pos) in enumerate(zip(motors, positions)):
        ax = axes[pos[0], pos[1]]
        ax.plot(time_array, history['ref_rpm'][i], 'b--', label='Reference RPM', linewidth=2)
        ax.plot(time_array, history['actual_rpm'][i], 'r-', label='Actual RPM', linewidth=1.5)
        ax.set_title(motor_name)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('RPM')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


#########################################################################################################
#########################################################################################################


def plot_all_agents_rpm(rpm_history):
    """
    Plot RPM comparison for all agents.
    """
    if not rpm_history:
        print("No RPM history found for any agents")
        return
    
    num_agents = len(rpm_history)
    fig, axes = plt.subplots(num_agents, 4, figsize=(16, 4*num_agents))
    if num_agents == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('RPM History Comparison - All Agents', fontsize=16)
    
    for agent_idx, agent_id in enumerate(sorted(rpm_history.keys())):
        history = rpm_history[agent_id]
        time_array = np.array(history['time'])
        
        for motor_idx in range(4):
            ax = axes[agent_idx, motor_idx]
            ax.plot(time_array, history['ref_rpm'][motor_idx], 'b--', label='Ref', linewidth=1.5)
            ax.plot(time_array, history['actual_rpm'][motor_idx], 'r-', label='Actual', linewidth=1)
            ax.set_title(f'Agent {agent_id} - Motor {motor_idx+1}')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('RPM')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


#########################################################################################################
#########################################################################################################

