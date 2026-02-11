#!/usr/bin/env python
import argparse
from threading import Thread
from rclpy.executors import SingleThreadedExecutor

from crazyflie_py import Crazyswarm
import numpy as np

from nmpc_cf2.cbf_emulate import CBFEmulate
from nmpc_cf2.cbf_swarm_controller import CBFMultiController


def _cf_name_from_obj(cf, fallback):
    for attr in ["name", "id", "prefix", "cf_id"]:
        if hasattr(cf, attr):
            value = getattr(cf, attr)
            if isinstance(value, str):
                return value
            if isinstance(value, int):
                return f"cf{value}"
    return fallback


def execute_trajectory(
    timeHelper,
    cf_list,
    target_positions,
    target_velocities,
    target_accelerations,
    cbf_emulate,
    rate=100,
):
    """
    Execute trajectory with CBF-based safety filtering for multi-agent deployment.
    
    Args:
        timeHelper: Crazyswarm time helper
        cf_list: List of Crazyflie objects to control
        target_positions: Array of target positions for each agent [num_agents, 3]
        target_velocities: Array of target velocities for each agent [num_agents, 3]
        target_accelerations: Array of target accelerations for each agent [num_agents, 3]
        rate: Control loop rate (Hz)
        cbf_emulate: CBFEmulate instance
    """

    num_agents = len(cf_list)

    print("[CBF_RUN] Starting trajectory execution with CBF safety filter")
    print(f"[CBF_RUN] Controlling {num_agents} agents")
    
    start_time = timeHelper.time()
    iteration = 0
    
    while not timeHelper.isShutdown():
        t = timeHelper.time() - start_time
        iteration += 1
        
        try:
            commands = cbf_emulate.controller_step(
                target_positions, target_velocities, target_accelerations
            )
        except Exception as e:
            print(f"[CBF_RUN] Error in controller: {e}")
            break

        for i, cmd in enumerate(commands):
            if cmd is None:
                continue

            position_next, velocity_next, acceleration_next, yaw_next, omega_next = cmd

            try:
                cf_list[i].cmdFullState(
                    position_next,
                    velocity_next,
                    acceleration_next,
                    yaw_next,
                    omega_next,
                )
            except Exception as e:
                print(f"[CBF_RUN] Error sending command to agent {i}: {e}")
        
        # Maintain control loop rate
        timeHelper.sleepForRate(rate)
        
        # Optional: Print periodic status
        if iteration % 100 == 0:
            print(f"[CBF_RUN] Iteration {iteration}, Time: {t:.2f}s")


def main():
    """Main function for multi-agent CBF-based deployment"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Multi-agent CBF deployment for Crazyflie')
    parser.add_argument('--num_drones', type=int, default=1, 
                       help='Number of drones to control (default: 1)')
    parser.add_argument('--duration', type=float, default=30.0,
                       help='Trajectory duration in seconds (default: 30.0)')
    parser.add_argument('--height', type=float, default=1.5,
                       help='Flight height in meters (default: 1.5)')
    parser.add_argument('--hoop_position', type=float, nargs=3, default=[0.0, 1.0, 1.0],
                       help='Hoop position as three floats: x y z (default: 0.0 1.0 1.0)')
    parser.add_argument('--hoop_direction', type=float, nargs=3, default=[0.0, 1.0, 0.0],
                       help='Hoop direction as three floats: x y z (default: 0.0 1.0 0.0)')
    args = parser.parse_args()
    num_drones = args.num_drones
    
    print(f"[CBF_RUN] Initializing Crazyswarm with {num_drones} drones")
    
    # Initialize Crazyswarm
    swarm = Crazyswarm()  # Already calls rclpy.init()
    timeHelper = swarm.timeHelper
    
    cf_list = swarm.allcfs.crazyflies[:num_drones]
    cf_names = [_cf_name_from_obj(cf, f"cf{i}") for i, cf in enumerate(cf_list)]

    cbf_controller = CBFMultiController(num_drones=num_drones, cbf_gamma=1.0)
    hoop_center = np.array(args.hoop_position)
    hoop_direction = np.array(args.hoop_direction)
    cbf_emulate = CBFEmulate(cf_names=cf_names, controller=cbf_controller, hoop_center=hoop_center, hoop_direction=hoop_direction)

    cbf_exec = SingleThreadedExecutor()
    cbf_exec.add_node(cbf_emulate)
    cbf_thread = Thread(target=cbf_exec.spin)
    cbf_thread.start()
    
    # Setup trajectory parameters
    Z = args.height
    vmax = 1.00
    accel = 0.10
    radius = 1.5
    
    # Create simple trajectory targets for each agent
    # Each agent gets a slightly offset target position
    target_positions = np.zeros((num_drones, 3))
    target_velocities = np.zeros((num_drones, 3))
    target_accelerations = np.zeros((num_drones, 3))
    
    for i in range(num_drones):
        target_positions[i] = np.array([0.5 * i, 0, Z])  # Spread agents along X-axis
        target_velocities[i] = np.array([0.0, 0.0, 0.0])
        target_accelerations[i] = np.array([0.0, 0.0, 0.0])
    
    try:
        # Takeoff phase
        print("[CBF_RUN] Taking off...")
        for i, cf in enumerate(cf_list[:num_drones]):
            cf.setParam("stabilizer.controller", 2)  # Set to controller mode 2
            cf.takeoff(targetHeight=Z * 0.3, duration=2.0)
        
        timeHelper.sleep(2.5)
        
        # Execute trajectory with CBF safety filter
        execute_trajectory(
            timeHelper=timeHelper,
            cf_list=cf_list[:num_drones],
            target_positions=target_positions,
            target_velocities=target_velocities,
            target_accelerations=target_accelerations,
            cbf_emulate=cbf_emulate,
            rate=100,
        )
        
        # Landing phase
        print("[CBF_RUN] Landing...")
        for cf in cf_list[:num_drones]:
            cf.notifySetpointsStop()
            cf.land(targetHeight=0.07, duration=Z + 1.0)
        
        timeHelper.sleep(Z + 1.5)
        
        print("[CBF_RUN] Mission complete")

        cbf_emulate.destroy_node()
        cbf_exec.shutdown()
        cbf_thread.join()
        
    except KeyboardInterrupt:
        print("[CBF_RUN] Keyboard interrupt received")
        for cf in cf_list[:num_drones]:
            cf.notifySetpointsStop()
            cf.emergency()
    except Exception as e:
        print(f"[CBF_RUN] Error during execution: {e}")
        for cf in cf_list[:num_drones]:
            try:
                cf.notifySetpointsStop()
                cf.emergency()
            except:
                pass

        cbf_emulate.destroy_node()
        cbf_exec.shutdown()
        cbf_thread.join()


if __name__ == "__main__":
    main()
