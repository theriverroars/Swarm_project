#!/usr/bin/env python
import os
from pathlib import Path
from threading import Thread
from rclpy.executors import SingleThreadedExecutor

from crazyflie_py import Crazyswarm
from nmpc_cf2.emulate import Emulate
import numpy as np
import pyquaternion
import datetime

from nmpc_cf2.nmpc_utils import lemniscate_trajectory


def main():
    swarm = Crazyswarm()  # already has rclpy.init()

    timeHelper = swarm.timeHelper
    cf = swarm.allcfs.crazyflies[0]
    rate = 100
    Z = 1.5
    vmax = 1.00
    accel = 0.10
    traj_Z = 1.5
    radius = 1.5
    cf_trajectory, cf_input, cf_time = lemniscate_trajectory(
        discretization_dt=0.01, radius=radius, z=traj_Z, lin_acc=accel, v_max=vmax
    )

    nmpc_emulate = Emulate(
        t_array=cf_time,
        ref_trajectory=cf_trajectory,
        ref_input=cf_input,
        emulate_state=["pos", "quat", "vel", "a_rate"],
    )
    nmpc_exec = SingleThreadedExecutor()
    nmpc_exec.add_node(nmpc_emulate)
    nmpc_thread = Thread(target=nmpc_exec.spin)
    nmpc_thread.start()

    nmpc_emulate.quad.state.pos = cf_trajectory[0, 0:3]
    nmpc_emulate.quad.state.quat = cf_trajectory[0, 3:7]
    nmpc_emulate.quad.state.vel = cf_trajectory[0, 7:10]
    nmpc_emulate.quad.state.a_rate = cf_trajectory[0, 10:13]

    try:
        cf.setParam("stabilizer.controller", 2)
        cf.takeoff(targetHeight=0.5, duration=2.0)
        timeHelper.sleep(2.5)
        executeTrajectory(
            timeHelper, cf, cf_trajectory, cf_input, cf_time, nmpc_emulate
        )
        cf.notifySetpointsStop()
        print("Going to Origin")
        cf.goTo(np.array([0, 0, Z]), 0, 3)
        timeHelper.sleep(3.5)
        cf.land(targetHeight=0.07, duration=Z + 1.0)
        timeHelper.sleep(Z + 1.0)

        # emu_node.destroy_node()
        # emu_exe.shutdown()
        # emu_thread.join()
        nmpc_emulate.destroy_node()
        nmpc_exec.shutdown()
        nmpc_thread.join()

    except KeyboardInterrupt:
        cf.notifySetpointsStop()
        cf.emergency()


def executeTrajectory(
    timeHelper, cf, cf_trajectory, cf_input, cf_time, nmpc_node, rate=100
):
    """
    Execute the trajectory on the Crazyflie
    cf_trajectory: position, quaternion, velocity, angular rate
    """

    print("Executing trajectory")
    print("Trajectory Duration: ", cf_time[-1])

    print("Going to initial trajectory point")
    init_position = cf_trajectory[0, 0:3]
    cf.goTo(init_position, 0, 5.0)
    timeHelper.sleep(5.5)

    print("Starting trajectory execution")
    start_time = timeHelper.time()
    while not timeHelper.isShutdown():
        t = timeHelper.time() - start_time
        if t > cf_time[-1] - 2.0:
            print("Trajectory execution finished")
            break

        try:
            position_next, velocity_next, acceleration_next, q_next, omega_next = (
                nmpc_node.controller()
            )
            yaw_next = quaternion_to_euler(q_next)[2]
        except Exception as e:
            print(e)
            print("Error in controller")
            break
        except KeyboardInterrupt:
            print("Keyboard interrupt received")
            cf.notifySetpointsStop()
            cf.emergency()
            break

        cf.cmdFullState(
            position_next,
            velocity_next,
            acceleration_next,
            yaw_next,
            omega_next,
        )

        timeHelper.sleepForRate(rate)


def quaternion_to_euler(q):
    q = pyquaternion.Quaternion(w=q[0], x=q[1], y=q[2], z=q[3])
    yaw, pitch, roll = q.yaw_pitch_roll
    return [roll, pitch, yaw]

if __name__ == "__main__":
    main()