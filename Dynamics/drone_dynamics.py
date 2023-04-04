" Dynamics for crazylflie 2.1"
import numpy as np
import pybullet as p

def _dynamics(params,
             forces):
        """Explicit dynamics implementation.

        Based on code written at the Dynamic Systems Lab by James Xu.

        Parameters
        ----------
        rpm : ndarray
            (4)-shaped array of ints containing the RPMs values of the 4 motors.
        
        

        """
        ## params
        KF = 3.16e-10
        KM = 7.94e-12
        L = 0.0397
        DRONE_MODEL = "C2FP"
        IXX = 2.3951e-5
        IYY = 2.3951e-5
        IZZ = 3.2347e-5
        GRAVITY = 9.81
        M = 0.027



        #### Current state #########################################
        pos = params['pos']
        quat = params['quat']
        rpy = params['rpy']
        vel = params['vel']
        rpy_rates = params['rpy_rates']
        TIMESTEP = params['dt']
        rotation = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        
        ## Compute Inertia Matrix #####################################
        J = np.diag([IXX, IYY, IZZ])
        J_INV = np.linalg.inv(J)
        
        
        #### Compute forces and torques ############################
        thrust = np.array([0, 0, np.sum(forces)])
        thrust_world_frame = np.dot(rotation, thrust)
        force_world_frame = thrust_world_frame - np.array([0, 0, GRAVITY])
        z_torques = forces*KM/KF
        z_torque = (-z_torques[0] + z_torques[1] - z_torques[2] + z_torques[3])
        if DRONE_MODEL== "CF2X":
            x_torque = (forces[0] + forces[1] - forces[2] - forces[3]) * (L/np.sqrt(2))
            y_torque = (- forces[0] + forces[1] + forces[2] - forces[3]) * (L/np.sqrt(2))
        elif DRONE_MODEL=="CF2P" or DRONE_MODEL== "HB":
            x_torque = (forces[1] - forces[3]) * L
            y_torque = (-forces[0] + forces[2]) * L
        torques = np.array([x_torque, y_torque, z_torque])
        torques = torques - np.cross(rpy_rates, np.dot(J, rpy_rates))
        rpy_rates_deriv = np.dot(J_INV, torques)
        acc = force_world_frame / M
        #### Update state ##########################################
        vel = vel + TIMESTEP * acc
        rpy_rates = rpy_rates + TIMESTEP * rpy_rates_deriv
        pos = pos + TIMESTEP * vel
        rpy = rpy + TIMESTEP * rpy_rates

        return pos, vel, rpy, rpy_rates, thrust, np.array([x_torque, y_torque, z_torque])
