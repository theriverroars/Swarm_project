import numpy as np
# from gym_cbf.envs.BaseAviary import BaseAviary
from scipy import integrate

class Quad3D():
    """"""

    def __init__(self):
        """ 
        Parameters
        ----------
        env : BaseAviary
            The PyBullet-based simulation environment.

        """
        self.g = 9.8
        """float: Gravity acceleration, in meters per second squared."""
        self.mass = 0.0316#0.0316
        """float: The mass of quad from environment."""
        self.inertia_xx =1.395e-5#env.J[0][0]
        """float: The inertia of quad around x axis."""
        self.arm_length = 0.046#env.L
        """float: The inertia of quad around x axis."""
        self.timestep = 0.01#env.TIMESTEP
        """float: Simulation and control timestep."""
        self.last_rpy = np.zeros(3)
        """ndarray: Store the last roll, pitch, and yaw."""
        self.kf_coeff = 3.16e-10#env.KF
        """float: RPMs to force coefficient."""
        self.km_coeff = 7.94e-12#env.KM
        """int: Flag switching beween implementations of u1."""
        self.p_coeff_position = {}
        """dict[str, float]: Proportional coefficient(s) for position control."""
        self.d_coeff_position = {}
        """dict[str, float]: Derivative coefficient(s) for position control."""

        self.matrix_u2rpm = np.array([ [1,   1,   1,   1],
                                       [-1/np.sqrt(2),   -1/np.sqrt(2),   1/np.sqrt(2),  1/np.sqrt(2)],
                                       [-1/np.sqrt(2),   1/np.sqrt(2),  1/np.sqrt(2),   -1/np.sqrt(2)],
                                       [-1,  1,   -1,  1] 
                                      ])

        self.matrix_u2rpm_inv = np.linalg.inv(self.matrix_u2rpm)

        self.p_coeff_position["x"] = 5*0.05  #0.0005
        self.d_coeff_position["x"] = 50 * 0.08 #0.05
        self.p_coeff_position["y"] = 5*0.05 #
        self.d_coeff_position["y"] = 50 * 0.08 #0
        self.p_coeff_position["z"] = 5*0.05  #0.0005
        self.d_coeff_position["z"] = 5 * 0.08 #0.05
        self.p_coeff_position["r"] = 0.5*0.07 #.0005 #0.7 * 0.7*0.9 
        self.d_coeff_position["r"] = 2 * 0.5 #.05#2 * 2.5 * 0.7 * 1.5*0.1
        self.p_coeff_position["p"] = 0.5*0.07  #0.7 * 0.7*0.95
        self.d_coeff_position["p"] = 2 * 0.5  #2 * 2.5 * 0.7 * 1.5
        self.p_coeff_position["ya"] = 0.7 * 0.07#.0005 #0.7 * 0.7
        self.d_coeff_position["ya"] = 2 * 0.5 #.05#2 * 2.5 * 0.7 * 1.5

        self.r_dd =[]
        self.p_dd =[]
        self.y_dd =[]

        self.r_d =[]
        self.p_d =[]
        self.y_d =[]

        self.time = []


        self.reset()

    def reset(self):
        """ Resets the controller counter."""
        self.control_counter = 0

    def compute_control(self,
                        current_position,
                        current_velocity,
                        current_rpy,
                        target_position,
                        target_velocity=np.zeros(3),
                        target_acceleration=np.zeros(3),
                        TIMESTEP= 0.01
                        ):
        """Computes the propellers' RPMs for the target state, given the current state.

        Parameters
        ----------
        current_position : ndarray
            (3,)-shaped array of floats containing global x, y, z, in meters.
        current_velocity : ndarray
            (3,)-shaped array of floats containing global vx, vy, vz, in m/s.
        current_rpy : ndarray
            (3,)-shaped array of floats containing roll, pitch, yaw, in rad.
        target_position : ndarray
            (3,)-shaped array of float containing global x, y, z, in meters.
        target_velocity : ndarray, optional
            (3,)-shaped array of floats containing global, in m/s.
        target_acceleration : ndarray, optional
            (3,)-shaped array of floats containing global, in m/s^2.

        Returns
        -------
        ndarray
            (4,)-shaped array of ints containing the desired RPMs of each propeller.
        """
        self.control_counter += 1
        self.timestep = TIMESTEP

        # print(current_rpy, self.last_rpy)

        # Compute roll, pitch, and yaw rates
        current_rpy_dot = (current_rpy - self.last_rpy) / self.timestep

        ## Calculate PD control in y, z
        x_ddot = self.pd_control(target_position[0],
                                 current_position[0],
                                 target_velocity[0],
                                 current_velocity[0],
                                 target_acceleration[0],
                                 "x"
                                 )

        y_ddot = self.pd_control(target_position[1],
                                 current_position[1],
                                 target_velocity[1],
                                 current_velocity[1],
                                 target_acceleration[1],
                                 "y"
                                 )
        z_ddot = self.pd_control(target_position[2],
                                 current_position[2],
                                 target_velocity[2],
                                 current_velocity[2],
                                 target_acceleration[2],
                                 "z"
                                 )

        # Calculate desired roll and rates given by PD
        desired_roll = np.arctan((x_ddot*np.sin(current_rpy[2]) - y_ddot*np.cos(current_rpy[2])) / (self.g + z_ddot)) #-y_ddot/(self.g + z_ddot)#
        desired_roll_dot = (desired_roll - current_rpy[0]) / self.timestep
        roll_ddot = (desired_roll_dot - current_rpy_dot[0]) / self.timestep

        desired_pitch = np.arctan((x_ddot*np.cos(current_rpy[2]) + y_ddot*np.sin(current_rpy[2]) )/ (self.g + z_ddot)) #x_ddot/(self.g + z_ddot)#
        desired_pitch_dot = (desired_pitch - current_rpy[1]) / self.timestep
        pitch_ddot = (desired_pitch_dot - current_rpy_dot[0]) / self.timestep
        print(x_ddot, y_ddot, z_ddot, roll_ddot)

        # Calculate thrust and moment given the PD input
        u_1 = self.mass * np.sqrt(x_ddot**2+y_ddot**2+(self.g + z_ddot)**2)
        u_2 = self.inertia_xx * roll_ddot
        u_3 = -self.inertia_xx * pitch_ddot
        # print(current_position[0])
        
        # Calculate RPMs
        u = np.array([ [u_1 / self.kf_coeff],
                       [u_2 / (self.arm_length*self.kf_coeff)],
                       [u_3 / (self.arm_length*self.kf_coeff)],
                       [0] ])
        propellers_rpm = np.dot(self.matrix_u2rpm_inv, u)

        for i in range(4):
            if propellers_rpm[i, 0]<0:
                propellers_rpm[i, 0] = 0

        # Command the turn rates of propellers 1 and 3
        propellers_1_rpm = np.sqrt(propellers_rpm[1, 0])
        propellers_3_rpm = np.sqrt(propellers_rpm[3, 0])
        propellers_0_rpm = np.sqrt(propellers_rpm[0, 0])
        propellers_2_rpm = np.sqrt(propellers_rpm[2, 0])

        # Print relevant output
        if self.control_counter%(1/self.timestep) == 0:
            print("current_position", current_position)
            print("current_velocity", current_velocity)
            print("target_position", target_position)
            print("target_velocity", target_velocity)
            print("target_acceleration", target_acceleration)

        # Store the last step's roll, pitch, and yaw
        self.last_rpy = current_rpy

        return x_ddot, y_ddot, z_ddot,np.array([propellers_0_rpm, propellers_1_rpm, propellers_2_rpm, propellers_3_rpm])

    def compute_xyz_ddot(self,
                        rpms, dt, current_rpy, current_rpy_rates,
                        ):
        """Computes the propellers' RPMs for the target state, given the current state.

        Parameters
        ----------
        ndarray
            (4,)-shaped array of ints containing the desired RPMs of each propeller.

        Returns
        -------
        x_ddot : float
            x_ddot values corresponding to the rpms.
        y_ddot : float
            y_ddot values corresponding to the rpms.
        z_ddot : float
            z_ddot values corresponding to the rpms.
        """

        propellers_rpm = np.square(rpms)
        u = np.dot(self.matrix_u2rpm, propellers_rpm)

        u1 = self.kf_coeff*u[0]
        u2 = self.arm_length*self.kf_coeff*u[1]
        u3 = self.arm_length*self.kf_coeff*u[2]

        r_ddot = u2/(self.inertia_xx)
        p_ddot = -u3/(self.inertia_xx)
        # print(r_ddot, p_ddot)

        # self.time.append(t)

        # self.r_dd.append(r_ddot)
        # self.p_dd.append(p_ddot)
        
        r_dot = r_ddot*dt + current_rpy_rates[0]
        p_dot = p_ddot*dt + current_rpy_rates[1]
        
        # self.r_d.append(r_dot)
        # self.p_d.append(p_dot)

        # print(len(self.p_d), len(self.time))
        
        r = r_dot*dt + current_rpy[0]
        p = p_dot*dt + current_rpy[1]

        # print('int_rp:', r_dot, p_dot)

        a = u1/(self.mass*np.sqrt((np.tan(r))**2 + (np.tan(p))**2 + 1))

        z_ddot = -self.g + a
        x_ddot = (np.tan(p)*a)
        y_ddot = -(np.tan(r)*a)

        return x_ddot, y_ddot, z_ddot

    def pd_control(self,
                   desired_position,
                   current_position,
                   desired_velocity,
                   current_velocity,
                   desired_acceleration,
                   opt
                   ):
        """Computes PD control for the acceleration minimizing position error.

        Parameters
        ----------
        desired_position :
            float: Desired global position.
        current_position :
            float: Current global position.
        desired_velocity :
            float: Desired global velocity.
        current_velocity :
            float: Current global velocity.
        desired_acceleration :
            float: Desired global acceleration.

        Returns
        -------
        float
            The commanded acceleration.
        """
        u = desired_acceleration + \
            self.d_coeff_position[opt] * (desired_velocity - current_velocity) + \
            self.p_coeff_position[opt] * (desired_position - current_position) 

        return u
