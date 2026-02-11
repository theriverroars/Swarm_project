import numpy as np


# --- Simulation Constants ---

GRAVITY_ACC = 9.81
SIM_TIMESTEP = 1/240.

# --- Default Quadrotor Parameters ---

quadrotor_params = {
"arm_length": 3.97e-2,                  # Motor-to-center distance (for torque calculation)
"safety_bound": 2e-1,                   # Safety radius for CBF
"diagonal_length": 1.44e-1,             # Diagonal arm length for CBF
"dist_center_base": 1e-2,               # Z-offset from center of mass to base
"motor_thrust_coeff": 3.16e-10,         # k_f (Thrust = k_f * rpm^2)
"motor_moment_coeff": 7.94e-12,         # k_m (Torque = k_m * rpm^2)
"motor_maximum_rpm": 2.e4,              # Maximum motor RPM
"PID_coeffs_position": np.array([
    [0.7 * 0.7, 0.7 * 0.7, 0.7 * 0.7],             # Kp
    [0., 0., 0.],                                  # Ki
    [2 * 0.5 * 0.7, 2 * 0.5 * 0.7, 2 * 0.5 * 0.7], # Kd
]),
"PID_coeffs_angle": np.array([
    [ 0.7 * 0.7,  0.7 * 0.7,  0.7 * 0.7],         # Kp
    [0., 0., 0.],                                 # Ki
    [2 * 10. * 0.7, 2 * 10. * 0.7, 2 * 10. * 0.7] # Kd
]),
}

# --- Quadrotor Data Handler ---

class Quadrotor():
    """
    Data structure to hold the state, parameters, and controller variables
    for a single quadrotor agent.
    """
    def __init__(self,
                 id:int,
                 mass:float,
                 arm_length:float, 
                 safety_bound:float,
                 diagonal_length:float,
                 dist_center_base:float,
                 inertia_matrix:np.ndarray,
                 motor_thrust_coeff:float, 
                 motor_moment_coeff:float,
                 motor_maximum_rpm:float,
                 PID_coeffs_position:np.ndarray,
                 PID_coeffs_angle:np.ndarray,
                ):
        """
        Initializes the Quadrotor instance with its physical parameters
        and control coefficients.

        Args:
            id (int): The unique ID for this agent (e.g., its PyBullet body ID).
            mass (float): Mass (m) of the quadrotor.
            arm_length (float): Motor-to-center distance (L).
            safety_bound (float): Safety radius (r_safety) for CBF.
            diagonal_length (float): Diagonal arm length (L_CBF) for CBF.
            dist_center_base (float): CoM offset (l_com) for CBF.
            inertia_matrix (np.ndarray): 3x3 inertia matrix (I).
            motor_thrust_coeff (float): Thrust coefficient (k_f).
            motor_moment_coeff (float): Moment coefficient (k_m).
            motor_maximum_rpm (float): Maximum allowed motor RPM.
            PID_coeffs_position (np.ndarray): 3x3 array [Kp, Ki, Kd] for position.
            PID_coeffs_angle (np.ndarray): 3x3 array [Kp, Ki, Kd] for attitude.
        """

        self.id = id # PyBullet body ID
        
        # --- Physical Parameters ---
        self.L = arm_length
        self.L_CBF = diagonal_length
        self.l_com = dist_center_base
        self.r_safety = safety_bound
        self.I = inertia_matrix
        self.m = mass
        
        # --- Motor Parameters ---
        self.kf = motor_thrust_coeff
        self.km = motor_moment_coeff
        self.max_rpm = motor_maximum_rpm
        
        # --- 12D State Variables ---
        self.pos_states = np.zeros(3, dtype="float") # [x, y, z]
        self.ang_states = np.zeros(3, dtype="float") # [phi, theta, psi]
        self.d_pos_states = np.zeros(3, dtype="float") # [x_dot, y_dot, z_dot]
        self.d_ang_states = np.zeros(3, dtype="float") # [w_x, w_y, w_z]
        
        # --- PID Controller Internal State ---
        self.prev_angle_error = np.zeros(3, dtype="float")
        self.prev_position_error = np.zeros(3, dtype="float")
        self.integral_angle_error = np.zeros_like(self.ang_states)
        self.integral_position_error = np.zeros_like(self.pos_states)
        
        # --- PID Gains ---
        self.PID_coeffs_pos = PID_coeffs_position
        self.PID_coeffs_ang = PID_coeffs_angle

