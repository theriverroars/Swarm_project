import numpy as np
from sympy import *

from quadrotor_info import Quadrotor


def norm(x, y, z):
    """Helper function for symbolic Euclidean norm."""
    return sqrt(x**2 + y**2 + z**2)


class CBFQPControllerDrone():
    """
    A Control Barrier Function (CBF) safety-filter implemented via a
    Quadratic Program (QP). This controller takes a nominal control input
    (e.g., from a PID controller) and modifies it to ensure safety
    constraints (like obstacle avoidance) are met.

    This class uses SymPy to symbolically define the drone's dynamics
    and the CBF, which allows for automatic differentiation.
    """
    def __init__(self, 
                 acceleration_gravity:float, 
                 CBF_gamma:float
                ):
        """
        Initializes the CBF-QP controller by defining the symbolic system
        dynamics and the control barrier function.

        Args:
            acceleration_gravity (float): The gravitational acceleration.
            CBF_gamma (float): The 'gamma' parameter for the CBF constraint
                               (class K function, h_dot >= -gamma*h).
                               Controls how aggressively the barrier is enforced.
        """
        
        self.gamma = CBF_gamma
        
        # --- Define Symbolic Variables ---
        # State variables (12D state for a quadrotor)
        self.sym_x, self.sym_y, self.sym_z = symbols('x y z') # Position
        self.sym_x_d, self.sym_y_d, self.sym_z_d = symbols('d_x d_y d_z') # Linear Velocity
        self.sym_phi, self.sym_theta, self.sym_psi = symbols('phi theta psi') # Euler Angles
        self.sym_w_1, self.sym_w_2, self.sym_w_3 = symbols('d_phi d_theta d_psi') # Angular Velocity
        
        # System parameters
        self.sym_Ixx, self.sym_Iyy, self.sym_Izz, self.sym_m = symbols('Ixx Iyy Izz m')
        self.sym_L, self.sym_l, self.sym_r = symbols('L_arm l_com r_safety') # Arm length, CoM offset, safety radius
        
        # Obstacle state variables
        self.sym_c_x, self.sym_c_y, self.sym_c_z = symbols('obst_cx, obst_cy, obst_cz')
        self.sym_c_x_d, self.sym_c_y_d, self.sym_c_z_d = symbols('d_obst_cx, d_obst_cy, d_obst_cz')

        # --- Define System Dynamics (x_dot = f(x) + g(x)u) ---
        
        # f(x) - The drift dynamics of the system
        self.f_sys = Matrix([
            self.sym_x_d, 
            self.sym_y_d,
            self.sym_z_d,
            0,  # x_dot_dot
            0,  # y_dot_dot
            -acceleration_gravity, # z_dot_dot
            self.sym_w_1 + self.sym_w_2*sin(self.sym_phi)*tan(self.sym_theta) + self.sym_w_3*cos(self.sym_phi)*tan(self.sym_theta), # phi_dot
            self.sym_w_2*cos(self.sym_phi) - self.sym_w_3*sin(self.sym_phi), # theta_dot
            (self.sym_w_2*sin(self.sym_phi) + self.sym_w_3*cos(self.sym_phi))/cos(self.sym_theta), # psi_dot
            (self.sym_Iyy - self.sym_Izz)*self.sym_w_2*self.sym_w_3/self.sym_Ixx, # w1_dot
            (self.sym_Izz - self.sym_Ixx)*self.sym_w_1*self.sym_w_3/self.sym_Iyy, # w2_dot
            (self.sym_Ixx - self.sym_Iyy)*self.sym_w_1*self.sym_w_2/self.sym_Izz  # w3_dot
        ])

        # g(x) - The control-affine input dynamics
        # Control inputs are [u1, u2, u3, u4] (scaled total thrust & torques)
        p = (cos(self.sym_psi)*sin(self.sym_theta)*cos(self.sym_phi) + sin(self.sym_psi)*sin(self.sym_phi))/self.sym_m
        q = (sin(self.sym_psi)*sin(self.sym_theta)*cos(self.sym_phi) - cos(self.sym_psi)*sin(self.sym_phi))/self.sym_m
        r = (cos(self.sym_theta)*cos(self.sym_phi))/self.sym_m

        self.g_sys = Matrix([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [p, p, p, p], # u1..u4 affect x_dot_dot
            [q, q, q, q], # u1..u4 affect y_dot_dot
            [r, r, r, r], # u1..u4 affect z_dot_dot
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [-self.sym_L/self.sym_Ixx, self.sym_L/self.sym_Ixx, self.sym_L/self.sym_Ixx, -self.sym_L/self.sym_Ixx], # w1_dot
            [self.sym_L/self.sym_Iyy, -self.sym_L/self.sym_Iyy, self.sym_L/self.sym_Iyy, -self.sym_L/self.sym_Iyy], # w2_dot
            [0, 0, 0, 0] # w3_dot (assuming z-torque is not a direct input here, or handled differently)
        ])
            
        # --- Define Control Barrier Function (CBF) for Obstacle Avoidance ---
        # h(x) >= 0 defines the safe set.
        
        # Rotation matrix components from body to world
        r_x = (cos(self.sym_psi)*sin(self.sym_theta)*cos(self.sym_phi) + sin(self.sym_psi)*sin(self.sym_phi))
        r_y = (sin(self.sym_psi)*sin(self.sym_theta)*cos(self.sym_phi) - cos(self.sym_psi)*sin(self.sym_phi))
        r_z = (cos(self.sym_theta)*cos(self.sym_phi))

        # Relative position: obstacle_pos - agent_pos_at_com
        p_rel_x = self.sym_c_x - (self.sym_x + self.sym_l*r_x)
        p_rel_y = self.sym_c_y - (self.sym_y + self.sym_l*r_y)
        p_rel_z = self.sym_c_z - (self.sym_z + self.sym_l*r_z)
        
        # Relative velocity: obstacle_vel - agent_vel_at_com
        v_rel_x = self.sym_c_x_d - (self.sym_x_d + self.sym_l*(-self.sym_w_3*r_y + self.sym_w_2*r_z))
        v_rel_y = self.sym_c_y_d - (self.sym_y_d + self.sym_l*(-self.sym_w_1*r_z + self.sym_w_3*r_x))
        v_rel_z = self.sym_c_z_d - (self.sym_z_d + self.sym_l*(-self.sym_w_2*r_x + self.sym_w_1*r_y))
        
        # C3BF Candidate
        # self.h = Matrix([
        #     p_rel_x*v_rel_x + p_rel_y*v_rel_y + p_rel_z*v_rel_z + norm(v_rel_x, v_rel_y, v_rel_z) * 
        #     sqrt(norm(p_rel_x, p_rel_y, p_rel_z)**2 - self.sym_r**2)
        # ])

        # HOCBF (High Order Control Barrier Function)
        # This is a common form of CBF for enforcing distance.
        self.h = Matrix([
            p_rel_x*v_rel_x + p_rel_y*v_rel_y + p_rel_z*v_rel_z + 
            sqrt(p_rel_x**2 + p_rel_y**2 + p_rel_z**2 - self.sym_r**2)
        ])
            
        # --- Pre-compute the symbolic gradient (Jacobian) of h(x) ---
        # This is the Lie derivative Lf_h = (dh/dx) * f(x)
        self.sym_partial_h_x = Matrix([[
            diff(self.h, self.sym_x),
            diff(self.h, self.sym_y),
            diff(self.h, self.sym_z),
            diff(self.h, self.sym_x_d),
            diff(self.h, self.sym_y_d),
            diff(self.h, self.sym_z_d),
            diff(self.h, self.sym_phi),
            diff(self.h, self.sym_theta),
            diff(self.h, self.sym_psi),
            diff(self.h, self.sym_w_1),
            diff(self.h, self.sym_w_2),
            diff(self.h, self.sym_w_3),
        ]])
        

    def solve_QP(self, 
                 agent:Quadrotor, 
                 u_ref:np.ndarray, 
                 obstacle_positions:np.ndarray, 
                 obstacle_velocities:np.ndarray
                ):
        """
        Solves the CBF-QP to find a minimal, safe correction to the
        reference control input.

        Finds u* = argmin ||u - u_ref||^2
        s.t.     Lg_h * u >= -Lf_h - gamma * h

        Args:
            agent (Quadrotor): The quadrotor object with its current state.
            u_ref (np.ndarray): The nominal control input (e.g., from PID).
            obstacle_positions (np.ndarray): Nx3 array of obstacle positions.
            obstacle_velocities (np.ndarray): Nx3 array of obstacle velocities.

        Returns:
            tuple:
                - (np.ndarray): The safe control *correction* (u_safe).
                                The final control is u_ref + u_safe.
                - (bool): True if a constraint was violated and a correction was needed.
        """
        
        num_obstacles = len(obstacle_positions)
        
        # --- Substitute numerical values into symbolic expressions ---
        # Create a dictionary of all agent state and parameter values
        agent_dict = {
            self.sym_x: agent.pos_states[0], 
            self.sym_y: agent.pos_states[1], 
            self.sym_z: agent.pos_states[2],
            self.sym_x_d: agent.d_pos_states[0], 
            self.sym_y_d: agent.d_pos_states[1], 
            self.sym_z_d: agent.d_pos_states[2],
            self.sym_phi: agent.ang_states[0], 
            self.sym_theta: agent.ang_states[1], 
            self.sym_psi: agent.ang_states[2],
            self.sym_w_1: agent.d_ang_states[0], 
            self.sym_w_2: agent.d_ang_states[1], 
            self.sym_w_3: agent.d_ang_states[2],
            self.sym_Ixx: agent.I[0, 0], 
            self.sym_Iyy: agent.I[1, 1], 
            self.sym_Izz: agent.I[2, 2], 
            self.sym_m: agent.m, 
            self.sym_L: agent.L_CBF,
            self.sym_l: agent.l_com,
            self.sym_r: agent.r_safety
        }
        
        # Evaluate f(x) and g(x) at the current state
        f_val = np.array(re(self.f_sys.xreplace(agent_dict))).astype("float").squeeze()
        g_val = np.array(re(self.g_sys.xreplace(agent_dict))).astype("float")

        h_list = []
        Lf_list = []
        Lg_list = []

        # --- Evaluate CBF constraints for each obstacle ---
        for i in range(num_obstacles):
            obstacle_dict = {
                self.sym_c_x: obstacle_positions[i, 0],
                self.sym_c_y: obstacle_positions[i, 1],
                self.sym_c_z: obstacle_positions[i, 2],
                self.sym_c_x_d: obstacle_velocities[i, 0],
                self.sym_c_y_d: obstacle_velocities[i, 1],
                self.sym_c_z_d: obstacle_velocities[i, 2],
            }
            
            # Combine agent and obstacle dictionaries for full substitution
            replacement_dict = {**agent_dict, **obstacle_dict}
            
            # h_i = h(x)
            h_i = np.array(self.h.xreplace(replacement_dict)).astype("float").squeeze()
            # d_h_x_i = dh/dx
            d_h_x_i = np.array(re(self.sym_partial_h_x.xreplace(replacement_dict))).astype("float").squeeze()
            
            h_list.append(h_i)
            # Lf_h = (dh/dx) * f(x)
            Lf_list.append(np.dot(d_h_x_i, f_val))
            # Lg_h = (dh/dx) * g(x)
            Lg_list.append(np.dot(d_h_x_i, g_val))
        
        h_vec = np.array(h_list)
        Lf_vec = np.array(Lf_list)
        Lg_mat = np.array(Lg_list)
        
        # --- Check for violations ---
        # Psi = Lf_h + Lg_h * u_ref + gamma * h
        # If Psi < 0, the constraint is violated by the reference control.
        Psi_vec = (self.gamma * h_vec) + Lf_vec + (Lg_mat @ u_ref.reshape(-1, 1)).squeeze()
        
        violating_indices = np.where(Psi_vec < 0)[0]
        if len(violating_indices) == 0:
            # No violations, no correction needed
            return np.zeros(4), False

        # --- Solve QP for minimal correction ---
        # We want to find u_safe such that:
        # Lg_h * (u_ref + u_safe) >= -Lf_h - gamma * h
        # Lg_h * u_safe >= - (Lf_h + Lg_h * u_ref + gamma * h)
        # A * u_safe >= b
        
        A = Lg_mat[violating_indices, :]
        b = -Psi_vec[violating_indices].reshape(-1, 1)
        
        # Solve the QP: argmin ||u_safe||^2 s.t. A*u_safe >= b
        # The analytical solution is u_safe = pinv(A) * b
        u_safe = np.linalg.pinv(A) @ b
        return u_safe.squeeze(), True

