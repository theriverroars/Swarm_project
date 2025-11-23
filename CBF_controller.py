import numpy as np
from sympy import *
from sympy import lambdify

import osqp
from quadrotor_info import Quadrotor
from scipy.sparse import csc_matrix


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
                 CBF_gamma:float,
                 hoop_center_coord=None,
                 hoop_direction=None,
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
        
        if hoop_center_coord is not None and hoop_direction is not None:
            self.hoop_x, self.hoop_y, self.hoop_z = hoop_center_coord
            hoop_x_dir, hoop_y_dir, hoop_z_dir = hoop_direction
            # store hoop direction as attributes (they're used as numeric constants inside symbolic expressions)
            self._hoop_dir = (float(hoop_x_dir), float(hoop_y_dir), float(hoop_z_dir))
            self._hoop_center = (float(self.hoop_x), float(self.hoop_y), float(self.hoop_z))
        else:
            self._hoop_dir = None
            self._hoop_center = None

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
        
        # Relative position & velocity projected in the plane of hoop
        if hoop_center_coord is not None and hoop_direction is not None:
            p_proj_mag = p_rel_x*hoop_x_dir + p_rel_y*hoop_y_dir + p_rel_z*hoop_z_dir
            p_rel_proj_x = p_rel_x - hoop_x_dir * p_proj_mag
            p_rel_proj_y = p_rel_y - hoop_y_dir * p_proj_mag
            p_rel_proj_z = p_rel_z - hoop_z_dir * p_proj_mag
            
            v_proj_mag = v_rel_x*hoop_x_dir + v_rel_y*hoop_y_dir + v_rel_z*hoop_z_dir
            v_rel_proj_x = v_rel_x - hoop_x_dir * v_proj_mag
            v_rel_proj_y = v_rel_y - hoop_y_dir * v_proj_mag
            v_rel_proj_z = v_rel_z - hoop_z_dir * v_proj_mag
    
        # C3BF Candidate
        # self.h_obst = Matrix([
        #     (p_rel_x*v_rel_x + p_rel_y*v_rel_y + p_rel_z*v_rel_z) + 
        #     sqrt(v_rel_x**2 + v_rel_y**2 + v_rel_z**2) * 
        #     sqrt(p_rel_x**2 + p_rel_y**2 + p_rel_z**2 - self.sym_r**2)
        # ])

        # HOCBF (High Order Control Barrier Function)
        # self.h_obst = Matrix([
        #     p_rel_x*v_rel_x + p_rel_y*v_rel_y + p_rel_z*v_rel_z + 
        #     0.25*((p_rel_x**2 + p_rel_y**2 + p_rel_z**2 - self.sym_r**2))
        # ])

        self.h_obst = Matrix([
            p_rel_x*v_rel_x + p_rel_y*v_rel_y + p_rel_z*v_rel_z + 
            0.3* sqrt(p_rel_x**2 + p_rel_y**2 + p_rel_z**2 - self.sym_r**2)
        ])

        # HOCBF for the "stay-in" constraint
        self.h_hoop = Matrix([
            -2* (p_rel_proj_x*v_rel_proj_x + p_rel_proj_y*v_rel_proj_y + p_rel_proj_z*v_rel_proj_z) + 
            self.gamma * ((self.sym_r*(0.25 + p_proj_mag**2))**2 - (p_rel_proj_x**2 + p_rel_proj_y**2 + p_rel_proj_z**2))
        ])
        
        # --- Pre-compute the symbolic gradient (Jacobian) of h(x) ---
        self.sym_partial_h_obst = Matrix([[
            diff(self.h_obst, self.sym_x),
            diff(self.h_obst, self.sym_y),
            diff(self.h_obst, self.sym_z),
            diff(self.h_obst, self.sym_x_d),
            diff(self.h_obst, self.sym_y_d),
            diff(self.h_obst, self.sym_z_d),
            diff(self.h_obst, self.sym_phi),
            diff(self.h_obst, self.sym_theta),
            diff(self.h_obst, self.sym_psi),
            diff(self.h_obst, self.sym_w_1),
            diff(self.h_obst, self.sym_w_2),
            diff(self.h_obst, self.sym_w_3),
        ]])
        
        self.sym_partial_h_hoop = Matrix([[
            diff(self.h_hoop, self.sym_x),
            diff(self.h_hoop, self.sym_y),
            diff(self.h_hoop, self.sym_z),
            diff(self.h_hoop, self.sym_x_d),
            diff(self.h_hoop, self.sym_y_d),
            diff(self.h_hoop, self.sym_z_d),
            diff(self.h_hoop, self.sym_phi),
            diff(self.h_hoop, self.sym_theta),
            diff(self.h_hoop, self.sym_psi),
            diff(self.h_hoop, self.sym_w_1),
            diff(self.h_hoop, self.sym_w_2),
            diff(self.h_hoop, self.sym_w_3),
        ]])

        # ===== LAMBDIFY: create fast numeric callables to avoid repeated xreplace/re(...) =====
        # Create ordered lists of agent and obstacle symbols for lambdify calls
        self._agent_symbols = [
            self.sym_x, self.sym_y, self.sym_z,
            self.sym_x_d, self.sym_y_d, self.sym_z_d,
            self.sym_phi, self.sym_theta, self.sym_psi,
            self.sym_w_1, self.sym_w_2, self.sym_w_3,
            self.sym_Ixx, self.sym_Iyy, self.sym_Izz,
            self.sym_m, self.sym_L, self.sym_l, self.sym_r
        ]

        self._obst_symbols = [
            self.sym_c_x, self.sym_c_y, self.sym_c_z,
            self.sym_c_x_d, self.sym_c_y_d, self.sym_c_z_d
        ]

        # Create namespace for lambdify to handle complex symbolic functions
        namespace = {
            'sqrt': np.sqrt, 
            'sin': np.sin, 
            'cos': np.cos, 
            'tan': np.tan,
            'abs': np.abs,
            'sign': np.sign,
            'Derivative': lambda *args: 0.0,  # Fallback for unevaluated derivatives
            're': np.real,
            'im': np.imag
        }
        
        # Lambdify f and g (depend only on agent state & params)
        self.f_func = lambdify(self._agent_symbols, self.f_sys, [namespace, 'numpy'])
        self.g_func = lambdify(self._agent_symbols, self.g_sys, [namespace, 'numpy'])

        # Lambdify CBFs and their gradients (depend on agent + obstacle symbols)
        self.h_obst_func = lambdify(self._agent_symbols + self._obst_symbols, self.h_obst, [namespace, 'numpy'])
        self.dh_obst_func = lambdify(self._agent_symbols + self._obst_symbols, self.sym_partial_h_obst, [namespace, 'numpy'])

        self.h_hoop_func = lambdify(self._agent_symbols + self._obst_symbols, self.h_hoop, [namespace, 'numpy'])
        self.dh_hoop_func = lambdify(self._agent_symbols + self._obst_symbols, self.sym_partial_h_hoop, [namespace, 'numpy'])

        # --- Setup the OSQP Solver ---
        self.solver = osqp.OSQP()
        # Flag to set up the solver on the first run
        self.solver_initialized = False

        # self.min_thrust_ratio = 0.3  # Minimum thrust as a ratio of weight
        # self.acceleration_gravity = acceleration_gravity
        
        
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

        Minimize: 1/2 x^T P x + q^T x
        Subject to: G x <= h
        Using OSQP solver:
            Our constraint is A * u_safe >= b
            osqp format is l <= A_osqp * x <= u

        Where: 
            x = u_safe
            A_osqp = A (from A * u_safe >= b)
            l = b (from A * u_safe >= b)
            u = infinity

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
        agent_vals = [
            float(agent.pos_states[0]), 
            float(agent.pos_states[1]), 
            float(agent.pos_states[2]),
            float(agent.d_pos_states[0]), 
            float(agent.d_pos_states[1]), 
            float(agent.d_pos_states[2]),
            float(agent.ang_states[0]), 
            float(agent.ang_states[1]), 
            float(agent.ang_states[2]),
            float(agent.d_ang_states[0]), 
            float(agent.d_ang_states[1]), 
            float(agent.d_ang_states[2]),
            float(agent.I[0, 0]), 
            float(agent.I[1, 1]), 
            float(agent.I[2, 2]), 
            float(agent.m), 
            float(agent.L_CBF),
            float(agent.l_com),
            float(agent.r_safety)
        ]
        
        # Evaluate f(x) and g(x) at the current state using lambdified functions
        f_val = np.array(self.f_func(*agent_vals)).astype("float").squeeze()
        g_val = np.array(self.g_func(*agent_vals)).astype("float")

        h_list = []
        Lf_list = []
        Lg_list = []

        # min_total_thrust = self.min_thrust_ratio * agent.m * self.acceleration_gravity

        # cos_theta = np.cos(agent.ang_states[1])
        # cos_phi = np.cos(agent.ang_states[0])
        # min_thrust_per_motor = min_total_thrust / (4 * cos_theta * cos_phi) if cos_theta * cos_phi > 0.1 else min_total_thrust / 4


        # --- Evaluate CBF constraints for each obstacle ---
        for i in range(num_obstacles):
            obst_vals = [
                float(obstacle_positions[i, 0]),
                float(obstacle_positions[i, 1]),
                float(obstacle_positions[i, 2]),
                float(obstacle_velocities[i, 0]),
                float(obstacle_velocities[i, 1]),
                float(obstacle_velocities[i, 2])
            ]
        
            # h_i = h(x) & d_h_x_i = dh/dx
            # Use lambdified functions for speed
            h_i = np.array(self.h_obst_func(*(agent_vals + obst_vals))).astype("float").squeeze()
            d_h_x_i = np.array(self.dh_obst_func(*(agent_vals + obst_vals))).astype("float").squeeze()

            h_list.append(h_i)
            Lf_list.append(np.dot(d_h_x_i, f_val))      # Lf_h = (dh/dx) * f(x)
            Lg_list.append(np.dot(d_h_x_i, g_val))      # Lg_h = (dh/dx) * g(x)
            
        # --- Evaluate CBF constraints for passing through hoop ---
        # Use the hoop center as a fake "obstacle" (same pattern as the obstacles above)
        if self._hoop_center is None:
            raise RuntimeError("Hoop center/direction were not provided at initialization, cannot evaluate hoop constraint.")

        obst_vals = [
            float(self._hoop_center[0]),
            float(self._hoop_center[1]),
            float(self._hoop_center[2]),
            0.0, 0.0, 0.0
        ]

        h_i = np.array(self.h_hoop_func(*(agent_vals + obst_vals))).astype("float").squeeze()
        d_h_x_i = np.array(self.dh_hoop_func(*(agent_vals + obst_vals))).astype("float").squeeze()

        h_list.append(h_i)
        Lf_list.append(np.dot(d_h_x_i, f_val))
        Lg_list.append(np.dot(d_h_x_i, g_val))
        
        h_vec = np.array(h_list)
        Lf_vec = np.array(Lf_list)
        Lg_mat = np.array(Lg_list)

        # A_thrust = np.eye(4)
        # l_thrust = np.full((4,), min_thrust_per_motor) - u_ref
        # u_thrust = np.full((4,), np.inf)
        
        # --- Check for violations ---
        # Psi = Lf_h + Lg_h * u_ref + gamma * h
        # If Psi < 0, the constraint is violated by the reference control.
        Psi_vec = (self.gamma * h_vec) + Lf_vec + (Lg_mat @ u_ref.reshape(-1, 1)).squeeze()
        
        violating_indices = np.where(Psi_vec < 0)[0]
        if len(violating_indices) == 0:
            # No violations, no correction needed
            return np.zeros(4), False
        
        # --- Solve the QP using osqp ---
        A_osqp = csc_matrix(Lg_mat)
        l = -Psi_vec
        u = np.full_like(l, np.inf)

        # A_osqp = csc_matrix(np.vstack([Lg_mat, A_thrust]))
        # l = np.hstack([-Psi_vec, l_thrust])
        # u = np.hstack([np.full_like(-Psi_vec, np.inf), u_thrust])

        # Check if problem dimensions have changed
        current_problem_size = A_osqp.shape[0]
        
        if not self.solver_initialized or not hasattr(self, '_last_problem_size') or self._last_problem_size != current_problem_size:
            # First run or problem size changed: setup the solver with new dimensions
            P = csc_matrix(np.eye(4) * 2)
            q = np.zeros(4)
            self.solver = osqp.OSQP()  # Create new solver instance
            self.solver.setup(P=P, q=q, A=A_osqp, l=l, u=u, verbose=False, warm_start=True)
            self.solver_initialized = True
            self._last_problem_size = current_problem_size
        else:
            # Subsequent runs with same dimensions: update the solver with new values
            # We must update A (Lg_mat) and l (-Psi_vec)
            self.solver.update(Ax=A_osqp.data, l=l, u=u)
    
        try:
            # Update the solver with the new constraints
            solution = self.solver.solve()
            
            if solution.info.status != 'solved':
                # Solver failed, fallback to a simple (but unstable) solution
                print("[Warning]: OSQP failed, using fallback.")
                u_safe, flag = self.fallback_solver(Lg_mat, Psi_vec)
                return u_safe, flag

            u_safe = solution.x
            return u_safe, True

        except Exception as e:
            # Solver failed, fallback to a simple (but unstable) solution
            print(f"[Error]: OSQP failed catastrophically: {e}")
            u_safe, flag = self.fallback_solver(Lg_mat, Psi_vec)
            return u_safe, flag


    def fallback_solver(self, Lg_mat, Psi_vec):
        """A simple fallback solver using pinv on the most-violating constraint."""
        most_violating_idx = np.argmin(Psi_vec)
        A_active = Lg_mat[most_violating_idx, :]
        b_active = -Psi_vec[most_violating_idx]
        
        # u_safe = pinv(A) * b
        u_safe = (np.linalg.pinv(A_active.reshape(1, -1)) * b_active).squeeze()
        return u_safe, True
    
    def CBF_value(self,
                  agent:Quadrotor,
                  obstacle_positions:np.ndarray,
                  obstacle_velocities:np.ndarray
                 ):
        """
        Computes the current value of the CBF for a given agent and obstacle.

        Args:
            agent (Quadrotor): The quadrotor object with its current state.
            obstacle_position (np.ndarray): 3D position of the obstacle.
            obstacle_velocity (np.ndarray): 3D velocity of the obstacle.

        Returns:
            float: The current value of the CBF h(x).
        """

        num_obstacles = len(obstacle_positions)
        
        # --- Substitute numerical values into symbolic expressions ---
        # Create a dictionary of all agent state and parameter values
        agent_vals = [
            float(agent.pos_states[0]), 
            float(agent.pos_states[1]), 
            float(agent.pos_states[2]),
            float(agent.d_pos_states[0]), 
            float(agent.d_pos_states[1]), 
            float(agent.d_pos_states[2]),
            float(agent.ang_states[0]), 
            float(agent.ang_states[1]), 
            float(agent.ang_states[2]),
            float(agent.d_ang_states[0]), 
            float(agent.d_ang_states[1]), 
            float(agent.d_ang_states[2]),
            float(agent.I[0, 0]), 
            float(agent.I[1, 1]), 
            float(agent.I[2, 2]), 
            float(agent.m), 
            float(agent.L_CBF),
            float(agent.l_com),
            float(agent.r_safety)
        ]

        h_list = []
        # --- Evaluate CBF constraints for each obstacle ---
        for i in range(num_obstacles):
            obst_vals = [
                float(obstacle_positions[i, 0]),
                float(obstacle_positions[i, 1]),
                float(obstacle_positions[i, 2]),
                float(obstacle_velocities[i, 0]),
                float(obstacle_velocities[i, 1]),
                float(obstacle_velocities[i, 2])
            ]
        
            # h_i = h(x) & d_h_x_i = dh/dx
            # Use lambdified functions for speed
            h_i = np.array(self.h_obst_func(*(agent_vals + obst_vals))).astype("float").squeeze()
            # d_h_x_i = np.array(self.dh_obst_func(*(agent_vals + obst_vals))).astype("float").squeeze()

            h_list.append(h_i)

        return h_list

