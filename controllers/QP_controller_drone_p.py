import sympy
import numpy as np
from sympy import symbols, diff, Matrix, simplify
from math import cos, sin, tan, pi, sqrt
from sympy import *
import time

# custom imports
from controllers.QP_controller import QP_Controller
from bots.drone import Drone


def norm(x, y, z):
    return sqrt(x**2 + y**2 + z**2)

class QP_Controller_Drone(QP_Controller):
    def __init__(self, gamma:float, obs_radius=0.5):
        """
        Constructor creates QP_Controller object for Multi Agent system where 
        each agent is a Drone model bot.        

        Parameters
        ----------
        gamma : float
            class k function is taken as simple function f(x) = \gamma x. The 
            valueo f gamma is assumed to 1 for all the agents for time being.
        no_of_agents : int
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.gamma = gamma
        self.u_ref = None
        self.u_star = None
        self.G = 9.81
        self.kf = 3.16e-10
        self.obs_r = obs_radius
        
    def set_reference_control(self, u_ref:np.ndarray):
        """
        sets the reference controller

        Parameters
        ----------
        u_ref : numpy.ndarray
            numpy array of size (n, 2) containing the reference controls.

        Returns
        -------
        None.

        """
        self.u_ref = u_ref
        self.u_star = np.copy(u_ref)
    
    def get_optimal_control(self):
        self.u_star = self.u_star / self.kf
        # print(self.u_star[1][0])
        if self.u_star[1][0]<0:
             self.u_star[1][0] =0
        if self.u_star[2][0]<0:
             self.u_star[2][0] =0
        if self.u_star[3][0]<0:
             self.u_star[3][0] =0
        if self.u_star[0][0]<0:
             self.u_star[0][0] =0
            
        propellers_1_rpm = sqrt(self.u_star[1][0])
        propellers_3_rpm = sqrt(self.u_star[3][0])
        propellers_0_rpm = sqrt(self.u_star[0][0])
        propellers_2_rpm = sqrt(self.u_star[2][0])

        return np.array([propellers_0_rpm, propellers_1_rpm,
                         propellers_2_rpm, propellers_3_rpm], dtype='float64')
    
    def get_reference_control(self):
        return self.u_ref
    
    def setup_QP(self, bot, c, c_d):
        """
        the function takes bot list and creates symbolic varaibles associated 
        with each bot required for computation in QP. The functions also 
        precomputes the required symbolic expressions for QP such as L_f, L_g,
        etc...
        Parameters
        ----------
        bot_list : list
            list of Drone bot objects 
            
        Returns
        -------
        None.

        """
        c_x, c_y, c_z = c
        c_x_d, c_y_d, c_z_d = c_d
        # Create placholders for symbolic expressions
        self.f = [0] # f Matrix in control system
        self.g = [0] # g Matrix in control system
        self.h = [[0]] # C3BF Matrix
        
        # Create placholder for terms in QP
        self.Psi = []
        self.Lg_h_T = [[]]
        self.Lg_h = [[]]

        bot.r = bot.encompassing_radius + self.obs_r
        
        # create state and parameter symbolic varaibles for each bot
        
        # symbols_string = 'x y z x_d y_d z_d phi theta psi w_1 w_2 w_3 L Ixx Iyy Izz m l r'
        # bot.x, bot.y, bot.z, bot.x_d, bot.y_d, bot.z_d, bot.phi, bot.theta, bot.psi, bot.w_1, bot.w_2, bot.w_3, bot.L, bot.Ixx, bot.Iyy, bot.Izz, bot.m, bot.l, bot.r =  symbols(symbols_string)
        self.f = Matrix([bot.x_d, 
                        bot.y_d,
                        bot.z_d,
                        0,
                        0, 
                        - self.G,
                        bot.w_1 + bot.w_2*sin(bot.phi)*tan(bot.theta) + bot.w_3*cos(bot.phi)*tan(bot.theta),
                        bot.w_2*cos(bot.phi) - bot.w_3*sin(bot.phi),
                        (bot.w_2*sin(bot.phi) + bot.w_3*cos(bot.phi))/cos(bot.theta),
                        (bot.Iyy - bot.Izz)*bot.w_2*bot.w_3/bot.Ixx,
                        (bot.Izz - bot.Ixx)*bot.w_1*bot.w_3/bot.Iyy,
                        (bot.Ixx - bot.Iyy)*bot.w_1*bot.w_2/bot.Izz])

        p = (cos(bot.psi)*sin(bot.theta)*cos(bot.phi) + sin(bot.psi)*sin(bot.phi))/bot.m
        q = (sin(bot.psi)*sin(bot.theta)*cos(bot.phi) - cos(bot.psi)*sin(bot.phi))/bot.m
        r = (cos(bot.theta)*cos(bot.phi))/bot.m
        self.g = Matrix([[0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [p, p, p, p],
                        [q, q, q, q],
                        [r, r, r, r],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [-bot.L*np.sqrt(0.5)/bot.Ixx, -bot.L*np.sqrt(0.5)/bot.Ixx, bot.L*np.sqrt(0.5)/bot.Ixx, bot.L*np.sqrt(0.5)/bot.Ixx],
                        [-bot.L*np.sqrt(0.5)/bot.Iyy, bot.L*np.sqrt(0.5)/bot.Iyy, bot.L*np.sqrt(0.5)/bot.Iyy, -bot.L*np.sqrt(0.5)/bot.Iyy],
                        [0, 0, 0, 0]])
            
        # for CBF h
        # Bot objects 
        r_x = (cos(bot.psi)*sin(bot.theta)*cos(bot.phi) + sin(bot.psi)*sin(bot.phi))
        r_y = (sin(bot.psi)*sin(bot.theta)*cos(bot.phi) - cos(bot.psi)*sin(bot.phi))
        r_z = (cos(bot.theta)*cos(bot.phi))

        # Relative position terms
        p_rel_x = c_x - (bot.x + bot.l*r_x)
        p_rel_y = c_y - (bot.y + bot.l*r_y)
        p_rel_z = c_z - (bot.z + bot.l*r_z)
        
        # Relative velocity terms
        v_rel_x = c_x_d - (bot.x_d + bot.l*(-bot.w_3*r_y + bot.w_2*r_z))
        v_rel_y = c_y_d - (bot.y_d + bot.l*(-bot.w_1*r_z + bot.w_3*r_x))
        v_rel_z = c_z_d - (bot.z_d + bot.l*(-bot.w_2*r_x + bot.w_1*r_y))
        
        # # C3BF Candidate
        # self.h = p_rel_x*v_rel_x + p_rel_y*v_rel_y + p_rel_z*v_rel_z \
        #     + norm(v_rel_x, v_rel_y, v_rel_z)*sqrt(norm(p_rel_x, p_rel_y, p_rel_z)**2 - bot.r**2)

        # Classical CBF
        # self.h = norm(c_x - bot.x, c_y - bot.y, c_z - bot.z)**2 -1
        
        r_x_by_phi = (-cos(bot.psi)*sin(bot.theta)*sin(bot.phi) + sin(bot.psi)*cos(bot.phi))
        r_y_by_phi = (-sin(bot.psi)*sin(bot.theta)*sin(bot.phi) - cos(bot.psi)*cos(bot.phi))
        r_z_by_phi = (-cos(bot.theta)*sin(bot.phi))
        
        r_x_by_theta = (cos(bot.psi)*cos(bot.theta)*cos(bot.phi))
        r_y_by_theta = (sin(bot.psi)*cos(bot.theta)*cos(bot.phi))
        r_z_by_theta = (-sin(bot.theta)*cos(bot.phi))
        
        r_x_by_psi = (-sin(bot.psi)*sin(bot.theta)*cos(bot.phi) + cos(bot.psi)*sin(bot.phi))
        r_y_by_psi = (cos(bot.psi)*sin(bot.theta)*cos(bot.phi) + sin(bot.psi)*sin(bot.phi))
        r_z_by_psi = 0

        p_rel_x_by_phi = - bot.l*r_x_by_phi
        p_rel_y_by_phi = - bot.l*r_y_by_phi
        p_rel_z_by_phi = - bot.l*r_z_by_phi

        p_rel_x_by_theta = - bot.l*r_x_by_theta
        p_rel_y_by_theta = - bot.l*r_y_by_theta
        p_rel_z_by_theta = - bot.l*r_z_by_theta

        p_rel_x_by_psi = - bot.l*r_x_by_psi
        p_rel_y_by_psi = - bot.l*r_y_by_psi
        p_rel_z_by_psi = - bot.l*r_z_by_psi

        v_rel_x_by_phi = - bot.l*(-bot.w_3*r_y_by_phi + bot.w_2*r_z_by_phi)
        v_rel_y_by_phi = - bot.l*(-bot.w_1*r_z_by_phi + bot.w_3*r_x_by_phi)
        v_rel_z_by_phi = - bot.l*(-bot.w_2*r_x_by_phi + bot.w_1*r_y_by_phi)

        v_rel_x_by_theta = - bot.l*(-bot.w_3*r_y_by_theta + bot.w_2*r_z_by_theta)
        v_rel_y_by_theta = - bot.l*(-bot.w_1*r_z_by_theta + bot.w_3*r_x_by_theta)
        v_rel_z_by_theta = - bot.l*(-bot.w_2*r_x_by_theta + bot.w_1*r_y_by_theta)

        v_rel_x_by_psi = - bot.l*(-bot.w_3*r_y_by_psi + bot.w_2*r_z_by_psi)
        v_rel_y_by_psi = - bot.l*(-bot.w_1*r_z_by_psi + bot.w_3*r_x_by_psi)
        v_rel_z_by_psi = - bot.l*(-bot.w_2*r_x_by_psi + bot.w_1*r_y_by_psi)

        # # C3BF Candidate
        # self.h = p_rel_x*v_rel_x + p_rel_y*v_rel_y + p_rel_z*v_rel_z + norm(v_rel_x, v_rel_y, v_rel_z)*sqrt(norm(p_rel_x, p_rel_y, p_rel_z)**2 - bot.r**2)
            
        # rho_h_by_rho_x = -v_rel_x + (-p_rel_x)*norm(v_rel_x, v_rel_y, v_rel_z)/sqrt(norm(p_rel_x, p_rel_y, p_rel_z)**2 - bot.r**2)
        # rho_h_by_rho_y = -v_rel_y + (-p_rel_y)*norm(v_rel_x, v_rel_y, v_rel_z)/sqrt(norm(p_rel_x, p_rel_y, p_rel_z)**2 - bot.r**2)
        # rho_h_by_rho_z = -v_rel_z + (-p_rel_z)*norm(v_rel_x, v_rel_y, v_rel_z)/sqrt(norm(p_rel_x, p_rel_y, p_rel_z)**2 - bot.r**2)
        # rho_h_by_rho_x_d = -p_rel_x + (-v_rel_x)*sqrt(norm(p_rel_x, p_rel_y, p_rel_z)**2 - bot.r**2)/norm(v_rel_x, v_rel_y, v_rel_z)
        # rho_h_by_rho_y_d = -p_rel_y + (-v_rel_y)*sqrt(norm(p_rel_x, p_rel_y, p_rel_z)**2 - bot.r**2)/norm(v_rel_x, v_rel_y, v_rel_z)
        # rho_h_by_rho_z_d = -p_rel_z + (-v_rel_z)*sqrt(norm(p_rel_x, p_rel_y, p_rel_z)**2 - bot.r**2)/norm(v_rel_x, v_rel_y, v_rel_z)
        # rho_h_by_rho_phi   = (p_rel_x_by_phi*v_rel_x + p_rel_y_by_phi*v_rel_y + p_rel_z_by_phi*v_rel_z) + (p_rel_x*v_rel_x_by_phi + p_rel_y*v_rel_y_by_phi + p_rel_z*v_rel_z_by_phi) + (v_rel_x*v_rel_x_by_phi + v_rel_y*v_rel_y_by_phi + v_rel_z*v_rel_z_by_phi)*sqrt(norm(p_rel_x, p_rel_y, p_rel_z)**2 - bot.r**2)/norm(v_rel_x, v_rel_y, v_rel_z) + (p_rel_x*p_rel_x_by_phi + p_rel_y*p_rel_y_by_phi + p_rel_z*p_rel_z_by_phi)*norm(v_rel_x, v_rel_y, v_rel_z)/ sqrt(norm(p_rel_x, p_rel_y, p_rel_z)**2 - bot.r**2)
        # rho_h_by_rho_theta = (p_rel_x_by_theta*v_rel_x + p_rel_y_by_theta*v_rel_y + p_rel_z_by_theta*v_rel_z) + (p_rel_x*v_rel_x_by_theta + p_rel_y*v_rel_y_by_theta + p_rel_z*v_rel_z_by_theta) + (v_rel_x*v_rel_x_by_theta + v_rel_y*v_rel_y_by_theta + v_rel_z*v_rel_z_by_theta)*sqrt(norm(p_rel_x, p_rel_y, p_rel_z)**2 - bot.r**2)/norm(v_rel_x, v_rel_y, v_rel_z) + (p_rel_x*p_rel_x_by_theta + p_rel_y*p_rel_y_by_theta + p_rel_z*p_rel_z_by_theta)*norm(v_rel_x, v_rel_y, v_rel_z)/ sqrt(norm(p_rel_x, p_rel_y, p_rel_z)**2 - bot.r**2)
        # rho_h_by_rho_psi   = (p_rel_x_by_psi*v_rel_x + p_rel_y_by_psi*v_rel_y + p_rel_z_by_psi*v_rel_z) + (p_rel_x*v_rel_x_by_psi + p_rel_y*v_rel_y_by_psi + p_rel_z*v_rel_z_by_psi) + (v_rel_x*v_rel_x_by_psi + v_rel_y*v_rel_y_by_psi + v_rel_z*v_rel_z_by_psi)*sqrt(norm(p_rel_x, p_rel_y, p_rel_z)**2 - bot.r**2)/norm(v_rel_x, v_rel_y, v_rel_z) + (p_rel_x*p_rel_x_by_psi + p_rel_y*p_rel_y_by_psi + p_rel_z*p_rel_z_by_psi)*norm(v_rel_x, v_rel_y, v_rel_z)/ sqrt(norm(p_rel_x, p_rel_y, p_rel_z)**2 - bot.r**2)
        # rho_h_by_rho_w_1 = bot.l*(r_z*p_rel_y - r_y*p_rel_z) + bot.l*(r_z*v_rel_y - r_y*v_rel_z)*sqrt(norm(p_rel_x, p_rel_y, p_rel_z)**2 - bot.r**2)/norm(v_rel_x, v_rel_y, v_rel_z)
        # rho_h_by_rho_w_2 = bot.l*(r_x*p_rel_z - r_z*p_rel_x) + bot.l*(r_x*v_rel_z - r_z*v_rel_x)*sqrt(norm(p_rel_x, p_rel_y, p_rel_z)**2 - bot.r**2)/norm(v_rel_x, v_rel_y, v_rel_z)
        # rho_h_by_rho_w_3 = bot.l*(r_y*p_rel_x - r_x*p_rel_y) + bot.l*(r_y*v_rel_x - r_x*v_rel_y)*sqrt(norm(p_rel_x, p_rel_y, p_rel_z)**2 - bot.r**2)/norm(v_rel_x, v_rel_y, v_rel_z)

        # Projection C3BF Candidate
        self.h = p_rel_x*v_rel_x + p_rel_y*v_rel_y + norm(v_rel_x, v_rel_y, 0)*sqrt(norm(p_rel_x, p_rel_y, 0)**2 - bot.r**2)
            
        rho_h_by_rho_x = -v_rel_x + (-p_rel_x)*norm(v_rel_x, v_rel_y, 0)/sqrt(norm(p_rel_x, p_rel_y, 0)**2 - bot.r**2)
        rho_h_by_rho_y = -v_rel_y + (-p_rel_y)*norm(v_rel_x, v_rel_y, 0)/sqrt(norm(p_rel_x, p_rel_y, 0)**2 - bot.r**2)
        rho_h_by_rho_z = 0
        rho_h_by_rho_x_d = -p_rel_x + (-v_rel_x)*sqrt(norm(p_rel_x, p_rel_y, 0)**2 - bot.r**2)/norm(v_rel_x, v_rel_y, 0)
        rho_h_by_rho_y_d = -p_rel_y + (-v_rel_y)*sqrt(norm(p_rel_x, p_rel_y, 0)**2 - bot.r**2)/norm(v_rel_x, v_rel_y, 0)
        rho_h_by_rho_z_d = 0
        rho_h_by_rho_phi   = (p_rel_x_by_phi*v_rel_x + p_rel_y_by_phi*v_rel_y + 0*0) + (p_rel_x*v_rel_x_by_phi + p_rel_y*v_rel_y_by_phi + 0*0) + (v_rel_x*v_rel_x_by_phi + v_rel_y*v_rel_y_by_phi + 0*0)*sqrt(norm(p_rel_x, p_rel_y, 0)**2 - bot.r**2)/norm(v_rel_x, v_rel_y, 0) + (p_rel_x*p_rel_x_by_phi + p_rel_y*p_rel_y_by_phi + 0*0)*norm(v_rel_x, v_rel_y, 0)/ sqrt(norm(p_rel_x, p_rel_y, 0)**2 - bot.r**2)
        rho_h_by_rho_theta = (p_rel_x_by_theta*v_rel_x + p_rel_y_by_theta*v_rel_y) + (p_rel_x*v_rel_x_by_theta + p_rel_y*v_rel_y_by_theta) + (v_rel_x*v_rel_x_by_theta + v_rel_y*v_rel_y_by_theta)*sqrt(norm(p_rel_x, p_rel_y, 0)**2 - bot.r**2)/norm(v_rel_x, v_rel_y, 0) + (p_rel_x*p_rel_x_by_theta + p_rel_y*p_rel_y_by_theta + 0*0)*norm(v_rel_x, v_rel_y, 0)/ sqrt(norm(p_rel_x, p_rel_y, 0)**2 - bot.r**2)
        rho_h_by_rho_psi   = (p_rel_x_by_psi*v_rel_x + p_rel_y_by_psi*v_rel_y + 0*0) + (p_rel_x*v_rel_x_by_psi + p_rel_y*v_rel_y_by_psi + 0*0) + (v_rel_x*v_rel_x_by_psi + v_rel_y*v_rel_y_by_psi + 0*0)*sqrt(norm(p_rel_x, p_rel_y, 0)**2 - bot.r**2)/norm(v_rel_x, v_rel_y, 0) + (p_rel_x*p_rel_x_by_psi + p_rel_y*p_rel_y_by_psi + 0*0)*norm(v_rel_x, v_rel_y, 0)/ sqrt(norm(p_rel_x, p_rel_y, 0)**2 - bot.r**2)
        rho_h_by_rho_w_1 = bot.l*(r_z*p_rel_y - r_y*0) + bot.l*(r_z*v_rel_y - r_y*0)*sqrt(norm(p_rel_x, p_rel_y, 0)**2 - bot.r**2)/norm(v_rel_x, v_rel_y, 0)
        rho_h_by_rho_w_2 = bot.l*(r_x*0 - r_z*p_rel_x) + bot.l*(r_x*0 - r_z*v_rel_x)*sqrt(norm(p_rel_x, p_rel_y, 0)**2 - bot.r**2)/norm(v_rel_x, v_rel_y, 0)
        rho_h_by_rho_w_3 = bot.l*(r_y*p_rel_x - r_x*p_rel_y) + bot.l*(r_y*v_rel_x - r_x*v_rel_y)*sqrt(norm(p_rel_x, p_rel_y, 0)**2 - bot.r**2)/norm(v_rel_x, v_rel_y, 0)
        
        Delta_h_wrt_bot = Matrix([[rho_h_by_rho_x, 
                                    rho_h_by_rho_y, 
                                    rho_h_by_rho_z,
                                    rho_h_by_rho_x_d, 
                                    rho_h_by_rho_y_d, 
                                    rho_h_by_rho_z_d,
                                    rho_h_by_rho_phi,
                                    rho_h_by_rho_theta, 
                                    rho_h_by_rho_psi,
                                    rho_h_by_rho_w_1, 
                                    rho_h_by_rho_w_2,
                                    rho_h_by_rho_w_3]])
        
        self.Lg_h_T = (Delta_h_wrt_bot*self.g).transpose()
        self.Lg_h = Delta_h_wrt_bot*self.g
        self.u_ref = self.u_ref.reshape((4,1))
        Lg_h_u = self.Lg_h * self.u_ref
        Lf_h = Delta_h_wrt_bot*self.f
        self.Psi = self.gamma*self.h
        self.Psi += Lf_h[0] + Lg_h_u[0]
                 
    def solve_QP(self, bot):
        """
        Solving Quadratic Program to set the optimal controls. This functions
        substitutes the values in symbolic expression and evalutes closed form
        solution to QP, modifies the reference control and sets the optimal 
        control

        Parameters
        ----------
        bot_list : list
            list of Drone bot objects .

        Returns
        -------
        TYPE: tuple of 2 numpy arrays
            first numpy array in the tuple returns of state of CBF if they are
            active or inactive 1 denotes active CBF and 0 denotes inactive CBF.
            second numpy array in the tupes returns the value of CBF since in 
            C3BF the value of the function is directly proportional to how 
            unsafe system is.
        """
        
        # build value substitution list        
        self.h = np.array(re(self.h))
        self.Psi = np.array(re(self.Psi))
        self.Lg_h_T = np.array(re(self.Lg_h_T))
        self.Lg_h = np.array(re(self.Lg_h))

        if self.Psi<0:
            self.u_safe = - np.matmul(self.Lg_h_T, np.linalg.inv(np.matmul(self.Lg_h,self.Lg_h_T).astype('float64'))).dot(self.Psi)
            # print(self.u_safe)
        else:
            self.u_safe = 0
        self.u_star = self.u_ref + self.u_safe

        if self.h< 0:
            print("Psi", self.Psi)
            print("h", self.h)
            print("u_safe =", self.u_safe, "u_ref =", self.u_ref)

        return self.h

        