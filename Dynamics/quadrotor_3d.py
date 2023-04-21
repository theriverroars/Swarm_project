import numpy as np
import scipy.integrate as integrate
from Dynamics.quaternion import Quaternion
from Dynamics.utils import RPYToRot, RotToQuat, RotToRPY
import Dynamics.params_3d as params

#State = namedtuple('State', 'pos vel rot omega')

class Quadrotor:
    """ Quadrotor class

    state  - 1 dimensional vector but used as 13 x 1. [x, y, z, xd, yd, zd, qw, qx, qy, qz, p, q, r]
             where [qw, qx, qy, qz] is quternion and [p, q, r] are angular velocity [roll_dot, pitch_dot, yaw_dot]
    F      - 1 x 1, thrust output from controller
    M      - 3 x 1, moments output from controller
    params - system parameters struct, arm_length, g, mass, etc.
    """

    def __init__(self, pos, attitude, vel, rates):
        """ pos = [x,y,z] attitude = [roll,pitch,yaw]
            """
        self.state = np.zeros(13)
        roll, pitch, yaw = attitude
        rot    = RPYToRot(roll, pitch, yaw)
        quat   = RotToQuat(rot)
        self.state[0] = pos[0]
        self.state[1] = pos[1]
        self.state[2] = pos[2]
	
        self.state[3] = vel[0]
        self.state[4] = vel[1]
        self.state[5] = vel[2]

        self.state[6] = quat[0]
        self.state[7] = quat[1]
        self.state[8] = quat[2]
        self.state[9] = quat[3]

        self.state[10] = rates[0]
        self.state[11] = rates[1]
        self.state[12] = rates[2]
        self.ode = integrate.ode(self.state_dot).set_integrator('vode',nsteps=500,method='bdf')

    def world_frame(self):
        """ position returns a 3x6 matrix
            where row is [x, y, z] column is m1 m2 m3 m4 origin h
            """
        origin = self.state[0:3]
        rot = self.Rotation().T
        wHb = np.r_[np.c_[rot,origin], np.array([[0, 0, 0, 1]])]
        quadBodyFrame = params.body_frame.T
        quadWorldFrame = wHb.dot(quadBodyFrame)
        world_frame = quadWorldFrame[0:3]
        return world_frame, origin

    def Rotation(self):
        rot = Quaternion(self.state[6:10]).as_rotation_matrix()
        return rot

    def get_state(self):
        return np.array([self.state[0:3],
                     self.state[3:6],
                     RotToRPY(self.Rotation()),
                     self.state[10:13]]).flatten()


    def state_dot(self, t,state,par):
        F,M = par
        #print("Par",type(par))
        #print("F",F)
        #print("M",M)
        x, y, z, xdot, ydot, zdot, qw, qx, qy, qz, p, q, r = state
        quat = np.array([qw,qx,qy,qz])

        bRw = Quaternion(quat).as_rotation_matrix() # world to body rotation matrix
        wRb = bRw.T # orthogonal matrix inverse = transpose
        # acceleration - Newton's second law of motion
        accel = 1.0 / params.mass * (wRb.dot(np.array([[0, 0, F]]).T)
                    - np.array([[0, 0, params.mass * params.g]]).T)
        # angular velocity - using quternion
        # http://www.euclideanspace.com/physics/kinematics/angularvelocity/
        K_quat = 2.0; # this enforces the magnitude 1 constraint for the quaternion
        quaterror = 1.0 - (qw**2 + qx**2 + qy**2 + qz**2)
        qdot = (-1.0/2) * np.array([[0, -p, -q, -r],
                                    [p,  0, -r,  q],
                                    [q,  r,  0, -p],
                                    [r, -q,  p,  0]]).dot(quat) + K_quat * quaterror * quat;

        # angular acceleration - Euler's equation of motion
        # https://en.wikipedia.org/wiki/Euler%27s_equations_(rigid_body_dynamics)
        omega = np.array([p,q,r])
        pqrdot = params.invI.dot( M.flatten() - np.cross(omega, params.I.dot(omega)) )
        #print("Angular acc using moment",pqrdot)
        state_dot = np.zeros(13)
        state_dot[0]  = xdot
        state_dot[1]  = ydot
        state_dot[2]  = zdot
        state_dot[3]  = accel[0]
        state_dot[4]  = accel[1]
        state_dot[5]  = accel[2]
        state_dot[6]  = qdot[0]
        state_dot[7]  = qdot[1]
        state_dot[8]  = qdot[2]
        state_dot[9]  = qdot[3]
        state_dot[10] = pqrdot[0]
        state_dot[11] = pqrdot[1]
        state_dot[12] = pqrdot[2]

        return state_dot

    def update(self, dt, F, M):
        #Mt = M[2]
        #prop_thrusts = params.invA.dot(np.r_[np.array([[F]]),M])
        #prop_thrusts_clamped = np.maximum(np.minimum(prop_thrusts, params.maxF/4), params.minF/4)
        ## F = np.sum(prop_thrusts_clamped)
        #M = params.A[1:].dot(prop_thrusts_clamped)
        #M = np.r_[M[:2],[Mt]]
        ## print(F)
        #F = np.clip(F, 0, 0.6)
        #M = np.clip(M, -0.05, 0.05)
        self.ode.set_initial_value(self.state,0).set_f_params([F,M])
        self.state = self.ode.integrate(self.ode.t + dt)
