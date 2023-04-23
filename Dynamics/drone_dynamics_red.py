" Dynamics for crazylflie 2.1"
import numpy as np
# import pybullet as p
from scipy.spatial.transform import Rotation as R

def drone_dynamics(params,
             forces):
        """Explicit dynamics implementation.

        Based on code written at the Dynamic Systems Lab by James Xu.

        Parameters
        ----------
        rpm : ndarray
            (4)-shaped array of ints containing the RPMs values of the 4 motors.
        
        

        """
        ## params
        KF = 3.1582e-10
        KM = 7.9379e-12
        thr_2_torque = 0.005964552
        L = 0.046#0.0397
        DRONE_MODEL = "C2FP"
        # IXX = 2.3951e-5
        # IYY = 2.3951e-5
        # IZZ = 3.2347e-5
        
        IXX = 4.122766748846959e-05
        IYY = 0.00010926514751224332
        IZZ = 0.00010065805138772742
        
#         IXX = 1.66e-5
#         IYY = 1.66e-5
#         IZZ = 2.93e-5
        
#         IXY = 0.83e-6
#         IYZ = 1.8e-6
#         IXZ = 0.72e-6
        
        
        M = 0.03762730431012316 #0.028#0.0316 #0.0366#0.0366#0.0316
        GRAVITY = 9.81*M



        #### Current state #########################################
        pos = params['pos']
        quat = params['quat']
        rpy = params['rpy']
        vel = params['vel']
        rpy_rates = params['rpy_rates']
        TIMESTEP = params['dt']
        
        r = R.from_euler('xyz', rpy, degrees=False)
        rotation = r.as_matrix()
        # rotation = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        
        ## Compute Inertia Matrix #####################################
        J = np.diag((IXX, IYY, IZZ))
        # J = np.array([[IXX,IXY,IXZ],[IXY, IYY, IYZ],[IXZ, IYZ, IZZ]])
        J_INV = np.linalg.inv(J)
        
        
        #### Compute forces and torques ############################
        thrust = np.array([0, 0, np.sum(forces)])
        thrust_world_frame = np.dot(rotation, thrust)
        force_world_frame = thrust_world_frame - np.array([0, 0, GRAVITY])
        
        acc = force_world_frame / M
        
        r = (acc[0]*np.sin(rpy[2]) - acc[1]*np.cos(rpy[2]) )/(9.81 + acc[2])
    	p = (acc[0]*np.cos(rpy[2]) + acc[1]*np.sin(rpy[2]) )/(9.81+ acc[2])

        return acc, r, p, np.sum(forces)
