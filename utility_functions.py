# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 14:50:16 2022

@author: My PC
"""
import math
import numpy as np
from scipy.spatial.transform import Rotation as R


def point_wrt_circle(x, y, c_x, c_y, r):
    return pow((x - c_x), 2) + pow((y - c_y), 2) - pow(r,2)

#def norm(x, y):
    #return math.sqrt(pow(x,2) + pow(y,2))

def norm(x, y, z):
    return math.sqrt(x**2 + y**2 + z**2)


def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def matrix_multiplication(A, B):
    result = [[sum(a * b for a, b in zip(A_row, B_col))
                        for B_col in zip(*B)]
                                for A_row in A]
    return result
    
## decompress quaternion data from stateZ
def decompressquat(comp):
    comp = np.uint32(comp)
    q = np.array([0., 0., 0., 0.], dtype=np.float32)
    
    M_SQRT1_2 = float(1/np.sqrt(2))
    mask = (1 << 9) - 1
    i_largest = comp >> 30
    sum_squares = 0.0
    
    for i in range(3, -1, -1):
        if i != i_largest:
            mag = comp & mask
            negbit = (comp >> 9) & 0x1
            comp = comp >> 10
            q[i] = M_SQRT1_2 * (float(mag))/mask
            if negbit == 1:
                q[i] = -q[i]
     
            sum_squares += q[i] * q[i]
            
    q[i_largest] = np.sqrt(1.0 - sum_squares)
    return q
    
    
## convert quaternion to euler

def comp_quat_to_euler(comp_quat):
    q = decompressquat(comp_quat)
    r = R.from_quat([q[0], q[1], q[2],q[3]])
    s= r.as_euler('xyz', degrees=False)
    return s[0],s[1],s[2]

## convert thrust
def convert_thrust_2_pwm(thrust):
    # Refer http://mikehamer.info/assets/papers/Crazyflie%20Modelling.pdf
    # thrust = = 2.130295e-11*cmd^2 + 1.032633e-6*cmd + 5.484560e-4
    a = 2.130295e-11
    b = 1.032633e-6
    c = 5.484560e-4
    pwm = np.roots([a, b, c - thrust])
    result = 0
    for val in pwm:
        if not np.iscomplex(val) and val > 0:
            result = val
    return int(result)
