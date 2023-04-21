import numpy as np
print("Crazyflie params")
mass = 0.0316#0.18 # kg
g = 9.81 # m/s^2
arm_length = 0.046#0.086 # meter
height = 0.029#0.05
'''
I = np.array([(0.00025, 0, 2.55e-6),
              (0, 0.000232, 0),
              (2.55e-6, 0, 0.0003738)]);
              
              
''' 

IXX = 1.66e-5
IYY = 1.66e-5
IZZ = 2.93e-5

IXY = 0.83e-6
IYZ = 1.8e-6
IXZ = 0.72e-6
# I = np.array([(0.000023263, 0, 0),
# 			(0, 0.000023263, 0),
# 			(0, 0, 0.00004232)])

I = np.array([[IXX,IXY,IXZ],[IXY, IYY, IYZ],[IXZ, IYZ, IZZ]])

invI = np.linalg.inv(I)

minF = 0.0
maxF = 2.0 * mass * g

km = 7.9379e-12
kf = 3.1582e-10
r = km / kf

L = arm_length
H = height
#  [ F  ]         [ F1 ]
#  | M1 |  = A *  | F2 |
#  | M2 |         | F3 |
#  [ M3 ]         [ F4 ]
A = np.array([[ 1,  1,  1,  1],
              [ 0,  L,  0, -L],
              [-L,  0,  L,  0],
              [ r, -r,  r, -r]])

invA = np.linalg.inv(A)

body_frame = np.array([(L, 0, 0, 1),
                       (0, L, 0, 1),
                       (-L, 0, 0, 1),
                       (0, -L, 0, 1),
                       (0, 0, 0, 1),
                       (0, 0, H, 1)])

B = np.array([[0, L,0, -L],
              [-L, 0, L,0]])
