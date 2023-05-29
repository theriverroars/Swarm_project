import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import math

## path for load

path_des = '/home/rajpal/github_dat/Drones-C3BF/results/save-flight--01.03.2023_13.24.04/'
path_traced = '/home/rajpal/github_dat/Drones-C3BF/data_log/tuning/dynamics_dataset Mon May 29 23-19-06 2023.csv'

df_traced = pd.read_csv(path_traced)

x = np.loadtxt(path_des + 'x0.csv',
                 delimiter=",", dtype=float)

 
t = x[:,0]
x = x[:,1]

y = np.loadtxt(path_des + 'y0.csv',
                 delimiter=",", dtype=float)
y = y[:,1]

z = np.loadtxt(path_des + 'z0.csv',
                 delimiter=",", dtype=float)
z = z[:,1]

T = np.loadtxt(path_des + 'x0.csv',
                 delimiter=",", dtype=float)
T = T[:,1]

roll = np.loadtxt(path_des + 'r0.csv',
                 delimiter=",", dtype=float)
roll = roll[:,1]

pitch = np.loadtxt(path_des + 'p0.csv',
                 delimiter=",", dtype=float)
pitch = pitch[:,1]

yaw = np.loadtxt(path_des + 'ya0.csv',
                 delimiter=",", dtype=float)
yaw = yaw[:,1]


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

from scipy.spatial.transform import Rotation as R

qx = []
qy = []
qz = []
qw = []
roll_des =[]
pitch_des=[]
yaw_des=[]
for i in range(df_traced['stateZ_quat'].shape[0]):

    q = decompressquat(df_traced['stateZ_quat'][i])
    qx.append(q[0])
    qy.append(q[1])
    qz.append(q[2])
    qw.append(q[3])
    r = R.from_quat([q[0], q[1], q[2], q[3]])
    s= r.as_euler('xyz', degrees=True)
    roll_des.append(s[0])
    pitch_des.append(s[1])
    yaw_des.append(s[2])
    

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(x, y, z,label = 'desired')
ax.plot3D(df_traced['stateZ_x']/1000, df_traced['stateZ_y']/1000, df_traced['stateZ_z']/1000,label = 'traced')
ax.scatter3D(x[0], y[0], z[0],label = 'pos_start')
ax.scatter3D(x[-1], y[-1], z[-1],label = 'pos_end')
ax.axis('equal')
plt.legend()
plt.show()

plt.figure()
plt.plot(t,np.degrees(roll),label = 'desired')
plt.plot((df_traced['stateZ_timestamp']-df_traced['stateZ_timestamp'][0])/1000,df_traced['stateZ_z']/1000,label = 'traced')
plt.ylabel('z(mm)')
plt.legend()
plt.show()

plt.figure()
plt.plot(t,np.degrees(roll),label = 'desired')
plt.plot((df_traced['stateZ_timestamp']-df_traced['stateZ_timestamp'][0])/1000,roll_des,label = 'traced')
plt.ylabel('roll')
plt.legend()
plt.show()

plt.figure()
# plt.plot(t,np.degrees(pitch),label = 'desired')
plt.plot((df_traced['stateZ_timestamp']-df_traced['stateZ_timestamp'][0])/1000,pitch_des,label = 'traced')
plt.legend()
plt.show()