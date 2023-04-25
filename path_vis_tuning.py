import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utility_functions import comp_quat_to_euler

path_traced = '/home/rajpal/github_dat/Drones-C3BF/data_log/dynamics_dataset Mon Apr 24 23-19-23 2023.csv'

df_traced = pd.read_csv(path_traced)

fig = plt.figure()
ax = plt.axes(projection='3d')
# ax.plot3D(x, y, z,label = 'desired')
ax.plot3D(df_traced['stateZ_x']/1000, df_traced['stateZ_y']/1000, df_traced['stateZ_z']/1000,label = 'traced')
# ax.scatter3D(x[0], y[0], z[0],label = 'pos_start')
# ax.scatter3D(x[-1], y[-1], z[-1],label = 'pos_end')
ax.axis('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

roll = []
pitch = []
yaw = []
for i in range(df_traced['stateZ_timestamp'].shape[0]):
    rpy = comp_quat_to_euler((df_traced['stateZ_quat'][i])) # in radians
    roll.append(rpy[0])
    pitch.append(rpy[1])
    yaw.append(rpy[2])

plt.figure()
plt.title('x vs time')
plt.plot((df_traced['stateZ_timestamp']- df_traced['stateZ_timestamp'][0])/1000,df_traced['stateZ_x'],label = 'traced')
# plt.plot(df_traced['des_timestamp'][:-400] - df_traced['des_timestamp'][0],df_traced['s_roll'][:-400],label = 'desired')
# plt.plot((df_traced['stateZ_timestamp']-df_traced['stateZ_timestamp'][0])/1000,roll_des,label = 'traced')
plt.legend()
plt.show()

plt.figure()
plt.title('y vs. time')
plt.plot((df_traced['stateZ_timestamp']- df_traced['stateZ_timestamp'][0])/1000,df_traced['stateZ_y'],label = 'traced')
# plt.plot(df_traced['des_timestamp'] - df_traced['des_timestamp'][0],df_traced['des_roll'],label = 'desired')
# plt.plot((df_traced['stateZ_timestamp']-df_traced['stateZ_timestamp'][0])/1000,pitch_des,label = 'traced')
plt.legend()
plt.show()

plt.figure()
plt.title('z vs. time')
plt.plot((df_traced['stateZ_timestamp']- df_traced['stateZ_timestamp'][0])/1000,df_traced['stateZ_z'],label = 'traced')
# plt.plot(df_traced['des_timestamp']- df_traced['des_timestamp'][0],df_traced['des_roll'],label = 'desired')
# plt.plot((df_traced['stateZ_timestamp']-df_traced['stateZ_timestamp'][0])/1000,pitch_des,label = 'traced')
plt.legend()
plt.show()






plt.figure()
plt.title('roll vs time')
plt.plot((df_traced['stateZ_timestamp']- df_traced['stateZ_timestamp'][0])/1000,np.degrees(roll),label = 'traced')
# plt.plot(df_traced['des_timestamp'][:-400] - df_traced['des_timestamp'][0],df_traced['s_roll'][:-400],label = 'desired')
# plt.plot((df_traced['stateZ_timestamp']-df_traced['stateZ_timestamp'][0])/1000,roll_des,label = 'traced')
plt.legend()
plt.show()

plt.figure()
plt.title('pitch vs. time')
plt.plot((df_traced['stateZ_timestamp']- df_traced['stateZ_timestamp'][0])/1000,np.degrees(pitch),label = 'traced')
# plt.plot(df_traced['des_timestamp'] - df_traced['des_timestamp'][0],df_traced['des_roll'],label = 'desired')
# plt.plot((df_traced['stateZ_timestamp']-df_traced['stateZ_timestamp'][0])/1000,pitch_des,label = 'traced')
plt.legend()
plt.show()

plt.figure()
plt.title('yaw vs. time')
plt.plot((df_traced['stateZ_timestamp']- df_traced['stateZ_timestamp'][0])/1000,np.degrees(yaw),label = 'traced')
# plt.plot(df_traced['des_timestamp']- df_traced['des_timestamp'][0],df_traced['des_roll'],label = 'desired')
# plt.plot((df_traced['stateZ_timestamp']-df_traced['stateZ_timestamp'][0])/1000,pitch_des,label = 'traced')
plt.legend()
plt.show()

plt.figure()
plt.title('cmd_thrust vs. time')
plt.plot((df_traced['stateZ_timestamp']- df_traced['stateZ_timestamp'][0])/1000,df_traced['cmd_thrust'],label = 'traced')
# plt.plot(df_traced['des_timestamp']- df_traced['des_timestamp'][0],df_traced['des_roll'],label = 'desired')
# plt.plot((df_traced['stateZ_timestamp']-df_traced['stateZ_timestamp'][0])/1000,pitch_des,label = 'traced')
plt.legend()
plt.show()