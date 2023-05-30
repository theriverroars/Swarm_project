import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import math
from paths import path_pars

## path for load

path = '/home/rajpal/github_dat/Drones-C3BF/data_log/tuning/dynamics_dataset Tue May 30 22-52-03 2023.csv'

df = pd.read_csv(path)


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(df['stateZ_x']/1000, df['stateZ_y']/1000, df['stateZ_z']/1000,label = 'traced')
ax.plot3D(df['des_x'], df['des_y'], df['des_z'],label = 'des')
ax.axis('equal')
plt.legend()
plt.show()


plt.figure()
plt.title('x vs time')
plt.plot((df['stateZ_timestamp']-df['stateZ_timestamp'][0])/1000,df['stateZ_x']/1000, label = 'traced')
plt.plot(df['des_timestamp']-df['des_timestamp'][0],df['des_x'], label = 'desired')
plt.legend()
plt.show()

plt.figure()
plt.title('y vs time')
plt.plot((df['stateZ_timestamp']-df['stateZ_timestamp'][0])/1000,df['stateZ_y']/1000, label = 'traced')
plt.plot(df['des_timestamp']-df['des_timestamp'][0],df['des_y'], label = 'desired')
plt.legend()
plt.show()

plt.figure()
plt.title('z vs time')
plt.plot((df['stateZ_timestamp']-df['stateZ_timestamp'][0])/1000,df['stateZ_z']/1000, label = 'traced')
plt.plot(df['des_timestamp']-df['des_timestamp'][0],df['des_z'], label = 'desired')
plt.legend()
plt.show()

plt.figure()
plt.title('vx vs time')
plt.plot((df['stateZ_timestamp']-df['stateZ_timestamp'][0])/1000,df['stateZ_vx']/1000, label = 'traced')
plt.plot(df['des_timestamp']-df['des_timestamp'][0],df['des_vx'], label = 'desired')
plt.legend()
plt.show()

plt.figure()
plt.title('vy vs time')
plt.plot((df['stateZ_timestamp']-df['stateZ_timestamp'][0])/1000,df['stateZ_vy']/1000, label = 'traced')
plt.plot(df['des_timestamp']-df['des_timestamp'][0],df['des_vy'], label = 'desired')
plt.legend()
plt.show()

plt.figure()
plt.title('vz vs time')
plt.plot((df['stateZ_timestamp']-df['stateZ_timestamp'][0])/1000,df['stateZ_vz']/1000, label = 'traced')
plt.plot(df['des_timestamp']-df['des_timestamp'][0],df['des_vz'], label = 'desired')
plt.legend()
plt.show()



