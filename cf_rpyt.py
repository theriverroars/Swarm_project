import numpy as np

import time
import pandas as pd
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.utils import uri_helper
# from cflib.positioning.position_hl_commander import PositionHlCommander
# from cflib.positioning.motion_commander import MotionCommander
# from traj.paths import path_pars
import numpy as np
import math

import logging
from threading import Event
from scipy.spatial.transform import Rotation

from mocaptools import sqrt, Pose, QtmWrapper
from utils import  comp_quat_to_euler,decompressquat, convert_thrust_2_pwm
from scipy.signal import savgol_filter
from controllers.QP_controller_drone import QP_Controller_Drone
from quad3d_ctrl1 import Quad3D


dt = 1/100
num_states = 12
num_inputs = 4


# URI to the Crazyflie to connect to
uri = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E702')
QTM_IP = '192.168.0.106' 
CF_BODY = 'cf'

# argparse this or replace with relative path  
# path_np = np.array(np.genfromtxt('/home/rajpal/CF programs/crazypaths/paths/circle_waypoints.csv',delimiter=',', skip_header=1, dtype=float).tolist())
# sequence = path_np[:,1:]
# Change the sequence according to your setup
#             x    y    z  YAW
# sequence = [
#     (0.0, 0.0, 0.4, 0),
#     (0.0, 0.0, 0.5, 0),
#     (0.1, -0.1, 0.5, 0),
#     (0.1, 0.1, 0.5, 0),
#     (-0.1, 0.1, 0.5, 0),
#     (-0.1, -0.1, 0.5, 0),
#     (0.0, 0.0, 0.5, 0),
#     (0.0, 0.0, 0.4, 0),
#     (0.0, 0.0, 0.0, 0.0)
# ]

# path parameters
t_run = 10
hieght = 0.4
t_lift = 5
t_land = 1
n_iters = 1
t_end = 14


deck_attached_event = Event()
logging.basicConfig(level=logging.ERROR)



def log_motor_callback(timestamp, data, logconf):
    global mocap, INPUTS, OUTPUTS
    INPUTS['motor_timestamp'].append(timestamp)
    # global start_time
    # t = time.time() - start_time
    # OUTPUTS['mocap_output_timestamp'].append(t)
    [INPUTS['{}'.format(i)].append(data['motor.{}'.format(i)]) for i in ('m1', 'm2', 'm3', 'm4')]
    # p = mocap.getpose()
    # OUTPUTS['mocap_x'].append(p.x)
    # OUTPUTS['mocap_y'].append(p.y)
    # OUTPUTS['mocap_z'].append(p.z)
    # rotation = Rotation.from_matrix(p.rotmatrix)
    # r = rotation.as_quat()
    # OUTPUTS['mocap_qx'].append(r[0])
    # OUTPUTS['mocap_qy'].append(r[1])
    # OUTPUTS['mocap_qz'].append(r[2])
    # OUTPUTS['mocap_qw'].append(r[3])
    # print(f'mocap: x: {p.x}, y: {p.y}, z: {p.z}')
  

def log_stabilizer_callback(timestamp, data, logconf):
    global INPUTS
    INPUTS['stabilizer_timestamp'].append(timestamp)
    INPUTS['stabilizer_thrust'].append(data['stabilizer.thrust'])

def log_controller_callback(timestamp, data, logconf):
    global INPUTS
    INPUTS['cmd_timestamp'].append(timestamp)
    INPUTS['cmd_thrust'].append(data['controller.cmd_thrust'])
    INPUTS['cmd_act_thrust'].append(data['controller.actuatorThrust'])
    INPUTS['cmd_roll'].append(data['controller.roll'])
    INPUTS['cmd_pitch'].append(data['controller.pitch'])
    INPUTS['cmd_yawrate'].append(data['controller.yawRate'])
    # INPUTS['cmd_rollrate'].append(data['controller.rollRate'])
    # INPUTS['cmd_pitchrate'].append(data['controller.pitchRate'])
    # INPUTS['cmd_yawrate'].append(data['controller.yawRate'])

def log_gyro_callback(timestamp, data, logconf):
    global OUTPUTS
    OUTPUTS['gyro_timestamp'].append(timestamp)
    OUTPUTS['gyro_wx'].append(data['gyro.x'])
    OUTPUTS['gyro_wy'].append(data['gyro.y'])
    OUTPUTS['gyro_wz'].append(data['gyro.z'])

def log_battery_voltage(timestamp, data, logconf):
    global INPUTS, start_time
    INPUTS['bat_timestamp'].append(timestamp)
    INPUTS['bat_volt'].append(data['pm.vbat'])

def log_state_callback(timestamp, data, logconf):
    global OUTPUTS, start_time
    OUTPUTS['state_timestamp'].append(timestamp)
    # OUTPUTS['state_x'].append(data['stateEstimate.x'])
    # OUTPUTS['state_y'].append(data['stateEstimate.y'])
    # OUTPUTS['state_z'].append(data['stateEstimate.z'])
    OUTPUTS['state_roll'].append(data['stateEstimate.roll'])
    OUTPUTS['state_pitch'].append(data['stateEstimate.pitch'])
    OUTPUTS['state_yaw'].append(data['stateEstimate.yaw'])
    OUTPUTS['state_qx'].append(data['stateEstimate.qx'])
    OUTPUTS['state_qy'].append(data['stateEstimate.qy'])
    OUTPUTS['state_qz'].append(data['stateEstimate.qz'])
    OUTPUTS['state_qw'].append(data['stateEstimate.qw'])


def log_state_rate_callback(timestamp, data, logconf):
    global OUTPUTS, start_time
    OUTPUTS['stateZ_timestamp'].append(timestamp)
    OUTPUTS['stateZ_x'].append(data['stateEstimateZ.x'])
    OUTPUTS['stateZ_y'].append(data['stateEstimateZ.y'])
    OUTPUTS['stateZ_z'].append(data['stateEstimateZ.z'])
    # OUTPUTS['stateZ_quat'].append(data['stateEstimateZ.quat'])
    # OUTPUTS['stateZ_vx'].append(data['stateEstimateZ.vx'])
    # OUTPUTS['stateZ_vy'].append(data['stateEstimateZ.vy'])
    # OUTPUTS['stateZ_vz'].append(data['stateEstimateZ.vz'])
    # OUTPUTS['stateZ_rollrate'].append(data['stateEstimateZ.rateRoll'])
    # OUTPUTS['stateZ_pitchrate'].append(data['stateEstimateZ.ratePitch'])
    # OUTPUTS['stateZ_yawrate'].append(data['stateEstimateZ.rateYaw'])

def wait_for_position_estimator(scf):
    print('Waiting for estimator to find position...')

    log_config = LogConfig(name='Kalman Variance', period_in_ms=10)
    log_config.add_variable('kalman.varPX', 'float')
    log_config.add_variable('kalman.varPY', 'float')
    log_config.add_variable('kalman.varPZ', 'float')

    var_y_history = [1000] * 10
    var_x_history = [1000] * 10
    var_z_history = [1000] * 10

    threshold = 0.001

    with SyncLogger(scf, log_config) as logger:
        for log_entry in logger:
            data = log_entry[1]

            var_x_history.append(data['kalman.varPX'])
            var_x_history.pop(0)
            var_y_history.append(data['kalman.varPY'])
            var_y_history.pop(0)
            var_z_history.append(data['kalman.varPZ'])
            var_z_history.pop(0)

            min_x = min(var_x_history)
            max_x = max(var_x_history)
            min_y = min(var_y_history)
            max_y = max(var_y_history)
            min_z = min(var_z_history)
            max_z = max(var_z_history)

            # print("{} {} {}".
            #       format(max_x - min_x, max_y - min_y, max_z - min_z))

            if (max_x - min_x) < threshold and (
                    max_y - min_y) < threshold and (
                    max_z - min_z) < threshold:
                break


def reset_estimator(scf):
    cf = scf.cf
    # activate Kalman estimator 
    cf.param.set_value('stabilizer.estimator', '2')

    # cf.param.set_value('locSrv.extQuatStdDev', 0.5)

    # reset estimator
    cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    cf.param.set_value('kalman.resetEstimation', '0')

    wait_for_position_estimator(cf)


def position_callback(timestamp, data, logconf):

    x = data['kalman.stateX']
    y = data['kalman.stateY']
    z = data['kalman.stateZ']
    print('kalman pos: ({}, {}, {})'.format(x, y, z))


def start_position_printing(scf):
    log_conf = LogConfig(name='Position', period_in_ms=10)
    log_conf.add_variable('kalman.stateX', 'float')
    log_conf.add_variable('kalman.stateY', 'float')
    log_conf.add_variable('kalman.stateZ', 'float')

    scf.cf.log.add_config(log_conf)
    log_conf.data_received_cb.add_callback(position_callback)
    log_conf.start()



def mocaplogging(pose):
    global OUTPUTS, mocap
    global start_time
    t = time.time() - start_time
    OUTPUTS['mocap_output_timestamp'].append(t)
    p = mocap.getpose()
    OUTPUTS['mocap_x'].append(p.x)
    OUTPUTS['mocap_y'].append(p.y)
    OUTPUTS['mocap_z'].append(p.z)
    rotation = Rotation.from_matrix(p.rotmatrix)
    r = rotation.as_quat()
    OUTPUTS['mocap_qx'].append(r[0])
    OUTPUTS['mocap_qy'].append(r[1])
    OUTPUTS['mocap_qz'].append(r[2])
    OUTPUTS['mocap_qw'].append(r[3])


def run_sequence(scf):
    cf = scf.cf
    global A,B,C,file,params,input_np,t_end,OUTPUTS
    cf.commander.send_setpoint(0,0,0,0)
    j = 0
    while np.absolute(OUTPUTS['stateZ_x'][-1])/1000 < 3 and np.absolute(OUTPUTS['stateZ_y'][-1])/1000 < 3 and (OUTPUTS['stateZ_z'][-1])/1000 < 0.7:
        t_now = time.time()
        t = t_now-t_in
        if t >= t_end:
            break
        ################################################################
        # put the controller here
        # controller
        # i/p - state - drone and obstacle
        # o/p - torques
        # call the model
        # i/p - torques
        # o/p - roll, pitch, yaw,z
        CTRL = Quad3D()
        thrusts = CTRL.compute_control(current_position=state[0:3],
                                        current_velocity=state[10:13],
                                        current_rpy=state[7:10],
                                        target_position=TARGET_POSITION[i, :],
                                        target_velocity=TARGET_VELOCITY[i, :],
                                        target_acceleration=TARGET_ACCELERATION[i, :]
                                        )


        gamma = 1
        qp = QP_Controller_Drone(gamma)
        u_ref = thrusts
        f_u_ref = kf * np.square(u_ref)
        qp.set_reference_control(f_u_ref)
        # print(obs_1[0:3], obs_1[10:13])
        qp.setup_QP(bot, obs_3[0:3], obs_3[10:13])
               

        # Simulation
        # Solve QP
        state_of_QP, value_of_h = qp.solve_QP(bot)
        
        # Bot Kinematics
        u_star = qp.get_optimal_control()
        thrusts = u_star

        thr = thrusts[0] + thrusts[1] + thrusts[2] + thrusts[3]

        ## rpy from the model

        ################################################################
        print('rpyt setpoints:',r,p,y,thr)
        # cfcommander.send_zdistance_setpoint(r, p, y, 0.4)

        cf.commander.send_setpoint(r,
                                    p,
                                    y,
                                    thr)
        time.sleep(0.01)

        j = j+1
  
    print('crazyflie gone crazy')
    cf.commander.send_stop_setpoint()
    # Make sure that the last packet leaves before the link is closed
    # since the message queue is not flushed before closing
    time.sleep(0.1)


if __name__ == '__main__':
    cflib.crtp.init_drivers()
    time_string = time.ctime().replace(':', '-')
    start_time = time.time()
    INPUTS = {'motor_timestamp': [], 'm1': [], 'm2': [], 'm3': [], 'm4': [], 'bat_timestamp': [], 'bat_volt':[],'cmd_timestamp': [],'cmd_thrust': [],'cmd_act_thrust':[], 'cmd_roll':[], 'cmd_pitch':[], 'cmd_yawrate':[], 'stabilizer_timestamp':[], 'stabilizer_thrust':[]}
    
    OUTPUTS = {'mocap_output_timestamp': [], 'mocap_x': [], 'mocap_y': [], 'mocap_z': [], 'mocap_qx': [], 'mocap_qy': [], 'mocap_qz': [], 'mocap_qw': [],
               'stateZ_timestamp':[],'stateZ_x':[],'stateZ_y':[],'stateZ_z':[],'stateZ_quat':[],'stateZ_vx':[],'stateZ_vy':[],'stateZ_vz':[],'stateZ_rollrate':[],'stateZ_pitchrate':[],'stateZ_yawrate':[]}# 'state_timestamp':[],'state_roll':[],'state_pitch':[], 'state_yaw':[],'state_qx':[],'state_qy':[],'state_qz':[],'state_qw':[]} #               'state_timestamp':[],'state_qx':[],'state_qy':[],'state_qz':[],'state_qw':[],'state_timestamp':[],'state_x':[], 'state_y':[],'state_z':[],'state_roll':[],'state_pitch':[], 'state_yaw':[],
 

    lg_motor = LogConfig(name='motor', period_in_ms=10) 
    [lg_motor.add_variable('motor.m{}'.format(i), 'uint16_t') for i in (1, 2, 3, 4)]
    
    bat_volt = LogConfig(name='pm.vbat', period_in_ms=10)
    bat_volt.add_variable('pm.vbat', 'float')
    
    lg_stab = LogConfig('stabilizer', period_in_ms=10)
    lg_stab.add_variable('stabilizer.thrust', 'float')
    
    lg_cont = LogConfig('controller', period_in_ms=10)
    lg_cont.add_variable('controller.cmd_thrust', 'float')
    lg_cont.add_variable('controller.actuatorThrust', 'float')
    # lg_cont.add_variable('controller.cmd_roll', 'float')
    # lg_cont.add_variable('controller.cmd_pitch', 'float')
    lg_cont.add_variable('controller.roll', 'float')
    lg_cont.add_variable('controller.pitch', 'float')
    lg_cont.add_variable('controller.yawRate', 'float')

    # lg_gyro = LogConfig('gyro', period_in_ms=10)
    # lg_gyro.add_variable('gyro.x', 'float')
    # lg_gyro.add_variable('gyro.y', 'float')
    # lg_gyro.add_variable('gyro.z', 'float')
    
    # lg_state = LogConfig('stateEstimate', period_in_ms=10)
    # # lg_state.add_variable('stateEstimate.x', 'float')
    # # lg_state.add_variable('stateEstimate.y', 'float')
    # # lg_state.add_variable('stateEstimate.z', 'float')
    # lg_state.add_variable('stateEstimate.roll', 'float')
    # lg_state.add_variable('stateEstimate.pitch', 'float')
    # lg_state.add_variable('stateEstimate.yaw', 'float')
    # lg_state.add_variable('stateEstimate.qx', 'float')
    # lg_state.add_variable('stateEstimate.qy', 'float')
    # lg_state.add_variable('stateEstimate.qz', 'float')
    # # lg_state.add_variable('stateEstimate.qw', 'float')






    lg_state_rate = LogConfig('stateEstimateZ', period_in_ms=10)
    lg_state_rate.add_variable('stateEstimateZ.x', 'int16_t')
    lg_state_rate.add_variable('stateEstimateZ.y', 'int16_t')
    lg_state_rate.add_variable('stateEstimateZ.z', 'int16_t')
    # lg_state_rate.add_variable('stateEstimateZ.quat', 'int32_t')
    # lg_state_rate.add_variable('stateEstimateZ.vx', 'int16_t')
    # lg_state_rate.add_variable('stateEstimateZ.vy', 'int16_t')
    # lg_state_rate.add_variable('stateEstimateZ.vz', 'int16_t')
    # # lg_state_rate.add_variable('stateEstimateZ.ax', 'int16_t')
    # # lg_state_rate.add_variable('stateEstimateZ.ay', 'int16_t')
    # # lg_state_rate.add_variable('stateEstimateZ.az', 'int16_t')
    # lg_state_rate.add_variable('stateEstimateZ.rateRoll', 'int16_t')
    # lg_state_rate.add_variable('stateEstimateZ.ratePitch', 'int16_t')
    # lg_state_rate.add_variable('stateEstimateZ.rateYaw', 'int16_t')
    
    # mocap = QtmWrapper(QTM_IP, CF_BODY)
    # time.sleep(5)

    # init_pos = mocap.getpose()
    # r_init = np.array([init_pos.x,init_pos.y,hieght])
    r_init = np.array([0., 0., hieght])
    t_in = time.time()

    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:
        time.sleep(1)
        scf.cf.log.add_config(lg_motor)
        scf.cf.log.add_config(bat_volt)
        scf.cf.log.add_config(lg_stab)
        scf.cf.log.add_config(lg_cont)
        # scf.cf.log.add_config(lg_state)   
        scf.cf.log.add_config(lg_state_rate)
        # scf.cf.log.add_config(lg_gyro)

        lg_motor.data_received_cb.add_callback(log_motor_callback)
        bat_volt.data_received_cb.add_callback(log_battery_voltage)
        lg_stab.data_received_cb.add_callback(log_stabilizer_callback)
        lg_cont.data_received_cb.add_callback(log_controller_callback)     
        # lg_state.data_received_cb.add_callback(log_state_callback)   
        lg_state_rate.data_received_cb.add_callback(log_state_rate_callback) 
        # lg_gyro.data_received_cb.add_callback(log_gyro_callback)       

        lg_state_rate.start()
        bat_volt.start()
        lg_motor.start() 
        lg_stab.start()
        lg_cont.start()
        # lg_state.start()
        # lg_gyro.start()
 
      
        # mocap.on_cf_pose = lambda pose: send_extpose_rot_matrix(scf.cf, pose[0], pose[1], pose[2], pose[3])
        # mocap.on_cf_pose = lambda pose: mocaplogging(pose)
        
        # start_position_printing(scf)

        # with MotionCommander(scf) as mc:
        #     mc.take_off(0.4)
        #     mc.stop()
        reset_estimator(scf)
        # run_sequence(scf, sequence)
        run_sequence(scf)
        lg_stab.stop()
        bat_volt.stop()
        lg_motor.stop() 
        lg_cont.stop()
        # lg_state.stop()
        # lg_gyro.stop()
        lg_state_rate.stop()

    # mocap.close()
    x = input('Do you want to write the file to CSV? (y/n): ')
    if x == 'y' or x == 'Y':
        ds = {**INPUTS, **OUTPUTS}
        DF = pd.DataFrame.from_dict(ds, orient='index')
        DF = DF.transpose()
        DF.to_csv('/home/rajpal/CF programs/pos_level_cont/datasets/dynamics_dataset {}.csv'.format(time_string))
        print("written to CSV")


        
