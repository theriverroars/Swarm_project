import numpy as np

import time
import pandas as pd
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.utils import uri_helper
from paths import path_pars
# from cflib.positioning.position_hl_commander import PositionHlCommander
# from cflib.positioning.motion_commander import MotionCommander
# from traj.paths import path_pars
import numpy as np
import math

import logging
from threading import Event
from scipy.spatial.transform import Rotation

from mocaptools import sqrt, Pose, QtmWrapper
from utility_functions import  comp_quat_to_euler,decompressquat, convert_thrust_2_pwm
from scipy.signal import savgol_filter
from quad3d_ctrl1 import Quad3D

from bots.drone import Drone
from controllers.QP_controller_drone_p import QP_Controller_Drone

# Creating Bots

bot1_config_file_path = 'bots//bot_config//drone1.json'
bot2_config_file_path = 'bots//bot_config//drone2.json'
bot3_config_file_path = 'bots//bot_config//drone3.json'

drone1 = Drone.from_JSON(bot1_config_file_path)
drone2 = Drone.from_JSON(bot2_config_file_path)

kf = 3.16e-10

bot = drone1

from scipy import integrate

from Dynamics.drone_dynamics_cart import drone_dynamics

dt = 1/100
num_states = 12
num_inputs = 4


# URI to the Crazyflie to connect to
uri = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E701')
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
t_run = 25
hieght = 0.5
t_lift = 5
t_land = 2
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


def log_stateZ_callback(timestamp, data, logconf):
    global OUTPUTS, start_time
    OUTPUTS['stateZ_timestamp'].append(timestamp)
    OUTPUTS['stateZ_x'].append(data['stateEstimateZ.x'])
    OUTPUTS['stateZ_y'].append(data['stateEstimateZ.y'])
    OUTPUTS['stateZ_z'].append(data['stateEstimateZ.z'])
    OUTPUTS['stateZ_quat'].append(data['stateEstimateZ.quat'])
    OUTPUTS['stateZ_vx'].append(data['stateEstimateZ.vx'])
    OUTPUTS['stateZ_vy'].append(data['stateEstimateZ.vy'])
    OUTPUTS['stateZ_vz'].append(data['stateEstimateZ.vz'])
    OUTPUTS['stateZ_rollrate'].append(data['stateEstimateZ.rateRoll'])
    OUTPUTS['stateZ_pitchrate'].append(data['stateEstimateZ.ratePitch'])
    OUTPUTS['stateZ_yawrate'].append(data['stateEstimateZ.rateYaw'])

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
    params = {}
    rpy_rates = np.array([0,0,0])
    t_int = 0
    time_int = []    

    acc_x_data = []
    acc_y_data = []
    acc_z_data = []

    CTRL = Quad3D()
    gamma = 1
    qp = QP_Controller_Drone(gamma, obs_radius=0.25)
        
    while np.absolute(OUTPUTS['stateZ_x'][-1])/1000 < 4.5 and np.absolute(OUTPUTS['stateZ_y'][-1])/1000 < 4.5 and (OUTPUTS['stateZ_z'][-1])/1000 < 2:
        t_now = time.time()
        t = t_now-t_in
        ## target values


 
        if t < t_lift:
            for i in range(10):
                cf.commander.send_position_setpoint(r_init[0],
                                                    r_init[1],
                                                    r_init[2],
                                                     0)
                time.sleep(0.001)

            INPUTS['des_timestamp'].append(t)
            INPUTS['des_x'].append(OUTPUTS['stateZ_x'][-1]/1000)
            INPUTS['des_y'].append(OUTPUTS['stateZ_x'][-1]/1000)
            INPUTS['des_z'].append(OUTPUTS['stateZ_z'][-1]/1000)
            INPUTS['des_vx'].append(OUTPUTS['stateZ_vx'][-1]/1000)
            INPUTS['des_vy'].append(OUTPUTS['stateZ_vy'][-1]/1000)
            INPUTS['des_vz'].append(OUTPUTS['stateZ_vz'][-1]/1000)

            INPUTS['cmd_ax'].append(0)
            INPUTS['cmd_ay'].append(0)
            INPUTS['cmd_az'].append(0)
        

        elif t < n_iters*t_run + t_lift:
            
            rd,rd_dot,rd_ddot = path_pars(t-t_lift,t_run,c = 0.35, tilt=0,rd_init = r_init,shape = 'line')
                ################################################################
            # put the controller here
            # controller
            # i/p - state - drone and obstacle
            # o/p - torques
            # call the model
            # i/p - torques
            # o/p - roll, pitch, yaw,z
            params['pos'] = np.array([OUTPUTS['stateZ_x'][-1], OUTPUTS['stateZ_y'][-1], OUTPUTS['stateZ_z'][-1]])/1000 # position in m
            params['vel'] = np.array([OUTPUTS['stateZ_vx'][-1], OUTPUTS['stateZ_vy'][-1], OUTPUTS['stateZ_vz'][-1]])/1000 # position in m
            # params['vel'] = xyz_dot
            params['quat'] = decompressquat(OUTPUTS['stateZ_quat'][-1])
            r_sen, p_sen, y_sen = comp_quat_to_euler((OUTPUTS['stateZ_quat'][-1])) # in radians
            params['rpy'] = np.array([r_sen, p_sen, y_sen])
            params['rpy_rates'] = np.array([OUTPUTS['stateZ_rollrate'][-1], OUTPUTS['stateZ_pitchrate'][-1], OUTPUTS['stateZ_yawrate'][-1]])/1000 # in radians/s
            # params['rpy_rates'] = rpy_rates
            params['dt'] = (OUTPUTS['stateZ_timestamp'][-1]-OUTPUTS['stateZ_timestamp'][-2])/1000 #s
            
            
            
            x_ddot, y_ddot, z_ddot, rpm = CTRL.compute_control(current_position=params['pos'] ,
                                            current_velocity=params['vel'],
                                            current_rpy=params['rpy'],
                                            target_position=rd,
                                            target_velocity=rd_dot,
                                            target_acceleration=rd_ddot,
                                            TIMESTEP=params['dt']
                                            )
            
            # print('current_rpy:',params['rpy_rates'][0], params['rpy_rates'][1])


            t_int += params['dt']
            time_int.append(t_int)

            # print('rpm',rpm)

            if t-t_lift>1:
                
                u_ref = rpm
                f_u_ref = 3.16e-10 * np.square(u_ref)
                qp.set_reference_control(f_u_ref)
                qp.setup_QP(bot, [2.0, 0, 0.35],[0,0,0])
                
                # Simulation
                # Solve QP
                value_of_h = qp.solve_QP(bot)
                # print("h", value_of_h)
                
                # Bot Kinematics
                u_star = qp.get_optimal_control()
                rpm = u_star

                x_ddot, y_ddot, z_ddot = CTRL.compute_xyz_ddot(rpm, params['dt'], params['rpy'], params['rpy_rates'])

            
            KF = 3.16e-10
            thrusts = KF*(rpm**2)

            bot.update_state(params['pos'], params['vel'], params['rpy'], params['dt'])



            # ## rpy from the dynamics

            # xyz_dot, acc = drone_dynamics(params, thrusts)

            # x_ddot = acc[0]
            # y_ddot = acc[1]
            # z_ddot = acc[2]


            acc_x_data.append(x_ddot)
            acc_y_data.append(y_ddot)
            acc_z_data.append(z_ddot)
            

            x_di = integrate.trapz(acc_x_data, time_int)
            y_di = integrate.trapz(acc_y_data, time_int)
            z_di = integrate.trapz(acc_z_data, time_int)
            net_thrust = np.sum(thrusts)
            net_thrust_pwm = np.clip(convert_thrust_2_pwm(net_thrust/4), 0, 65535)
            ################################################################
            # print('rpyt setpoints:', thrusts, net_thrust ,net_thrust_pwm)
            # print('acc:', x_ddot, y_ddot, z_ddot)
            # print('acc1:', x_ddot1, y_ddot1, z_ddot1)
            # cfcommander.send_zdistance_setpoint(r, p, y, 0.4)

            INPUTS['des_timestamp'].append(t)
            INPUTS['des_x'].append(rd[0])
            INPUTS['des_y'].append(rd[1])
            INPUTS['des_z'].append(rd[2])

            INPUTS['des_vx'].append(rd_dot[0])
            INPUTS['des_vy'].append(rd_dot[1])
            INPUTS['des_vz'].append(rd_dot[2])

            INPUTS['cmd_ax'].append(x_ddot)
            INPUTS['cmd_ay'].append(y_ddot)
            INPUTS['cmd_az'].append(z_ddot)
 
            for i in range(10):
                cf.commander.send_velocity_world_setpoint(x_di,
                                                          y_di,
                                                          z_di,#xyz_dot[2],
                                                          0)
                
                # cf.commander.send_zdistance_setpoint(0,
                #                                      -2,
                #                                      0,
                #                                      r_init[2])
                
                # cf.commander.send_setpoint(math.degrees(rpy[0]),
                #                 math.degrees(rpy[1]),
                #                 math.degrees(rpy_rates[2]),
                #                 net_thrust_pwm)
                time.sleep(0.001)

            j = j+1
            
        
        elif t < n_iters*t_run + t_lift + t_land:
            for i in range(10):
                cf.commander.send_position_setpoint(rd[0],
                                                    rd[1],
                                                    0.05,
                                                     0)
                time.sleep(0.001)
        
        elif t >= n_iters*t_run + t_lift + t_land:

            INPUTS['des_timestamp'].append(t)
            INPUTS['des_x'].append(rd[0])
            INPUTS['des_y'].append(rd[1])
            INPUTS['des_z'].append(0)
            INPUTS['des_vx'].append(0)
            INPUTS['des_vy'].append(0)
            INPUTS['des_vz'].append(OUTPUTS['stateZ_vz'][-1]/1000)

            INPUTS['cmd_ax'].append(0)
            INPUTS['cmd_ay'].append(0)
            INPUTS['cmd_az'].append(0)
            
            break



        
  
    print('crazyflie gone crazy')
    cf.commander.send_stop_setpoint()
    # Make sure that the last packet leaves before the link is closed
    # since the message queue is not flushed before closing
    time.sleep(0.1)


if __name__ == '__main__':
    cflib.crtp.init_drivers()
    time_string = time.ctime().replace(':', '-')
    start_time = time.time()
    INPUTS = {'motor_timestamp': [], 'm1': [], 'm2': [], 'm3': [], 'm4': [], 'bat_timestamp': [], 'bat_volt':[],'cmd_timestamp': [],'cmd_thrust': [],'cmd_act_thrust':[], 'cmd_roll':[], 'cmd_pitch':[], 'cmd_yawrate':[], 'stabilizer_timestamp':[],'des_x':[],'des_y':[],'des_z':[], 'des_timestamp':[],'des_vx':[],'des_vy':[],'des_vz':[],'cmd_ax':[],'cmd_ay':[],'cmd_az':[] }
    
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






    lg_stateZ = LogConfig('stateEstimateZ', period_in_ms=10)
    lg_stateZ.add_variable('stateEstimateZ.x', 'int16_t')
    lg_stateZ.add_variable('stateEstimateZ.y', 'int16_t')
    lg_stateZ.add_variable('stateEstimateZ.z', 'int16_t')
    lg_stateZ.add_variable('stateEstimateZ.quat', 'int32_t')
    lg_stateZ.add_variable('stateEstimateZ.vx', 'int16_t')
    lg_stateZ.add_variable('stateEstimateZ.vy', 'int16_t')
    lg_stateZ.add_variable('stateEstimateZ.vz', 'int16_t')
    # # lg_stateZ.add_variable('stateEstimateZ.ax', 'int16_t')
    # # lg_stateZ.add_variable('stateEstimateZ.ay', 'int16_t')
    # # lg_stateZ.add_variable('stateEstimateZ.az', 'int16_t')
    lg_stateZ.add_variable('stateEstimateZ.rateRoll', 'int16_t')
    lg_stateZ.add_variable('stateEstimateZ.ratePitch', 'int16_t')
    lg_stateZ.add_variable('stateEstimateZ.rateYaw', 'int16_t')
    
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
        # scf.cf.log.add_config(lg_stab)
        scf.cf.log.add_config(lg_cont)
        # scf.cf.log.add_config(lg_state)   
        scf.cf.log.add_config(lg_stateZ)
        # scf.cf.log.add_config(lg_gyro)

        lg_motor.data_received_cb.add_callback(log_motor_callback)
        bat_volt.data_received_cb.add_callback(log_battery_voltage)
        # lg_stab.data_received_cb.add_callback(log_stabilizer_callback)
        lg_cont.data_received_cb.add_callback(log_controller_callback)     
        # lg_state.data_received_cb.add_callback(log_state_callback)   
        lg_stateZ.data_received_cb.add_callback(log_stateZ_callback) 
        # lg_gyro.data_received_cb.add_callback(log_gyro_callback)       

        lg_stateZ.start()
        bat_volt.start()
        lg_motor.start() 
        # lg_stab.start()
        # lg_cont.start()
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
        # lg_stab.stop()
        bat_volt.stop()
        lg_motor.stop() 
        # lg_cont.stop()
        # lg_state.stop()
        # lg_gyro.stop()
        lg_stateZ.stop()

    # mocap.close()
    x = input('Do you want to write the file to CSV? (y/n): ')
    if x == 'y' or x == 'Y':
        ds = {**INPUTS, **OUTPUTS}
        DF = pd.DataFrame.from_dict(ds, orient='index')
        DF = DF.transpose()
        DF.to_csv('/home/rajpal/github_dat/Drones-C3BF/data_log/tuning/dynamics_dataset {}.csv'.format(time_string))
        print("written to CSV")


        
