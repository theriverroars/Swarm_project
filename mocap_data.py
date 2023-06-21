import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.utils import uri_helper

from mocap.mocaptools2 import sqrt, Pose, QtmWrapper
from mocapSDK import QTMSDK
import logging
from threading import Event
from scipy.spatial.transform import Rotation

import time
import pandas as pd

# URI to the Crazyflie to connect to
DEFAULT_URI = 'radio://0/80/2M/E7E7E7E701'
CF_BODY = 'cf2'#args.rigid_body_name
OBS_1 = 'obs_1'
# OBS_TEMP = 'obs_temp'
QTM_IP = '192.168.0.2'
# URI to the Crazyflie to connect to
uri = uri_helper.uri_from_env(default=DEFAULT_URI)

deck_attached_event = Event()
logging.basicConfig(level=logging.ERROR)

send_full_pose = False

def log_state_callback(timestamp, data, logconf):
    global OUTPUTS, start_time, mocap
    OUTPUTS['stateZ_timestamp'].append(timestamp)
    OUTPUTS['cf_x'].append(data['stateEstimateZ.x']/1000)
    OUTPUTS['cf_y'].append(data['stateEstimateZ.y']/1000)
    OUTPUTS['cf_z'].append(data['stateEstimateZ.z']/1000)
    # OUTPUTS['stateZ_quat'].append(data['stateEstimateZ.quat'])
    # OUTPUTS['stateZ_vx'].append(data['stateEstimateZ.vx']/1000)
    # OUTPUTS['stateZ_vy'].append(data['stateEstimateZ.vy']/1000)
    # OUTPUTS['stateZ_vz'].append(data['stateEstimateZ.vz']/1000)
    # OUTPUTS['stateZ_rollrate'].append(data['stateEstimateZ.rateRoll']/1000)
    # OUTPUTS['stateZ_pitchrate'].append(data['stateEstimateZ.ratePitch']/1000)
    # OUTPUTS['stateZ_yawrate'].append(data['stateEstimateZ.rateYaw']/1000)

    # bodydict = mocap.getpose()[1].x
    
    # cf_mocap_data = bodydict['cf2']
    OUTPUTS['cf_mocap_data_x'].append(mocap.getpose()[0].x)
    OUTPUTS['cf_mocap_data_y'].append(mocap.getpose()[0].y)
    OUTPUTS['cf_mocap_data_z'].append(mocap.getpose()[0].z)

    # obs_1_data = bodydict['obs_1']
    
    OUTPUTS['obs_x'].append(mocap.getpose()[1].x)
    OUTPUTS['obs_y'].append(mocap.getpose()[1].y)
    OUTPUTS['obs_z'].append(mocap.getpose()[1].z)
    
    # p_obs_temp = mocap_obs_temp.getpose()
    # OUTPUTS['obs_temp_x'].append(p_obs_temp.x)
    # OUTPUTS['obs_temp_y'].append(p_obs_temp.y)
    # OUTPUTS['obs_temp_z'].append(p_obs_temp.z)

def wait_for_position_estimator(scf):
    print('Waiting for estimator to find position...')

    log_config = LogConfig(name='Kalman Variance', period_in_ms=500)
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

def send_extpose_rot_matrix(cf, x, y, z, rot):
    """
    Send the current Crazyflie X, Y, Z position and attitude as a (3x3)
    rotaton matrix. This is going to be forwarded to the Crazyflie's
    position estimator.
    """
    # qw = _sqrt(1 + rot[0][0] + rot[1][1] + rot[2][2]) / 2
    # qx = _sqrt(1 + rot[0][0] - rot[1][1] - rot[2][2]) / 2
    # qy = _sqrt(1 - rot[0][0] + rot[1][1] - rot[2][2]) / 2
    # qz = _sqrt(1 - rot[0][0] - rot[1][1] + rot[2][2]) / 2

    # # Normalize the quaternion
    # ql = math.sqrt(qx ** 2 + qy ** 2 + qz ** 2 + qw ** 2)
    # # print('Trying to send poses')

    # if send_full_pose:
    #     cf.extpos.send_extpose(x, y, z, qx / ql, qy / ql, qz / ql, qw / ql)
    quat = Rotation.from_matrix(rot).as_quat()

    if send_full_pose:
        cf.extpos.send_extpose(x, y, z, quat[0], quat[1], quat[2], quat[3])
    else:
        # print("Sending position", x,y,z)
        cf.extpos.send_extpos(x, y, z)

# def mocaplogging(mocap):
#     global OUTPUTS
#     global start_time
#     t = time.time() - start_time
#     # OUTPUTS['mocap_output_timestamp'].append(t)
#     p = mocap.getpose()
#     OUTPUTS['obs_x'].append(p.x)
#     OUTPUTS['obs_y'].append(p.y)
#     OUTPUTS['obs_z'].append(p.z)



def reset_estimator(scf):
    scf.cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    scf.cf.param.set_value('kalman.resetEstimation', '0')

    # time.sleep(1)
    wait_for_position_estimator(scf)


def activate_kalman_estimator(scf):
    scf.cf.param.set_value('stabilizer.estimator', '2')

    # Set the std deviation for the quaternion data pushed into the
    # kalman filter. The default value seems to be a bit too low.
    # cf.param.set_value('locSrv.extPosStdDev', 0.009)
    # cf.param.set_value('locSrv.extQuatStdDev', 0.06)
    # cf.param.set_value('motion.adaptive', 1)


if __name__ == '__main__':
    cflib.crtp.init_drivers()
    time_string = time.ctime().replace(':', '-')
    start_time = time.time()

    # OUTPUTS = {'stateZ_timestamp':[],'stateZ_quat':[],'stateZ_yawrate':[]}#,'state_qx':[],'state_qy':[],'state_qz':[],'state_qw':[]} #               'state_timestamp':[],'state_qx':[],'state_qy':[],'state_qz':[],'state_qw':[],'state_timestamp':[],'state_x':[], 'state_y':[],'state_z':[],'state_roll':[],'state_pitch':[], 'state_yaw':[],
 

    
    OUTPUTS = {'stateZ_timestamp':[], 'cf_x':[], 'cf_y':[], 'cf_z':[], 'obs_x':[], 'obs_y':[], 'obs_z':[], 'cf_mocap_data_x':[], 'cf_mocap_data_y':[], 'cf_mocap_data_z':[]}#, 'obs_temp_x':[], 'obs_temp_y':[], 'obs_temp_z':[]}
    

    lg_state = LogConfig('stateEstimateZ', period_in_ms=10)
    lg_state.add_variable('stateEstimateZ.x', 'int16_t')
    lg_state.add_variable('stateEstimateZ.y', 'int16_t')
    lg_state.add_variable('stateEstimateZ.z', 'int16_t')
    lg_state.add_variable('stateEstimateZ.quat', 'int32_t')
    lg_state.add_variable('stateEstimateZ.vx', 'int16_t')
    lg_state.add_variable('stateEstimateZ.vy', 'int16_t')
    lg_state.add_variable('stateEstimateZ.vz', 'int16_t')

    # lg_state_rate.add_variable('stateEstimateZ.rateRoll', 'int16_t')
    # lg_state_rate.add_variable('stateEstimateZ.ratePitch', 'int16_t')
    # lg_state_rate.add_variable('stateEstimateZ.rateYaw', 'int16_t')
    qtm_bodies = [CF_BODY, OBS_1]
    mocap = QtmWrapper(QTM_IP, qtm_bodies)
    # time.sleep(5)
    # mocap_obs = QTMSDK(QTM_IP, OBS_1)
    # mocap_obs_temp = QtmWrapper(QTM_IP, OBS_TEMP)


    # mocap_cf.on_cf_pose = lambda pose: send_extpose_rot_matrix(cf, pose[0], pose[1], pose[2], pose[3])


    time.sleep(5)
    # bodydict = mocap_cf.getxy()
    # obs_1_data = bodydict['obs_1']
    # print("received new data")
    # print(bodydict)
    # print(obs_1_data)
    # exit()
    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:
        time.sleep(1)
        cf = scf.cf
        # scf.cf.log.add_config(lg_motor)
        # scf.cf.log.add_config(bat_volt)
        scf.cf.log.add_config(lg_state)

        # lg_motor.data_received_cb.add_callback(log_motor_callback)
        # bat_volt.data_received_cb.add_callback(log_battery_voltage)   
        lg_state.data_received_cb.add_callback(log_state_callback)     

        # lg_motor.start() 
        # bat_volt.start()
        lg_state.start()


        mocap.on_cf_pose = lambda pose: send_extpose_rot_matrix(cf, pose[0], pose[1], pose[2], pose[3])

        # print(bodydict)

        

            # run_sequence(scf, sequence)
        activate_kalman_estimator(scf)
        reset_estimator(scf)

        # mocaplogging(mocap_obs)


        time.sleep(10)

        # lg_motor.stop() 
        # bat_volt.stop()
        lg_state.stop()

    mocap.close()
    # mocap_obs.close()
    # mocap_obs_temp.close()
    
    x = input('Do you want to write the file to CSV? (y/n): ')
    if x == 'y' or x == 'Y':
        ds = {**OUTPUTS}
        DF = pd.DataFrame.from_dict(ds, orient='index')
        DF = DF.transpose()
        DF.to_csv('/home/rajpal/github_dat/Drones-C3BF/data_log/tuning/{}.csv'.format(time_string))
        print("written to CSV")


    