import time
import random
import numpy as np
import pybullet as p

from gym_cbf.envs.CtrlAviary import CtrlAviary
from gym_cbf.utils.Logger import Logger
from gym_cbf.utils.utils import sync
from gym_cbf.utils.enums import DroneModel
from quad3d_ctrl import Quad3D

from bots.drone import Drone
from controllers.QP_controller_drone import QP_Controller_Drone

# Creating Bots

bot1_config_file_path = 'bots//bot_config//drone1.json'
bot2_config_file_path = 'bots//bot_config//drone2.json'
bot3_config_file_path = 'bots//bot_config//drone3.json'

drone1 = Drone.from_JSON(bot1_config_file_path)
drone2 = Drone.from_JSON(bot2_config_file_path)

kf = 3.16e-10

bot = drone1

DURATION = 25
"""int: The duration of the simulation in seconds."""
GUI = True
"""bool: Whether to use PyBullet graphical interface."""
RECORD = False
"""bool: Whether to save a video under /files/videos. Requires ffmpeg"""

if __name__ == "__main__":

    # Create the environment
    env = CtrlAviary(num_drones=6,
                     drone_model=DroneModel.CF2P,
                     initial_xyzs=np.array([[.0, 0, 0.5], [1, .10, 0.9], [1, .10, 1.1], [0, 0.0, 2.5], [1, .10, 1.5], [1, .10, 1.7]]),
                     gui=GUI,
                     record=RECORD
                     )
    PYB_CLIENT = env.getPyBulletClient() #returns client ID of currently running simulation

    # Initialize the LOGGER  
    LOGGER = Logger(logging_freq_hz=env.SIM_FREQ,
                    num_drones=1,
                    )

    # Initialize the CONTROLLERS
    CTRL_0 = Quad3D(env=env)
    CTRL_1 = Quad3D(env=env)
    CTRL_2 = Quad3D(env=env)
    CTRL_3 = Quad3D(env=env)
    CTRL_4 = Quad3D(env=env)
    CTRL_5 = Quad3D(env=env)

    # Initialize the action
    action = {}
    obs = env.reset()
    state = obs["0"]["state"]
    action["0"] = CTRL_0.compute_control(current_position=state[0:3],
                                         current_velocity=state[10:13],
                                         current_rpy=state[7:10],
                                         target_position=state[0:3],
                                         target_velocity=np.zeros(3),
                                         target_acceleration=np.zeros(3)
                                         )
    


    c = [1, 0.1, 0.9]
    state = obs["1"]["state"]
    action["1"] = CTRL_1.compute_control(current_position=state[0:3],
                                         current_velocity=state[10:13],
                                         current_rpy=state[7:10],
                                         target_position= c,
                                         target_velocity=np.zeros(3),
                                         target_acceleration=np.zeros(3)
                                         )

    c = [1, 0.1, 1.1]
    state = obs["2"]["state"]
    action["2"] = CTRL_2.compute_control(current_position=state[0:3],
                                         current_velocity=state[10:13],
                                         current_rpy=state[7:10],
                                         target_position= c,
                                         target_velocity=np.zeros(3),
                                         target_acceleration=np.zeros(3)
                                         )
    
    c = [0, 0.0, 2.5]
    state = obs["3"]["state"]
    action["3"] = CTRL_3.compute_control(current_position=state[0:3],
                                         current_velocity=state[10:13],
                                         current_rpy=state[7:10],
                                         target_position= c,
                                         target_velocity=np.zeros(3),
                                         target_acceleration=np.zeros(3)
                                         )

    c = [1, 0.1, 1.5]
    state = obs["4"]["state"]
    action["4"] = CTRL_4.compute_control(current_position=state[0:3],
                                         current_velocity=state[10:13],
                                         current_rpy=state[7:10],
                                         target_position= c,
                                         target_velocity=np.zeros(3),
                                         target_acceleration=np.zeros(3)
                                         )

    c = [1, 0.1, 1.7]
    state = obs["5"]["state"]
    action["5"] = CTRL_5.compute_control(current_position=state[0:3],
                                         current_velocity=state[10:13],
                                         current_rpy=state[7:10],
                                         target_position= c,
                                         target_velocity=np.zeros(3),
                                         target_acceleration=np.zeros(3)
                                         )

    # Initialize the target trajectory   
    TARGET_POSITION = np.array([[0, 0, 0.5+0.02*i] for i in range(DURATION*env.SIM_FREQ)])
    TARGET_VELOCITY = np.zeros([DURATION * env.SIM_FREQ, 3])
    TARGET_ACCELERATION = np.zeros([DURATION * env.SIM_FREQ, 3])

    # Derive the target trajectory to obtain target velocities and accelerations
    TARGET_VELOCITY[1:, :] = (TARGET_POSITION[1:, :] - TARGET_POSITION[0:-1, :]) / env.SIM_FREQ
    TARGET_ACCELERATION[1:, :] = (TARGET_VELOCITY[1:, :] - TARGET_VELOCITY[0:-1, :]) / env.SIM_FREQ

    TARGET_POSITION_1 = np.array([[1, 0.1, 0.9] for i in range(DURATION*env.SIM_FREQ)])
    TARGET_VELOCITY_1 = np.zeros([DURATION * env.SIM_FREQ, 3])
    TARGET_ACCELERATION_1 = np.zeros([DURATION * env.SIM_FREQ, 3])

    # Derive the target trajectory to obtain target velocities and accelerations
    TARGET_VELOCITY_1[1:, :] = (TARGET_POSITION_1[1:, :] - TARGET_POSITION_1[0:-1, :]) / env.SIM_FREQ
    TARGET_ACCELERATION_1[1:, :] = (TARGET_VELOCITY_1[1:, :] - TARGET_VELOCITY_1[0:-1, :]) / env.SIM_FREQ

    state_1 = obs["1"]["state"]

    TARGET_POSITION_2 = np.array([[1, .10, 1.1] for i in range(DURATION*env.SIM_FREQ)])
    TARGET_VELOCITY_2 = np.zeros([DURATION * env.SIM_FREQ, 3])
    TARGET_ACCELERATION_2 = np.zeros([DURATION * env.SIM_FREQ, 3])

    # Derive the target trajectory to obtain target velocities and accelerations
    TARGET_VELOCITY_2[1:, :] = (TARGET_POSITION_2[1:, :] - TARGET_POSITION_2[0:-1, :]) / env.SIM_FREQ
    TARGET_ACCELERATION_2[1:, :] = (TARGET_VELOCITY_2[1:, :] - TARGET_VELOCITY_2[0:-1, :]) / env.SIM_FREQ

    state_2 = obs["2"]["state"]

    TARGET_POSITION_3 = np.array([[0, 0.0, 2.5] for i in range(DURATION*env.SIM_FREQ)])
    TARGET_VELOCITY_3 = np.zeros([DURATION * env.SIM_FREQ, 3])
    TARGET_ACCELERATION_3 = np.zeros([DURATION * env.SIM_FREQ, 3])

    # Derive the target trajectory to obtain target velocities and accelerations
    TARGET_VELOCITY_3[1:, :] = (TARGET_POSITION_3[1:, :] - TARGET_POSITION_3[0:-1, :]) / env.SIM_FREQ
    TARGET_ACCELERATION_3[1:, :] = (TARGET_VELOCITY_3[1:, :] - TARGET_VELOCITY_3[0:-1, :]) / env.SIM_FREQ

    state_3 = obs["3"]["state"]

    TARGET_POSITION_4 = np.array([[1, .1, 1.5] for i in range(DURATION*env.SIM_FREQ)])
    TARGET_VELOCITY_4 = np.zeros([DURATION * env.SIM_FREQ, 3])
    TARGET_ACCELERATION_4 = np.zeros([DURATION * env.SIM_FREQ, 3])

    # Derive the target trajectory to obtain target velocities and accelerations
    TARGET_VELOCITY_4[1:, :] = (TARGET_POSITION_4[1:, :] - TARGET_POSITION_4[0:-1, :]) / env.SIM_FREQ
    TARGET_ACCELERATION_4[1:, :] = (TARGET_VELOCITY_4[1:, :] - TARGET_VELOCITY_4[0:-1, :]) / env.SIM_FREQ

    state_4 = obs["4"]["state"]

    TARGET_POSITION_5 = np.array([[1, .1, 1.7] for i in range(DURATION*env.SIM_FREQ)])
    TARGET_VELOCITY_5 = np.zeros([DURATION * env.SIM_FREQ, 3])
    TARGET_ACCELERATION_5 = np.zeros([DURATION * env.SIM_FREQ, 3])

    # Derive the target trajectory to obtain target velocities and accelerations
    TARGET_VELOCITY_5[1:, :] = (TARGET_POSITION_5[1:, :] - TARGET_POSITION_5[0:-1, :]) / env.SIM_FREQ
    TARGET_ACCELERATION_5[1:, :] = (TARGET_VELOCITY_5[1:, :] - TARGET_VELOCITY_5[0:-1, :]) / env.SIM_FREQ

    state_5 = obs["5"]["state"]

    # Run the simulation    
    START = time.time()
    for i in range(0, DURATION*env.SIM_FREQ):

        # Secret control performance booster
        # if i/env.SIM_FREQ>3 and i%30==0 and i/env.SIM_FREQ<10: p.loadURDF("duck_vhacd.urdf", [random.gauss(0, 0.3), random.gauss(0, 0.3), 3], p.getQuaternionFromEuler([random.randint(0, 360),random.randint(0, 360),random.randint(0, 360)]), physicsClientId=PYB_CLIENT)
        start = time.time()

        # Step the simulation 
        obs, _, _, _ = env.step(action)
        if (i%20) == 0:
            obs_1 = state_1
            obs_2 = state_2
            obs_3 = state_3
            obs_4 = state_4
            obs_5 = state_5

        # Compute control for drone 0
        state = obs["0"]["state"]
        action["0"] = CTRL_0.compute_control(current_position=state[0:3],
                                             current_velocity=state[10:13],
                                             current_rpy=state[7:10],
                                             target_position=TARGET_POSITION[i, :],
                                             target_velocity=TARGET_VELOCITY[i, :],
                                             target_acceleration=TARGET_ACCELERATION[i, :]
                                             )
                                             
        # Controller 
        # QP Parameters
        gamma = 1
        qp = QP_Controller_Drone(gamma)
        u_ref = action["0"]
        f_u_ref = kf * np.square(u_ref)
        qp.set_reference_control(f_u_ref)
        # print(obs_1[0:3], obs_1[10:13])
        qp.setup_QP(bot, obs_3[0:3], obs_3[10:13])
               

        # Simulation
        # Solve QP
        state_of_QP, value_of_h = qp.solve_QP(bot)
        
        # Bot Kinematics
        u_star = qp.get_optimal_control()

        action["0"] = u_star

        bot.update_state(state[0:3], state[10:13], state[7:10], 1/env.SIM_FREQ)

        # Compute control for drone 1
        state_1 = obs["1"]["state"]
        action["1"] = CTRL_1.compute_control(current_position=state_1[0:3],
                                             current_velocity=state_1[10:13],
                                             current_rpy=state_1[7:10],
                                             target_position=TARGET_POSITION_1[i, :],
                                             target_velocity=TARGET_VELOCITY_1[i, :],
                                             target_acceleration=TARGET_ACCELERATION_1[i, :]
                                             )

        # Compute control for drone 2
        state_2 = obs["2"]["state"]
        action["2"] = CTRL_2.compute_control(current_position=state_2[0:3],
                                             current_velocity=state_2[10:13],
                                             current_rpy=state_2[7:10],
                                             target_position=TARGET_POSITION_2[i, :],
                                             target_velocity=TARGET_VELOCITY_2[i, :],
                                             target_acceleration=TARGET_ACCELERATION_2[i, :]
                                             )
        
        # Compute control for drone 3
        state_3 = obs["3"]["state"]
        action["3"] = CTRL_3.compute_control(current_position=state_3[0:3],
                                             current_velocity=state_3[10:13],
                                             current_rpy=state_3[7:10],
                                             target_position=TARGET_POSITION_3[i, :],
                                             target_velocity=TARGET_VELOCITY_3[i, :],
                                             target_acceleration=TARGET_ACCELERATION_3[i, :]
                                             )

        # Compute control for drone 4
        state_4 = obs["4"]["state"]
        action["4"] = CTRL_4.compute_control(current_position=state_4[0:3],
                                             current_velocity=state_4[10:13],
                                             current_rpy=state_4[7:10],
                                             target_position=TARGET_POSITION_4[i, :],
                                             target_velocity=TARGET_VELOCITY_4[i, :],
                                             target_acceleration=TARGET_ACCELERATION_4[i, :]
                                             )

        # Compute control for drone 5
        state_5 = obs["5"]["state"]
        action["5"] = CTRL_5.compute_control(current_position=state_5[0:3],
                                             current_velocity=state_5[10:13],
                                             current_rpy=state_5[7:10],
                                             target_position=TARGET_POSITION_5[i, :],
                                             target_velocity=TARGET_VELOCITY_5[i, :],
                                             target_acceleration=TARGET_ACCELERATION_5[i, :]
                                             )
            
        # print('timestep 3:', time.time()-start)
        # Log drone 0 
        LOGGER.log(drone=0, timestamp=i/env.SIM_FREQ, state=state)

        # Printout
        if i%env.SIM_FREQ == 0:
            env.render()

        # Sync the simulation
        if GUI:
            sync(i, START, env.TIMESTEP)

    # Close the environment 
    env.close()

    # Save the simulation results 
    LOGGER.save()

    # Save the simulation results 
    LOGGER.save_as_csv()

    # Plot the simulation results 
    LOGGER.plot()
