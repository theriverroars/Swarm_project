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

DURATION = 15
"""int: The duration of the simulation in seconds."""
GUI = True
"""bool: Whether to use PyBullet graphical interface."""
RECORD = False
"""bool: Whether to save a video under /files/videos. Requires ffmpeg"""

if __name__ == "__main__":

    # Create the environment
    env = CtrlAviary(num_drones=2,
                     drone_model=DroneModel.CF2P,
                     initial_xyzs=np.array([ [.0, 1, 1], [1., .0, 1.]]), #, [-.3, .0, 1.], [.3, .0, .15] 
                     gui=GUI,
                     record=RECORD
                     )
    PYB_CLIENT = env.getPyBulletClient()

    # Initialize the LOGGER  
    LOGGER = Logger(logging_freq_hz=env.SIM_FREQ,
                    num_drones=2,
                    )

    # Initialize the CONTROLLERS
    CTRL_0 = Quad3D(env=env)
    CTRL_1 = Quad3D(env=env)

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

    c = [1, 0, 1]
    state = obs["1"]["state"]
    action["1"] = CTRL_1.compute_control(current_position=state[0:3],
                                         current_velocity=state[10:13],
                                         current_rpy=state[7:10],
                                         target_position= c,
                                         target_velocity=np.zeros(3),
                                         target_acceleration=np.zeros(3)
                                         )

    # Initialize the target trajectory   
    TARGET_POSITION = np.array([[np.sin(0.002*i), np.cos(0.002*i), 1] for i in range(DURATION*env.SIM_FREQ)])
    TARGET_VELOCITY = np.zeros([DURATION * env.SIM_FREQ, 3])
    TARGET_ACCELERATION = np.zeros([DURATION * env.SIM_FREQ, 3])

    # Derive the target trajectory to obtain target velocities and accelerations
    TARGET_VELOCITY[1:, :] = (TARGET_POSITION[1:, :] - TARGET_POSITION[0:-1, :]) / env.SIM_FREQ
    TARGET_ACCELERATION[1:, :] = (TARGET_VELOCITY[1:, :] - TARGET_VELOCITY[0:-1, :]) / env.SIM_FREQ

    TARGET_POSITION_1 = np.array([[np.cos(0.002*i), np.sin(0.002*i), 1] for i in range(DURATION*env.SIM_FREQ)])
    TARGET_VELOCITY_1 = np.zeros([DURATION * env.SIM_FREQ, 3])
    TARGET_ACCELERATION_1 = np.zeros([DURATION * env.SIM_FREQ, 3])

    # Derive the target trajectory to obtain target velocities and accelerations
    TARGET_VELOCITY_1[1:, :] = (TARGET_POSITION_1[1:, :] - TARGET_POSITION_1[0:-1, :]) / env.SIM_FREQ
    TARGET_ACCELERATION_1[1:, :] = (TARGET_VELOCITY_1[1:, :] - TARGET_VELOCITY_1[0:-1, :]) / env.SIM_FREQ

    state_1 = obs["1"]["state"]

    # Run the simulation    
    START = time.time()
    for i in range(0, DURATION*env.SIM_FREQ):

        #  Secret control performance booster
        # if i/env.SIM_FREQ>3 and i%30==0 and i/env.SIM_FREQ<10: p.loadURDF("duck_vhacd.urdf", [random.gauss(0, 0.3), random.gauss(0, 0.3), 3], p.getQuaternionFromEuler([random.randint(0, 360),random.randint(0, 360),random.randint(0, 360)]), physicsClientId=PYB_CLIENT)
        start = time.time()

        # Step the simulation 
        obs, _, _, _ = env.step(action)
        if (i%60) == 0:
            obs_1 = state_1
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
        qp.setup_QP(bot, obs_1[0:3], obs_1[10:13])
        

        # Simulation
        # Solve QP
        state_of_QP, value_of_h = qp.solve_QP(bot)
        # print('u_ref',u_ref)
        
        # Bot Kinematics
        u_star = qp.get_optimal_control()
        # print('u_star',u_star)

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

    # Plot the simulation results 
    LOGGER.plot()
