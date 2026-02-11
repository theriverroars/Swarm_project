#!/usr/bin/env python

import rclpy
from rclpy.node import Node
from nmpc_cf2_interfaces.msg import FullPose
from geometry_msgs.msg import Vector3, Quaternion
from std_msgs.msg import String
from pathlib import Path

import numpy as np

from nmpc_cf2.nmpc_utils import (
    Quadrotor3D,
    Quadrotor3DMPC,
    get_reference_chunk,
    separate_variables,
)


class Emulate(Node):

    def __init__(self, t_array, ref_trajectory, ref_input, emulate_state):
        super().__init__("emulate")
        self.get_logger().info("Emulate node started")
        self.get_logger().info(f"Emulate state: {emulate_state}")

        self.t_array = t_array
        self.ref_trajectory = ref_trajectory
        self.ref_input = ref_input
        self.emulate_state = emulate_state

        if self.emulate_state is None:
            self.get_logger().info("Emulating all states")

        self.full_pose_sub = self.create_subscription(
            FullPose, "/cf231/full_pose", self.full_pose_callback, 1
        )

        self.quad = Quadrotor3D(emulate_state=self.emulate_state)

        ## * CALLBACK LISTS
        self._position_callback_list = ["vel", "quat", "a_rate"]
        self._velocity_callback_list = ["quat", "a_rate"]
        self._attitude_callback_list = ["a_rate"]
        self._a_rate_callback_list = []

        self.controller_init()

    def controller(self):
        # get reference trajectory chunk
        ref_traj_chunk, ref_u_chunk = get_reference_chunk(
            self.ref_trajectory,
            self.ref_input,
            self.quad.step_counter,
            self.n_mpc_node,
            1,
        )
        # Set the reference for the OCP
        model_ind = self.quad_mpc.set_reference(
            x_reference=separate_variables(ref_traj_chunk), u_reference=ref_u_chunk
        )

        w_opt, x_pred = self.quad_mpc.optimize(use_model=model_ind, return_x=True)
        ref_u = np.squeeze(np.array(w_opt[:4]))

        return self.quad_mpc.simulate(ref_u)

    def controller_init(self):
        self.q_diagonal = np.array(
            [10, 10, 10, 5, 5, 5, 0.5, 0.5, 0.5, 0.05, 0.05, 0.05]
        )
        self.r_diagonal = np.array([1, 1, 1, 1])

        self.q_mask = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

        self.quad = Quadrotor3D(self.emulate_state)

        ## MPC parameters
        self.t_horizon = 1.0
        self.n_mpc_node = 100
        optimisation_dt = self.t_horizon / self.n_mpc_node

        self.quad_mpc = Quadrotor3DMPC(
            quadrotor=self.quad,
            simulation_dt=0.01,
            n_mpc_node=self.n_mpc_node,
            t_horizon=self.t_horizon,
            q_cost=self.q_diagonal,
            r_cost=self.r_diagonal,
            q_mask=self.q_mask,
        )

    def full_pose_callback(self, msg: FullPose):
        # remove this after testing
        assert isinstance(msg, FullPose), "msg should be of type FullPose"

        # Create a map of all state types to their values from the message
        state_map = {
            "pos": np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]),
            "vel": np.array([msg.velocity.linear.x, msg.velocity.linear.y, msg.velocity.linear.z]),
            "quat": np.array([msg.pose.orientation.w, msg.pose.orientation.x, 
                              msg.pose.orientation.y, msg.pose.orientation.z]),
            "a_rate": np.array([msg.velocity.angular.x, msg.velocity.angular.y, msg.velocity.angular.z])
        }

        # Complete list of all state types
        all_state_types = ["pos", "vel", "quat", "a_rate"]
        
        # For each state type that should NOT be emulated (i.e., should be taken from the message)
        for state_type in all_state_types:
            if state_type not in self.emulate_state:
                # Update the state from the message
                if state_type == "pos":
                    self.quad.state.pos = state_map["pos"]
                elif state_type == "vel":
                    self.quad.state.vel = state_map["vel"]
                elif state_type == "quat":
                    self.quad.state.quat = state_map["quat"]
                elif state_type == "a_rate":
                    self.quad.state.a_rate = state_map["a_rate"]
