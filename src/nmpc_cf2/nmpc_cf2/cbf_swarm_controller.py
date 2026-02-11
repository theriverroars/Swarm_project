#!/usr/bin/env python
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import pyquaternion

CBF_CONTROLLER_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "CBF_controller")
)
if CBF_CONTROLLER_PATH not in sys.path:
    sys.path.insert(0, CBF_CONTROLLER_PATH)

from controllers_hub import CBFQPControllerDrone, NominalPIDControllerDrone
from quadrotor_info import Quadrotor, quadrotor_params, GRAVITY_ACC, SIM_TIMESTEP


def quaternion_to_euler(q):
    q = pyquaternion.Quaternion(w=q[0], x=q[1], y=q[2], z=q[3])
    yaw, pitch, roll = q.yaw_pitch_roll
    return [roll, pitch, yaw]


class CBFMultiController:
    def __init__(self, num_drones: int, cbf_gamma: float = 1.0):
        self.num_drones = num_drones
        self.cbf_gamma = cbf_gamma

        self.cbf_controller = CBFQPControllerDrone(
            acceleration_gravity=GRAVITY_ACC,
            CBF_gamma=self.cbf_gamma,
        )
        self.pid_controller = NominalPIDControllerDrone(
            acceleration_gravity=GRAVITY_ACC,
            time_step_size=SIM_TIMESTEP,
        )

        self.agents: List[Quadrotor] = []
        self._has_state: List[bool] = []
        for idx in range(num_drones):
            agent = Quadrotor(
                id=idx,
                mass=0.034,
                inertia_matrix=np.diag([1.4e-5, 1.4e-5, 2.17e-5]),
                **quadrotor_params,
            )
            self.agents.append(agent)
            self._has_state.append(False)

    def has_state(self, index: int) -> bool:
        return self._has_state[index]

    def update_agent_state(
        self,
        index: int,
        position: np.ndarray,
        velocity: np.ndarray,
        quaternion: np.ndarray,
        angular_velocity: np.ndarray,
    ) -> None:
        agent = self.agents[index]
        agent.pos_states = np.array(position)
        agent.d_pos_states = np.array(velocity)
        agent.ang_states = np.array(quaternion_to_euler(quaternion))
        agent.d_ang_states = np.array(angular_velocity)
        self._has_state[index] = True

    def update_agent_from_full_pose(self, index: int, msg) -> None:
        position = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        )
        velocity = np.array(
            [msg.velocity.linear.x, msg.velocity.linear.y, msg.velocity.linear.z]
        )
        quaternion = np.array(
            [
                msg.pose.orientation.w,
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
            ]
        )
        angular_velocity = np.array(
            [msg.velocity.angular.x, msg.velocity.angular.y, msg.velocity.angular.z]
        )
        self.update_agent_state(index, position, velocity, quaternion, angular_velocity)

    def compute_priority(self, agent, hoop_center, hoop_direction, weights, cbf_controller, agents_list, pass_through_hoop):
        # Compute CBF values for this agent against all others
        obstacle_positions = np.asarray([a.pos_states for a in agents_list if a != agent])
        obstacle_velocities = np.asarray([a.d_pos_states for a in agents_list if a != agent])
        h_values = cbf_controller.CBF_value(
            agent=agent,
            obstacle_positions=obstacle_positions,
            obstacle_velocities=obstacle_velocities,
        )
        distances = np.linalg.norm(hoop_center - agent.pos_states)
        velocities_towards_hoop = np.dot((hoop_center - agent.pos_states), agent.d_pos_states) / (distances + 1e-6)
        if not pass_through_hoop:
            weights = [0, 0, weights[2]]
        priority_value = weights[0]*distances - weights[1]*velocities_towards_hoop - weights[2]*np.sum(h_values)
        return priority_value

    def compute_commands(
        self,
        target_positions: np.ndarray,
        target_velocities: np.ndarray,
        target_accelerations: np.ndarray,
        hoop_center: np.ndarray = None,
        hoop_direction: np.ndarray = None,
        weights: list = None,
    ) -> List[Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]]]:
        # Default weights if not provided
        if weights is None:
            weights = [0.5, 0.2, 0.3]
        if hoop_center is None:
            hoop_center = np.zeros(3)
        if hoop_direction is None:
            hoop_direction = np.array([0., 1., 0.])

        commands = []
        num_agents = self.num_drones
        agents_list = self.agents
        eps = 0.001

        # Compute initial sign for pass_through_hoop logic
        initial_sign = [np.sign(np.dot(hoop_center - agent.pos_states, hoop_direction)) for agent in agents_list]

        # Compute priorities for all agents
        priority_list = []
        for j, agent in enumerate(agents_list):
            to_hoop_vec = hoop_center - agent.pos_states
            pass_through_hoop = 1 if np.sign(np.dot(to_hoop_vec, hoop_direction)) == initial_sign[j] else 0
            priority_value = self.compute_priority(agent, hoop_center, hoop_direction, weights, self.cbf_controller, agents_list, pass_through_hoop)
            priority_list.append((j, priority_value))
        # Sort by priority value (lower = higher priority)
        priority_list = sorted(priority_list, key=lambda x: x[1])

        # Assign ranks (equal within threshold)
        ranked_list = []
        current_rank = 1
        prev_value = None
        for idx, (agent_id, p_val) in enumerate(priority_list):
            if prev_value is None:
                ranked_list.append((agent_id, p_val, current_rank))
                prev_value = p_val
                continue
            if abs(p_val - prev_value) <= eps:
                ranked_list.append((agent_id, p_val, current_rank))
            else:
                current_rank += 1
                ranked_list.append((agent_id, p_val, current_rank))
            prev_value = p_val

        # For each agent, compute command using dynamic priority
        for i in range(num_agents):
            if not self._has_state[i]:
                commands.append(None)
                continue
            agent = agents_list[i]
            target_pos = (
                target_positions[i]
                if len(target_positions.shape) > 1
                else target_positions
            )
            target_vel = (
                target_velocities[i]
                if len(target_velocities.shape) > 1
                else target_velocities
            )
            target_acc = (
                target_accelerations[i]
                if len(target_accelerations.shape) > 1
                else target_accelerations
            )
            ref_propellers_rpm = self.pid_controller.compute_PID_control(
                agent=agent,
                target_linear_position=target_pos,
                target_linear_velocity=target_vel,
                target_linear_acceleration=target_acc,
            )
            ref_thrusts = self.pid_controller.thrust_from_rpm(ref_propellers_rpm)
            # Get current agent's rank
            current_rank = next(rank for agent_id, _, rank in ranked_list if agent_id == i)
            # Agents with higher priority (lower rank number) other than itself
            higher_priority_agents = [
                agent_id for agent_id, _, rank in ranked_list
                if rank <= current_rank and agent_id != i
            ]
            obstacle_pos = np.asarray([agents_list[k].pos_states for k in higher_priority_agents]) if higher_priority_agents else np.empty((0, 3))
            obstacle_vel = np.asarray([agents_list[k].d_pos_states for k in higher_priority_agents]) if higher_priority_agents else np.empty((0, 3))
            safety_thrust, safety_flag = self.cbf_controller.solve_QP(
                agent=agent,
                u_ref=ref_thrusts,
                obstacle_positions=obstacle_pos,
                obstacle_velocities=obstacle_vel,
            )
            if safety_flag:
                total_thrust = ref_thrusts + safety_thrust
            else:
                total_thrust = ref_thrusts
            _ = self.pid_controller.rpm_from_thrust(total_thrust)
            z_acc = (total_thrust[0] * self.pid_controller.agent.kf) / agent.m
            desired_accel = np.array([0, 0, z_acc])
            desired_yaw = agent.ang_states[2]
            desired_omega = agent.d_ang_states
            commands.append(
                (
                    agent.pos_states,
                    agent.d_pos_states,
                    desired_accel,
                    desired_yaw,
                    desired_omega,
                )
            )
        return commands
