#!/usr/bin/env python

import rclpy
from rclpy.node import Node

from nmpc_cf2_interfaces.msg import FullPose

from nmpc_cf2.cbf_swarm_controller import CBFMultiController


class CBFEmulate(Node):
    def __init__(self, cf_names, controller: CBFMultiController, hoop_center=None, hoop_direction=None):
        super().__init__("cbf_emulate")
        self.cf_names = list(cf_names)
        self.controller = controller
        self._subscriptions = []
        self.hoop_center = hoop_center if hoop_center is not None else np.array([0.0, 1.0, 1.0])
        self.hoop_direction = hoop_direction if hoop_direction is not None else np.array([0.0, 1.0, 0.0])
        for idx, cf_name in enumerate(self.cf_names):
            topic = f"/{cf_name}/full_pose"
            sub = self.create_subscription(
                FullPose,
                topic,
                lambda msg, i=idx: self.full_pose_callback(msg, i),
                1,
            )
            self._subscriptions.append(sub)
        self.get_logger().info(
            f"CBFEmulate subscribed to {len(self.cf_names)} full_pose topics"
        )

    def full_pose_callback(self, msg: FullPose, index: int):
        self.controller.update_agent_from_full_pose(index, msg)

    def has_all_states(self) -> bool:
        return all(self.controller.has_state(i) for i in range(self.controller.num_drones))

    def controller_step(self, target_positions, target_velocities, target_accelerations):
        return self.controller.compute_commands(
            target_positions,
            target_velocities,
            target_accelerations,
            hoop_center=self.hoop_center,
            hoop_direction=self.hoop_direction
        )
