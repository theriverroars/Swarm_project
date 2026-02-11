#!/usr/bin/env python

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import time

from geometry_msgs.msg import Pose
from motion_capture_tracking_interfaces.msg import NamedPoseArray
from nmpc_cf2_interfaces.msg import FullPose, FullPoseArray
from crazyflie_interfaces.msg import LogDataGeneric

import numpy as np


class SingleMarker2FullPose(Node):
    def __init__(self):

        super().__init__(
            "singlemarker2fullpose",
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )

        self._params = self._param_to_dict(self._parameters)
        self.get_logger().info(f"SingleMarker2FullPose received params: {self._params}")
        self.dt = 1 / self._params["mocap_deadline"]
        self.crazyflies = self._params["crazyflies"]

        mocap_pose_topic = "/poses"
        mocap_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            deadline=Duration(seconds=0, nanoseconds=1e9 / 100.0),
        )

        self.full_pose_callback_group = (
            ReentrantCallbackGroup()
        )  ## Not sure if this is the right way to do it
        self.timer_callback_group = MutuallyExclusiveCallbackGroup()

        self.latest_pose = {}
        self.old_pose = {cf: FullPose() for cf in self.crazyflies}
        self.latest_velocity = {}
        self.latest_quaternion = {}
        self.latest_gyroscope = {}

        self.position_sub = {}
        self.velocity_sub = {}
        self.quaternion_sub = {}
        self.gyroscope_sub = {}

        # it is assumed that each crazyflie is publishing velocity, compressed quaternion, and angular velocity
        # on the topics /<cf>/velocity, /<cf>/quaternion, and /<cf>/gyroscope, respectively
        # create subscribers for the crazyflies to receive the latest velocity quaternion, and angular velocity
        # the position is subscribed from the motion capture system, which is assumed to be publishing on the topic /poses
        # the name of the tracked crazyflie is in the /poses/name field, and the corresponding pose is in the /poses/pose field
        self.position_sub = self.create_subscription(
            NamedPoseArray,
            "/poses",
            self.position_callback,
            qos_profile=mocap_qos_profile,
            callback_group=self.full_pose_callback_group,
        )
        for cf in self.crazyflies:
            # Assumption: the array of poses in /poses is ordered the same as the array of crazyflies in the params
            self.velocity_sub[cf] = self.create_subscription(
                LogDataGeneric,
                f"/{cf}/velocity",
                lambda msg: self.velocity_callback(msg, cf),
                1,
                callback_group=self.full_pose_callback_group,
            )

            self.quaternion_sub[cf] = self.create_subscription(
                LogDataGeneric,
                f"/{cf}/quaternion",
                lambda msg: self.quaternion_callback(msg, cf),
                1,
                callback_group=self.full_pose_callback_group,
            )

            self.gyroscope_sub[cf] = self.create_subscription(
                LogDataGeneric,
                f"/{cf}/gyroscope",
                lambda msg: self.gyro_callback(msg, cf),
                1,
                callback_group=self.full_pose_callback_group,
            )

        # create a new publisher for each of the crazyflies present in the params
        self.cf_publishers = {}
        for cf in self.crazyflies:
            self.cf_publishers[cf] = self.create_publisher(
                FullPose, f"/{cf}/full_pose", 1
            )

        timer_period = 0.01
        self.timer = self.create_timer(
            timer_period,
            self.publish_full_pose,
            callback_group=self.timer_callback_group,
        )
        self.i = 0

    def _param_to_dict(self, param_ros):
        """
        Turn ROS 2 parameters from the node into a dict
        """
        tree = {}
        for item in param_ros:
            t = tree
            for part in item.split("."):
                if part == item.split(".")[-1]:
                    t = t.setdefault(part, param_ros[item].value)
                else:
                    t = t.setdefault(part, {})
        return tree

    def position_callback(self, msg: NamedPoseArray):
        # Assumption: the array of poses in /poses is ordered the same as the array of crazyflies in the params
        # TODO: check this assumption, both with the source code and experimentally
        for i in range(len(msg.poses)):  # no. of objects detected
            for cf in self.crazyflies:  # no. of crazyflies
                if msg.poses[i].name == cf:

                    self.latest_velocity[cf] = (
                        np.array(
                            [
                                (
                                    msg.poses[i].pose.position.x
                                    - self.old_pose[cf].pose.position.x
                                ),
                                (
                                    msg.poses[i].pose.position.y
                                    - self.old_pose[cf].pose.position.y
                                ),
                                (
                                    msg.poses[i].pose.position.z
                                    - self.old_pose[cf].pose.position.z
                                ),
                            ]
                        )
                        / 0.01
                    )

                    # self.latest_velocity[cf] = (
                    #     msg.poses[i].pose.position - self.old_pose[cf]
                    # ) / 0.01
                    self.latest_pose[cf] = msg.poses[
                        i
                    ].pose.position  # type: geometry_msgs.msg.Point

                    self.old_pose[cf] = msg.poses[i]

                    # self.get_logger().info(f"Received position for {cf}: {self.latest_pose[cf]}")

        # self.publish_full_pose() # made redundant by the timer

    def velocity_callback(self, msg: LogDataGeneric, cf: str):
        # Assumption: the velocity is being published by the crazyflie server.
        # The velocity is assumed to be in the /<cf>/velocity topic

        self.latest_velocity[cf] = msg.values
        # self.publish_full_pose() # made redundant by the timer

        # log the msg received
        # self.get_logger().info(f"Received velocity for {cf}: {msg.values}")

    def gyro_callback(self, msg: LogDataGeneric, cf: str):
        # Assumption: the gyroscope is being published by the crazyflie server.
        # The gyroscope is assumed to be in the /<cf>/gyroscope topic

        # convert the gyroscope values to radians/sec
        msg.values = [x * np.pi / 180 for x in msg.values]

        self.latest_gyroscope[cf] = msg.values
        # self.publish_full_pose() # made redundant by the timer
        # log the msg received
        # self.get_logger().info(f"Received gyroscope for {cf}: {msg.values}")

    def quaternion_callback(self, msg: LogDataGeneric, cf: str):
        # Assumption: the quaternion is being published by the crazyflie server.
        # The quaternion is assumed to be in the /<cf>/quaternion topic
        # NOTE: the quaternion is compressed and is of the type uint32

        self.latest_quaternion[cf] = msg.values
        # self.publish_full_pose() # made redundant by the timer

        # log the msg received
        # self.get_logger().info(f"Received quaternion for {cf}: {self.latest_quaternion[cf]}")

    def publish_full_pose(self):
        # send full pose of each crazyflie

        for cf in self.crazyflies:
            if (
                cf in self.latest_pose
                and cf in self.latest_velocity
                and cf in self.latest_gyroscope
                and cf in self.latest_quaternion
            ):
                msg = FullPose()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = "world"
                msg.name = cf
                msg.pose.position = self.latest_pose[cf]

                # NOTE: The quaternion coming from the decompression function is in the form [x, y, z, w]
                msg.pose.orientation.x = self.latest_quaternion[cf][0]
                msg.pose.orientation.y = self.latest_quaternion[cf][1]
                msg.pose.orientation.z = self.latest_quaternion[cf][2]
                msg.pose.orientation.w = self.latest_quaternion[cf][3]

                msg.velocity.linear.x = self.latest_velocity[cf][0]
                msg.velocity.linear.y = self.latest_velocity[cf][1]
                msg.velocity.linear.z = self.latest_velocity[cf][2]

                msg.velocity.angular.x = self.latest_gyroscope[cf][0]
                msg.velocity.angular.y = self.latest_gyroscope[cf][1]
                msg.velocity.angular.z = self.latest_gyroscope[cf][2]

                self.cf_publishers[cf].publish(msg)

                self.i += 1
            else:
                self.get_logger().info(f"Did not receive position for {cf}")

    @staticmethod
    def decompress_quaternion(comp):
        """Decompress a quaternion

        see quatcompress.h in the firmware for definitions

        Args:
            comp int: A 32-bit number

        Returns:
            np array: q = [x, y, z, w]
        """
        q = np.zeros(4)
        mask = (1 << 9) - 1
        i_largest = comp >> 30
        sum_squares = 0
        for i in range(3, -1, -1):
            if i != i_largest:
                mag = comp & mask
                negbit = (comp >> 9) & 0x1
                comp = comp >> 10
                q[i] = mag / mask / np.sqrt(2)
                if negbit == 1:
                    q[i] = -q[i]
                sum_squares += q[i] * q[i]
        q[i_largest] = np.sqrt(1.0 - sum_squares)

        return q


def main(args=None):
    rclpy.init(args=args)

    executor = MultiThreadedExecutor(num_threads=2)

    singlemarker2fullpose = SingleMarker2FullPose()
    executor.add_node(singlemarker2fullpose)
    executor.spin()
    executor.shutdown()
    singlemarker2fullpose.destroy_node()
    rclpy.shutdown()