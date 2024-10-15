#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from linkattacher_msgs.srv import AttachLink, DetachLink
from geometry_msgs.msg import Pose
from pymoveit2 import MoveIt2
from pymoveit2.robots import ur5
from threading import Thread

class RobotArmController(Node):
    def __init__(self):
        super().__init__('robot_arm_controller')

        # Create ROS2 service clients for attaching and detaching the gripper
        self.attach_service = self.create_client(AttachLink, '/GripperMagnetON')
        self.detach_service = self.create_client(DetachLink, '/GripperMagnetOFF')

        # Wait for services to be available
        while not self.attach_service.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('AttachLink service not available, waiting again...')
        while not self.detach_service.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('DetachLink service not available, waiting again...')

        # Create MoveIt 2 interface for controlling the UR5 robot
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=ur5.joint_names(),
            base_link_name=ur5.base_link_name(),
            end_effector_name=ur5.end_effector_name(),
            group_name=ur5.MOVE_GROUP_ARM,
        )

        # Start the executor in a background thread for handling callbacks
        executor = rclpy.executors.MultiThreadedExecutor(2)
        executor.add_node(self)
        executor_thread = Thread(target=executor.spin, daemon=True)
        executor_thread.start()

        self.move_arm_to_positions()

    def move_arm_to_positions(self):
        # Predefined positions (w.r.t BaseLink)
        P1 = [0.20, -0.47, 0.65]
        P2 = [0.75, 0.49, -0.05]
        P3 = [0.75, -0.23, -0.05]
        D = [-0.69, 0.10, 0.44]

        # Move to P1, Attach the box, then move to D
        self.move_arm(P1)
        self.attach_box('obj_1')
        self.move_arm(D)
        self.detach_box('obj_1')

        # Move to P2, Detach the box, then move to D
        self.move_arm(P2)
        self.attach_box('obj_3')
        self.move_arm(D)
        self.detach_box('obj_3')

        # Move to P3, Detach the box, then move to D
        self.move_arm(P3)
        self.attach_box('obj_49')
        self.move_arm(D)
        self.detach_box('obj_49')

    def move_arm(self, position):
        # Function to move the robot to the desired position using MoveIt2
        self.get_logger().info(f"Moving arm to position: {position}")
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = position
        quat_xyzw = [0.0, 0.0, 0.0, 1.0]  # Default orientation
        self.moveit2.move_to_pose(position=position, quat_xyzw=quat_xyzw, cartesian=False)
        self.moveit2.wait_until_executed()

    def attach_box(self, box_name):
        # Function to call the service to attach the box
        self.get_logger().info(f"Attaching box: {box_name}")
        req = AttachLink.Request()
        req.model1_name = box_name
        req.link1_name = 'link'
        req.model2_name = 'ur5'
        req.link2_name = 'wrist_3_link'
        self.attach_service.call_async(req)

    def detach_box(self, box_name):
        # Function to call the service to detach the box
        self.get_logger().info(f"Detaching box: {box_name}")
        req = DetachLink.Request()
        req.model1_name = box_name
        req.link1_name = 'link'
        req.model2_name = 'ur5'
        req.link2_name = 'wrist_3_link'
        self.detach_service.call_async(req)


def main(args=None):
    rclpy.init(args=args)
    robot_arm_controller = RobotArmController()
    rclpy.spin(robot_arm_controller)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
