#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose

from ur_draw_cmake.srv import MoveToPose



class MoveClient(Node):
    def __init__(self):
        super().__init__('move_group_python_interface')
        self.get_logger().info("Initializing MoveClient...")
        self.client = self.create_client(MoveToPose, 'moveit_to_pose')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        self.request = MoveToPose.Request()

    def send_request(self, pose):
        self.request.target_pose = pose
        self.get_logger().info("Sending request to move to the target pose...")
        return self.client.call(self.request)




def main(args=None):
    rclpy.init(args=args)

    move_client = MoveClient()

    # --- Define the target pose here ---
    # Position in meters, orientation as a quaternion
    target_pose = Pose()
    target_pose.position.x = 0.28  # Example X position
    target_pose.position.y = -0.2  # Example Y position
    target_pose.position.z = 0.5  # Example Z position
    target_pose.orientation.w = 1.0 # Example orientation (identity quaternion, no rotation)

    move_client.send_request(target_pose)

    rclpy.spin_once(move_client)
    move_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()