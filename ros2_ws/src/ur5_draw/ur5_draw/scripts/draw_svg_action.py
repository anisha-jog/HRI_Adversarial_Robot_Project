import sys
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from ur5_draw_actions.action import DrawStroke
from geometry_msgs.msg import Pose
from tf2_ros import TransformListener, Buffer
import numpy as np

import moveit_commander
import moveit_msgs.msg



class DrawSVGAction(Node):
    def __init__(self):
        super().__init__('draw_svg_action')

        # Create callback group for action server
        self.callback_group = ReentrantCallbackGroup()

        # Create action server
        self._action_server = ActionServer(
            self,
            DrawStroke,
            'draw_svg',
            self.execute_callback,
            callback_group=self.callback_group
        )

        # Publisher for robot commands (you'll need to modify this based on your UR5 setup)
        self.robot_command_pub = self.create_publisher(
            Pose,
            '/ur5/command_pose',
            10
        )

        # TF listener for coordinate transformations
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.get_logger().info('Draw SVG Action Server has started')

    async def execute_callback(self, goal_handle):
        self.get_logger().info('Executing drawing action...')

        feedback_msg = DrawSVG.Feedback()
        result = DrawSVG.Result()

        try:
            # Get coordinates from goal
            coords = goal_handle.request.coordinates

            # Process each coordinate
            for i, coord in enumerate(coords):
                # Convert image coordinates to robot coordinates
                robot_pose = self.image_to_robot_coords(coord.x, coord.y)

                # Send command to robot
                self.robot_command_pub.publish(robot_pose)

                # Update feedback
                feedback_msg.current_point = i
                goal_handle.publish_feedback(feedback_msg)

                # Add small delay for robot movement
                await rclpy.sleep(0.1)

            goal_handle.succeed()
            result.success = True
            return result

        except Exception as e:
            self.get_logger().error(f'Failed to execute drawing action: {str(e)}')
            goal_handle.abort()
            result.success = False
            return result

    def image_to_robot_coords(self, image_x, image_y):
        # TODO: Implement coordinate transformation from image frame to robot frame
        # This will depend on your specific setup and calibration
        robot_pose = Pose()
        # Add transformation logic here
        return robot_pose

class DrawPointTest(Node):
    def __init__():
        super().__init__('draw_point')
        moveit_commander.roscpp_initialize


def main(args=None):
    rclpy.init(args=args)

    draw_svg_action = DrawSVGAction()

    try:
        rclpy.spin(draw_svg_action)
    except KeyboardInterrupt:
        pass
    finally:
        draw_svg_action.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()