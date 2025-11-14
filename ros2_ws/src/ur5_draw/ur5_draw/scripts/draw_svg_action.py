#!/usr/bin/env python3
import sys
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import Pose, Point, PoseStamped, Quaternion
from visualization_msgs.msg import Marker
# from tf2_ros import TransformListener, Buffer, TransformException
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_geometry_msgs import do_transform_pose
from copy import deepcopy
import numpy as np

from ur_draw_cmake.srv import DrawStroke

PEN_HEIGHT = .05
HOME_POSE = Pose(position=Point(x=0.20, y=0.0, z=0.5),orientation=Quaternion(w=.707,x=-.707))
TEST_STROKE = [
    (0.1, 0.1),
    (0.1, 0.2),
    (0.2, 0.2),
    (0.2, 0.1),
    (0.1, 0.1),
]

class DrawSVG(Node):
    def __init__(self):
        super().__init__('draw_svg')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.client = self.create_client(DrawStroke, 'moveit_draw_stroke')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self.request = DrawStroke.Request()

        self.image_to_world_t = self.get_img_transform('image_frame', 'world')
        self.get_logger().info('Transform successfully retrieved and cached.')

        self.img_viz_pub = self.create_publisher(Marker, 'image_viz', 10)
        self.img_viz = Marker()
        self.img_viz.header.frame_id = 'world'
        self.img_viz.type = Marker.LINE_STRIP
        self.img_viz.scale.x = 0.002
        self.img_viz.color.a = 1.0
        self.img_viz.color.r = 1.0
        self.img_viz.points = []

    def get_img_transform(self, source_frame, target_frame):
        wait_duration_sec = 5.0
        start_time = self.get_clock().now()
        transform_available = False

        while rclpy.ok() and (self.get_clock().now() - start_time).nanoseconds / 1e9 < wait_duration_sec:
            if self.tf_buffer.can_transform(target_frame, source_frame, rclpy.time.Time()):
                self.get_logger().info('Transform available!')
                transform_available = True
                break

            # Spin the node once to allow the TransformListener to process incoming TF messages
            rclpy.spin_once(self, timeout_sec=0.1)

        if not transform_available:
             self.get_logger().error(f"Failed to find transform between {source_frame} and {target_frame} within {wait_duration_sec}s.")
             # Exit the node setup gracefully or raise an error
             sys.exit(1)

        # Now that we know the transform exists, we can perform the lookup (with a short timeout)
        try:
            return self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.1) # Short timeout since we just confirmed it exists
            )

        except TransformException as ex:
             # Should not happen if can_transform passed, but good practice to catch it
             self.get_logger().error(f"Transform lookup failed after waiting: {ex}")
             sys.exit(1)

    def send_traj_request(self, poses):
        self.request.target_poses = poses
        self.get_logger().info("Sending request to move to the target pose...")
        return self.client.call_async(self.request)

    def draw_stroke_traj(self, points):
        pose_list = []
        for point in points:
            x, y = point
            img_pose = Pose()
            img_pose.position.x = x
            img_pose.position.y = y
            img_pose.position.z = PEN_HEIGHT

            img_pose.orientation.w = 0.707
            img_pose.orientation.x = -0.707
            img_pose.orientation.y = 0.0
            img_pose.orientation.z = 0.0

            world_pose = do_transform_pose(img_pose, self.image_to_world_t)
            self.des_pose.pose = world_pose
            self.des_pose.header.stamp = self.get_clock().now().to_msg()
            self.des_pose_pub.publish(self.des_pose)
            pose_list.append(world_pose)
            self.img_viz.points.append(world_pose.position)
            self.img_viz.header.stamp = self.get_clock().now().to_msg()
            self.img_viz_pub.publish(self.img_viz)

        future = self.send_traj_request(pose_list)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info(f"Move completed {'' if future.result().success else 'un'}successfully. {future.result().message}")
        else:
            self.get_logger().error("Service call failed.")

def main(args=None):
    rclpy.init(args=args)

    draw_svg_node = DrawSVG()

    try:
        # draw_svg_node.go_home()
        # draw_svg_node.draw_stroke(TEST_STROKE)
        draw_svg_node.draw_stroke_traj(TEST_STROKE)
        # rclpy.spin(draw_svg_node)
    except KeyboardInterrupt:
        pass
    finally:
        draw_svg_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()