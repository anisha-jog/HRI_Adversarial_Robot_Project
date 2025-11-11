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

from ur_draw_cmake.srv import MoveToPose

PEN_HEIGHT = .3
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
        self.client = self.create_client(MoveToPose, 'moveit_to_pose')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self.request = MoveToPose.Request()

        self.image_to_world_t = self.get_img_transform('image_frame', 'world')
        # self.tool_to_frame = self.get_img_transform('tool0', 'pen_frame')
        # self.tool_to_frame = self.get_img_transform('pen_frame', 'tool0')
        self.get_logger().info('Transform successfully retrieved and cached.')
        self.des_pose_pub = self.create_publisher(PoseStamped,'des_pose',10)
        self.des_pose = PoseStamped()
        self.des_pose.header.frame_id = "world"
        self.img_viz_pub = self.create_publisher(Marker, 'image_viz', 10)
        self.img_viz = Marker()
        self.img_viz.header.frame_id = 'world'
        self.img_viz.type = Marker.LINE_STRIP
        self.img_viz.scale.x = 0.002
        self.img_viz.color.a = 1.0
        self.img_viz.color.r = 1.0
        self.img_viz.points = [] # May need to set a starting point

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

    def go_home(self):
        home_pose_world = do_transform_pose(HOME_POSE, self.image_to_world_t)
        # home_pose_world = do_transform_pose(do_transform_pose(HOME_POSE, self.image_to_world_t),self.tool_to_frame)
        future = self.send_request(home_pose_world)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info("Returned to home position successfully.")
        else:
            self.get_logger().error("Service call to return home failed.")

    def send_request(self, pose):
        self.request.target_pose = pose
        self.get_logger().info("Sending request to move to the target pose...")
        return self.client.call_async(self.request)

    def draw_point(self, x, y):
        img_pose = Pose()
        img_pose.position.x = x
        img_pose.position.y = y
        img_pose.position.z = 0.0

        img_pose.orientation.w = 0.707
        img_pose.orientation.x = -0.707
        img_pose.orientation.y = 0.0
        img_pose.orientation.z = 0.0


        world_pose = do_transform_pose(img_pose, self.image_to_world_t)
        # tool_pose = Pose(position=world_pose.position,orientation=world_pose.orientation)
        tool_pose = deepcopy(world_pose)
        tool_pose.position.z = PEN_HEIGHT
        # tool_pose = do_transform_pose(world_pose,self.tool_to_frame)
        self.des_pose.pose = tool_pose
        self.des_pose.header.stamp = self.get_clock().now().to_msg()
        self.des_pose_pub.publish(self.des_pose)


        # future = self.send_request(world_pose)
        future = self.send_request(tool_pose)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info(f"Move completed successfully. {future.result()}")
            t = self.get_img_transform("tool0","world")
            self.get_logger().info(f"T:({t.transform.translation.x },{t.transform.translation.y },{t.transform.translation.z }) {t.transform.rotation }")
            self.get_logger().info(f"P:({self.des_pose.pose.position.x },{self.des_pose.pose.position.y },{self.des_pose.pose.position.z }) {self.des_pose.pose.orientation }")
        else:
            self.get_logger().error("Service call failed.")

        self.img_viz.points.append(world_pose.position)
        self.img_viz.header.stamp = self.get_clock().now().to_msg()
        self.img_viz_pub.publish(self.img_viz)

    def draw_stroke(self, points):
        for point in points:
            self.draw_point(point[0], point[1])
        self.go_home()




def main(args=None):
    rclpy.init(args=args)

    draw_svg_node = DrawSVG()

    try:
        # draw_svg_node.go_home()
        draw_svg_node.draw_stroke(TEST_STROKE)
        # rclpy.spin(draw_svg_node)
    except KeyboardInterrupt:
        pass
    finally:
        draw_svg_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()