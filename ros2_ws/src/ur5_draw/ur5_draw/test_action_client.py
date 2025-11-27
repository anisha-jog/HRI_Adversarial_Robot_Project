#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import Point
import cv2

from ur_draw_cmake.action import DrawStrokes
from ur_draw_cmake.msg import Stroke
from ur5_draw.image_to_svg import image_to_lines


class DrawSVGActionClient(Node):
    def __init__(self):
        super().__init__('draw_svg_action_client')
        
        self._action_client = ActionClient(
            self,
            DrawStrokes,
            'draw_strokes'
        )
        
        self.get_logger().info('Waiting for action server...')
        self._action_client.wait_for_server()
        self.get_logger().info('Action server available!')
    
    def send_goal(self, image_path):
        """Load image, convert to strokes, and send goal to action server."""
        # Load and process image
        self.get_logger().info(f'Loading image from: {image_path}')
        img = cv2.imread(image_path)
        
        if img is None:
            self.get_logger().error(f'Failed to load image: {image_path}')
            return None
        
        height, width, channels = img.shape
        self.get_logger().info(f'Image dimensions: {width} x {height}')
        
        # Convert image to strokes
        self.get_logger().info('Converting image to strokes...')
        strokes = image_to_lines(img)
        self.get_logger().info(f'Generated {len(strokes)} strokes')
        
        # Create goal message
        goal_msg = DrawStrokes.Goal()
        goal_msg.img_width = float(width)
        goal_msg.img_length = float(height)
        
        # Convert strokes to ROS messages
        for stroke_points in strokes:
            stroke_msg = Stroke()
            for x, y in stroke_points:
                stroke_msg.points.append(Point(x=x, y=y, z=0.0))
            goal_msg.strokes.append(stroke_msg)
        
        self.get_logger().info('Sending goal to action server...')
        
        # Send goal with feedback callback
        send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        
        send_goal_future.add_done_callback(self.goal_response_callback)
        
        return send_goal_future
    
    def goal_response_callback(self, future):
        """Called when the action server accepts/rejects the goal."""
        goal_handle = future.result()
        
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected by action server')
            return
        
        self.get_logger().info('Goal accepted by action server')
        
        # Get result
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.get_result_callback)
    
    def feedback_callback(self, feedback_msg):
        """Called when feedback is received from the action server."""
        feedback = feedback_msg.feedback
        self.get_logger().info(
            f'Feedback: Stroke {feedback.current_stroke}/{feedback.total_strokes} '
            f'({feedback.percent_complete:.1f}%) - {feedback.status_message}'
        )
    
    def get_result_callback(self, future):
        """Called when the action completes."""
        result = future.result().result
        
        if result.success:
            self.get_logger().info(
                f'SUCCESS! Completed {result.strokes_completed} strokes. '
                f'Message: {result.message}'
            )
        else:
            self.get_logger().error(
                f'FAILED! Completed {result.strokes_completed} strokes. '
                f'Message: {result.message}'
            )
        
        # Shutdown after result
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    
    action_client = DrawSVGActionClient()
    
    # Path to your test image
    image_path = '/home/studioadmin/HRI_Adversarial_Robot_Project/ros2_ws/test.jpg'
    
    # Send goal
    future = action_client.send_goal(image_path)
    
    if future is None:
        action_client.get_logger().error('Failed to send goal')
        action_client.destroy_node()
        rclpy.shutdown()
        return
    
    # Spin until the action completes
    try:
        rclpy.spin(action_client)
    except KeyboardInterrupt:
        action_client.get_logger().info('Keyboard interrupt, canceling goal...')
        # Optionally cancel the goal here if needed
    finally:
        action_client.destroy_node()


if __name__ == '__main__':
    main()