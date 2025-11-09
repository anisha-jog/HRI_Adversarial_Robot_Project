from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # 1. Launch Arguments
    enable_sim = LaunchConfiguration('enable_sim')
    robot_ip = LaunchConfiguration('robot_ip')
    ur_type = LaunchConfiguration('ur_type')
    x_offset = LaunchConfiguration('x_offset')
    y_offset = LaunchConfiguration('y_offset')

    # Argument to switch between real robot and simulation
    enable_sim_arg = DeclareLaunchArgument(
        'enable_sim',
        default_value='false',
        description='Set to true to launch in simulation mode (using fake hardware) or false for real robot.',
    )

    set_x_offset = DeclareLaunchArgument(
        'x_offset',
        default_value='0.5',
        description='image frame x offset from base_link frame',
    )
    set_y_offset = DeclareLaunchArgument(
        'y_offset',
        default_value='-0.5',
        description='image frame y offset from base_link frame',
    )
    # Robot IP (only needed when enable_sim is false)
    robot_ip_arg = DeclareLaunchArgument(
        'robot_ip',
        default_value='192.168.56.50',
        description='The IP address of the real UR robot (only used if enable_sim is false).',
    )

    # UR Type (used for both)
    ur_type_arg = DeclareLaunchArgument(
        'ur_type',
        default_value='ur5',
        description='The type of UR robot (e.g., ur5, ur10).',
    )


    # Find the UR Driver package share directory
    ur_driver_pkg = FindPackageShare('ur_robot_driver')

    # Define common arguments for the ur_control.launch.py
    # Note: The ur_robot_driver launch file handles both real and fake hardware
    # using 'use_fake_hardware' argument.
    ur_control_launch = PathJoinSubstitution([
        ur_driver_pkg,
        'launch',
        'ur_control.launch.py'
    ])

    # Conditional Launch for UR Driver (Real Hardware)
    # When enable_sim is false, use the provided robot_ip
    ur_driver_real = IncludeLaunchDescription(
        ur_control_launch,
        condition=UnlessCondition(enable_sim),
        launch_arguments={
            'ur_type': ur_type,
            'robot_ip': robot_ip,
            'use_fake_hardware': 'false',
            'headless_mode': 'true',
            'use_sim_time': 'false', # Important for real robot
            'initial_joint_controller': 'joint_position_controller',
            'launch_rviz': 'false'
        }.items(),
    )

    # Conditional Launch for UR Driver (Simulation/Fake Hardware)
    # When enable_sim is true, set use_fake_hardware to true
    ur_sim_launch = PathJoinSubstitution([
        FindPackageShare('ur_simulation_gz'),
        'launch',
        'ur_sim_control.launch.py'
    ])
    ur_driver_sim = IncludeLaunchDescription(
            ur_sim_launch,
            condition=IfCondition(enable_sim),
            launch_arguments={
                'ur_type': ur_type,
                'gazebo_gui': 'false',
                # 'initial_joint_controller': 'joint_position_controller',
                'initial_joint_controller': 'joint_trajectory_controller',
                'launch_rviz': 'false',
            }.items(),
    )

    moveit_config = IncludeLaunchDescription(
        PathJoinSubstitution([ FindPackageShare('ur_moveit_config'), 'launch', 'ur_moveit.launch.py' ]),
        launch_arguments={
            'ur_type': ur_type,
            'use_sim_time': enable_sim,
            'launch_rviz': 'true',
        }.items(),
    )

    static_img_frame_pub = Node(
                package='tf2_ros',
                executable='static_transform_publisher',
                arguments=[
                    '--x', x_offset, '--y', y_offset, '--z', '0',
                    '--yaw', '0', '--pitch', '0', '--roll',
                    '0', '--frame-id', 'base_link', '--child-frame-id', 'image_frame']
            )

    static_pen_frame_pub = Node(
                package='tf2_ros',
                executable='static_transform_publisher',
                arguments=[
                    # TODO adjust these values based on our setup
                    '--x', '0.02', '--y', '0', '--z', '0.01',
                    '--yaw', '0', '--pitch', '-1.57725', '--roll',
                    '0', '--frame-id', 'tool0', '--child-frame-id', 'pen_frame']
            )
    # This node needs to be built and installed via your package's setup.py
    # We assume this node is in a package named 'my_ur5_control'
    # moveit_position_sender_node = Node(
    #     package='my_ur5_control', # Replace with your package name
    #     executable='moveit_position_client',
    #     name='moveit_position_client',
    #     output='screen',
    #     parameters=[
    #         {'use_sim_time': enable_sim} # Use sim time if simulation is enabled
    #     ]
    # )

    # bass source /ros2_ws/install/setup.bash
    # ros2 launch ur5_draw ur5_draw.launch.py enable_sim:=true launch_rviz:=false
    # ros2 launch ur5_draw ur5_draw.launch.py enable_sim:=true launch_rviz:=false initial_joint_controller:=forward_position_controller
    # ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur5 use_sim_time:=true launch_rviz:=true


    return LaunchDescription([
        enable_sim_arg,
        robot_ip_arg,
        ur_type_arg,
        set_x_offset,
        set_y_offset,
        static_img_frame_pub,
        static_pen_frame_pub,

        # Launches the appropriate UR driver setup
        ur_driver_real,
        ur_driver_sim,
        moveit_config,

        # Launches your custom node
        # moveit_position_sender_node,
    ])