from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # 1. Launch Arguments
    enable_sim = LaunchConfiguration('enable_sim')
    robot_ip = LaunchConfiguration('robot_ip')
    ur_type = LaunchConfiguration('ur_type')

    # Argument to switch between real robot and simulation
    enable_sim_arg = DeclareLaunchArgument(
        'enable_sim',
        default_value='false',
        description='Set to true to launch in simulation mode (using fake hardware) or false for real robot.',
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
            'launch_rviz': 'true'
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
            # 'gazebo_gui': 'false',
            # 'launch_rviz': 'true'
        }.items(),
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


    return LaunchDescription([
        enable_sim_arg,
        robot_ip_arg,
        ur_type_arg,

        # Launches the appropriate UR driver setup
        ur_driver_real,
        ur_driver_sim,

        # Launches your custom node
        # moveit_position_sender_node,
    ])