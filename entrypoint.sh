#!/bin/bash
# Source the ROS 2 setup file
source /opt/venv/scripts/activate
source /opt/ros/jazzy/setup.bash
# Execute the command passed to the container (CMD or docker run command)
exec "$@"