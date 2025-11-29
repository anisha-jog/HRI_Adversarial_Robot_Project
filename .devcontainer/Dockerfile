# Use the official ROS 2 Jazzy desktop image as the base
FROM osrf/ros:humble-desktop

# Set environment variables for non-interactive install
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:fish-shell/release-4

# Install necessary packages: UR Driver, OpenCV, venv, and build tools
RUN apt-get update && apt-get install -y \
    ros-humble-moveit \
    ros-humble-ur-robot-driver \
    ros-humble-ur-simulation-gz \
    ros-humble-ur-bringup \
    python3-opencv \
    python3-pip \
    python3-venv \
    nano \
    fish \
    tmux \
    net-tools \
    iputils-ping \
    dos2unix \
    git \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Install fish
RUN fish -c "curl -sL https://git.io/fisher | source && fisher install jorgebucaran/fisher"

RUN python3 -m pip install Pillow \
    numpy \
    opencv-python \
    svgwrite \
    google-generativeai \
    requests \
    potracer \
    scikit-image

# Copy custom entrypoint script
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN dos2unix /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# --- Workspace and Venv Setup ---
# WORKDIR /ros2_ws

# Default command: keeps the container running and ready for interactive use
CMD ["bash"]