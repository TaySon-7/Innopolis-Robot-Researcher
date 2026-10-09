FROM ros:jazzy-ros-base-noble

ARG DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        mesa-utils \
        python3-colcon-common-extensions \
        ros-jazzy-rmw-cyclonedds-cpp \
        ros-jazzy-rosbridge-suite \
        ros-jazzy-turtlebot3-gazebo \
        ros-jazzy-turtlebot3-teleop \
    && rm -rf /var/lib/apt/lists/*

# Keep the large ROS/Gazebo layer above stable. NumPy and PyYAML already come
# in with the ROS packages; only pip is needed for the small LLM dependency.
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Python packages that are not in apt. Copied before the workspace so a change
# to the ROS sources does not force a reinstall of these.
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir --break-system-packages -r /tmp/requirements.txt

# Keep Nav2 separate from the existing ROS/Gazebo and Python layers. Only the
# navigation servers/plugins used by did_agent are installed (no AMCL/SLAM).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ros-jazzy-nav2-behaviors \
        ros-jazzy-nav2-bt-navigator \
        ros-jazzy-nav2-controller \
        ros-jazzy-nav2-lifecycle-manager \
        ros-jazzy-nav2-navfn-planner \
        ros-jazzy-nav2-planner \
        ros-jazzy-nav2-regulated-pure-pursuit-controller \
        ros-jazzy-tf2-ros-py \
    && rm -rf /var/lib/apt/lists/*

ENV ROS_DOMAIN_ID=30 \
    RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    TURTLEBOT3_MODEL=burger \
    LIBGL_ALWAYS_SOFTWARE=1

WORKDIR /opt/did_ws
COPY ros2_ws/src ./src

RUN source /opt/ros/jazzy/setup.bash \
    && colcon build \
    && rm -rf build log

COPY docker/entrypoint.sh /did-entrypoint.sh
RUN chmod +x /did-entrypoint.sh

ENTRYPOINT ["/did-entrypoint.sh"]
CMD ["bash"]
