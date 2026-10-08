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
