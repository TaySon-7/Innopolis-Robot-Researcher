from math import nan

from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseArray
from nav_msgs.msg import Odometry

from did_judge.judge_node import physical_model_pose
from did_judge.judge_node import physical_pose_fresh
from did_judge.judge_node import stamped_message_after


def pose_array(stamp_ns: int, *, x: float = 1.0, y: float = 2.0) -> PoseArray:
    message = PoseArray()
    message.header.stamp.sec = stamp_ns // 1_000_000_000
    message.header.stamp.nanosec = stamp_ns % 1_000_000_000
    pose = Pose()
    pose.position.x = x
    pose.position.y = y
    pose.orientation.w = 1.0
    message.poses = [pose]
    return message


def test_physical_model_pose_rejects_empty_stale_and_nonfinite_frames():
    assert physical_model_pose(PoseArray(), 10) is None
    assert physical_model_pose(pose_array(10), 10) is None
    assert physical_model_pose(pose_array(11, x=nan), 10) is None


def test_physical_model_pose_accepts_the_first_fresh_model_pose():
    message = pose_array(12, x=-0.55, y=0.55)

    pose = physical_model_pose(message, 10)

    assert pose is message.poses[0]


def test_physical_pose_fresh_fails_closed_when_pose_stream_stops():
    assert not physical_pose_fresh(None, 1_000_000_000)
    assert physical_pose_fresh(1_000_000_000, 1_750_000_000)
    assert not physical_pose_fresh(1_000_000_000, 1_750_000_001)
    assert not physical_pose_fresh(1_200_000_001, 1_000_000_000)


def test_reset_timestamp_gate_rejects_queued_odometry_from_removed_robot():
    stale = Odometry()
    stale.header.stamp.sec = 10
    fresh = Odometry()
    fresh.header.stamp.sec = 10
    fresh.header.stamp.nanosec = 1

    assert not stamped_message_after(stale, 10_000_000_000)
    assert stamped_message_after(fresh, 10_000_000_000)
