import assert from 'node:assert/strict'
import test from 'node:test'

import {
  appendTimedPose,
  interpolateTimedPose,
  lidarOrigin,
  projectLidarHit,
  quaternionYaw,
  rosClockReset,
  synchronizedLidarPose,
} from '../src/lidarSync.ts'

const close = (actual, expected, tolerance = 1e-9) => {
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} != ${expected}`)
}

test('interpolates translation and yaw at the scan timestamp', () => {
  const history = [
    { stamp: 10, x: 0, y: 1, yaw: 0 },
    { stamp: 11, x: 2, y: 3, yaw: Math.PI / 2 },
  ]
  const pose = interpolateTimedPose(history, 10.5)
  assert.ok(pose)
  close(pose.x, 1)
  close(pose.y, 2)
  close(pose.yaw, Math.PI / 4)
})

test('interpolates yaw through the pi boundary instead of rotating backwards', () => {
  const degrees = (value) => (value * Math.PI) / 180
  const pose = interpolateTimedPose([
    { stamp: 1, x: 0, y: 0, yaw: degrees(170) },
    { stamp: 2, x: 0, y: 0, yaw: degrees(-170) },
  ], 1.5)
  assert.ok(pose)
  close(Math.abs(pose.yaw), Math.PI)
})

test('waits for odometry newer than the scan', () => {
  assert.equal(interpolateTimedPose([
    { stamp: 3, x: 0, y: 0, yaw: 0 },
  ], 3.1), null)
})

test('does not project a scan from an earlier simulation epoch', () => {
  assert.equal(interpolateTimedPose([
    { stamp: 100, x: 2, y: 2, yaw: 1 },
    { stamp: 101, x: 3, y: 3, yaw: 1.5 },
  ], 0.2), null)
})

test('drops stale odometry when simulation time resets', () => {
  assert.equal(rosClockReset(100, 0.2), true)
  const history = appendTimedPose([
    { stamp: 100, x: 2, y: 2, yaw: 1 },
  ], { stamp: 0.2, x: -2, y: -0.5, yaw: 0 })
  assert.deepEqual(history, [{ stamp: 0.2, x: -2, y: -0.5, yaw: 0 }])
})

test('projects the scan from the physical lidar frame in world coordinates', () => {
  const origin = lidarOrigin({ x: 1, y: 2, yaw: Math.PI / 2 }, -0.032)
  close(origin.x, 1)
  close(origin.y, 1.968)
  const hit = projectLidarHit(origin, -Math.PI / 2, 2)
  close(hit.x, 3)
  close(hit.y, 1.968)
})

test('uses the normalized Gazebo world pose for lidar', () => {
  const world = [
    { stamp: 10, x: -1.58, y: -0.39, yaw: 1.08 },
    { stamp: 11, x: -1.57, y: -0.38, yaw: 1.10 },
  ]

  const pose = synchronizedLidarPose(world, 10.5)

  assert.ok(pose)
  close(pose.x, -1.575)
  close(pose.y, -0.385)
  close(pose.yaw, 1.09)
})

test('waits for a newer Gazebo pose instead of using another coordinate frame', () => {
  const world = [{ stamp: 10, x: -1.5, y: -0.4, yaw: 1 }]

  assert.equal(synchronizedLidarPose(world, 10.1), null)
  assert.equal(synchronizedLidarPose([], 10.1), null)
})

test('extracts planar yaw from the Gazebo model quaternion', () => {
  close(quaternionYaw(0, 0, Math.sin(Math.PI / 6), Math.cos(Math.PI / 6)), Math.PI / 3)
})
