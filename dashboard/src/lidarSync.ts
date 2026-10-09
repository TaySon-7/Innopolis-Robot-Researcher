export interface LidarPose {
  x: number
  y: number
  yaw: number
}

export interface TimedLidarPose extends LidarPose {
  stamp: number
}

const CLOCK_RESET_TOLERANCE = 0.25
const STAMP_EPSILON = 1e-6

export function rosClockReset(previous: number | undefined, next: number): boolean {
  return previous !== undefined && next < previous - CLOCK_RESET_TOLERANCE
}

export function quaternionYaw(x: number, y: number, z: number, w: number): number {
  return Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
}

function angleDelta(from: number, to: number): number {
  return Math.atan2(Math.sin(to - from), Math.cos(to - from))
}

/** Keep a short, ordered odometry history and discard it on a ROS clock reset. */
export function appendTimedPose(
  history: TimedLidarPose[],
  sample: TimedLidarPose,
  maxAgeSeconds = 3,
): TimedLidarPose[] {
  if (![sample.stamp, sample.x, sample.y, sample.yaw].every(Number.isFinite)) {
    return history
  }
  const last = history.at(-1)
  if (rosClockReset(last?.stamp, sample.stamp)) {
    return [sample]
  }

  const next = history.filter((item) => Math.abs(item.stamp - sample.stamp) > STAMP_EPSILON)
  next.push(sample)
  next.sort((left, right) => left.stamp - right.stamp)
  const newest = next.at(-1)?.stamp ?? sample.stamp
  return next.filter((item) => item.stamp >= newest - maxAgeSeconds).slice(-160)
}

/** Pose at the scan timestamp. Null means a newer odometry sample is still needed. */
export function interpolateTimedPose(
  history: TimedLidarPose[],
  stamp: number,
): LidarPose | null {
  if (!Number.isFinite(stamp) || history.length === 0) return null
  const first = history[0]
  const last = history[history.length - 1]
  if (stamp > last.stamp + STAMP_EPSILON) return null
  // A stamp well before the retained history belongs to another simulation
  // epoch (or arrived too late to transform safely). Waiting is preferable to
  // projecting it with an unrelated pose and throwing the cloud across the map.
  if (stamp < first.stamp - CLOCK_RESET_TOLERANCE) return null
  if (stamp <= first.stamp + STAMP_EPSILON) {
    return { x: first.x, y: first.y, yaw: first.yaw }
  }

  for (let index = 1; index < history.length; index += 1) {
    const right = history[index]
    if (stamp > right.stamp + STAMP_EPSILON) continue
    const left = history[index - 1]
    const duration = right.stamp - left.stamp
    if (duration <= STAMP_EPSILON) {
      return { x: right.x, y: right.y, yaw: right.yaw }
    }
    const fraction = Math.max(0, Math.min(1, (stamp - left.stamp) / duration))
    return {
      x: left.x + (right.x - left.x) * fraction,
      y: left.y + (right.y - left.y) * fraction,
      yaw: left.yaw + angleDelta(left.yaw, right.yaw) * fraction,
    }
  }
  return null
}

/**
 * Resolve a scan against Gazebo ground truth when it is available.
 *
 * Once the world-pose stream has started, do not silently fall back to wheel
 * odometry while waiting for its next sample: the two frames can differ after
 * slip, and mixing them is what makes one scan jump across the arena.
 */
export function synchronizedLidarPose(
  worldHistory: TimedLidarPose[],
  odometryHistory: TimedLidarPose[],
  stamp: number,
): LidarPose | null {
  return interpolateTimedPose(
    worldHistory.length ? worldHistory : odometryHistory,
    stamp,
  )
}

export function lidarOrigin(pose: LidarPose, xOffset: number): LidarPose {
  return {
    x: pose.x + xOffset * Math.cos(pose.yaw),
    y: pose.y + xOffset * Math.sin(pose.yaw),
    yaw: pose.yaw,
  }
}

export function projectLidarHit(
  origin: LidarPose,
  relativeAngle: number,
  range: number,
): { x: number; y: number } {
  const angle = origin.yaw + relativeAngle
  return {
    x: origin.x + range * Math.cos(angle),
    y: origin.y + range * Math.sin(angle),
  }
}
