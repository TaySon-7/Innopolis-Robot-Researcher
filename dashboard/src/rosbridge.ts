import { useCallback, useEffect, useRef, useState } from 'react'

export type ConnectionStatus = 'connecting' | 'connected' | 'disconnected'

export interface Pose2D {
  x: number
  y: number
  yaw: number
}

export interface LaserScan {
  origin: Pose2D
  angleMin: number
  angleIncrement: number
  rangeMin: number
  rangeMax: number
  ranges: number[]
}

export interface SampleState {
  x: number
  y: number
  collected: boolean
}

export interface ScoreState {
  scenario: string
  battery: number
  collected: number
  samplesTotal: number
  distanceTravelled: number
  finished: boolean
}

export interface RosSnapshot {
  status: ConnectionStatus
  pose: Pose2D
  linearVelocity: number
  angularVelocity: number
  battery: number | null
  sampleSignal: number | null
  score: ScoreState
  samples: SampleState[]
  scan: LaserScan | null
  trail: Array<{ x: number; y: number }>
  simTimeSeconds: number | null
  lastEvent: string | null
}

export interface TriggerResult {
  success: boolean
  message: string
}

const BASE = { x: -2, y: -0.5 }
const LIDAR_X_OFFSET = -0.032
const DEFAULT_SAMPLES: SampleState[] = [
  { x: -1.5, y: -0.5, collected: false },
  { x: -0.75, y: 0.25, collected: false },
  { x: 0.25, y: -0.5, collected: false },
]

const initialSnapshot = (): RosSnapshot => ({
  status: 'connecting',
  pose: { x: BASE.x, y: BASE.y, yaw: 0 },
  linearVelocity: 0,
  angularVelocity: 0,
  battery: null,
  sampleSignal: null,
  score: {
    scenario: 'easy',
    battery: 60,
    collected: 0,
    samplesTotal: 3,
    distanceTravelled: 0,
    finished: false,
  },
  samples: DEFAULT_SAMPLES,
  scan: null,
  trail: [{ x: BASE.x, y: BASE.y }],
  simTimeSeconds: null,
  lastEvent: null,
})

type SnapshotListener = (snapshot: RosSnapshot) => void
type JsonObject = Record<string, unknown>

function rosbridgeUrl(): string {
  const configured = import.meta.env.VITE_ROSBRIDGE_URL as string | undefined
  if (configured) return configured
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${protocol}//${window.location.host}/ros`
}

function number(value: unknown, fallback = 0): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback
}

function object(value: unknown): JsonObject {
  return value !== null && typeof value === 'object' ? (value as JsonObject) : {}
}

function eventText(data: JsonObject): string {
  switch (data.event) {
    case 'sample_collected':
      return `Образец собран · ${number(data.collected)} из 3`
    case 'false_collect':
      return 'Рядом нет образца: подведите робота ближе'
    default:
      return typeof data.event === 'string' ? data.event : 'Получено событие судьи'
  }
}

class RosbridgeClient {
  private socket: WebSocket | null = null
  private snapshot = initialSnapshot()
  private listeners = new Set<SnapshotListener>()
  private reconnectTimer: number | null = null
  private reconnectAttempt = 0
  private stopped = false
  private sequence = 0
  private serviceCalls = new Map<
    string,
    {
      resolve: (result: TriggerResult) => void
      reject: (error: Error) => void
      timeout: number
    }
  >()

  connect(): void {
    this.stopped = false
    this.openSocket()
  }

  destroy(): void {
    this.stopped = true
    if (this.reconnectTimer !== null) window.clearTimeout(this.reconnectTimer)
    if (this.socket?.readyState === WebSocket.OPEN) this.publishVelocity(0, 0)
    this.socket?.close()
    this.socket = null
    for (const call of this.serviceCalls.values()) {
      window.clearTimeout(call.timeout)
      call.reject(new Error('ROS-соединение закрыто'))
    }
    this.serviceCalls.clear()
  }

  subscribe(listener: SnapshotListener): () => void {
    this.listeners.add(listener)
    listener(this.snapshot)
    return () => this.listeners.delete(listener)
  }

  publishVelocity(linear: number, angular: number): boolean {
    return this.send({
      op: 'publish',
      topic: '/cmd_vel',
      msg: {
        header: {
          stamp: { sec: 0, nanosec: 0 },
          frame_id: '',
        },
        twist: {
          linear: { x: linear, y: 0, z: 0 },
          angular: { x: 0, y: 0, z: angular },
        },
      },
    })
  }

  callTrigger(service: '/did/collect' | '/did/finish'): Promise<TriggerResult> {
    const id = `service:${++this.sequence}`
    return new Promise((resolve, reject) => {
      const timeout = window.setTimeout(() => {
        this.serviceCalls.delete(id)
        reject(new Error('Судья не ответил за 5 секунд'))
      }, 5000)

      this.serviceCalls.set(id, { resolve, reject, timeout })
      const sent = this.send({
        op: 'call_service',
        id,
        service,
        type: 'std_srvs/srv/Trigger',
        args: {},
      })
      if (!sent) {
        window.clearTimeout(timeout)
        this.serviceCalls.delete(id)
        reject(new Error('Нет соединения с ROS'))
      }
    })
  }

  private openSocket(): void {
    if (this.stopped) return
    this.setSnapshot({ ...this.snapshot, status: 'connecting' })
    const socket = new WebSocket(rosbridgeUrl())
    this.socket = socket

    socket.addEventListener('open', () => {
      if (this.socket !== socket) return
      this.reconnectAttempt = 0
      this.setSnapshot({ ...this.snapshot, status: 'connected' })
      this.send({
        op: 'advertise',
        topic: '/cmd_vel',
        type: 'geometry_msgs/msg/TwistStamped',
      })
      this.addSubscriptions()
    })

    socket.addEventListener('message', (event) => {
      if (this.socket !== socket || typeof event.data !== 'string') return
      try {
        this.handleMessage(JSON.parse(event.data) as JsonObject)
      } catch {
        // rosbridge can emit non-JSON diagnostics; they are safe to ignore here.
      }
    })

    socket.addEventListener('close', () => {
      if (this.socket !== socket) return
      this.socket = null
      this.setSnapshot({
        ...this.snapshot,
        status: 'disconnected',
        linearVelocity: 0,
        angularVelocity: 0,
      })
      if (!this.stopped) this.scheduleReconnect()
    })

    socket.addEventListener('error', () => socket.close())
  }

  private scheduleReconnect(): void {
    const delay = Math.min(5000, 500 * 2 ** this.reconnectAttempt)
    this.reconnectAttempt += 1
    this.reconnectTimer = window.setTimeout(() => this.openSocket(), delay)
  }

  private addSubscriptions(): void {
    const subscriptions = [
      ['/odom', 'nav_msgs/msg/Odometry', 50],
      ['/scan', 'sensor_msgs/msg/LaserScan', 100],
      ['/did/battery', 'std_msgs/msg/Float32', 100],
      ['/did/sample_sensor', 'std_msgs/msg/Float32', 100],
      ['/did/score', 'std_msgs/msg/String', 100],
      ['/did/events', 'std_msgs/msg/String', 0],
      ['/clock', 'rosgraph_msgs/msg/Clock', 100],
    ] as const

    for (const [topic, type, throttleRate] of subscriptions) {
      this.send({
        op: 'subscribe',
        id: `sub:${topic}`,
        topic,
        type,
        throttle_rate: throttleRate,
        queue_length: 1,
      })
    }
  }

  private handleMessage(data: JsonObject): void {
    if (data.op === 'service_response' && typeof data.id === 'string') {
      const call = this.serviceCalls.get(data.id)
      if (!call) return
      window.clearTimeout(call.timeout)
      this.serviceCalls.delete(data.id)
      const values = object(data.values)
      call.resolve({
        success: Boolean(values.success),
        message: typeof values.message === 'string' ? values.message : '',
      })
      return
    }

    if (data.op !== 'publish' || typeof data.topic !== 'string') return
    const message = object(data.msg)

    switch (data.topic) {
      case '/odom':
        this.handleOdometry(message)
        break
      case '/scan':
        this.handleScan(message)
        break
      case '/did/battery':
        this.setSnapshot({ ...this.snapshot, battery: number(message.data) })
        break
      case '/did/sample_sensor':
        this.setSnapshot({ ...this.snapshot, sampleSignal: number(message.data) })
        break
      case '/did/score':
        this.handleScore(message)
        break
      case '/did/events':
        this.handleEvent(message)
        break
      case '/clock': {
        const clock = object(message.clock)
        this.setSnapshot({
          ...this.snapshot,
          simTimeSeconds: number(clock.sec) + number(clock.nanosec) / 1e9,
        })
        break
      }
    }
  }

  private handleOdometry(message: JsonObject): void {
    const poseRoot = object(object(message.pose).pose)
    const position = object(poseRoot.position)
    const orientation = object(poseRoot.orientation)
    const twist = object(object(message.twist).twist)
    const linear = object(twist.linear)
    const angular = object(twist.angular)
    const qx = number(orientation.x)
    const qy = number(orientation.y)
    const qz = number(orientation.z)
    const qw = number(orientation.w, 1)
    const yaw = Math.atan2(
      2 * (qw * qz + qx * qy),
      1 - 2 * (qy * qy + qz * qz),
    )
    const pose = {
      x: BASE.x + number(position.x),
      y: BASE.y + number(position.y),
      yaw,
    }
    const previous = this.snapshot.trail.at(-1)
    const moved =
      !previous || Math.hypot(pose.x - previous.x, pose.y - previous.y) >= 0.025
    const trail = moved
      ? [...this.snapshot.trail, { x: pose.x, y: pose.y }].slice(-500)
      : this.snapshot.trail

    this.setSnapshot({
      ...this.snapshot,
      pose,
      trail,
      linearVelocity: number(linear.x),
      angularVelocity: number(angular.z),
    })
  }

  private handleScan(message: JsonObject): void {
    const ranges = Array.isArray(message.ranges)
      ? message.ranges.map((value) => number(value, Number.NaN))
      : []
    this.setSnapshot({
      ...this.snapshot,
      scan: {
        // Freeze the transform at scan receipt. Reusing the newest odometry for
        // an older scan makes the entire point cloud appear to follow the robot.
        origin: {
          x: this.snapshot.pose.x + LIDAR_X_OFFSET * Math.cos(this.snapshot.pose.yaw),
          y: this.snapshot.pose.y + LIDAR_X_OFFSET * Math.sin(this.snapshot.pose.yaw),
          yaw: this.snapshot.pose.yaw,
        },
        angleMin: number(message.angle_min),
        angleIncrement: number(message.angle_increment),
        rangeMin: number(message.range_min),
        rangeMax: number(message.range_max, 3.5),
        ranges,
      },
    })
  }

  private handleScore(message: JsonObject): void {
    if (typeof message.data !== 'string') return
    try {
      const data = object(JSON.parse(message.data))
      const worldPose = object(data.world_pose)
      const incomingSamples = Array.isArray(data.samples) ? data.samples : []
      const samples = incomingSamples.length
        ? incomingSamples.map((value) => {
            const sample = object(value)
            return {
              x: number(sample.x),
              y: number(sample.y),
              collected: Boolean(sample.collected),
            }
          })
        : this.snapshot.samples
      this.setSnapshot({
        ...this.snapshot,
        battery: number(data.battery, this.snapshot.battery ?? 0),
        pose: {
          ...this.snapshot.pose,
          x: number(worldPose.x, this.snapshot.pose.x),
          y: number(worldPose.y, this.snapshot.pose.y),
        },
        samples,
        score: {
          scenario: typeof data.scenario === 'string' ? data.scenario : 'easy',
          battery: number(data.battery, 60),
          collected: number(data.collected),
          samplesTotal: number(data.samples_total, samples.length || 3),
          distanceTravelled: number(data.distance_travelled),
          finished: Boolean(data.finished),
        },
      })
    } catch {
      // Keep the last valid score if the message is malformed.
    }
  }

  private handleEvent(message: JsonObject): void {
    if (typeof message.data !== 'string') return
    try {
      this.setSnapshot({
        ...this.snapshot,
        lastEvent: eventText(object(JSON.parse(message.data))),
      })
    } catch {
      this.setSnapshot({ ...this.snapshot, lastEvent: message.data })
    }
  }

  private send(payload: JsonObject): boolean {
    if (this.socket?.readyState !== WebSocket.OPEN) return false
    this.socket.send(JSON.stringify(payload))
    return true
  }

  private setSnapshot(snapshot: RosSnapshot): void {
    this.snapshot = snapshot
    for (const listener of this.listeners) listener(snapshot)
  }
}

export function useRosbridge() {
  const clientRef = useRef<RosbridgeClient | null>(null)
  const [snapshot, setSnapshot] = useState<RosSnapshot>(initialSnapshot)

  useEffect(() => {
    const client = new RosbridgeClient()
    clientRef.current = client
    const unsubscribe = client.subscribe(setSnapshot)
    client.connect()
    return () => {
      unsubscribe()
      client.destroy()
      clientRef.current = null
    }
  }, [])

  const publishVelocity = useCallback((linear: number, angular: number) => {
    return clientRef.current?.publishVelocity(linear, angular) ?? false
  }, [])

  const callTrigger = useCallback((service: '/did/collect' | '/did/finish') => {
    const client = clientRef.current
    if (!client) return Promise.reject(new Error('ROS-клиент не готов'))
    return client.callTrigger(service)
  }, [])

  return { snapshot, publishVelocity, callTrigger }
}
