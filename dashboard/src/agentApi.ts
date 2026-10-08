import { useCallback, useEffect, useRef, useState } from 'react'

export type AgentConnection = 'connecting' | 'connected' | 'disconnected'
export type NavigationMode = 'goto' | 'search'
export type CostRun = [number, number, number, number]
export type MaskRun = [number, number, number]

export interface AgentCostmap {
  version: number
  knowledge_version?: number
  knowledge: CostRun[]
  terrain: CostRun[]
  wall_cost: CostRun[]
  total: CostRun[]
  blocked: MaskRun[]
}

export interface AgentJournalEntry {
  t?: number
  kind?: 'hypothesis' | 'result' | 'decision' | 'agent' | string
  status?: 'open' | 'confirmed' | 'rejected' | string
  title?: string
  text?: string
}

export interface JudgeEvent {
  t?: number
  event?: string
  [key: string]: unknown
}

export interface AgentNavigation {
  status?: string
  replans?: number
  waypoints?: Array<[number, number]>
}

export interface AgentRuntimeState {
  t?: number
  battery?: number
  score?: number
  sensor?: { value?: number; noise_estimate?: number }
  current?: { plan_id?: string; index?: number; type?: string; state?: string }
  return_cost_estimate?: number
  anomaly?: {
    battery_deviation?: boolean
    penalties_burst?: boolean
    sensor_noise_up?: boolean
  }
  navigation?: AgentNavigation
}

export interface AgentRunStatus {
  state?: 'idle' | 'running' | 'done' | 'failed' | 'preempted' | string
  subgoal?: string
  reason?: string
  data?: Record<string, unknown>
}

export interface AgentScore {
  scenario?: string
  t?: number
  battery?: number
  collected?: number
  samples_total?: number
  distance_travelled?: number
  collisions?: number
  false_collects?: number
  hazard_hits?: number
  score?: number
  finished?: boolean
}

export interface AgentSnapshot {
  pose: { x: number; y: number; yaw: number } | null
  state: AgentRuntimeState
  status: AgentRunStatus
  score: AgentScore
  trail: Array<[number, number]>
  events: JudgeEvent[]
  journal: AgentJournalEntry[]
  costmap: AgentCostmap
  plan: string
  collected_at: Array<[number, number]>
  scenario: string | null
}

export interface ScenarioZone {
  id?: string
  shape?: 'circle' | 'rectangle' | string
  x?: number
  y?: number
  radius?: number
  x_min?: number
  x_max?: number
  y_min?: number
  y_max?: number
  cost_multiplier?: number
  penalty?: number
}

export interface AgentTruth {
  name?: string
  at?: number
  base?: { x: number; y: number }
  samples?: Array<{ x: number; y: number }>
  soil_zones?: ScenarioZone[]
  hazard_zones?: ScenarioZone[]
}

export interface AgentGeometry {
  source?: 'gazebo_scene' | 'navigation_map' | string
  scene_service?: string
  floor: Array<[number, number]>
  pillars: Array<{ x: number; y: number; r: number }>
  wall: number
  bounds: {
    xmin: number
    xmax: number
    ymin: number
    ymax: number
  }
  resolution: number
  origin: [number, number]
}

interface ApiResult {
  ok: boolean
  error?: string
}

const EMPTY_SNAPSHOT: AgentSnapshot = {
  pose: null,
  state: {},
  status: {},
  score: {},
  trail: [],
  events: [],
  journal: [],
  costmap: {
    version: -1,
    knowledge: [],
    terrain: [],
    wall_cost: [],
    total: [],
    blocked: [],
  },
  plan: '',
  collected_at: [],
  scenario: null,
}

async function getJson<T>(url: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(url, { cache: 'no-store', signal })
  if (!response.ok) throw new Error(`API ответил ${response.status}`)
  return response.json() as Promise<T>
}

async function postJson(url: string, body: unknown): Promise<ApiResult> {
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  const result = (await response.json().catch(() => ({
    ok: false,
    error: `API ответил ${response.status}`,
  }))) as ApiResult
  if (!response.ok || !result.ok) {
    throw new Error(result.error || `API ответил ${response.status}`)
  }
  return result
}

export function useAgentApi() {
  const [snapshot, setSnapshot] = useState<AgentSnapshot>(EMPTY_SNAPSHOT)
  const [truth, setTruth] = useState<AgentTruth | null>(null)
  const [geometry, setGeometry] = useState<AgentGeometry | null>(null)
  const [connection, setConnection] = useState<AgentConnection>('connecting')
  const [lastError, setLastError] = useState<string | null>(null)
  const scenarioRef = useRef<string | null>(null)
  const truthFetchedAtRef = useRef(0)

  useEffect(() => {
    let stopped = false
    let timer: number | null = null
    const controller = new AbortController()

    const poll = async () => {
      try {
        const next = await getJson<AgentSnapshot>('/api/state', controller.signal)
        if (stopped) return
        setSnapshot(next)
        setConnection('connected')
        setLastError(null)
        const now = Date.now()
        const scenarioChanged = next.scenario !== scenarioRef.current
        if (next.scenario && (scenarioChanged || now - truthFetchedAtRef.current >= 1000)) {
          scenarioRef.current = next.scenario
          truthFetchedAtRef.current = now
          const nextTruth = await getJson<AgentTruth>('/api/truth', controller.signal)
          if (!stopped) setTruth(nextTruth)
        }
      } catch (error) {
        if (stopped || controller.signal.aborted) return
        setConnection('disconnected')
        setLastError(error instanceof Error ? error.message : 'Agent API недоступен')
      } finally {
        if (!stopped) timer = window.setTimeout(poll, 300)
      }
    }

    void getJson<AgentGeometry>('/api/geometry', controller.signal)
      .then((value) => {
        if (!stopped) setGeometry(value)
      })
      .catch(() => undefined)
    void poll()

    return () => {
      stopped = true
      controller.abort()
      if (timer !== null) window.clearTimeout(timer)
    }
  }, [])

  const request = useCallback(async (url: string, body: unknown) => {
    try {
      const result = await postJson(url, body)
      setLastError(null)
      return result
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Команда агента не отправлена'
      setLastError(message)
      throw error
    }
  }, [])

  const sendGoto = useCallback(
    (x: number, y: number, mode: NavigationMode = 'goto') =>
      request('/api/goto', { x, y, mode }),
    [request],
  )
  const sendCommand = useCallback(
    (cmd: 'auto' | 'stop') => request('/api/command', { cmd }),
    [request],
  )
  const sendPlan = useCallback(
    (subgoals: Array<Record<string, unknown>>) =>
      request('/api/plan', { plan_id: 'react-ui', subgoals }),
    [request],
  )

  return {
    snapshot,
    truth,
    geometry,
    connection,
    lastError,
    sendGoto,
    sendCommand,
    sendPlan,
  }
}
