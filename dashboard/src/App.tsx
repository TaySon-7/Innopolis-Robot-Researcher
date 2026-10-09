import {
  BatteryMedium,
  Bot,
  Box,
  CircleStop,
  Gauge,
  LocateFixed,
  Radio,
  RotateCcw,
  Ruler,
  SatelliteDish,
  Search,
  Settings2,
  Waypoints,
  Zap,
} from 'lucide-react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { AgentHypothesisPanel, AgentJournal, AgentPanel, AgentPlanPanel } from './AgentPanels'
import { ArenaCanvas } from './ArenaCanvas'
import type { MapMode, PlannerLayer } from './ArenaCanvas'
import { useAgentApi } from './agentApi'
import type { AgentScenarioPreview, Difficulty, NavigationBackend, NavigationMode } from './agentApi'
import { useRosbridge } from './rosbridge'
import { ScenarioSetup } from './ScenarioSetup'
import type { ScenarioMode } from './ScenarioSetup'

type Motion = 'forward' | 'reverse' | 'left' | 'right' | null

const MOTION_KEYS: Record<string, Exclude<Motion, null>> = {
  w: 'forward',
  arrowup: 'forward',
  x: 'reverse',
  arrowdown: 'reverse',
  a: 'left',
  arrowleft: 'left',
  d: 'right',
  arrowright: 'right',
}

function fixed(value: number | null, digits: number, suffix = ''): string {
  return value === null ? '—' : `${value.toFixed(digits)}${suffix}`
}

function formatRunTime(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) return 'RUN --:--:--.---'
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  const remaining = seconds % 60
  return `RUN ${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${remaining.toFixed(3).padStart(6, '0')}`
}

function App() {
  const { snapshot, publishVelocity, callTrigger } = useRosbridge()
  const {
    snapshot: agentSnapshot,
    truth: agentTruth,
    geometry: agentGeometry,
    connection: agentConnection,
    lastError: agentError,
    sendGoto,
    sendCommand,
    sendPlan,
    selectNavigationBackend,
    selectScenario,
    previewScenario,
  } = useAgentApi()
  const [view, setView] = useState<'setup' | 'dashboard'>('setup')
  const [difficulty, setDifficulty] = useState<Difficulty>('easy')
  const [difficultyTouched, setDifficultyTouched] = useState(false)
  const [scenarioMode, setScenarioMode] = useState<ScenarioMode>('standard')
  const [scenarioSeed, setScenarioSeed] = useState('7')
  const [scenarioPreview, setScenarioPreview] = useState<AgentScenarioPreview | null>(null)
  const [previewBusy, setPreviewBusy] = useState(false)
  const [previewError, setPreviewError] = useState<string | null>(null)
  const [scenarioBusy, setScenarioBusy] = useState(false)
  const [scenarioError, setScenarioError] = useState<string | null>(null)
  const [linearSpeed, setLinearSpeed] = useState(0.12)
  const [angularSpeed, setAngularSpeed] = useState(0.6)
  const [motion, setMotion] = useState<Motion>(null)
  const manualContextRef = useRef<{ backend?: NavigationBackend; episodeId?: number }>({})
  const [serviceBusy, setServiceBusy] = useState(false)
  const [agentBusy, setAgentBusy] = useState(false)
  const [pendingNavigationBackend, setPendingNavigationBackend] = useState<NavigationBackend | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [showTrail, setShowTrail] = useState(true)
  const [showPath, setShowPath] = useState(true)
  const [mapMode, setMapMode] = useState<MapMode>('knowledge')
  const [plannerLayer, setPlannerLayer] = useState<PlannerLayer>('total')
  const [mapHover, setMapHover] = useState<{ x: number; y: number } | null>(null)
  const connected = snapshot.status === 'connected'
  const scenarioName = useMemo(() => {
    if (scenarioMode === 'standard') return difficulty
    if (!/^\d+$/.test(scenarioSeed)) return null
    const numericSeed = Number(scenarioSeed)
    if (!Number.isSafeInteger(numericSeed) || numericSeed < 0 || numericSeed > 2_147_483_647) {
      return null
    }
    return `${difficulty}@${numericSeed}`
  }, [difficulty, scenarioMode, scenarioSeed])

  const command = useMemo(() => {
    switch (motion) {
      case 'forward':
        return { linear: linearSpeed, angular: 0 }
      case 'reverse':
        return { linear: -linearSpeed, angular: 0 }
      case 'left':
        return { linear: 0, angular: angularSpeed }
      case 'right':
        return { linear: 0, angular: -angularSpeed }
      default:
        return { linear: 0, angular: 0 }
    }
  }, [angularSpeed, linearSpeed, motion])

  const stop = useCallback(() => {
    setMotion(null)
    publishVelocity(0, 0)
  }, [publishVelocity])

  const startMotion = useCallback((next: Motion) => {
    if (motion === next) return
    void sendCommand('stop').catch(() => undefined)
    setMotion(next)
  }, [motion, sendCommand])

  const emergencyStop = useCallback(() => {
    stop()
    void sendCommand('stop')
      .then(() => setNotice('Робот и агент остановлены'))
      .catch(() => setNotice('Ручное движение остановлено, Agent API недоступен'))
  }, [sendCommand, stop])

  useEffect(() => {
    const backend = agentSnapshot.state.navigation?.backend
    const episodeId = agentSnapshot.state.episode_id
    const previous = manualContextRef.current
    const changed = (backend !== undefined && previous.backend !== undefined && backend !== previous.backend)
      || (episodeId !== undefined && previous.episodeId !== undefined && episodeId !== previous.episodeId)
    // Retain the last confirmed values across an empty reset snapshot.
    manualContextRef.current = {
      backend: backend ?? previous.backend,
      episodeId: episodeId ?? previous.episodeId,
    }
    if (changed && motion !== null) stop()
  }, [agentSnapshot.state.navigation?.backend, agentSnapshot.state.episode_id, motion, stop])

  useEffect(() => {
    if (agentBusy || agentConnection !== 'connected' || pendingNavigationBackend === null
      || agentSnapshot.state.navigation?.backend !== pendingNavigationBackend) return
    setNotice(`Агент подтвердил навигацию ${pendingNavigationBackend === 'nav2' ? 'Nav2' : 'Наша (A*)'}`)
    setPendingNavigationBackend(null)
  }, [agentBusy, agentConnection, agentSnapshot.state.navigation?.backend, pendingNavigationBackend])

  useEffect(() => {
    if (view !== 'dashboard' || !connected || !motion) return
    publishVelocity(command.linear, command.angular)
    const timer = window.setInterval(
      () => publishVelocity(command.linear, command.angular),
      100,
    )
    return () => {
      window.clearInterval(timer)
      publishVelocity(0, 0)
    }
  }, [command, connected, motion, publishVelocity, view])

  useEffect(() => {
    const keyDown = (event: KeyboardEvent) => {
      if (view !== 'dashboard') return
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement
        || event.target instanceof HTMLTextAreaElement || event.metaKey || event.ctrlKey) return
      const key = event.key.toLowerCase()
      if (key === 's' || key === ' ') {
        event.preventDefault()
        emergencyStop()
        return
      }
      const next = MOTION_KEYS[key]
      if (next && connected) {
        event.preventDefault()
        // Revoked holds require a new press; repeated arrow keys must not scroll either.
        if (!event.repeat) startMotion(next)
      }
    }
    const keyUp = (event: KeyboardEvent) => {
      const released = MOTION_KEYS[event.key.toLowerCase()]
      if (released) setMotion((current) => (current === released ? null : current))
    }
    const safetyStop = () => stop()
    window.addEventListener('keydown', keyDown)
    window.addEventListener('keyup', keyUp)
    window.addEventListener('blur', safetyStop)
    window.addEventListener('pointerup', safetyStop)
    document.addEventListener('visibilitychange', safetyStop)
    return () => {
      window.removeEventListener('keydown', keyDown)
      window.removeEventListener('keyup', keyUp)
      window.removeEventListener('blur', safetyStop)
      window.removeEventListener('pointerup', safetyStop)
      document.removeEventListener('visibilitychange', safetyStop)
    }
  }, [connected, emergencyStop, startMotion, stop, view])

  useEffect(() => {
    if (!connected) setMotion(null)
  }, [connected])

  useEffect(() => {
    if (difficultyTouched) return
    const [current, activeSeed] = agentSnapshot.scenario?.split('@') ?? []
    if (current === 'easy' || current === 'medium' || current === 'hard') {
      setDifficulty(current)
      if (activeSeed && /^\d+$/.test(activeSeed)) {
        setScenarioMode('seeded')
        setScenarioSeed(String(Number(activeSeed)))
      } else {
        setScenarioMode('standard')
      }
    }
  }, [agentSnapshot.scenario, difficultyTouched])

  useEffect(() => {
    if (view !== 'setup') return
    if (!scenarioName) {
      setScenarioPreview(null)
      setPreviewBusy(false)
      setPreviewError('Seed должен быть числом от 0 до 2147483647')
      return
    }
    if (agentConnection !== 'connected') {
      setScenarioPreview(null)
      setPreviewBusy(false)
      setPreviewError(null)
      return
    }
    let cancelled = false
    setScenarioPreview(null)
    setPreviewError(null)
    setPreviewBusy(true)
    const timer = window.setTimeout(() => {
      void previewScenario(scenarioName)
        .then((next) => {
          if (!cancelled) setScenarioPreview(next)
        })
        .catch((error) => {
          if (!cancelled) {
            setPreviewError(error instanceof Error ? error.message : 'Не удалось создать превью')
          }
        })
        .finally(() => {
          if (!cancelled) setPreviewBusy(false)
        })
    }, scenarioMode === 'seeded' ? 250 : 0)
    return () => {
      cancelled = true
      window.clearTimeout(timer)
    }
  }, [agentConnection, previewScenario, scenarioMode, scenarioName, view])

  const runService = async (service: '/did/collect' | '/did/finish') => {
    stop()
    setServiceBusy(true)
    try {
      const result = await callTrigger(service)
      const prefix = result.success ? 'Готово' : 'Не выполнено'
      setNotice(`${prefix}: ${result.message || 'судья ответил без сообщения'}`)
    } catch (error) {
      setNotice(error instanceof Error ? error.message : 'Ошибка обращения к судье')
    } finally {
      setServiceBusy(false)
    }
  }

  const runAgentAction = async (
    action: () => Promise<unknown>,
    successMessage: string,
  ) => {
    stop()
    setAgentBusy(true)
    try {
      await action()
      setNotice(successMessage)
    } catch (error) {
      setNotice(error instanceof Error ? error.message : 'Команда агента не отправлена')
    } finally {
      setAgentBusy(false)
    }
  }

  const startScenario = async () => {
    if (!scenarioName || scenarioPreview?.name !== scenarioName) return
    stop()
    setScenarioBusy(true)
    setScenarioError(null)
    try {
      await selectScenario(scenarioName)
      setNotice(`Сценарий ${scenarioName.toUpperCase()} создан — карта готова к работе`)
      setView('dashboard')
    } catch (error) {
      setScenarioError(error instanceof Error ? error.message : 'Не удалось создать сценарий')
    } finally {
      setScenarioBusy(false)
    }
  }

  const restartScenario = async () => {
    const activeScenario = agentSnapshot.scenario
    if (!activeScenario || scenarioBusy) return
    stop()
    setScenarioBusy(true)
    setNotice(`Перезапускаю Gazebo и сценарий ${activeScenario.toUpperCase()}…`)
    try {
      await selectScenario(activeScenario)
      setNotice(`Gazebo и сценарий ${activeScenario.toUpperCase()} перезапущены`)
    } catch (error) {
      setNotice(error instanceof Error ? error.message : 'Не удалось перезапустить симуляцию')
    } finally {
      setScenarioBusy(false)
    }
  }

  const openScenarioSetup = () => {
    stop()
    void sendCommand('stop').catch(() => undefined)
    setScenarioError(null)
    setView('setup')
  }

  const navigateFromMap = useCallback((x: number, y: number, mode: NavigationMode) => {
    stop()
    setAgentBusy(true)
    void sendGoto(Number(x.toFixed(3)), Number(y.toFixed(3)), mode)
      .then(() => setNotice(
        mode === 'search'
          ? `Агент исследует область ${x.toFixed(2)} · ${y.toFixed(2)}`
          : `Маршрут построен к ${x.toFixed(2)} · ${y.toFixed(2)}`,
      ))
      .catch((error) => setNotice(
        error instanceof Error ? error.message : 'Не удалось построить маршрут',
      ))
      .finally(() => setAgentBusy(false))
  }, [sendGoto, stop])

  const connectionLabel = {
    connected: 'ROS подключён',
    connecting: 'Подключение к ROS',
    disconnected: 'ROS недоступен',
  }[snapshot.status]
  const battery = snapshot.battery
  const batteryPercent = battery === null ? 0 : Math.max(0, Math.min(100, (battery / 60) * 100))
  const samplePercent = Math.max(0, Math.min(100, (snapshot.sampleSignal ?? 0) * 100))
  const normalizedHeading = ((snapshot.pose.yaw * 180) / Math.PI + 360) % 360
  const heading = normalizedHeading > 359.95 ? 0 : normalizedHeading
  const mapTrail = agentSnapshot.trail.length
    ? agentSnapshot.trail.map(([x, y]) => ({ x, y }))
    : snapshot.trail
  const visibleSamples = useMemo(() => {
    if (mapMode !== 'truth') return snapshot.samples.filter((sample) => sample.collected)
    if (!agentTruth?.samples?.length) return snapshot.samples
    return agentTruth.samples.map((sample) => {
      const judged = snapshot.samples.find(
        (item) => Math.hypot(item.x - sample.x, item.y - sample.y) < 0.01,
      )
      return { ...sample, collected: judged?.collected ?? false }
    })
  }, [agentTruth, mapMode, snapshot.samples])
  const scanReturns = snapshot.scan
    ? snapshot.scan.ranges.reduce(
        (count, range) =>
          Number.isFinite(range) &&
          range >= snapshot.scan!.rangeMin &&
          range <= snapshot.scan!.rangeMax
            ? count + 1
            : count,
        0,
      )
    : 0
  const missionMessage = snapshot.score.finished
    ? 'Миссия завершена на базе'
    : snapshot.score.collected >= snapshot.score.samplesTotal
      ? 'Все образцы собраны — возвращайтесь на базу'
      : (snapshot.sampleSignal ?? 0) > 0.8
        ? 'Сильный сигнал: образец совсем рядом'
        : 'Исследуйте арену и следите за сигналом датчика'

  if (view === 'setup') {
    return (
      <ScenarioSetup
        selected={difficulty}
        mode={scenarioMode}
        seed={scenarioSeed}
        scenarioName={scenarioName}
        activeScenario={agentSnapshot.scenario}
        connection={agentConnection}
        geometry={agentGeometry}
        preview={scenarioPreview}
        previewBusy={previewBusy}
        previewError={previewError}
        busy={scenarioBusy}
        error={scenarioError}
        onSelect={(next) => {
          setDifficultyTouched(true)
          setDifficulty(next)
        }}
        onModeChange={(next) => {
          setDifficultyTouched(true)
          setScenarioMode(next)
        }}
        onSeedChange={(next) => {
          setDifficultyTouched(true)
          setScenarioSeed(next)
        }}
        onRandomSeed={() => {
          const value = new Uint32Array(1)
          crypto.getRandomValues(value)
          setDifficultyTouched(true)
          setScenarioSeed(String(value[0] % 2_147_483_648))
        }}
        onStart={() => void startScenario()}
        onJoin={() => {
          // Just look at the episode that is already running. Selecting a
          // difficulty must not be the price of watching it.
          setView('dashboard')
        }}
        onAbort={() => {
          void runAgentAction(
            () => sendCommand('stop'),
            'Прогон прерван: робот и планировщик остановлены',
          )
          setView('dashboard')
        }}
      />
    )
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark" aria-hidden="true">
            <Bot size={23} strokeWidth={1.8} />
          </div>
          <div>
            <h1>Robot Researcher</h1>
            <p>TB3 BURGER <span>·</span> {agentSnapshot.scenario?.toUpperCase() ?? 'LEVEL 0'}</p>
          </div>
        </div>
        <div className="connection-group">
          <button className="topbar-action" type="button" onClick={openScenarioSetup}>
            <Settings2 size={15} />Сценарий
          </button>
          <button
            className="topbar-action topbar-restart"
            type="button"
            disabled={scenarioBusy || agentConnection !== 'connected' || !agentSnapshot.scenario}
            onClick={() => void restartScenario()}
            title="Пересоздать Burger на базе в Gazebo и заново запустить текущий сценарий"
          >
            <RotateCcw className={scenarioBusy ? 'is-spinning' : undefined} size={15} />
            {scenarioBusy ? 'Перезапуск…' : 'Перезапустить'}
          </button>
          <div className={`connection connection--${agentConnection}`} role="status">
            <span className="connection-dot" />
            <span>{agentConnection === 'connected' ? 'Агент готов' : 'Agent API недоступен'}</span>
            <span className="connection-address">/api</span>
          </div>
          <div className={`connection connection--${snapshot.status}`} role="status">
            <span className="connection-dot" />
            <span>{connectionLabel}</span>
            <span className="connection-address">/ros</span>
          </div>
        </div>
      </header>

      <main className="dashboard-grid">
        <section className="panel map-panel">
          <div className="panel-header map-header">
            <div>
              <span className="eyebrow">LIVE MAP</span>
              <h2>Арена, лидар и маршрут</h2>
            </div>
            <div className="map-meta">
              <span>{formatRunTime(agentSnapshot.score.t)}</span>
              <span>
                {mapHover
                  ? `КУРСОР ${mapHover.x.toFixed(2)} · ${mapHover.y.toFixed(2)}`
                  : `X ${snapshot.pose.x.toFixed(2)} · Y ${snapshot.pose.y.toFixed(2)}`}
              </span>
              <span>LIDAR {scanReturns}/{snapshot.scan?.ranges.length ?? 0}</span>
            </div>
          </div>
          <div className="map-stage">
            <ArenaCanvas
              pose={snapshot.pose}
              scan={snapshot.scan}
              samples={visibleSamples}
              trail={mapTrail}
              waypoints={agentSnapshot.state.navigation?.waypoints ?? []}
              costmap={agentSnapshot.costmap}
              geometry={agentGeometry}
              truth={agentTruth}
              showTrail={showTrail}
              showPath={showPath}
              mapMode={mapMode}
              plannerLayer={plannerLayer}
              onNavigate={navigateFromMap}
              onHover={setMapHover}
            />
            <div className="map-layers" aria-label="Слои карты">
              <div className="map-mode-group" role="radiogroup" aria-label="Режим карты">
                <button type="button" aria-pressed={mapMode === 'truth'} onClick={() => setMapMode('truth')}>Истина</button>
                <button type="button" aria-pressed={mapMode === 'knowledge'} onClick={() => setMapMode('knowledge')}>Знания</button>
                <button type="button" aria-pressed={mapMode === 'planner'} onClick={() => setMapMode('planner')}>Планировщик</button>
              </div>
              {mapMode === 'planner' && (
                <div className="planner-layer-group" role="radiogroup" aria-label="Слой стоимости">
                  <button type="button" aria-pressed={plannerLayer === 'terrain'} onClick={() => setPlannerLayer('terrain')}>terrain</button>
                  <button type="button" aria-pressed={plannerLayer === 'wall_cost'} onClick={() => setPlannerLayer('wall_cost')}>wall_cost</button>
                  <button type="button" aria-pressed={plannerLayer === 'total'} onClick={() => setPlannerLayer('total')}>сумма</button>
                </div>
              )}
              <div className="map-overlay-toggles">
                <label><input type="checkbox" checked={showTrail} onChange={(event) => setShowTrail(event.target.checked)} />След</label>
                <label><input type="checkbox" checked={showPath} onChange={(event) => setShowPath(event.target.checked)} />Маршрут</label>
              </div>
            </div>
            <div className={`map-cost-legend map-cost-legend--${mapMode}`}>
              {mapMode === 'truth' && <><span>×1</span><i className="cost-gradient" /><span>×5</span><b>актуально @ {Math.floor(agentTruth?.at ?? 0)} с</b></>}
              {mapMode === 'knowledge' && <><span>неизвестно</span><i className="unknown-swatch" /><span>&lt;1</span><i className="cost-gradient" /><span>&gt;1</span></>}
              {mapMode === 'planner' && <><span>{plannerLayer}</span><i className={plannerLayer === 'wall_cost' ? 'wall-gradient' : 'cost-gradient'} /><span>дороже</span><i className="blocked-swatch" /><span>блок</span></>}
            </div>
            <div className="map-status">
              <span><i className="legend-dot robot" />Burger</span>
              <span><i className="legend-dot lidar" />Лидар /scan</span>
              <span><i className="legend-dot pillar" />Столбы ×9</span>
              {mapMode === 'truth' && <span><i className="legend-dot sample" />Образец</span>}
              <span><i className="legend-dot route" />Маршрут</span>
              <span>{agentGeometry?.source === 'gazebo_scene' ? 'Геометрия Gazebo' : 'Геометрия карты'}</span>
            </div>
            <div className="map-scale"><span />1 м</div>
            <div className="map-click-hint">Клик — ехать · Shift+клик — искать образец</div>
          </div>
        </section>

        <section className="panel telemetry-panel telemetry-panel--wide">
          <div className="panel-header">
            <div>
              <span className="eyebrow">TELEMETRY</span>
              <h2>Состояние робота</h2>
            </div>
            <Radio className={connected ? 'live-icon' : ''} size={19} />
          </div>

          <div className="telemetry-body">
            <div className="meter-card">
              <div className="meter-heading">
                <span><BatteryMedium size={16} />Батарея</span>
                <strong>{battery === null ? '—' : `${battery.toFixed(1)} / 60`}</strong>
              </div>
              <div className="meter-track" role="progressbar" aria-valuenow={battery ?? 0} aria-valuemax={60}>
                <span className="meter-fill battery-fill" style={{ width: `${batteryPercent}%` }} />
              </div>
            </div>

            <div className="meter-card">
              <div className="meter-heading">
                <span><SatelliteDish size={16} />Сигнал образца</span>
                <strong>{fixed(snapshot.sampleSignal, 3)}</strong>
              </div>
              <div className="meter-track" role="progressbar" aria-valuenow={snapshot.sampleSignal ?? 0} aria-valuemax={1}>
                <span className="meter-fill sample-fill" style={{ width: `${samplePercent}%` }} />
              </div>
            </div>

            <div className="mission-card">
              <div className="mission-label"><span className="pulse" />ТЕКУЩАЯ ЗАДАЧА</div>
              <p>{notice ?? snapshot.lastEvent ?? missionMessage}</p>
            </div>

            <div className="telemetry-cards">
              <article><LocateFixed size={16} /><span>Позиция</span><strong>{snapshot.pose.x.toFixed(2)} · {snapshot.pose.y.toFixed(2)}</strong><small>метры, world</small></article>
              <article><RotateCcw size={16} /><span>Курс</span><strong>{heading.toFixed(1)}°</strong><small>от оси X</small></article>
              <article><Gauge size={16} /><span>Скорость</span><strong>{snapshot.linearVelocity.toFixed(3)}</strong><small>м/с</small></article>
              <article><Waypoints size={16} /><span>Вращение</span><strong>{snapshot.angularVelocity.toFixed(3)}</strong><small>рад/с</small></article>
              <article><Ruler size={16} /><span>Путь</span><strong>{snapshot.score.distanceTravelled.toFixed(2)}</strong><small>метра</small></article>
              <article><Box size={16} /><span>Собрано</span><strong>{snapshot.score.collected} / {snapshot.score.samplesTotal}</strong><small>образца</small></article>
              <article><Gauge size={16} /><span>Счёт</span><strong>{agentSnapshot.score.score?.toFixed(0) ?? '—'}</strong><small>баллов</small></article>
              <article><Bot size={16} /><span>Столкновения</span><strong>{agentSnapshot.score.collisions ?? 0}</strong><small>штрафных</small></article>
              <article><Search size={16} /><span>Ложные сборы</span><strong>{agentSnapshot.score.false_collects ?? 0}</strong><small>попыток</small></article>
              <article><Zap size={16} /><span>Опасные зоны</span><strong>{agentSnapshot.score.hazard_hits ?? 0}</strong><small>попаданий</small></article>
            </div>

            <div className="service-actions">
              <button disabled={!connected || serviceBusy} onClick={() => void runService('/did/collect')}>
                <Box size={16} />Собрать образец
              </button>
              <button disabled={!connected || serviceBusy} onClick={() => void runService('/did/finish')}>
                <LocateFixed size={16} />Финиш на базе
              </button>
            </div>
          </div>
        </section>

        <aside className="side-rail">
          <AgentJournal journal={agentSnapshot.journal} events={agentSnapshot.events} />

          <AgentHypothesisPanel hypotheses={agentSnapshot.hypotheses ?? []} />

          <AgentPlanPanel
            raw={agentSnapshot.plan}
            runningPlanId={agentSnapshot.state?.current?.plan_id ?? ''}
            controlMode={agentSnapshot.state.control_mode}
            runState={agentSnapshot.status.state}
            finished={agentSnapshot.state.finished}
          />

          <AgentPanel
            snapshot={agentSnapshot}
            connection={agentConnection}
            error={agentError}
            busy={agentBusy}
            onNavigationBackend={(backend) => void runAgentAction(
              async () => {
                setPendingNavigationBackend(null)
                await selectNavigationBackend(backend)
                setPendingNavigationBackend(backend)
              },
              `Запрошена навигация ${backend === 'nav2' ? 'Nav2' : 'Наша (A*)'} — ждём подтверждения агента`,
            )}
            onAuto={() => void runAgentAction(() => sendCommand('auto'), 'Автономный режим запущен')}
            onLLM={() => void runAgentAction(() => sendCommand('llm'), 'Управление передано LLM-планировщику')}
            onStop={() => void runAgentAction(() => sendCommand('stop'), 'Агент остановлен')}
            onCollect={() => void runAgentAction(
              () => sendPlan([{ type: 'collect' }]),
              'Команда сбора передана агенту',
            )}
            onHome={() => void runAgentAction(
              () => sendPlan([{ type: 'return_to_base' }]),
              'Агент возвращается на базу',
            )}
          />
        </aside>

        <section className="panel control-panel">
          <div className="panel-header control-heading">
            <div>
              <span className="eyebrow">MANUAL CONTROL</span>
              <h2>Безопасное управление</h2>
            </div>
            <p><Zap size={14} />Команда идёт, пока удерживается клавиша</p>
          </div>

          <div className="control-body">
            <div className="drive-column">
              <div className="drive-pad" aria-label="Управление движением">
                <DriveButton label="W" motion="forward" current={motion} disabled={!connected} onStart={startMotion} onStop={stop} />
                <DriveButton label="A" motion="left" current={motion} disabled={!connected} onStart={startMotion} onStop={stop} />
                <button className="drive-key stop-key" type="button" disabled={!connected} onClick={emergencyStop} aria-label="Остановить робота и агента">S</button>
                <DriveButton label="D" motion="right" current={motion} disabled={!connected} onStart={startMotion} onStop={stop} />
                <DriveButton label="X" motion="reverse" current={motion} disabled={!connected} onStart={startMotion} onStop={stop} />
              </div>
              <span className="keyboard-note">Также работают стрелки</span>
            </div>

            <div className="speed-controls">
              <label>
                <span><b>Линейная скорость</b><output>{linearSpeed.toFixed(2)} м/с</output></span>
                <input
                  type="range"
                  min="0.02"
                  max="0.22"
                  step="0.01"
                  value={linearSpeed}
                  onChange={(event) => setLinearSpeed(Number(event.target.value))}
                />
                <small><i>0.02</i><i>0.22</i></small>
              </label>
              <label>
                <span><b>Угловая скорость</b><output>{angularSpeed.toFixed(1)} рад/с</output></span>
                <input
                  type="range"
                  min="0.1"
                  max="1.5"
                  step="0.1"
                  value={angularSpeed}
                  onChange={(event) => setAngularSpeed(Number(event.target.value))}
                />
                <small><i>0.1</i><i>1.5</i></small>
              </label>
            </div>

            <button className="emergency-stop" type="button" disabled={!connected} onClick={emergencyStop}>
              <span><CircleStop size={25} />СТОП</span>
              <small>мгновенный ноль</small>
            </button>
          </div>
        </section>
      </main>

      <footer>
        <span>ROS 2 Jazzy</span><i />
        <span>Gazebo Harmonic</span><i />
        <span>rosbridge :9090</span>
        <span className="footer-right">Схема арены · координаты в метрах</span>
      </footer>
    </div>
  )
}

interface DriveButtonProps {
  label: string
  motion: Exclude<Motion, null>
  current: Motion
  disabled: boolean
  onStart: (motion: Motion) => void
  onStop: () => void
}

function DriveButton({ label, motion, current, disabled, onStart, onStop }: DriveButtonProps) {
  return (
    <button
      className={`drive-key drive-${motion}${current === motion ? ' is-active' : ''}`}
      type="button"
      disabled={disabled}
      onPointerDown={(event) => {
        event.currentTarget.setPointerCapture(event.pointerId)
        onStart(motion)
      }}
      onPointerUp={onStop}
      onPointerCancel={onStop}
      onLostPointerCapture={onStop}
      aria-label={`Движение: ${motion}`}
    >
      {label}
    </button>
  )
}

export default App
