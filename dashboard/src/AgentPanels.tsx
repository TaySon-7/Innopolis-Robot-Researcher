import {
  Bot,
  BrainCircuit,
  House,
  Play,
  Search,
  Square,
} from 'lucide-react'
import { useMemo, useState } from 'react'
import { describeSubgoal, parsePlan } from './agentApi'
import type {
  AgentConnection,
  AgentControlMode,
  AgentHypothesis,
  AgentJournalEntry,
  AgentSnapshot,
  JudgeEvent,
  NavigationBackend,
} from './agentApi'

interface AgentPanelProps {
  snapshot: AgentSnapshot
  connection: AgentConnection
  error: string | null
  busy: boolean
  onAuto: () => void
  onLLM: () => void
  onStop: () => void
  onCollect: () => void
  onHome: () => void
  onNavigationBackend: (backend: NavigationBackend) => void
}

const RUN_LABEL: Record<string, string> = {
  idle: 'Ожидание',
  running: 'Выполняет план',
  done: 'Готов',
  failed: 'Ошибка',
  preempted: 'Остановлен',
}

const CONTROL_LABEL: Record<AgentControlMode, string> = {
  llm: 'LLM-планировщик',
  fallback: 'Резервный алгоритм',
  autonomous: 'Автономная политика',
  manual: 'Ручное управление',
  stopped: 'Остановлен',
}

export function AgentPanel({
  snapshot,
  connection,
  error,
  busy,
  onAuto,
  onLLM,
  onStop,
  onCollect,
  onHome,
  onNavigationBackend,
}: AgentPanelProps) {
  const state = snapshot.state
  const status = snapshot.status
  const autonomous = state.control_mode === 'autonomous'
    || (!state.control_mode && state.current?.plan_id === 'auto')
  const controlLabel = state.finished ? 'Эпизод завершён'
    : state.control_mode ? CONTROL_LABEL[state.control_mode]
      : autonomous ? CONTROL_LABEL.autonomous : 'Режим не сообщён'
  const navigation = state.navigation ?? {}
  const navigationReady = connection === 'connected' && navigation.ready === true
  const navigationMessage = connection !== 'connected'
    ? 'Agent API недоступен — состояние навигации не подтверждено'
    : navigation.reason || (navigation.ready === true ? 'Готова к движению'
      : navigation.ready === false ? 'Навигация не готова' : 'Ожидание состояния навигации')
  const runState = status.state ?? 'idle'
  const anomalies = [
    ['battery_deviation', 'Расход вне прогноза'],
    ['penalties_burst', 'Серия штрафов'],
    ['sensor_noise_up', 'Шум датчика вырос'],
  ] as const

  return (
    <section className="panel agent-panel">
      <div className="panel-header">
        <div>
          <span className="eyebrow">AGENT NAVIGATION</span>
          <h2>Автономный агент</h2>
        </div>
        <div className={`agent-state agent-state--${runState}`}>
          <span />{RUN_LABEL[runState] ?? runState}
        </div>
      </div>
      <div className="agent-body">
        <div className="agent-now">
          <BrainCircuit size={18} />
          <div>
            <strong>Управление: {controlLabel}</strong>
            <strong>{status.subgoal || state.current?.type || 'Ожидает команду'}</strong>
            <span>
              {error ?? status.reason ?? (
                connection === 'connected'
                  ? `Навигация: ${navigation.status ?? 'idle'}`
                  : 'Agent API недоступен'
              )}
            </span>
          </div>
        </div>

        <div className="agent-navigation">
          <label htmlFor="navigation-backend">Навигация</label>
          <select
            id="navigation-backend"
            value={navigation.backend ?? ''}
            disabled={busy || connection !== 'connected' || !navigation.available_backends?.length}
            aria-describedby="navigation-status navigation-switch-hint"
            onChange={(event) => {
              const backend = event.target.value
              if (backend === 'custom' || backend === 'nav2') onNavigationBackend(backend)
            }}
          >
            {!navigation.backend && <option value="" disabled>Ожидание данных</option>}
            <option value="custom" disabled={!navigation.available_backends?.includes('custom')}>Наша (A*)</option>
            <option value="nav2" disabled={!navigation.available_backends?.includes('nav2')}>Nav2</option>
          </select>
          <span id="navigation-status" role="status" className={navigationReady ? 'is-ready' : 'is-warning'}>
            {navigationMessage}
          </span>
          <small id="navigation-switch-hint">Смена навигации останавливает текущий план. Сценарий сохраняется.</small>
        </div>

        <div className="agent-metrics">
          <span>Перепланирований <b>{navigation.replans ?? 0}</b></span>
          <span>Точек маршрута <b>{navigation.waypoints?.length ?? 0}</b></span>
          <span>Цена возврата <b>{state.return_cost_estimate?.toFixed(1) ?? '—'}</b></span>
        </div>

        <div className="agent-flags">
          {anomalies.map(([key, label]) => (
            <span className={state.anomaly?.[key] ? 'is-warning' : ''} key={key}>
              {label}
            </span>
          ))}
        </div>

        <div className="agent-actions">
          <button
            className="agent-primary"
            disabled={busy || connection !== 'connected'}
            onClick={autonomous ? onLLM : onAuto}
            title={autonomous
              ? 'Вернуть управление LLM-планировщику'
              : 'Передать управление автономной политике агента'}
          >
            <Play size={15} />{autonomous ? 'Вернуть LLM' : 'Автономно'}
          </button>
          {!autonomous && state.control_mode !== 'llm' && (
            <button disabled={busy || connection !== 'connected'} onClick={onLLM}
              title="Передать управление LLM-планировщику">
              <BrainCircuit size={15} />LLM
            </button>
          )}
          <button className="agent-stop" disabled={busy || connection !== 'connected'} onClick={onStop}>
            <Square size={14} />Стоп
          </button>
          <button disabled={busy || connection !== 'connected'} onClick={onCollect}>
            <Search size={15} />Собрать здесь
          </button>
          <button disabled={busy || connection !== 'connected'} onClick={onHome}>
            <House size={15} />На базу
          </button>
        </div>

        <p className="agent-map-hint"><Bot size={13} />Клик по карте — ехать, Shift+клик — искать и собрать</p>
      </div>
    </section>
  )
}

const JOURNAL_KIND: Record<string, string> = {
  llm: 'LLM',
  robot: 'робот',
  search: 'поиск',
  collect: 'сбор',
  hypothesis: 'гипотеза',
  result: 'вывод',
  decision: 'решение',
  agent: 'агент',
}

const EVENT_LABEL: Record<string, string> = {
  collision: 'Столкновение',
  false_collect: 'Ложный сбор',
  hazard_hit: 'Опасная зона',
  sample_collected: 'Образец собран',
}

function journalStatus(entry: AgentJournalEntry) {
  if (!entry.status || entry.status === 'open') return ''
  return entry.status === 'confirmed'
    ? ' · подтверждена'
    : entry.status === 'rejected'
      ? ' · отвергнута'
      : ` · ${entry.status}`
}

function eventDetails(event: JudgeEvent) {
  return Object.entries(event)
    .filter(([key]) => key !== 'event' && key !== 't')
    .map(([key, value]) => `${key}: ${String(value)}`)
    .join(' · ')
}

interface AgentJournalProps {
  journal: AgentJournalEntry[]
  events: JudgeEvent[]
}

interface AgentHypothesisProps {
  hypotheses: AgentHypothesis[]
}

const HYPOTHESIS_STATUS: Record<string, string> = {
  open: 'проверяется',
  confirmed: 'подтверждена',
  rejected: 'не подтвердилась',
}

/**
 * The experiment's hypotheses, as a table rather than as prose.
 *
 * This is the part of the run a judge reads to decide whether the agent was
 * doing science or driving around. A claim is only worth showing next to the
 * rule that would settle it and the verdict that came back, so all three are
 * on the row rather than behind a click — a hypothesis with no verdict is
 * visibly unfinished, which is what it is.
 */
export function AgentHypothesisPanel({ hypotheses }: AgentHypothesisProps) {
  const counts = useMemo(() => {
    const tally = { open: 0, confirmed: 0, rejected: 0 }
    for (const item of hypotheses) {
      if (item.status === 'confirmed') tally.confirmed += 1
      else if (item.status === 'rejected') tally.rejected += 1
      else tally.open += 1
    }
    return tally
  }, [hypotheses])

  return (
    <section className="panel hypothesis-panel">
      <div className="panel-header">
        <div>
          <span className="eyebrow">НАУЧНЫЙ ЦИКЛ</span>
          <h2>Гипотезы</h2>
        </div>
        <div className="hypothesis-counts">
          <span>в работе <b>{counts.open}</b></span>
          <span>подтверждено <b>{counts.confirmed}</b></span>
          <span>отброшено <b>{counts.rejected}</b></span>
        </div>
      </div>
      <div className="hypothesis-list">
        {!hypotheses.length && (
          <p className="hypothesis-empty">
            Гипотез пока нет. Планировщик формулирует их, когда у данных
            появляется смысл, который можно опровергнуть измерением.
          </p>
        )}
        {hypotheses.slice().reverse().map((item) => (
          <article className="hypothesis" key={item.id}>
            <header>
              <span className={`hypothesis-status is-${item.status ?? 'open'}`}>
                {HYPOTHESIS_STATUS[item.status ?? 'open'] ?? item.status}
              </span>
              <strong>{item.claim}</strong>
            </header>
            <dl>
              <dt>Проверка</dt>
              <dd>{item.testable}</dd>
              <dt>Замер</dt>
              <dd>{item.measurement}</dd>
            </dl>
            {item.verdict && <p className="hypothesis-verdict">{item.verdict}</p>}
          </article>
        ))}
      </div>
    </section>
  )
}

interface AgentPlanProps {
  raw: string
  /** The plan the executor says it is actually running right now. */
  runningPlanId: string
  controlMode?: AgentControlMode
  runState?: string
  finished?: boolean
}

/**
 * The last published plan, as it is on the wire.
 *
 * Shown verbatim on purpose. During a demo the interesting part is not that a
 * plan exists but which subgoals it names and why, and that reasoning is the
 * planner's only outward-facing account of itself.
 *
 * The stale check matters more than it looks. The dashboard keeps the last
 * plan string for ever, so when the planner hands the episode to the agent's
 * own behaviour the panel would still show a plan while the robot drives
 * something else entirely. Without the check the screen claims the model is
 * driving when it is not.
 */
export function AgentPlanPanel({ raw, runningPlanId, controlMode, runState, finished }: AgentPlanProps) {
  const plan = useMemo(() => parsePlan(raw), [raw])
  const [open, setOpen] = useState(true)

  const active = !!plan && !!runningPlanId && runningPlanId === plan.plan_id
    && runState === 'running' && !finished && controlMode !== 'stopped'
    && controlMode !== 'autonomous'
  const source = plan?.source ?? 'unknown'
  const sourceLabel: Record<string, string> = {
    llm: 'LLM', fallback: 'резервный алгоритм', budget: 'контроль запаса энергии',
    manual: 'ручная команда', signal: 'алгоритм по сигналу',
    auto_collect: 'автоматический сбор', unknown: 'не указан',
  }
  const decision = plan?.decision
  const formatEnergy = (value: number | null | undefined) => value == null ? '—' : value.toFixed(1)

  return (
    <section className="panel plan-panel">
      <div className="panel-header">
        <div>
          <span className="eyebrow">ПЛАН ДЕЙСТВИЙ</span>
          <h2>{plan?.plan_id ?? 'плана нет'}</h2>
        </div>
        <button
          type="button"
          className="panel-toggle"
          aria-expanded={open}
          onClick={() => setOpen((value) => !value)}
        >
          {open ? 'свернуть' : 'развернуть'}
        </button>
      </div>

      {plan && (
        <p className={`plan-stale${active ? ' is-live' : ''}`}>
          {active ? 'Выполняется' : 'Последний опубликованный план'}.
          {' '}Источник: {sourceLabel[source] ?? source}.
          {controlMode && <> Управление: {CONTROL_LABEL[controlMode]}.</>}
        </p>
      )}

      {!plan && (
        <p className="plan-empty">
          План ещё не опубликован.
          {controlMode && <> Управление: {CONTROL_LABEL[controlMode]}.</>}
        </p>
      )}

      {plan && open && (
        <>
          {plan.goal_selection && (
            <p className="plan-why">Цель: {plan.goal_selection.goal_id}.
              {decision && <> Требуется по оценке: {formatEnergy(decision.required_battery)},
                {' '}батарея при выборе: {formatEnergy(decision.battery)}.
                {' '}Движение: {formatEnergy(decision.energy_to_goal)},
                {' '}поиск: {formatEnergy(decision.energy_search)},
                {' '}возврат: {formatEnergy(decision.energy_home)}.
                {' '}Итог включает запас; это оценка, а не гарантия возврата.</>}
            </p>
          )}
          <ol className="plan-subgoals">
            {plan.subgoals?.map((subgoal, index) => (
              <li key={index}>
                <span className="plan-subgoal-type">{subgoal.type}</span>
                <span className="plan-subgoal-body">{describeSubgoal(subgoal)}</span>
              </li>
            ))}
          </ol>
          {plan.explanation && <p className="plan-why">{plan.explanation}</p>}
        </>
      )}
    </section>
  )
}

export function AgentJournal({ journal, events }: AgentJournalProps) {
  const [tab, setTab] = useState<'journal' | 'events'>('journal')
  const items = tab === 'journal' ? journal : events

  return (
    <section className="panel journal-panel">
      <div className="journal-header">
        <div className="journal-tabs" role="tablist" aria-label="Журнал агента">
          <button className={tab === 'journal' ? 'is-active' : ''} onClick={() => setTab('journal')}>
            Журнал агента
          </button>
          <button className={tab === 'events' ? 'is-active' : ''} onClick={() => setTab('events')}>
            События судьи
          </button>
        </div>
        <span>{items.length}</span>
      </div>
      <div className="journal-feed">
        {!items.length && (
          <div className="journal-empty">
            {tab === 'journal'
              ? 'Здесь появятся гипотезы, выводы и решения агента.'
              : 'Штрафов и собранных образцов пока нет.'}
          </div>
        )}
        {tab === 'journal'
          ? journal.slice().reverse().map((entry, index) => (
              <article className="journal-entry" key={`${entry.t ?? 0}-${index}`}>
                <time>{entry.t === undefined ? '—' : `${Math.round(entry.t)} с`}</time>
                <div>
                  <strong>
                    <i className={`journal-tag journal-tag--${entry.kind ?? 'agent'}`}>
                      {JOURNAL_KIND[entry.kind ?? ''] ?? entry.kind ?? 'агент'}{journalStatus(entry)}
                    </i>
                    {entry.title || 'Запись агента'}
                  </strong>
                  {entry.text && <p>{entry.text}</p>}
                </div>
              </article>
            ))
          : events.slice().reverse().map((event, index) => (
              <article className="journal-entry" key={`${event.t ?? 0}-${index}`}>
                <time>{event.t === undefined ? '—' : `${Math.round(event.t)} с`}</time>
                <div>
                  <strong>{EVENT_LABEL[event.event ?? ''] ?? event.event ?? 'Событие'}</strong>
                  {eventDetails(event) && <p>{eventDetails(event)}</p>}
                </div>
              </article>
            ))}
      </div>
    </section>
  )
}
