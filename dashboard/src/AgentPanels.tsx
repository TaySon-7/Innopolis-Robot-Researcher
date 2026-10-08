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
  AgentJournalEntry,
  AgentSnapshot,
  JudgeEvent,
} from './agentApi'

interface AgentPanelProps {
  snapshot: AgentSnapshot
  connection: AgentConnection
  error: string | null
  busy: boolean
  onAuto: () => void
  onStop: () => void
  onCollect: () => void
  onHome: () => void
}

const RUN_LABEL: Record<string, string> = {
  idle: 'Ожидание',
  running: 'Выполняет план',
  done: 'Готов',
  failed: 'Ошибка',
  preempted: 'Остановлен',
}

export function AgentPanel({
  snapshot,
  connection,
  error,
  busy,
  onAuto,
  onStop,
  onCollect,
  onHome,
}: AgentPanelProps) {
  const state = snapshot.state
  const status = snapshot.status
  const navigation = state.navigation ?? {}
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
          <button className="agent-primary" disabled={busy || connection !== 'connected'} onClick={onAuto}>
            <Play size={15} />Автономно
          </button>
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

interface AgentPlanProps {
  raw: string
  /** The plan the executor says it is actually running right now. */
  runningPlanId: string
}

/**
 * The plan the LLM last published, as it is on the wire.
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
export function AgentPlanPanel({ raw, runningPlanId }: AgentPlanProps) {
  const plan = useMemo(() => parsePlan(raw), [raw])
  const [open, setOpen] = useState(true)

  const autonomous = runningPlanId === 'auto'
  const stale = !autonomous && !!plan && !!runningPlanId
    && runningPlanId !== plan.plan_id

  return (
    <section className="panel plan-panel">
      <div className="panel-header">
        <div>
          <span className="eyebrow">ПЛАН LLM</span>
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

      {autonomous && (
        <p className="plan-stale is-live">
          Исполняет автономный режим — LLM сейчас не управляет. План ниже
          последний, что он опубликовал.
        </p>
      )}
      {stale && (
        <p className="plan-stale">
          Устарело: исполнитель выполняет {runningPlanId}.
        </p>
      )}

      {!plan && (
        <p className="plan-empty">
          Исполнитель работает по своей политике. Планировщик либо не запущен,
          либо ещё не опубликовал план.
        </p>
      )}

      {plan && open && (
        <>
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
