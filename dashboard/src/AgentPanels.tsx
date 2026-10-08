import {
  Bot,
  BrainCircuit,
  House,
  Play,
  Search,
  Square,
} from 'lucide-react'
import { useState } from 'react'
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
