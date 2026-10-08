import {
  AlertTriangle,
  ArrowRight,
  Bot,
  Check,
  FlaskConical,
  Gauge,
  Layers3,
  MapPinned,
  Radio,
  RefreshCw,
  Sparkles,
} from 'lucide-react'
import type {
  AgentConnection,
  AgentGeometry,
  AgentScenarioPreview,
  Difficulty,
  ScenarioPreviewEvent,
  ScenarioZone,
} from './agentApi'

export type ScenarioMode = 'standard' | 'seeded'

interface DifficultySpec {
  id: Difficulty
  title: string
  subtitle: string
  samples: number
  soils: number
  events: string
  shortEvents: string
  tone: 'easy' | 'medium' | 'hard'
}

const DIFFICULTIES: DifficultySpec[] = [
  {
    id: 'easy',
    title: 'Easy',
    subtitle: 'Знакомство с ареной',
    samples: 3,
    soils: 1,
    events: 'Нет',
    shortEvents: 'Стабильная среда без событий',
    tone: 'easy',
  },
  {
    id: 'medium',
    title: 'Medium',
    subtitle: 'Больше целей и грунтов',
    samples: 5,
    soils: 3,
    events: 'Нет',
    shortEvents: 'Стабильная среда без событий',
    tone: 'medium',
  },
  {
    id: 'hard',
    title: 'Hard',
    subtitle: 'Динамическая среда',
    samples: 7,
    soils: 4,
    events: 'Смена грунта, новая опасность, сбой датчика',
    shortEvents: '3 скрытых события во время прогона',
    tone: 'hard',
  },
]

const SAMPLE_POINTS: Record<Difficulty, Array<[number, number]>> = {
  easy: [[35, 57], [56, 42], [73, 59]],
  medium: [[29, 40], [43, 45], [57, 59], [73, 43], [51, 25]],
  hard: [[29, 40], [43, 45], [57, 59], [73, 43], [51, 25], [41, 72], [62, 72]],
}

interface ScenarioSetupProps {
  selected: Difficulty
  mode: ScenarioMode
  seed: string
  scenarioName: string | null
  activeScenario: string | null
  connection: AgentConnection
  geometry: AgentGeometry | null
  preview: AgentScenarioPreview | null
  previewBusy: boolean
  previewError: string | null
  busy: boolean
  error: string | null
  onSelect: (difficulty: Difficulty) => void
  onModeChange: (mode: ScenarioMode) => void
  onSeedChange: (seed: string) => void
  onRandomSeed: () => void
  onStart: () => void
}

function normalizeScenario(value: string | null): Difficulty | null {
  const name = value?.split('@')[0]
  return name === 'easy' || name === 'medium' || name === 'hard' ? name : null
}

function sampleWord(count: number): string {
  return count === 3 ? 'образца' : 'образцов'
}

function soilWord(count: number): string {
  return count === 1 ? 'грунтовая зона' : 'грунтовые зоны'
}

function ArenaPreview({ difficulty }: { difficulty: Difficulty }) {
  return (
    <svg className="scenario-arena-preview" viewBox="0 0 100 100" aria-hidden="true">
      <polygon points="20,22 50,8 80,22 91,50 80,78 50,92 20,78 9,50" />
      {[32, 50, 68].flatMap((x) => [34, 50, 66].map((y) => (
        <circle key={`${x}-${y}`} className="scenario-pillar" cx={x} cy={y} r="2.4" />
      )))}
      {SAMPLE_POINTS[difficulty].map(([x, y], index) => (
        <circle key={`${x}-${y}`} className="scenario-sample" cx={x} cy={y} r="2.1" data-index={index} />
      ))}
      <path className="scenario-route" d="M22 58 C34 76, 48 63, 55 52 S72 38, 79 52" />
      <circle className="scenario-robot" cx="22" cy="58" r="3.6" />
    </svg>
  )
}

function zoneCentre(zone: ScenarioZone): [number, number] {
  if (zone.shape === 'circle') return [zone.x ?? 0, zone.y ?? 0]
  return [
    ((zone.x_min ?? 0) + (zone.x_max ?? 0)) / 2,
    ((zone.y_min ?? 0) + (zone.y_max ?? 0)) / 2,
  ]
}

function ZoneShape({ zone, className }: { zone: ScenarioZone; className: string }) {
  if (zone.shape === 'circle') {
    return (
      <circle
        className={className}
        cx={zone.x ?? 0}
        cy={zone.y ?? 0}
        r={zone.radius ?? 0}
      />
    )
  }
  return (
    <rect
      className={className}
      x={zone.x_min ?? 0}
      y={zone.y_min ?? 0}
      width={Math.max(0, (zone.x_max ?? 0) - (zone.x_min ?? 0))}
      height={Math.max(0, (zone.y_max ?? 0) - (zone.y_min ?? 0))}
      rx="0.04"
    />
  )
}

function eventText(event: ScenarioPreviewEvent): string {
  const time = `${Math.round(event.at)} с`
  if (event.type === 'soil_change') {
    return `${time} · ${String(event.zone)} станет ×${event.cost_multiplier?.toFixed(1)}`
  }
  if (event.type === 'hazard_appear') return `${time} · появится опасная зона`
  if (event.type === 'sensor_fault') {
    return `${time} · шум датчика на ${Math.round(event.duration ?? 0)} с`
  }
  return `${time} · ${event.type}`
}

function ExactScenarioPreview({
  preview,
  geometry,
  busy,
  error,
}: {
  preview: AgentScenarioPreview | null
  geometry: AgentGeometry | null
  busy: boolean
  error: string | null
}) {
  const bounds = geometry?.bounds ?? { xmin: -3.3, xmax: 3.3, ymin: -3.0, ymax: 3.0 }
  const width = bounds.xmax - bounds.xmin
  const height = bounds.ymax - bounds.ymin
  const floor = geometry?.floor ?? [
    [-2.8, -1.7], [-1.7, -2.5], [1.7, -2.5], [2.8, -1.7],
    [2.8, 1.7], [1.7, 2.5], [-1.7, 2.5], [-2.8, 1.7],
  ]
  const changes = new Map(
    (preview?.events ?? [])
      .filter((event) => event.type === 'soil_change' && typeof event.zone === 'string')
      .map((event) => [event.zone as string, event]),
  )

  return (
    <section className="scenario-exact-preview" aria-live="polite">
      <div className="scenario-preview-heading">
        <div>
          <span className="eyebrow">EXACT LAYOUT</span>
          <h3>Предпросмотр раскладки</h3>
        </div>
        {preview && <span className="scenario-preview-name">{preview.name.toUpperCase()}</span>}
      </div>

      <div className="scenario-preview-grid">
        <div className="scenario-preview-map">
          <svg
            viewBox={`${bounds.xmin} ${-bounds.ymax} ${width} ${height}`}
            preserveAspectRatio="xMidYMid meet"
            role="img"
            aria-label="Точное расположение грунтовых зон и образцов на арене"
          >
            <defs>
              <clipPath id="scenario-floor-clip" clipPathUnits="userSpaceOnUse">
                <polygon points={floor.map(([x, y]) => `${x},${y}`).join(' ')} transform="scale(1 -1)" />
              </clipPath>
              <pattern id="scenario-hazard-pattern" width="0.14" height="0.14" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
                <line x1="0" y1="0" x2="0" y2="0.14" />
              </pattern>
            </defs>
            <g transform="scale(1 -1)">
              <polygon className="scenario-map-floor" points={floor.map(([x, y]) => `${x},${y}`).join(' ')} />
              <g clipPath="url(#scenario-floor-clip)">
                {preview?.soil_zones.map((zone) => (
                  <ZoneShape key={zone.id} zone={zone} className="scenario-map-soil" />
                ))}
                {preview?.future_hazard_zones.map((zone) => (
                  <ZoneShape key={zone.id} zone={zone} className="scenario-map-hazard" />
                ))}
              </g>
              {geometry?.pillars.map((pillar, index) => (
                <circle
                  key={`${pillar.x}-${pillar.y}-${index}`}
                  className="scenario-map-pillar"
                  cx={pillar.x}
                  cy={pillar.y}
                  r={pillar.r}
                />
              ))}
              {preview?.samples.map((sample) => (
                <g key={sample.id ?? `${sample.x}-${sample.y}`}>
                  <circle className="scenario-map-sample-ring" cx={sample.x} cy={sample.y} r="0.14" />
                  <circle className="scenario-map-sample-dot" cx={sample.x} cy={sample.y} r="0.075" />
                </g>
              ))}
              {preview?.base && (
                <g>
                  <circle className="scenario-map-base-ring" cx={preview.base.x} cy={preview.base.y} r="0.20" />
                  <circle className="scenario-map-base-dot" cx={preview.base.x} cy={preview.base.y} r="0.075" />
                </g>
              )}
            </g>

            {preview?.soil_zones.map((zone) => {
              const [x, y] = zoneCentre(zone)
              const change = zone.id ? changes.get(zone.id) : undefined
              const current = zone.cost_multiplier?.toFixed(1) ?? '1.0'
              const label = change
                ? `×${current}→${change.cost_multiplier?.toFixed(1)}`
                : `×${current}`
              return <text key={`label-${zone.id}`} className="scenario-map-soil-label" x={x} y={-y}>{label}</text>
            })}
            {preview?.future_hazard_zones.map((zone) => {
              const [x, y] = zoneCentre(zone)
              return (
                <text key={`hazard-label-${zone.id}`} className="scenario-map-hazard-label" x={x} y={-y}>
                  ! {Math.round(zone.appears_at ?? 0)}с
                </text>
              )
            })}
            {preview?.samples.map((sample, index) => (
              <text
                key={`sample-label-${sample.id ?? index}`}
                className="scenario-map-sample-label"
                x={sample.x + 0.13}
                y={-sample.y - 0.12}
              >
                {sample.id?.toUpperCase() ?? index + 1}
              </text>
            ))}
            {preview?.base && (
              <text className="scenario-map-base-label" x={preview.base.x + 0.24} y={-preview.base.y + 0.05}>БАЗА</text>
            )}
          </svg>
          {busy && (
            <div className="scenario-preview-overlay">
              <RefreshCw className="is-spinning" size={18} /> Генерируем точную раскладку…
            </div>
          )}
          {!busy && error && (
            <div className="scenario-preview-overlay is-error"><AlertTriangle size={18} /> {error}</div>
          )}
          {!busy && !error && !preview && (
            <div className="scenario-preview-overlay">Введите корректный seed</div>
          )}
        </div>

        <aside className="scenario-preview-details">
          <div className="scenario-preview-stats">
            <span><b>{preview?.samples.length ?? '—'}</b> образцов</span>
            <span><b>{preview?.soil_zones.length ?? '—'}</b> грунтов</span>
            <span><b>{preview?.events.length ?? '—'}</b> событий</span>
          </div>
          <div className="scenario-preview-legend">
            <span><i className="is-sample" />Образец</span>
            <span><i className="is-soil" />Медленный грунт</span>
            <span><i className="is-hazard" />Будущая опасность</span>
            <span><i className="is-base" />База</span>
          </div>
          <div className="scenario-preview-events">
            <span className="eyebrow">DYNAMIC EVENTS</span>
            {preview?.events.length
              ? preview.events.map((event, index) => (
                  <p key={`${event.type}-${event.at}-${index}`}><b>{index + 1}</b>{eventText(event)}</p>
                ))
              : <p className="is-empty">Во время прогона среда не меняется</p>}
          </div>
        </aside>
      </div>
    </section>
  )
}

export function ScenarioSetup({
  selected,
  mode,
  seed,
  scenarioName,
  activeScenario,
  connection,
  geometry,
  preview,
  previewBusy,
  previewError,
  busy,
  error,
  onSelect,
  onModeChange,
  onSeedChange,
  onRandomSeed,
  onStart,
}: ScenarioSetupProps) {
  const active = normalizeScenario(activeScenario)
  const chosen = DIFFICULTIES.find((item) => item.id === selected) ?? DIFFICULTIES[0]
  const connected = connection === 'connected'

  return (
    <div className="scenario-shell">
      <header className="topbar scenario-topbar">
        <div className="brand">
          <div className="brand-mark" aria-hidden="true"><Bot size={23} strokeWidth={1.8} /></div>
          <div>
            <h1>Robot Researcher</h1>
            <p>TB3 BURGER <span>·</span> SIMULATION SETUP</p>
          </div>
        </div>
        <div className={`connection connection--${connection}`} role="status">
          <span className="connection-dot" />
          <span>{connected ? 'Симулятор готов' : connection === 'connecting' ? 'Подключение' : 'Симулятор недоступен'}</span>
          <span className="connection-address">/api</span>
        </div>
      </header>

      <main className="scenario-main">
        <section className="scenario-intro">
          <div>
            <span className="eyebrow">SIMULATION SETUP</span>
            <h2>Настройка исследовательской миссии</h2>
            <p>
              Выберите сложность и фиксированную либо воспроизводимую seed-раскладку.
              Предпросмотр показывает точные позиции образцов, грунтов и будущих опасностей.
            </p>
          </div>
          <div className="scenario-flow" aria-label="Этапы запуска">
            <span className="is-active"><b>01</b> Сценарий</span>
            <i />
            <span><b>02</b> Карта</span>
            <i />
            <span><b>03</b> Миссия</span>
          </div>
        </section>

        <section className="panel scenario-panel">
          <div className="scenario-panel-heading">
            <div>
              <span className="eyebrow">DIFFICULTY</span>
              <h3>Уровень сложности</h3>
            </div>
            {active && (
              <span className="active-scenario-badge">
                <Radio size={13} /> Сейчас запущен {activeScenario?.toUpperCase()}
              </span>
            )}
          </div>

          <div className="difficulty-grid" role="radiogroup" aria-label="Уровень сложности">
            {DIFFICULTIES.map((item) => {
              const checked = selected === item.id
              return (
                <button
                  key={item.id}
                  className={`difficulty-card difficulty-card--${item.tone}${checked ? ' is-selected' : ''}`}
                  type="button"
                  role="radio"
                  aria-checked={checked}
                  onClick={() => onSelect(item.id)}
                >
                  <div className="difficulty-card-top">
                    <span className="difficulty-index">0{DIFFICULTIES.indexOf(item) + 1}</span>
                    <span className="difficulty-check">{checked && <Check size={14} strokeWidth={3} />}</span>
                  </div>
                  <ArenaPreview difficulty={item.id} />
                  <div className="difficulty-copy">
                    <strong>{item.title}</strong>
                    <span>{item.subtitle}</span>
                  </div>
                  <div className="difficulty-metrics">
                    <span><FlaskConical size={14} /><b>{item.samples}</b> {sampleWord(item.samples)}</span>
                    <span><Layers3 size={14} /><b>{item.soils}</b> {soilWord(item.soils)}</span>
                  </div>
                  <p className={item.id === 'hard' ? 'has-events' : ''}>
                    {item.id === 'hard' ? <AlertTriangle size={14} /> : <Gauge size={14} />}
                    {item.shortEvents}
                  </p>
                </button>
              )
            })}
          </div>

          <div className="scenario-generation">
            <div className="scenario-generation-copy">
              <span className="eyebrow">LAYOUT SOURCE</span>
              <strong>Как создать раскладку</strong>
              <p>Одинаковые сложность и seed всегда дают одинаковую карту.</p>
            </div>
            <div className="scenario-mode-switch" role="radiogroup" aria-label="Тип раскладки">
              <button
                type="button"
                role="radio"
                aria-checked={mode === 'standard'}
                className={mode === 'standard' ? 'is-active' : ''}
                onClick={() => onModeChange('standard')}
              >
                Стандартная
                <small>из ТЗ</small>
              </button>
              <button
                type="button"
                role="radio"
                aria-checked={mode === 'seeded'}
                className={mode === 'seeded' ? 'is-active' : ''}
                onClick={() => onModeChange('seeded')}
              >
                По seed
                <small>случайная</small>
              </button>
            </div>
            <div className={`scenario-seed-control${mode === 'seeded' ? ' is-enabled' : ''}`}>
              <label htmlFor="scenario-seed">SEED</label>
              <input
                id="scenario-seed"
                type="text"
                inputMode="numeric"
                pattern="[0-9]*"
                maxLength={10}
                value={seed}
                disabled={mode !== 'seeded'}
                onChange={(event) => onSeedChange(event.target.value.replace(/\D/g, ''))}
                aria-describedby="scenario-seed-help"
              />
              <button
                type="button"
                disabled={mode !== 'seeded'}
                onClick={onRandomSeed}
                title="Создать другой seed"
              >
                <RefreshCw size={15} /> Новая
              </button>
              <small id="scenario-seed-help">
                {scenarioName ?? 'Введите число от 0 до 2147483647'}
              </small>
            </div>
          </div>

          <ExactScenarioPreview
            preview={preview}
            geometry={geometry}
            busy={previewBusy}
            error={previewError}
          />

          <div className="scenario-lower">
            <div className="scenario-table-wrap">
              <table className="scenario-table">
                <thead>
                  <tr>
                    <th>Сценарий</th>
                    <th>Образцы</th>
                    <th>Грунтовые зоны с медленной проходимостью</th>
                    <th>События во время прогона</th>
                  </tr>
                </thead>
                <tbody>
                  {DIFFICULTIES.map((item) => (
                    <tr key={item.id} className={selected === item.id ? 'is-selected' : ''}>
                      <th>{item.id}</th>
                      <td>{item.samples}</td>
                      <td>{item.soils}</td>
                      <td>{item.events}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <aside className="scenario-launch-card">
              <span className="eyebrow">SELECTED SCENARIO</span>
              <div className="scenario-launch-title">
                <strong>{chosen.title}{mode === 'seeded' ? ` @ ${seed || '—'}` : ''}</strong>
                <span>{chosen.samples} / {chosen.soils}</span>
              </div>
              <p>
                {chosen.samples} {sampleWord(chosen.samples)}, {chosen.soils} {soilWord(chosen.soils)}.
                {' '}{chosen.id === 'hard' ? 'Среда меняется во время миссии.' : 'Среда остаётся стабильной.'}
              </p>
              {error && <div className="scenario-error" role="alert"><AlertTriangle size={15} />{error}</div>}
              <button
                className="scenario-start"
                type="button"
                disabled={!connected || busy || previewBusy || !scenarioName || !preview || Boolean(previewError)}
                onClick={onStart}
              >
                {busy ? <RefreshCw className="is-spinning" size={17} /> : <Sparkles size={17} />}
                <span>{busy ? 'Создаём сценарий…' : `Запустить ${scenarioName?.toUpperCase() ?? chosen.title}`}</span>
                {!busy && <ArrowRight size={17} />}
              </button>
              {!connected && !error && <small>Кнопка станет доступна после подключения Agent API</small>}
              {connected && previewBusy && <small>Сначала дождитесь точного предпросмотра</small>}
              {connected && !previewBusy && preview && <small><MapPinned size={12} /> Запустится именно показанная раскладка</small>}
            </aside>
          </div>
        </section>
      </main>
    </div>
  )
}
