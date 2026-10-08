import { useEffect, useRef, useState } from 'react'
import type { AgentGeometry, AgentTruth, NavigationMode, ScenarioZone } from './agentApi'
import type { LaserScan, Pose2D, SampleState } from './rosbridge'

interface ArenaCanvasProps {
  pose: Pose2D
  scan: LaserScan | null
  samples: SampleState[]
  trail: Array<{ x: number; y: number }>
  waypoints: Array<[number, number]>
  costmapRuns: Array<[number, number, number, number]>
  geometry: AgentGeometry | null
  truth: AgentTruth | null
  showTrail: boolean
  showPath: boolean
  showCostmap: boolean
  showTruth: boolean
  onNavigate?: (x: number, y: number, mode: NavigationMode) => void
  onHover?: (point: { x: number; y: number } | null) => void
}

const BASE = { x: -2, y: -0.5 }

// Exact world geometry from turtlebot3_world/model.sdf and wall.dae.
// wall.dae uses inches and is scaled by 0.25 in SDF.
const OUTER_HEX = [
  { x: 3.2996, y: 0 },
  { x: 1.6498, y: 2.8575 },
  { x: -1.6498, y: 2.8575 },
  { x: -3.2996, y: 0 },
  { x: -1.6498, y: -2.8575 },
  { x: 1.6498, y: -2.8575 },
]
const INNER_HEX = [
  { x: 2.9329, y: 0 },
  { x: 1.4665, y: 2.54 },
  { x: -1.4665, y: 2.54 },
  { x: -2.9329, y: 0 },
  { x: -1.4665, y: -2.54 },
  { x: 1.4665, y: -2.54 },
]
const PILLARS = [-1.1, 0, 1.1].flatMap((x) =>
  [-1.1, 0, 1.1].map((y) => ({ x, y })),
)

type ScreenProjector = (x: number, y: number) => [number, number]

function tracePolygon(
  context: CanvasRenderingContext2D,
  points: Array<{ x: number; y: number }>,
  screen: ScreenProjector,
) {
  context.beginPath()
  points.forEach((point, index) => {
    const [x, y] = screen(point.x, point.y)
    if (index === 0) context.moveTo(x, y)
    else context.lineTo(x, y)
  })
  context.closePath()
}

function pointInPolygon(x: number, y: number, polygon: Array<{ x: number; y: number }>) {
  let inside = false
  for (let index = 0, previous = polygon.length - 1; index < polygon.length; previous = index++) {
    const a = polygon[index]
    const b = polygon[previous]
    const crosses = a.y > y !== b.y > y && x < ((b.x - a.x) * (y - a.y)) / (b.y - a.y) + a.x
    if (crosses) inside = !inside
  }
  return inside
}

function traceZone(
  context: CanvasRenderingContext2D,
  zone: ScenarioZone,
  screen: ScreenProjector,
  scale: number,
) {
  context.beginPath()
  if (
    zone.shape === 'circle' &&
    typeof zone.x === 'number' &&
    typeof zone.y === 'number' &&
    typeof zone.radius === 'number'
  ) {
    const [x, y] = screen(zone.x, zone.y)
    context.arc(x, y, zone.radius * scale, 0, Math.PI * 2)
    return
  }
  if (
    typeof zone.x_min === 'number' &&
    typeof zone.x_max === 'number' &&
    typeof zone.y_min === 'number' &&
    typeof zone.y_max === 'number'
  ) {
    const [left, top] = screen(zone.x_min, zone.y_max)
    const [right, bottom] = screen(zone.x_max, zone.y_min)
    context.roundRect(left, top, right - left, bottom - top, 7)
  }
}

export function ArenaCanvas({
  pose,
  scan,
  samples,
  trail,
  waypoints,
  costmapRuns,
  geometry,
  truth,
  showTrail,
  showPath,
  showCostmap,
  showTruth,
  onNavigate,
  onHover,
}: ArenaCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [hover, setHover] = useState<{ x: number; y: number } | null>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const context = canvas.getContext('2d')
    if (!context) return

    const draw = () => {
      const rect = canvas.getBoundingClientRect()
      const ratio = window.devicePixelRatio || 1
      canvas.width = Math.max(1, Math.floor(rect.width * ratio))
      canvas.height = Math.max(1, Math.floor(rect.height * ratio))
      context.setTransform(ratio, 0, 0, ratio, 0, 0)
      context.clearRect(0, 0, rect.width, rect.height)

      // The camera is fixed to the world frame. Only the robot and scan change.
      const scale = Math.min((rect.width - 54) / 7.15, (rect.height - 42) / 6.25)
      const centerX = rect.width / 2
      const centerY = rect.height / 2
      const screen: ScreenProjector = (x, y) => [
        centerX + x * scale,
        centerY - y * scale,
      ]

      // Solid outer wall and the inner drivable floor.
      tracePolygon(context, OUTER_HEX, screen)
      context.fillStyle = '#1a323a'
      context.strokeStyle = '#49616a'
      context.lineWidth = 1.3
      context.fill()
      context.stroke()

      tracePolygon(context, INNER_HEX, screen)
      context.fillStyle = '#07171f'
      context.strokeStyle = '#36535c'
      context.lineWidth = 1
      context.fill()
      context.stroke()

      // Metric grid is clipped to the arena so its frame can never appear to pan.
      context.save()
      tracePolygon(context, INNER_HEX, screen)
      context.clip()
      context.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace'
      context.textAlign = 'center'
      context.textBaseline = 'top'
      for (let value = -3; value <= 3; value += 1) {
        const [gx] = screen(value, 0)
        const [, gy] = screen(0, value)
        context.strokeStyle = value === 0 ? '#294650' : '#173039'
        context.lineWidth = value === 0 ? 1.2 : 1
        context.beginPath()
        context.moveTo(gx, centerY - 3.1 * scale)
        context.lineTo(gx, centerY + 3.1 * scale)
        context.moveTo(centerX - 3.55 * scale, gy)
        context.lineTo(centerX + 3.55 * scale, gy)
        context.stroke()
        if (value !== 0) {
          context.fillStyle = '#5f7a7b'
          context.fillText(String(value), gx, centerY + 8)
        }
      }
      context.restore()

      if (showCostmap && geometry && costmapRuns.length) {
        context.save()
        tracePolygon(context, INNER_HEX, screen)
        context.clip()
        const resolution = geometry.resolution
        for (const [row, start, end, value] of costmapRuns) {
          if (value < 1.15) continue
          const worldX = geometry.origin[0] + start * resolution
          const worldY = geometry.origin[1] + row * resolution
          const [left, top] = screen(worldX, worldY + resolution)
          const [right, bottom] = screen(
            geometry.origin[0] + (end + 1) * resolution,
            worldY,
          )
          const alpha = Math.min(0.42, 0.08 + (value - 1) * 0.09)
          context.fillStyle = `rgba(242, 163, 58, ${alpha})`
          context.fillRect(left, top, right - left, bottom - top)
        }
        context.restore()
      }

      if (showTruth && truth) {
        context.save()
        context.lineWidth = 1.3
        context.setLineDash([6, 5])
        for (const zone of truth.soil_zones ?? []) {
          traceZone(context, zone, screen, scale)
          context.fillStyle = 'rgba(242, 163, 58, 0.1)'
          context.strokeStyle = 'rgba(242, 163, 58, 0.8)'
          context.fill()
          context.stroke()
        }
        for (const zone of truth.hazard_zones ?? []) {
          traceZone(context, zone, screen, scale)
          context.fillStyle = 'rgba(255, 96, 108, 0.12)'
          context.strokeStyle = 'rgba(255, 96, 108, 0.9)'
          context.fill()
          context.stroke()
        }
        context.setLineDash([])
        context.restore()
      }

      if (showTrail && trail.length > 1) {
        context.strokeStyle = 'rgba(37, 217, 199, 0.36)'
        context.lineWidth = 2
        context.beginPath()
        trail.forEach((point, index) => {
          const [x, y] = screen(point.x, point.y)
          if (index === 0) context.moveTo(x, y)
          else context.lineTo(x, y)
        })
        context.stroke()
      }

      if (showPath && waypoints.length) {
        context.save()
        context.strokeStyle = 'rgba(110, 231, 168, 0.9)'
        context.lineWidth = 2.2
        context.lineCap = 'round'
        context.lineJoin = 'round'
        context.setLineDash([7, 6])
        context.beginPath()
        const [startX, startY] = screen(pose.x, pose.y)
        context.moveTo(startX, startY)
        waypoints.forEach(([x, y]) => {
          const [pointX, pointY] = screen(x, y)
          context.lineTo(pointX, pointY)
        })
        context.stroke()
        context.setLineDash([])
        const goal = waypoints.at(-1)
        if (goal) {
          const [goalX, goalY] = screen(goal[0], goal[1])
          context.fillStyle = '#6ee7a8'
          context.strokeStyle = 'rgba(110, 231, 168, 0.4)'
          context.lineWidth = 7
          context.beginPath()
          context.arc(goalX, goalY, 4, 0, Math.PI * 2)
          context.stroke()
          context.fill()
        }
        context.restore()
      }

      const [baseX, baseY] = screen(BASE.x, BASE.y)
      context.fillStyle = 'rgba(242, 163, 58, 0.08)'
      context.strokeStyle = '#f2a33a'
      context.lineWidth = 1.5
      context.beginPath()
      context.arc(baseX, baseY, 25, 0, Math.PI * 2)
      context.fill()
      context.stroke()
      context.fillStyle = '#f2a33a'
      context.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace'
      context.fillText('БАЗА', baseX, baseY + 29)

      samples.forEach((sample, index) => {
        const [x, y] = screen(sample.x, sample.y)
        context.globalAlpha = sample.collected ? 0.35 : 1
        context.fillStyle = sample.collected ? '#6ee7a8' : '#bd8aff'
        context.shadowColor = sample.collected ? '#6ee7a8' : '#bd8aff'
        context.shadowBlur = sample.collected ? 0 : 10
        context.beginPath()
        context.arc(x, y, 6, 0, Math.PI * 2)
        context.fill()
        context.shadowBlur = 0
        context.strokeStyle = '#071119'
        context.lineWidth = 2
        context.stroke()
        context.fillStyle = '#a9bdba'
        context.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace'
        context.fillText(String(index + 1), x, y + 10)
        context.globalAlpha = 1
      })

      // Nine 0.15 m radius cylinders from model.sdf, in a 3 × 3 grid.
      PILLARS.forEach(({ x, y }, index) => {
        const [px, py] = screen(x, y)
        const radius = Math.max(7, 0.15 * scale)
        const gradient = context.createRadialGradient(
          px - radius * 0.35,
          py - radius * 0.4,
          radius * 0.15,
          px,
          py,
          radius,
        )
        gradient.addColorStop(0, '#607982')
        gradient.addColorStop(0.55, '#36515a')
        gradient.addColorStop(1, '#1b3038')
        context.fillStyle = gradient
        context.strokeStyle = '#718991'
        context.lineWidth = 1
        context.beginPath()
        context.arc(px, py, radius, 0, Math.PI * 2)
        context.fill()
        context.stroke()
        context.fillStyle = 'rgba(219, 233, 232, 0.42)'
        context.font = '8px ui-monospace, SFMono-Regular, Menlo, monospace'
        context.textBaseline = 'middle'
        context.fillText(String(index + 1), px, py + 0.5)
      })

      if (scan && scan.ranges.length) {
        const hits = scan.ranges.flatMap((range, index) => {
          if (!Number.isFinite(range) || range < scan.rangeMin || range > scan.rangeMax) {
            return []
          }
          const angle = scan.origin.yaw + scan.angleMin + index * scan.angleIncrement
          const worldX = scan.origin.x + range * Math.cos(angle)
          const worldY = scan.origin.y + range * Math.sin(angle)
          const [x, y] = screen(worldX, worldY)
          return [{ index, range, worldX, worldY, x, y }]
        })

        context.save()
        tracePolygon(context, OUTER_HEX, screen)
        context.clip()

        // A restrained subset of rays makes the scan direction readable without
        // turning the arena into a solid blue fan.
        const [originX, originY] = screen(scan.origin.x, scan.origin.y)
        const beamStride = Math.max(1, Math.ceil(hits.length / 48))
        context.strokeStyle = 'rgba(82, 166, 255, 0.16)'
        context.lineWidth = 0.8
        context.beginPath()
        hits.forEach((hit, hitIndex) => {
          if (hitIndex % beamStride !== 0) return
          context.moveTo(originX, originY)
          context.lineTo(hit.x, hit.y)
        })
        context.stroke()

        // Adjacent returns from the same surface form short contours. A distance
        // gate prevents a line from jumping between a pillar and the outer wall.
        context.strokeStyle = 'rgba(127, 187, 255, 0.88)'
        context.lineWidth = 1.45
        context.lineCap = 'round'
        context.lineJoin = 'round'
        context.shadowColor = 'rgba(91, 172, 255, 0.72)'
        context.shadowBlur = 5
        context.beginPath()
        let previous: (typeof hits)[number] | null = null
        hits.forEach((hit) => {
          const joinsPrevious =
            previous !== null &&
            hit.index === previous.index + 1 &&
            Math.hypot(hit.worldX - previous.worldX, hit.worldY - previous.worldY) < 0.22
          if (joinsPrevious) context.lineTo(hit.x, hit.y)
          else context.moveTo(hit.x, hit.y)
          previous = hit
        })
        context.stroke()

        // Draw returns above the arena geometry so contacts on pillars and walls
        // remain visible instead of being hidden by their fills.
        context.fillStyle = '#8bc0ff'
        context.shadowBlur = 4
        hits.forEach((hit) => {
          context.beginPath()
          context.arc(hit.x, hit.y, 1.75, 0, Math.PI * 2)
          context.fill()
        })
        context.restore()
      }

      const [robotX, robotY] = screen(pose.x, pose.y)
      context.save()
      context.translate(robotX, robotY)
      context.rotate(-pose.yaw)
      context.fillStyle = 'rgba(37, 217, 199, 0.12)'
      context.beginPath()
      context.arc(0, 0, 32, 0, Math.PI * 2)
      context.fill()
      context.fillStyle = '#132b34'
      context.strokeStyle = '#3e6570'
      context.lineWidth = 1.5
      context.beginPath()
      context.roundRect(-19, -15, 38, 30, 7)
      context.fill()
      context.stroke()
      context.fillStyle = '#f2a33a'
      context.fillRect(-24, -13, 6, 26)
      context.fillRect(18, -13, 6, 26)
      context.fillStyle = '#25d9c7'
      context.beginPath()
      context.arc(0, 0, 12, 0, Math.PI * 2)
      context.fill()
      context.fillStyle = '#071119'
      context.beginPath()
      context.arc(0, 0, 4, 0, Math.PI * 2)
      context.fill()
      context.strokeStyle = '#25d9c7'
      context.fillStyle = '#25d9c7'
      context.lineWidth = 2.5
      context.beginPath()
      context.moveTo(8, 0)
      context.lineTo(34, 0)
      context.stroke()
      context.beginPath()
      context.moveTo(34, 0)
      context.lineTo(26, -5)
      context.lineTo(26, 5)
      context.closePath()
      context.fill()
      context.restore()

      if (hover) {
        const [hoverX, hoverY] = screen(hover.x, hover.y)
        context.save()
        context.strokeStyle = 'rgba(37, 217, 199, 0.85)'
        context.fillStyle = 'rgba(6, 16, 24, 0.86)'
        context.lineWidth = 1.2
        context.beginPath()
        context.arc(hoverX, hoverY, Math.max(7, 0.12 * scale), 0, Math.PI * 2)
        context.stroke()
        context.beginPath()
        context.moveTo(hoverX - 6, hoverY)
        context.lineTo(hoverX + 6, hoverY)
        context.moveTo(hoverX, hoverY - 6)
        context.lineTo(hoverX, hoverY + 6)
        context.stroke()
        const label = `${hover.x.toFixed(2)} · ${hover.y.toFixed(2)}`
        context.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace'
        const width = context.measureText(label).width + 12
        context.fillRect(hoverX - width / 2, hoverY - 27, width, 17)
        context.fillStyle = '#8eece2'
        context.textAlign = 'center'
        context.textBaseline = 'middle'
        context.fillText(label, hoverX, hoverY - 18.5)
        context.restore()
      }
    }

    const observer = new ResizeObserver(draw)
    observer.observe(canvas)
    draw()
    return () => observer.disconnect()
  }, [costmapRuns, geometry, hover, pose, samples, scan, showCostmap, showPath, showTrail, showTruth, trail, truth, waypoints])

  const pointFromPointer = (clientX: number, clientY: number) => {
    const canvas = canvasRef.current
    if (!canvas) return null
    const rect = canvas.getBoundingClientRect()
    const scale = Math.min((rect.width - 54) / 7.15, (rect.height - 42) / 6.25)
    const x = (clientX - rect.left - rect.width / 2) / scale
    const y = (rect.height / 2 - (clientY - rect.top)) / scale
    return pointInPolygon(x, y, INNER_HEX) ? { x, y } : null
  }

  return (
    <canvas
      ref={canvasRef}
      className="arena-canvas"
      aria-label="Схема арены с положением TurtleBot3, маршрутом, лучами и возвратами лидара"
      onPointerMove={(event) => {
        const point = pointFromPointer(event.clientX, event.clientY)
        setHover(point)
        onHover?.(point)
      }}
      onPointerLeave={() => {
        setHover(null)
        onHover?.(null)
      }}
      onClick={(event) => {
        const point = pointFromPointer(event.clientX, event.clientY)
        if (point) onNavigate?.(point.x, point.y, event.shiftKey ? 'search' : 'goto')
      }}
    />
  )
}
