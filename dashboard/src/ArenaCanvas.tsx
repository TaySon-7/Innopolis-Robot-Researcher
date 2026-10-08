import { useEffect, useRef, useState } from 'react'
import type {
  AgentCostmap,
  AgentGeometry,
  CostRun,
  MaskRun,
  AgentTruth,
  NavigationMode,
  ScenarioZone,
} from './agentApi'
import type { LaserScan, Pose2D, SampleState } from './rosbridge'

export type MapMode = 'truth' | 'knowledge' | 'planner'
export type PlannerLayer = 'terrain' | 'wall_cost' | 'total'

interface ArenaCanvasProps {
  pose: Pose2D
  scan: LaserScan | null
  samples: SampleState[]
  trail: Array<{ x: number; y: number }>
  waypoints: Array<[number, number]>
  costmap: AgentCostmap
  geometry: AgentGeometry | null
  truth: AgentTruth | null
  showTrail: boolean
  showPath: boolean
  mapMode: MapMode
  plannerLayer: PlannerLayer
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
  [-1.1, 0, 1.1].map((y) => ({ x, y, r: 0.15 })),
)

type ScreenProjector = (x: number, y: number) => [number, number]
type ArenaPoint = { x: number; y: number }
type ArenaBounds = { xmin: number; xmax: number; ymin: number; ymax: number }

const FALLBACK_BOUNDS: ArenaBounds = {
  xmin: -3.55,
  xmax: 3.55,
  ymin: -3.1,
  ymax: 3.1,
}

function arenaFloor(geometry: AgentGeometry | null): ArenaPoint[] {
  if (!geometry?.floor || geometry.floor.length < 3) return INNER_HEX
  return geometry.floor.map(([x, y]) => ({ x, y }))
}

/**
 * The box the view is fitted to.
 *
 * `bounds` is the extent of the *walls*, not of the drivable floor: with the
 * Gazebo scene it spans 7.3 m while the robot can only reach the middle 5.4 m.
 * Fitting the view to that box shrinks the arena and makes every passage look
 * narrower than it is. So the floor wins when it is a real outline, and
 * otherwise the walls are inset by their own thickness.
 */
function arenaBounds(geometry: AgentGeometry | null): ArenaBounds {
  if (!geometry) return FALLBACK_BOUNDS
  const floor = geometry.floor
  // A real arena outline has at least eight vertices. The Gazebo scene's floor
  // extraction currently yields six and covers only part of the arena, so a
  // short polygon is ignored rather than fitted to.
  if (floor && floor.length >= 8) {
    const xs = floor.map(([x]) => x)
    const ys = floor.map(([, y]) => y)
    return {
      xmin: Math.min(...xs),
      xmax: Math.max(...xs),
      ymin: Math.min(...ys),
      ymax: Math.max(...ys),
    }
  }
  const bounds = geometry.bounds ?? FALLBACK_BOUNDS
  const inset = Math.max(0.1, geometry.wall ?? 0.2)
  return {
    xmin: bounds.xmin + inset,
    xmax: bounds.xmax - inset,
    ymin: bounds.ymin + inset,
    ymax: bounds.ymax - inset,
  }
}

function arenaView(width: number, height: number, bounds: ArenaBounds) {
  const worldWidth = Math.max(0.1, bounds.xmax - bounds.xmin)
  const worldHeight = Math.max(0.1, bounds.ymax - bounds.ymin)
  const worldCenterX = (bounds.xmin + bounds.xmax) / 2
  const worldCenterY = (bounds.ymin + bounds.ymax) / 2
  const scale = Math.min(
    Math.max(1, width - 54) / worldWidth,
    Math.max(1, height - 42) / worldHeight,
  )
  const centerX = width / 2
  const centerY = height / 2
  const screen: ScreenProjector = (x, y) => [
    centerX + (x - worldCenterX) * scale,
    centerY - (y - worldCenterY) * scale,
  ]
  return { scale, screen, worldCenterX, worldCenterY }
}

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

function zoneCentre(zone: ScenarioZone): [number, number] | null {
  if (typeof zone.x === 'number' && typeof zone.y === 'number') return [zone.x, zone.y]
  if (
    typeof zone.x_min === 'number' &&
    typeof zone.x_max === 'number' &&
    typeof zone.y_min === 'number' &&
    typeof zone.y_max === 'number'
  ) {
    return [(zone.x_min + zone.x_max) / 2, (zone.y_min + zone.y_max) / 2]
  }
  return null
}

function terrainColour(value: number, alpha = 0.58): string {
  if (value < 0.98) {
    const blend = Math.max(0, Math.min(1, (value - 0.5) / 0.5))
    const hue = 202 - blend * 42
    return `hsla(${hue}, 72%, 49%, ${alpha})`
  }
  if (value <= 1.08) return `rgba(116, 137, 139, ${alpha * 0.62})`
  const heat = Math.max(0, Math.min(1, (value - 1) / 4))
  const hue = 52 - heat * 49
  return `hsla(${hue}, 88%, 57%, ${alpha})`
}

function runRectangle(
  run: CostRun | MaskRun,
  geometry: AgentGeometry,
  screen: ScreenProjector,
) {
  const [row, start, end] = run
  const resolution = geometry.resolution
  const worldX = geometry.origin[0] + start * resolution
  const worldY = geometry.origin[1] + row * resolution
  const [left, top] = screen(worldX, worldY + resolution)
  const [right, bottom] = screen(
    geometry.origin[0] + (end + 1) * resolution,
    worldY,
  )
  return { left, top, width: right - left, height: bottom - top }
}

function drawValueRuns(
  context: CanvasRenderingContext2D,
  runs: CostRun[],
  geometry: AgentGeometry,
  screen: ScreenProjector,
  colour: (value: number) => string,
) {
  for (const run of runs) {
    const box = runRectangle(run, geometry, screen)
    context.fillStyle = colour(run[3])
    context.fillRect(box.left, box.top, box.width + 0.4, box.height + 0.4)
  }
}

function blockedPattern(context: CanvasRenderingContext2D): CanvasPattern | null {
  const tile = document.createElement('canvas')
  tile.width = 10
  tile.height = 10
  const ink = tile.getContext('2d')
  if (!ink) return null
  ink.fillStyle = 'rgba(255, 96, 108, 0.13)'
  ink.fillRect(0, 0, 10, 10)
  ink.strokeStyle = 'rgba(255, 135, 143, 0.52)'
  ink.lineWidth = 1.2
  ink.beginPath()
  ink.moveTo(-2, 10)
  ink.lineTo(10, -2)
  ink.moveTo(4, 12)
  ink.lineTo(12, 4)
  ink.stroke()
  return context.createPattern(tile, 'repeat')
}

export function ArenaCanvas({
  pose,
  scan,
  samples,
  trail,
  waypoints,
  costmap,
  geometry,
  truth,
  showTrail,
  showPath,
  mapMode,
  plannerLayer,
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

      // The camera is fixed to the world frame. Geometry comes from the exact
      // occupancy map used by navigation, including the extra side facets in
      // the stock TurtleBot3 world.
      const floor = arenaFloor(geometry)
      const bounds = arenaBounds(geometry)
      const pillars = (geometry?.pillars?.length ? geometry.pillars : PILLARS)
        .slice()
        .sort((a, b) => a.x - b.x || a.y - b.y)
      const { scale, screen } = arenaView(rect.width, rect.height, bounds)

      // Solid outer wall and the inner drivable floor.
      if (geometry?.floor?.length) {
        tracePolygon(context, floor, screen)
        context.strokeStyle = '#1a323a'
        context.lineJoin = 'miter'
        context.miterLimit = 4
        const wallDepth = Math.max(0.22, Math.min(0.34, geometry.wall * 2.5))
        context.lineWidth = wallDepth * scale * 2
        context.stroke()
      } else {
        tracePolygon(context, OUTER_HEX, screen)
        context.fillStyle = '#1a323a'
        context.strokeStyle = '#49616a'
        context.lineWidth = 1.3
        context.fill()
        context.stroke()
      }

      tracePolygon(context, floor, screen)
      context.fillStyle = '#07171f'
      context.strokeStyle = '#36535c'
      context.lineWidth = 1
      context.fill()
      context.stroke()

      // Metric grid is clipped to the arena so its frame can never appear to pan.
      context.save()
      tracePolygon(context, floor, screen)
      context.clip()
      context.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace'
      context.textAlign = 'center'
      context.textBaseline = 'top'
      const [, axisY] = screen(0, 0)
      for (let value = -3; value <= 3; value += 1) {
        const [gx, top] = screen(value, bounds.ymax)
        const [, bottom] = screen(value, bounds.ymin)
        const [left, gy] = screen(bounds.xmin, value)
        const [right] = screen(bounds.xmax, value)
        context.strokeStyle = value === 0 ? '#294650' : '#173039'
        context.lineWidth = value === 0 ? 1.2 : 1
        context.beginPath()
        context.moveTo(gx, top)
        context.lineTo(gx, bottom)
        context.moveTo(left, gy)
        context.lineTo(right, gy)
        context.stroke()
        if (value !== 0) {
          context.fillStyle = '#5f7a7b'
          context.fillText(String(value), gx, axisY + 8)
        }
      }
      context.restore()

      if (mapMode === 'knowledge' && geometry) {
        context.save()
        tracePolygon(context, floor, screen)
        context.clip()
        context.fillStyle = 'rgba(103, 112, 118, 0.38)'
        context.fillRect(0, 0, rect.width, rect.height)
        drawValueRuns(context, costmap.knowledge ?? [], geometry, screen, (value) =>
          terrainColour(value, 0.7),
        )
        context.restore()
      }

      if (mapMode === 'planner' && geometry) {
        const runs = costmap[plannerLayer] ?? []
        context.save()
        tracePolygon(context, floor, screen)
        context.clip()
        context.fillStyle = plannerLayer === 'wall_cost'
          ? 'rgba(52, 92, 112, 0.18)'
          : 'rgba(116, 137, 139, 0.18)'
        context.fillRect(0, 0, rect.width, rect.height)
        drawValueRuns(context, runs, geometry, screen, (value) => {
          if (plannerLayer === 'wall_cost') {
            const strength = Math.max(0, Math.min(1, value / 4))
            return `hsla(${205 + strength * 65}, 76%, 60%, ${0.18 + strength * 0.56})`
          }
          return terrainColour(value, 0.68)
        })
        const pattern = blockedPattern(context)
        if (pattern) {
          context.fillStyle = pattern
          for (const run of costmap.blocked ?? []) {
            const box = runRectangle(run, geometry, screen)
            context.fillRect(box.left, box.top, box.width + 0.4, box.height + 0.4)
          }
        }
        context.restore()
      }

      if (mapMode === 'truth' && truth) {
        context.save()
        context.lineWidth = 1.3
        for (const zone of truth.soil_zones ?? []) {
          traceZone(context, zone, screen, scale)
          const multiplier = zone.cost_multiplier ?? 1
          context.fillStyle = terrainColour(multiplier, 0.42)
          context.strokeStyle = terrainColour(multiplier, 0.92)
          context.fill()
          context.stroke()
          const centre = zoneCentre(zone)
          if (centre) {
            const [labelX, labelY] = screen(...centre)
            const label = `×${multiplier.toFixed(1)}`
            context.font = '700 11px ui-monospace, SFMono-Regular, Menlo, monospace'
            context.textAlign = 'center'
            context.textBaseline = 'middle'
            context.lineWidth = 3.5
            context.strokeStyle = 'rgba(5, 14, 20, 0.84)'
            context.strokeText(label, labelX, labelY)
            context.fillStyle = '#fff1cf'
            context.fillText(label, labelX, labelY)
          }
        }
        for (const zone of truth.hazard_zones ?? []) {
          traceZone(context, zone, screen, scale)
          context.setLineDash([5, 4])
          context.fillStyle = 'rgba(255, 96, 108, 0.22)'
          context.strokeStyle = 'rgba(255, 96, 108, 0.9)'
          context.fill()
          context.stroke()
          const centre = zoneCentre(zone)
          if (centre) {
            const [labelX, labelY] = screen(...centre)
            context.setLineDash([])
            context.font = '700 9px ui-monospace, SFMono-Regular, Menlo, monospace'
            context.textAlign = 'center'
            context.textBaseline = 'middle'
            context.fillStyle = '#ffadb4'
            context.fillText('ОПАСНО', labelX, labelY)
          }
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

      // Pillars come from the same occupancy map as the navigation costmap.
      pillars.forEach(({ x, y, r }, index) => {
        const [px, py] = screen(x, y)
        const radius = Math.max(7, r * scale)
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
        const [clipLeft, clipTop] = screen(bounds.xmin, bounds.ymax)
        const [clipRight, clipBottom] = screen(bounds.xmax, bounds.ymin)
        context.beginPath()
        context.rect(clipLeft, clipTop, clipRight - clipLeft, clipBottom - clipTop)
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
  }, [costmap, geometry, hover, mapMode, plannerLayer, pose, samples, scan, showPath, showTrail, trail, truth, waypoints])

  const pointFromPointer = (clientX: number, clientY: number) => {
    const canvas = canvasRef.current
    if (!canvas) return null
    const rect = canvas.getBoundingClientRect()
    const floor = arenaFloor(geometry)
    const bounds = arenaBounds(geometry)
    const { scale, worldCenterX, worldCenterY } = arenaView(rect.width, rect.height, bounds)
    const x = worldCenterX + (clientX - rect.left - rect.width / 2) / scale
    const y = worldCenterY + (rect.height / 2 - (clientY - rect.top)) / scale
    return pointInPolygon(x, y, floor) ? { x, y } : null
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
