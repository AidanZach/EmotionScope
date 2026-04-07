import { useRef, useEffect } from 'react'
import { oklchToRgb } from '../utils/oklch'

/**
 * Valence spectrum with dual animated markers for user + model.
 * Markers smoothly interpolate to new positions via spring physics.
 */

const STOPS = [
  { pos: 0.00, L: 0.25, C: 0.14, H: 15 },
  { pos: 0.10, L: 0.30, C: 0.18, H: 10 },
  { pos: 0.20, L: 0.35, C: 0.12, H: 300 },
  { pos: 0.30, L: 0.45, C: 0.16, H: 25 },
  { pos: 0.40, L: 0.40, C: 0.10, H: 270 },
  { pos: 0.50, L: 0.58, C: 0.03, H: 240 },
  { pos: 0.60, L: 0.60, C: 0.03, H: 250 },
  { pos: 0.70, L: 0.65, C: 0.08, H: 220 },
  { pos: 0.80, L: 0.75, C: 0.10, H: 85 },
  { pos: 0.90, L: 0.82, C: 0.15, H: 85 },
  { pos: 1.00, L: 0.82, C: 0.15, H: 85 },
]

function valenceToX(valence, width, pad) {
  return pad + ((valence + 1) / 2) * (width - pad * 2)
}

// Simple spring: approaches target smoothly
function springStep(current, target, speed = 0.08) {
  return current + (target - current) * speed
}

export default function ValenceStrip({
  userValence = null,
  modelValence = null,
  width = 260,
  height = 22,
}) {
  const canvasRef = useRef(null)
  const animRef = useRef(null)
  // Animated marker positions (interpolated, not snapped)
  const markersRef = useRef({
    userX: width / 2,
    modelX: width / 2,
    userTarget: width / 2,
    modelTarget: width / 2,
  })

  // Update targets when props change
  useEffect(() => {
    const pad = 6
    markersRef.current.userTarget =
      userValence !== null ? valenceToX(userValence, width, pad) : -10
    markersRef.current.modelTarget =
      modelValence !== null ? valenceToX(modelValence, width, pad) : -10
  }, [userValence, modelValence, width])

  // Draw loop with animated markers
  useEffect(() => {
    const cv = canvasRef.current
    if (!cv) return
    const ctx = cv.getContext('2d')
    const dpr = window.devicePixelRatio || 1
    cv.width = width * dpr
    cv.height = (height + 18) * dpr
    ctx.scale(dpr, dpr)
    cv.style.width = `${width}px`
    cv.style.height = `${height + 18}px`

    const pad = 6
    const barH = height

    // Pre-render the gradient (doesn't change)
    const gradCanvas = document.createElement('canvas')
    gradCanvas.width = width
    gradCanvas.height = barH
    const gctx = gradCanvas.getContext('2d')
    const grad = gctx.createLinearGradient(pad, 0, width - pad, 0)
    for (const stop of STOPS) {
      const [r, g, b] = oklchToRgb(stop.L, stop.C, stop.H)
      grad.addColorStop(stop.pos, `rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)})`)
    }
    gctx.fillStyle = grad
    gctx.beginPath()
    gctx.roundRect(pad, 0, width - pad * 2, barH, 4)
    gctx.fill()

    function draw() {
      const m = markersRef.current
      // Smooth step toward targets
      m.userX = springStep(m.userX, m.userTarget, 0.1)
      m.modelX = springStep(m.modelX, m.modelTarget, 0.1)

      ctx.clearRect(0, 0, width, height + 18)

      // Draw cached gradient
      ctx.drawImage(gradCanvas, 0, 0)

      // Draw markers
      const drawMarker = (x, label, offset) => {
        if (x < 0) return // offscreen = no data

        // Line on the bar
        ctx.fillStyle = '#fff'
        ctx.shadowColor = '#fff'
        ctx.shadowBlur = 4
        ctx.beginPath()
        ctx.roundRect(x - 1, 1, 2, barH - 2, 1)
        ctx.fill()
        ctx.shadowBlur = 0

        // Triangle below
        const triY = barH + 2
        ctx.fillStyle = '#888'
        ctx.beginPath()
        ctx.moveTo(x, triY)
        ctx.lineTo(x - 3, triY + 4)
        ctx.lineTo(x + 3, triY + 4)
        ctx.closePath()
        ctx.fill()

        // Label
        ctx.fillStyle = '#777780'
        ctx.font = '9px -apple-system, "Segoe UI", system-ui, sans-serif'
        ctx.textAlign = 'center'
        ctx.fillText(label, x, triY + 14)
      }

      drawMarker(m.userX, 'you', 0)
      drawMarker(m.modelX, 'model', 0)

      animRef.current = requestAnimationFrame(draw)
    }

    animRef.current = requestAnimationFrame(draw)

    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current)
    }
  }, [width, height])

  return (
    <canvas
      ref={canvasRef}
      style={{ display: 'block', margin: '0 auto' }}
    />
  )
}
