import { useRef, useEffect, useCallback } from 'react'

/**
 * Interactive conversation arc — click a segment to replay that emotion state.
 *
 * Top band = user emotion, bottom band = model emotion.
 * Most recent on the right. Clicked segment gets a highlight border.
 */
export default function Timeline({
  entries = [],
  selectedIndex = null,
  onSelect = () => {},
  width = 260,
  height = 28,
}) {
  const canvasRef = useRef(null)
  // Store segment layout for hit testing
  const layoutRef = useRef({ startX: 0, segW: 0, gap: 1 })

  useEffect(() => {
    const cv = canvasRef.current
    if (!cv) return
    const ctx = cv.getContext('2d')
    const dpr = window.devicePixelRatio || 1
    cv.width = width * dpr
    cv.height = height * dpr
    ctx.scale(dpr, dpr)
    cv.style.width = `${width}px`
    cv.style.height = `${height}px`

    // Background
    ctx.fillStyle = '#12121a'
    ctx.clearRect(0, 0, width, height)
    ctx.beginPath()
    ctx.roundRect(0, 0, width, height, 4)
    ctx.fill()

    if (!entries.length) return

    const gap = 1
    const segW = Math.max(8, Math.min((width - gap) / entries.length - gap, 28))
    const totalW = entries.length * (segW + gap) - gap
    const startX = width - totalW

    layoutRef.current = { startX, segW, gap }

    for (let i = 0; i < entries.length; i++) {
      const e = entries[i]
      const x = startX + i * (segW + gap)
      const halfH = height / 2

      // Top half: user emotion color
      ctx.fillStyle = e.userColor || '#222'
      ctx.fillRect(x, 1, segW, halfH - 1.5)

      // Bottom half: model emotion color
      ctx.fillStyle = e.modelColor || '#222'
      ctx.fillRect(x, halfH + 0.5, segW, halfH - 1.5)

      // Selected indicator — bright border around the segment
      if (i === selectedIndex) {
        ctx.strokeStyle = '#ffffff'
        ctx.lineWidth = 1.5
        ctx.strokeRect(x - 0.5, 0.5, segW + 1, height - 1)
      }
    }
  }, [entries, selectedIndex, width, height])

  const handleClick = useCallback((e) => {
    const cv = canvasRef.current
    if (!cv || !entries.length) return

    const rect = cv.getBoundingClientRect()
    const x = e.clientX - rect.left
    const { startX, segW, gap } = layoutRef.current

    // Hit test: which segment was clicked?
    for (let i = 0; i < entries.length; i++) {
      const segX = startX + i * (segW + gap)
      if (x >= segX && x <= segX + segW) {
        // Toggle: click same segment again to deselect
        onSelect(i === selectedIndex ? null : i)
        return
      }
    }
    // Clicked outside any segment
    onSelect(null)
  }, [entries, selectedIndex, onSelect])

  return (
    <canvas
      ref={canvasRef}
      onClick={handleClick}
      style={{
        display: 'block',
        margin: '0 auto',
        borderRadius: 4,
        cursor: entries.length ? 'pointer' : 'default',
      }}
    />
  )
}
