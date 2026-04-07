import { useState, useCallback, useRef, useEffect } from 'react'
import ChatPanel from './components/ChatPanel'
import EmotionOrbCanvas from './components/EmotionOrb'
import ValenceStrip from './components/ValenceStrip'
import Timeline from './components/Timeline'
import DetailPanel from './components/DetailPanel'
import ExplainerPanel from './components/ExplainerPanel'
import Legend from './components/Legend'
import TestGallery from './components/TestGallery'
import { emotionToColor } from './utils/palette'
import './app.css'

function EmotionLabel({ emotionState }) {
  if (!emotionState) return <div className="orb-emotion-label">—</div>
  const top = (emotionState.top_emotions || []).slice(0, 3)
  if (!top.length) return <div className="orb-emotion-label">—</div>

  return (
    <div className="orb-emotion-breakdown">
      {top.map(([name, score]) => (
        <div key={name} className="orb-emotion-row">
          <span className="orb-emotion-swatch" style={{ background: emotionToColor(name) }} />
          <span className="orb-emotion-name">{name}</span>
          <span className="orb-emotion-score">{score.toFixed(3)}</span>
        </div>
      ))}
    </div>
  )
}

function EmotionMetrics({ emotionState }) {
  if (!emotionState) return null
  return (
    <div className="orb-metrics">
      v: {emotionState.valence?.toFixed(2) ?? '—'}
      {'  '}a: {emotionState.arousal?.toFixed(2) ?? '—'}
    </div>
  )
}

// ── Resizable sidebar hook ──
function useSidebarResize(initialWidth = 350) {
  const [width, setWidth] = useState(initialWidth)
  const isDragging = useRef(false)
  const dragRef = useRef(null)

  useEffect(() => {
    const handleMouseMove = (e) => {
      if (!isDragging.current) return
      const newWidth = window.innerWidth - e.clientX
      setWidth(Math.max(280, Math.min(600, newWidth)))
    }
    const handleMouseUp = () => {
      isDragging.current = false
      if (dragRef.current) dragRef.current.classList.remove('dragging')
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
    }
    window.addEventListener('mousemove', handleMouseMove)
    window.addEventListener('mouseup', handleMouseUp)
    return () => {
      window.removeEventListener('mousemove', handleMouseMove)
      window.removeEventListener('mouseup', handleMouseUp)
    }
  }, [])

  const startDrag = useCallback((e) => {
    isDragging.current = true
    if (dragRef.current) dragRef.current.classList.add('dragging')
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'
    e.preventDefault()
  }, [])

  return { width, dragRef, startDrag }
}

export default function App() {
  const [mode, setMode] = useState('chat')
  const [userEmotion, setUserEmotion] = useState(null)
  const [modelEmotion, setModelEmotion] = useState(null)
  const [timeline, setTimeline] = useState([])
  const [selectedTurn, setSelectedTurn] = useState(null)
  const [backendInfo, setBackendInfo] = useState(null)
  // Speaker mode: "single" (default, reliable) or "dual" (experimental)
  const [speakerMode, setSpeakerMode] = useState('single')

  const { width: sidebarWidth, dragRef, startDrag } = useSidebarResize(350)

  useState(() => {
    fetch('http://localhost:8000/health')
      .then(r => r.json())
      .then(setBackendInfo)
      .catch(() => {})
  }, [])

  const hasSpeakerSep = backendInfo?.speaker_separation === true
  const isDual = speakerMode === 'dual' && hasSpeakerSep

  // Labels that honestly describe what each orb shows
  // Single mode: same vectors, different time points in the forward pass.
  // Dual mode: different vector sets (other-speaker vs current-speaker).
  const topOrbLabel = isDual
    ? "Model's read of your emotional state"
    : "Model's emotional state — processing your message"
  const bottomOrbLabel = isDual
    ? "Model's own emotional state"
    : "Model's emotional state — beginning response"

  const handleEmotionUpdate = useCallback(({ userEmotion: ue, modelEmotion: me }) => {
    setUserEmotion(ue)
    setModelEmotion(me)
    setTimeline((prev) => [
      ...prev.slice(-29),
      {
        userColor: ue?.color_hex || '#333',
        modelColor: me?.color_hex || '#333',
        userEmotion: ue,
        modelEmotion: me,
      },
    ])
    setSelectedTurn(null)
  }, [])

  const handleTimelineSelect = useCallback((index) => {
    setSelectedTurn(index)
    if (index !== null && timeline[index]) {
      setUserEmotion(timeline[index].userEmotion)
      setModelEmotion(timeline[index].modelEmotion)
    }
  }, [timeline])

  const timelineColors = timeline.map((t) => ({
    userColor: t.userColor,
    modelColor: t.modelColor,
  }))

  // Orb sizes scale with sidebar width
  const topOrbSize = Math.max(140, Math.min(sidebarWidth * 0.48, 220))
  const bottomOrbSize = Math.max(160, Math.min(sidebarWidth * 0.55, 260))
  const stripWidth = Math.max(200, sidebarWidth - 80)

  return (
    <div className="app">
      <header className="app-header">
        <div className="app-header-row">
          <h1>EmotionScope</h1>
          {backendInfo && (
            <span className="app-model-badge">
              {backendInfo.model?.split('/').pop()} &middot; layer {backendInfo.probe_layer}/{backendInfo.n_layers} &middot; {backendInfo.n_emotions} emotions
            </span>
          )}
        </div>
        <p className="app-subtitle">
          Real-time emotion vectors from the model's residual stream
        </p>
        <div className="app-mode-toggle">
          <button
            className={mode === 'chat' ? 'active' : ''}
            onClick={() => setMode('chat')}
          >Chat</button>
          <button
            className={mode === 'test' ? 'active' : ''}
            onClick={() => setMode('test')}
          >Test Gallery</button>
        </div>
      </header>

      {mode === 'test' ? (
        <TestGallery />
      ) : (
      <main className="app-main">
        <div className="panel-chat">
          <ChatPanel
            onEmotionUpdate={handleEmotionUpdate}
            highlightedTurn={selectedTurn}
            speakerMode={speakerMode}
          />
        </div>

        <div className="panel-sidebar-wrapper">
          <div
            ref={dragRef}
            className="panel-sidebar-drag"
            onMouseDown={startDrag}
          />
          <div className="panel-sidebar" style={{ width: sidebarWidth }}>
            {/* Replay indicator */}
            {selectedTurn !== null && (
              <div className="replay-indicator" onClick={() => handleTimelineSelect(null)}>
                Viewing turn {selectedTurn + 1} — click to return to live
              </div>
            )}

            {/* Speaker mode toggle */}
            {hasSpeakerSep && (
              <div className="speaker-toggle">
                <button
                  className={`speaker-toggle-btn ${speakerMode === 'single' ? 'active' : ''}`}
                  onClick={() => setSpeakerMode('single')}
                >
                  Single vector
                </button>
                <button
                  className={`speaker-toggle-btn ${speakerMode === 'dual' ? 'active' : ''}`}
                  onClick={() => setSpeakerMode('dual')}
                >
                  Dual speaker
                  <span className="speaker-toggle-exp">experimental</span>
                </button>
              </div>
            )}

            {/* ── Section 1: Top orb ── */}
            <div className="sidebar-section">
              <div className="section-header">{topOrbLabel}</div>
              <EmotionOrbCanvas emotionState={userEmotion} size={topOrbSize} />
              <EmotionLabel emotionState={userEmotion} />
              <EmotionMetrics emotionState={userEmotion} />
            </div>

            <div className="sidebar-divider" />

            {/* ── Section 2: Bottom orb ── */}
            <div className="sidebar-section">
              <div className="section-header">{bottomOrbLabel}</div>
              <EmotionOrbCanvas emotionState={modelEmotion} size={bottomOrbSize} />
              <EmotionLabel emotionState={modelEmotion} />
              <EmotionMetrics emotionState={modelEmotion} />
            </div>

            {/* ── Valence spectrum ── */}
            <div className="sidebar-section">
              <div className="section-header">Valence Spectrum</div>
              <ValenceStrip
                userValence={userEmotion?.valence ?? null}
                modelValence={modelEmotion?.valence ?? null}
                width={stripWidth}
                height={20}
              />
            </div>

            <DetailPanel userState={userEmotion} modelState={modelEmotion} />

            <div className="sidebar-section">
              <div className="section-header">Conversation Arc</div>
              <Timeline
                entries={timelineColors}
                selectedIndex={selectedTurn}
                onSelect={handleTimelineSelect}
                width={stripWidth}
                height={28}
              />
            </div>

            <Legend />
            <ExplainerPanel hasSpeakerSep={isDual} />
          </div>
        </div>
      </main>
      )}
    </div>
  )
}
