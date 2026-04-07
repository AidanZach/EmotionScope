import { useState } from 'react'
import { emotionToColor } from '../utils/palette'

/**
 * Collapsible score breakdown panel.
 * Shows top 5 emotions as per-emotion-colored horizontal bars.
 * Toggleable between user and model readings.
 */
export default function DetailPanel({ userState, modelState }) {
  const [expanded, setExpanded] = useState(false)
  const [viewing, setViewing] = useState('model') // 'model' | 'user'

  const state = viewing === 'model' ? modelState : userState
  if (!state || !state.top_emotions) return null

  const top = state.top_emotions.slice(0, 5)
  const maxScore = top.length ? Math.max(...top.map(([, s]) => Math.abs(s)), 0.01) : 1

  return (
    <div className="detail-panel">
      <button
        className="detail-header"
        onClick={() => setExpanded(!expanded)}
      >
        <span className="detail-header-title">Score Breakdown</span>
        <span className="detail-header-chevron">{expanded ? '▾' : '▸'}</span>
      </button>

      {expanded && (
        <div className="detail-content">
          {/* Toggle between user and model */}
          <div className="detail-toggle">
            <button
              className={`detail-toggle-btn ${viewing === 'user' ? 'active' : ''}`}
              onClick={() => setViewing('user')}
              disabled={!userState}
            >
              you
            </button>
            <button
              className={`detail-toggle-btn ${viewing === 'model' ? 'active' : ''}`}
              onClick={() => setViewing('model')}
              disabled={!modelState}
            >
              model
            </button>
          </div>

          <div className="detail-bars">
            {top.map(([name, score]) => {
              const barColor = emotionToColor(name)
              const pct = (Math.abs(score) / maxScore) * 100
              return (
                <div key={name} className="detail-bar-row">
                  <span className="detail-bar-name">{name}</span>
                  <div className="detail-bar-track">
                    <div
                      className="detail-bar-fill"
                      style={{
                        width: `${pct}%`,
                        background: barColor,
                        opacity: 0.4 + (Math.abs(score) / maxScore) * 0.6,
                      }}
                    />
                  </div>
                  <span className="detail-bar-score">{score.toFixed(3)}</span>
                </div>
              )
            })}
          </div>

          <div className="detail-numbers">
            v={state.valence?.toFixed(2) ?? '—'}{' '}
            a={state.arousal?.toFixed(2) ?? '—'}{' '}
            cplx={state.complexity?.toFixed(2) ?? '—'}
          </div>
        </div>
      )}
    </div>
  )
}
