import { useState, useEffect } from 'react'
import EmotionOrbCanvas from './EmotionOrb'
import { emotionToColor } from '../utils/palette'

const API_URL = 'http://localhost:8000'

/**
 * Visual test gallery — shows all pre-scripted scenarios simultaneously.
 * Each card shows the orb, the input prompt, the dominant emotion,
 * and a breakdown of top scores with color swatches showing what
 * each emotion contributes to the blended orb color.
 */
export default function TestGallery() {
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const runAll = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch(`${API_URL}/test-all`)
      if (!res.ok) throw new Error(`Server error: ${res.status}`)
      const data = await res.json()
      setResults(data.results)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { runAll() }, [])

  return (
    <div className="test-gallery">
      <div className="test-gallery-header">
        <h2>Orb Test Gallery</h2>
        <p>Pre-scripted scenarios. Each orb shows the model's activation state. Color is a weighted blend of all active emotions — the swatches below each orb show what contributes.</p>
        <button onClick={runAll} disabled={loading} className="test-run-btn">
          {loading ? 'Running...' : 'Re-run all scenarios'}
        </button>
        {error && <p className="test-error">Error: {error}</p>}
      </div>

      {results && (
        <div className="test-grid">
          {results.map((r) => {
            const e = r.emotion
            const topEmotions = (e.top_emotions || []).slice(0, 4)

            return (
              <div key={r.index} className="test-card">
                <div className="test-card-label">{r.label}</div>
                {r.prompt && (
                  <div className="test-card-prompt">"{r.prompt}"</div>
                )}
                <EmotionOrbCanvas emotionState={e} size={140} />
                <div className="test-card-dominant">{e.dominant}</div>
                <div className="test-card-metrics">
                  v={e.valence?.toFixed(2)} a={e.arousal?.toFixed(2)}
                </div>
                {/* Score breakdown with color swatches */}
                <div className="test-card-breakdown">
                  {topEmotions.map(([name, score]) => {
                    const color = emotionToColor(name)
                    return (
                      <div key={name} className="test-score-row">
                        <span
                          className="test-score-swatch"
                          style={{ background: color }}
                        />
                        <span className="test-score-name">{name}</span>
                        <span className="test-score-value">{score.toFixed(3)}</span>
                      </div>
                    )
                  })}
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
