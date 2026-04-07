import { useState, useEffect } from 'react'
import EmotionOrbCanvas from './EmotionOrb'

const API_URL = 'http://localhost:8000'

/**
 * Visual test gallery — shows all pre-scripted scenarios simultaneously.
 * Each orb renders the emotion state for a different scenario.
 * Use this during visualization development to see the full range at once.
 *
 * Access via: http://localhost:5173/test (add route in App.jsx)
 * Or render directly as a component.
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

  // Auto-run on mount
  useEffect(() => { runAll() }, [])

  return (
    <div className="test-gallery">
      <div className="test-gallery-header">
        <h2>Orb Test Gallery</h2>
        <p>12 pre-scripted emotional scenarios. Each orb shows the model's internal state.</p>
        <button onClick={runAll} disabled={loading} className="test-run-btn">
          {loading ? 'Running...' : 'Re-run all scenarios'}
        </button>
        {error && <p className="test-error">Error: {error}</p>}
      </div>

      {results && (
        <div className="test-grid">
          {results.map((r) => {
            const e = r.emotion
            return (
              <div key={r.index} className="test-card">
                <div className="test-card-label">{r.label}</div>
                <EmotionOrbCanvas emotionState={e} size={140} />
                <div className="test-card-dominant">{e.dominant}</div>
                <div className="test-card-metrics">
                  v={e.valence?.toFixed(2)} a={e.arousal?.toFixed(2)}
                </div>
                <div className="test-card-scores">
                  {(e.top_emotions || []).slice(0, 3).map(([n, s]) => (
                    <span key={n} className="test-score-chip">
                      {n} {s.toFixed(2)}
                    </span>
                  ))}
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
