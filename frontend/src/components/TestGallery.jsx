import { useState, useEffect } from 'react'
import EmotionOrbCanvas from './EmotionOrb'
import { emotionToColor } from '../utils/palette'

const API_URL = 'http://localhost:8000'

function Disclaimer({ expanded, onToggle }) {
  return (
    <div className="test-disclaimer">
      <button className="test-disclaimer-header" onClick={onToggle}>
        <span className="test-disclaimer-icon">i</span>
        <span>Interpreting these results</span>
        <span className="test-disclaimer-chevron">{expanded ? '▾' : '▸'}</span>
      </button>
      {expanded && (
        <div className="test-disclaimer-body">
          <p>
            <strong>What you're seeing:</strong> Each orb shows the model's
            residual stream activation at layer 22 (84.6% depth) projected onto
            20 emotion direction vectors via cosine similarity. The dominant
            emotion is the highest-scoring direction. Color is a weighted blend
            of all active directions.
          </p>
          <p>
            <strong>Signal magnitude is small.</strong> Scores range from 0.05
            to 0.25 — statistically significant (3-12x above the random baseline
            of ~0.021 for d=2304) but the emotion directions explain only a small
            fraction of the residual stream's total variance. These are not
            strong, confident classifications — they are subtle directional
            tendencies in a high-dimensional space.
          </p>
          <p>
            <strong>Vector entanglement.</strong> The angry, hostile, and
            frustrated vectors share 56-62% cosine similarity — they point in
            nearly the same direction and cannot be cleanly separated at 2B
            scale. When you see "hostile" as the dominant reading on an anger
            scenario, it's because these vectors are entangled, not because the
            model misread the emotion. The angry vector responds to situational
            injustice (e.g., "someone keyed my car") better than to declared
            anger (e.g., "I am furious").
          </p>
          <p>
            <strong>Baseline bias.</strong> The "afraid" vector has an elevated
            baseline across many prompts — any scenario involving uncertainty,
            medical context, or asking for help activates it moderately. The
            Tylenol test measures whether afraid <em>increases monotonically</em> with
            danger (it does: rho = 1.000), not whether it is absent at safe
            dosages.
          </p>
          <p>
            <strong>Perfect scores, small samples.</strong> 100% top-3 accuracy
            on 12 scenarios and rho = 1.000 on 7 dosage levels should be read as
            strong gate passes, not as claims of literal perfection. A single
            rank swap or classification miss would drop these to ~0.96 and 91.7%
            respectively.
          </p>
          <p>
            <strong>Single model.</strong> All results are from Gemma 2 2B IT
            (2.6B parameters). The entanglement patterns, baseline biases, and
            separation quality may differ substantially on larger models or
            different architectures. This toolkit is designed for cross-model
            comparison — swap the model and re-extract to compare.
          </p>
        </div>
      )}
    </div>
  )
}

export default function TestGallery() {
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [disclaimerOpen, setDisclaimerOpen] = useState(false)

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
        <p>
          Validation scenarios from the research paper. Each orb shows the
          model's activation state probed via the same chat-templated path as
          the live demo. Color is a weighted blend of all active emotion
          directions — the swatches below show what contributes.
        </p>
        <button onClick={runAll} disabled={loading} className="test-run-btn">
          {loading ? 'Running...' : 'Re-run all scenarios'}
        </button>
        {error && <p className="test-error">Error: {error}</p>}
      </div>

      <Disclaimer expanded={disclaimerOpen} onToggle={() => setDisclaimerOpen(!disclaimerOpen)} />

      {results && (
        <div className="test-grid">
          {results.map((r) => {
            const e = r.emotion
            const topEmotions = (e.top_emotions || []).slice(0, 4)
            // Check if dominant matches any expected emotion
            const expected = (r.expected || '').split(',').map(s => s.trim()).filter(Boolean)
            const dominantHit = expected.length === 0 || expected.includes(e.dominant)
            const top3Names = topEmotions.slice(0, 3).map(([n]) => n)
            const top3Hit = expected.length === 0 || expected.some(ex => top3Names.includes(ex))

            return (
              <div key={r.index} className={`test-card ${top3Hit ? '' : 'test-card-miss'}`}>
                <div className="test-card-label">{r.label}</div>
                {r.prompt && (
                  <div className="test-card-prompt">"{r.prompt}"</div>
                )}
                <EmotionOrbCanvas emotionState={e} size={140} />
                <div className="test-card-dominant">{e.dominant}</div>
                {expected.length > 0 && (
                  <div className={`test-card-expected ${top3Hit ? 'hit' : 'miss'}`}>
                    expected: {r.expected}
                    {top3Hit ? ' ✓' : ' ✗'}
                  </div>
                )}
                <div className="test-card-metrics">
                  v={e.valence?.toFixed(2)} a={e.arousal?.toFixed(2)}
                </div>
                <div className="test-card-breakdown">
                  {topEmotions.map(([name, score]) => {
                    const color = emotionToColor(name)
                    const isExpected = expected.includes(name)
                    return (
                      <div key={name} className="test-score-row">
                        <span
                          className="test-score-swatch"
                          style={{ background: color }}
                        />
                        <span className={`test-score-name ${isExpected ? 'test-score-hit' : ''}`}>
                          {name}
                        </span>
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
