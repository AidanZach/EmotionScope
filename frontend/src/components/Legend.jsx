import { useState } from 'react'

/**
 * Visual legend that maps every orb property to what it actually means
 * in terms of the model's internals — grounded in the real math,
 * written so a non-ML person can follow it.
 */

const SECTIONS = [
  {
    id: 'color',
    icon: '●',
    title: 'Color',
    short: 'Which emotion is strongest',
    detail: `The orb's color is a blend of the active emotions, weighted by how
strongly each one is firing. Each of the 20 tracked emotions has a fixed
color — warm golds for joy, deep violets for fear, teal for calm.

Technically: we extract a direction vector for each emotion from the model's
neural activations. At inference time, we measure cosine similarity between
the model's current activation (at the probe layer, typically ~80% depth) and
each emotion vector. The color blends the top-scoring emotions proportionally.`,
  },
  {
    id: 'size',
    icon: '◐',
    title: 'Size',
    short: 'How strong the dominant emotion is',
    detail: `A larger orb means the model's activation is strongly aligned with a
specific emotion direction. A small orb means no emotion is particularly
active — the model is in a relatively neutral state.

Technically: size maps to "intensity" — the highest cosine similarity score
among all 20 emotion vectors. Range: 0 (no alignment) to ~0.7 (strong
alignment). Mapped to orb scale 0.7x – 1.3x.`,
  },
  {
    id: 'motion',
    icon: '↝',
    title: 'Motion speed',
    short: 'Arousal — emotional energy level',
    detail: `Fast, churning motion means high arousal — the model is processing
something activating (fear, excitement, anger). Slow, gentle breathing
means low arousal (calm, reflective, gloomy).

Arousal is independent of positive/negative. Both "enthusiastic" (+0.9) and
"desperate" (+0.9) are high arousal. Both "calm" (−0.5) and "gloomy" (−0.3)
are low arousal.

Technically: arousal is the weighted average of each emotion's arousal
metadata, weighted by its cosine similarity score. Mapped to distortion
animation speed: 0.5 (calm) to 4.0 (agitated).`,
  },
  {
    id: 'surface',
    icon: '◍',
    title: 'Surface texture',
    short: 'Complexity — how many emotions compete',
    detail: `A smooth, glassy surface means one emotion dominates clearly. A rough,
turbulent surface means the activation is distributed across multiple emotion
directions rather than concentrated in one.

Technically: complexity is the normalized Shannon entropy of the active
emotion scores (those above 0.05). If one emotion has 90% of the total
score → low entropy → smooth. If three emotions split evenly → high
entropy → distorted. Range: 0.0 (single emotion) to 1.0 (uniform
distribution). Mapped to mesh distortion: 0.08 (glass) to 0.53 (lava).`,
  },
  {
    id: 'particles',
    icon: '✦',
    title: 'Particles',
    short: 'Arousal above the alarm threshold',
    detail: `Particles only appear when arousal exceeds 0.4 — meaning emotions like
fear, anger, desperation, or intense excitement are dominant. The more
particles and the faster they move, the higher the arousal.

This is a pre-attentive signal: you notice particles before you read any
text. If you see them, the model is processing something emotionally
intense — even if its text output sounds calm.

Count scales from 15 (arousal = 0.4) to 55 (arousal = 1.0).`,
  },
  {
    id: 'glow',
    icon: '◉',
    title: 'Glow',
    short: 'Strength of the emotional signal',
    detail: `The halo of light around the orb intensifies with the dominant emotion's
strength. A bright, wide glow means the model's activation is strongly aligned
with a particular emotion direction. A dim glow means weak or ambiguous
activation — no single emotion direction dominates.

Technically: bloom intensity maps to the same "intensity" value as size —
the max cosine similarity score. Emissive intensity ranges from 0.3
(neutral) to 1.2 (strong emotion).`,
  },
  {
    id: 'strip',
    icon: '━',
    title: 'Valence strip',
    short: 'Positive ↔ negative on one axis',
    detail: `The gradient bar compresses the activation onto a single axis: negative
valence (left) to positive valence (right). Two markers show the readings
from each orb — one from processing your message, one from the response start.

When the markers are far apart, the model's activation shifted between
reading your message and beginning its response — e.g., it processed
distressing content but its response-start activation leans toward calm.

Technically: valence is the weighted average of each emotion's valence
metadata (happy = +0.8, desperate = −0.9, etc.), weighted by cosine
similarity scores. The metadata is from psychology (Russell, 1980), not
derived from the model.`,
  },
  {
    id: 'timeline',
    icon: '▪▪▪',
    title: 'Conversation arc',
    short: 'Emotional trajectory over time',
    detail: `Each column is one conversation turn. Top band = the model's activation
while processing your message. Bottom band = the model's activation at
response start. Color matches what the orbs showed at that moment.
Click any segment to replay that turn's emotion state.

Look for: escalation (colors shifting darker/warmer over time), shifts in
the model's response-start state, or divergence between the two bands
(the model's activation changes between reading and responding).`,
  },
]

function LegendRow({ section, isOpen, onToggle }) {
  return (
    <div className="legend-row">
      <button className="legend-row-header" onClick={onToggle}>
        <span className="legend-row-icon">{section.icon}</span>
        <span className="legend-row-title">{section.title}</span>
        <span className="legend-row-short">{section.short}</span>
        <span className="legend-row-chevron">{isOpen ? '▾' : '▸'}</span>
      </button>
      {isOpen && (
        <div className="legend-row-detail">
          {section.detail.split('\n\n').map((para, i) => (
            <p key={i}>{para}</p>
          ))}
        </div>
      )}
    </div>
  )
}

export default function Legend() {
  const [expanded, setExpanded] = useState(false)
  const [openRow, setOpenRow] = useState(null)

  return (
    <div className="legend-panel">
      <button
        className="legend-header"
        onClick={() => { setExpanded(!expanded); if (expanded) setOpenRow(null) }}
      >
        <span className="legend-header-title">Visual Guide</span>
        <span className="legend-header-chevron">{expanded ? '▾' : '▸'}</span>
      </button>

      {expanded && (
        <div className="legend-content">
          <p className="legend-intro">
            Each visual property maps to a specific measurement from inside the
            model's neural network. Tap any row to see the details.
          </p>
          {SECTIONS.map((s) => (
            <LegendRow
              key={s.id}
              section={s}
              isOpen={openRow === s.id}
              onToggle={() => setOpenRow(openRow === s.id ? null : s.id)}
            />
          ))}
        </div>
      )}
    </div>
  )
}
