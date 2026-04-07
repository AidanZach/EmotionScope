import { useState } from 'react'

/**
 * Collapsible "How This Works" explainer panel.
 * Honest about what we can and can't detect at different scales.
 */
export default function ExplainerPanel({ hasSpeakerSep = false }) {
  const [expanded, setExpanded] = useState(false)

  return (
    <div className="explainer-panel">
      <button
        className="explainer-header"
        onClick={() => setExpanded(!expanded)}
      >
        <span className="explainer-header-icon">i</span>
        <span className="explainer-header-title">How this works</span>
        <span className="explainer-header-chevron">{expanded ? '▾' : '▸'}</span>
      </button>

      {expanded && (
        <div className="explainer-content">
          <p>
            EmotionScope reads the residual stream activations at a specific
            layer inside the model's transformer — the same kind of internal
            representations that Anthropic discovered drive behavior in Claude,
            including reward hacking and strategic deception.
          </p>
          <p>
            We extract <em>emotion direction vectors</em> — directions in the
            model's activation space that correspond to specific emotions. During
            conversation, we measure how strongly the model's current activation
            aligns with each direction via cosine similarity.
          </p>
          <p>
            The top orb shows the emotional content the model detects while
            processing your message. The bottom orb shows the model's internal
            state as it begins generating its response. These are the same
            emotion vectors measured at two different points in the forward pass.
          </p>
          {hasSpeakerSep ? (
            <p>
              <em>Dual-speaker mode</em> uses separate vector sets extracted from
              two-speaker dialogues: "current-speaker" vectors (the model's own
              emotional state) and "other-speaker" vectors (the model's
              representation of another person's emotional state). Anthropic found
              these are geometrically distinct in Claude. This is experimental —
              behavioral accuracy varies by model scale and training procedure.
            </p>
          ) : (
            <p>
              At this model scale, we cannot reliably separate the model's
              representation of <em>its own</em> emotional state from its
              representation of <em>the user's</em> emotional state. Anthropic
              demonstrated this separation in a frontier-scale model (Claude Sonnet
              4.5); our extraction produces geometrically distinct vectors but
              their behavioral accuracy is mixed at 2B parameters. Both orbs
              therefore use the same emotion vectors — what differs is <em>when</em> in
              the processing pipeline each reading is taken.
            </p>
          )}
          <p>
            What you see here is <em>not</em> the model's text output — it's the
            internal activation state underneath. The model can produce calm,
            helpful text while its residual stream registers distress.
          </p>
          <div className="explainer-links">
            <a
              href="https://transformer-circuits.pub/2026/emotions/index.html"
              target="_blank"
              rel="noopener noreferrer"
            >
              Read Anthropic's paper
            </a>
            <a
              href="https://github.com/AidanZach/EmotionScope"
              target="_blank"
              rel="noopener noreferrer"
            >
              View on GitHub
            </a>
          </div>
        </div>
      )}
    </div>
  )
}
