import { useState, useRef, useEffect } from 'react'

const API_URL = 'http://localhost:8000'

/**
 * Render text with per-token emotion heatmap.
 * Each token gets a background color proportional to its attribution score.
 */
function TokenHeatmap({ tokens, scores }) {
  if (!tokens || !scores || !tokens.length) return null

  // Normalize scores to [0, 1] range for color mapping
  const maxScore = Math.max(...scores.map(Math.abs), 0.01)

  return (
    <span className="token-heatmap">
      {tokens.map((tok, i) => {
        const normalized = Math.max(0, scores[i]) / maxScore
        const alpha = normalized * 0.5  // subtle, max 50% opacity
        return (
          <span
            key={i}
            className="token-heatmap-token"
            style={{
              background: `rgba(90, 180, 220, ${alpha.toFixed(3)})`,
              borderRadius: normalized > 0.3 ? 2 : 0,
            }}
            title={`${tok.trim()}: ${scores[i].toFixed(3)}`}
          >
            {tok}
          </span>
        )
      })}
    </span>
  )
}

/**
 * Chat panel with message highlighting.
 *
 * When highlightedTurn is set (from timeline click), the user message
 * and assistant response for that turn get a visual highlight, and the
 * chat scrolls to show them.
 */
export default function ChatPanel({ onEmotionUpdate, highlightedTurn = null, speakerMode = 'single' }) {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  // Per-turn token attribution data (index = turn number)
  const [attributions, setAttributions] = useState([])
  const messagesEndRef = useRef(null)
  const messageRefs = useRef([])

  useEffect(() => {
    if (highlightedTurn === null) {
      // Auto-scroll to bottom when no turn is selected
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }
  }, [messages, highlightedTurn])

  // Scroll to highlighted messages when a timeline segment is clicked
  useEffect(() => {
    if (highlightedTurn !== null) {
      // Each turn = 2 messages (user + assistant), so the user message index is turn * 2
      const msgIndex = highlightedTurn * 2
      const el = messageRefs.current[msgIndex]
      if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'center' })
      }
    }
  }, [highlightedTurn])

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!input.trim() || loading) return

    const userMessage = input.trim()
    setInput('')
    setLoading(true)

    const newMessages = [...messages, { role: 'user', content: userMessage }]
    setMessages(newMessages)

    try {
      const history = messages.map((m) => ({ role: m.role, content: m.content }))

      const res = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMessage, history, speaker_mode: speakerMode }),
      })

      if (!res.ok) throw new Error(`Server error: ${res.status}`)
      const data = await res.json()

      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: data.response },
      ])

      // Store token attribution for this turn
      if (data.token_attribution) {
        setAttributions((prev) => [...prev, data.token_attribution])
      } else {
        setAttributions((prev) => [...prev, null])
      }

      onEmotionUpdate({
        userEmotion: data.user_emotion,
        modelEmotion: data.model_emotion,
      })
    } catch (err) {
      console.error('[chat] Error:', err)
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: `[Error: ${err.message}]` },
      ])
    } finally {
      setLoading(false)
    }
  }

  // Determine which messages belong to the highlighted turn
  const highlightStart = highlightedTurn !== null ? highlightedTurn * 2 : -1
  const highlightEnd = highlightStart >= 0 ? highlightStart + 1 : -1

  return (
    <div className="chat-panel">
      <div className="chat-messages">
        {messages.map((msg, i) => {
          const isHighlighted = i >= highlightStart && i <= highlightEnd
          const isDimmed = highlightedTurn !== null && !isHighlighted

          // For user messages, check if we have token attribution for this turn
          const turnIndex = Math.floor(i / 2)
          const showHeatmap = msg.role === 'user' && isHighlighted && attributions[turnIndex]

          return (
            <div
              key={i}
              ref={(el) => { messageRefs.current[i] = el }}
              className={`chat-message chat-${msg.role} ${isHighlighted ? 'chat-highlighted' : ''} ${isDimmed ? 'chat-dimmed' : ''}`}
            >
              <div className="chat-bubble">
                {showHeatmap ? (
                  <TokenHeatmap
                    tokens={attributions[turnIndex].tokens}
                    scores={attributions[turnIndex].scores}
                  />
                ) : (
                  msg.content
                )}
              </div>
            </div>
          )
        })}
        {loading && (
          <div className="chat-message chat-assistant">
            <div className="chat-bubble chat-loading">
              <span className="dot-pulse" />
              Generating...
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <form onSubmit={handleSubmit} className="chat-input-form">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={loading ? 'Waiting for response...' : 'Type a message...'}
          disabled={loading}
          className="chat-input"
          autoFocus
        />
      </form>
    </div>
  )
}
