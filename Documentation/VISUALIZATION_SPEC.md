# EmotionScope Visualization Design Specification

**Version 0.1 — April 5, 2026**
**Author: Aidan Zach + Architectural AI**

> This document defines the complete visual system for EmotionScope's emotion
> visualization. It should be read and internalized before writing any visualization
> code. Every decision here is deliberate. Don't shortcut it.

---

## 1. Design Philosophy

We are visualizing something that has never been visualized before: the internal
emotional state of a language model during inference, and simultaneously, the
model's computational assessment of the user's emotional state. These are real
signals extracted from the residual stream — not sentiment analysis of the output
text, but measurements of what's happening inside the model before it writes a
single word.

The visualization must communicate three things:

1. **What emotion is present** — the primary channel
2. **How intense it is** — the secondary channel
3. **What the relationship between user and model states is** — the novel contribution

It must communicate these pre-attentively (without requiring conscious analysis)
while remaining available for deeper inspection by researchers.

### What this is NOT

- Not a sentiment analyzer badge ("positive" / "negative" / "neutral")
- Not a simple colored dot that requires a lookup table
- Not a decoration or ambient effect — it carries real data
- Not a toy — this is a research instrument that also needs to be beautiful

---

## 2. The Orb: Primary Visual Element

### 2.1 Form: Lava Lamp Blob

The orb is a luminous, amorphous body of fluid light. Not a rigid sphere — a soft,
organic shape that slowly morphs between rounded forms, like a lava lamp element
floating in darkness.

**Why this form:**
- An amorphous shape feels alive in a way a perfect sphere doesn't
- The shape itself can carry information (smooth = coherent emotion, distorted = complex)
- It avoids the "UI widget" feeling of a circle with a border
- It creates a sense of looking at something with an inner life

**Physical properties:**
- Base diameter: ~120-160px (adjustable)
- No hard edge — the boundary is a luminous falloff, not a geometric line
- The shape is defined by noise-displaced radial points, creating organic contours
- A specular highlight near upper-left gives dimensionality (glass marble effect)
- A soft radial glow extends beyond the body, creating the sense of emitted light

### 2.2 Color: OKLCH Emotion Palette

Color is the primary data channel. It communicates **what emotion category** is
active. We use the OKLCH perceptual color space because:

- Equal perceptual steps in all directions (unlike HSV where green dominates)
- Natural, sophisticated palette (no neon, no garish transitions)
- Smooth interpolation between any two emotions

**The palette** (OKLCH values — Lightness, Chroma, Hue):

```
Emotion Cluster     L      C      H      Character
─────────────────────────────────────────────────────
Joy / Enthusiasm    0.82   0.15   85°    Morning sunlight, warm gold
Love / Care         0.80   0.12   60°    Candlelight, honey
Confidence / Pride  0.70   0.13   70°    Burnished bronze
Hope                0.75   0.10   85°    Dawn horizon
Calm / Serenity     0.65   0.08   220°   Deep lake at dusk, still teal
Curiosity           0.72   0.12   175°   Bioluminescence, seafoam
Reflective          0.60   0.03   250°   Overcast morning
Surprise            0.80   0.14   200°   Lightning flash, bright cyan
Nervousness         0.55   0.09   290°   Static charge, lavender
Frustration         0.50   0.10   35°    Rusted iron
Guilt               0.38   0.08   310°   Bruise purple
Anger               0.45   0.16   25°    Cooling ember
Sadness             0.40   0.10   270°   Twilight indigo
Fear                0.35   0.12   300°   Pre-storm violet
Desperation         0.30   0.18   10°    Blood moon, deep dark red
Hostility           0.25   0.14   15°    Volcanic glass
Brooding            0.35   0.04   260°   Charcoal undertow
Gloomy              0.42   0.03   240°   Fog bank
Neutral             0.58   0.03   240°   Warm gray
```

**Key design pattern:** Lightness carries emotional weight.
- Positive emotions: L 0.65–0.85 (light, lifted)
- Negative emotions: L 0.25–0.50 (dark, heavy)
- Neutral: L 0.55–0.65 (mid-range)

Chroma never exceeds 0.18 — everything stays muted and sophisticated.

**When multiple emotions are active:** The orb's color is a weighted blend of the
top active emotions' colors, weighted by their scores. This produces natural
intermediate hues. When two emotions with very different hues compete (e.g., joy
at 80° and fear at 300°), the fluid shows visible color currents — distinct threads
of each hue swirling together rather than averaging to a muddy middle. This is the
"complexity made visible" feature.

### 2.3 Intensity: Size and Brightness

**How intense the emotion is** maps to the most universally understood visual
variables: how big and how bright the orb is.

- Weak emotion (max score < 0.2): Small body (~80px), dim glow, feels dormant
- Moderate emotion (max score 0.2–0.5): Normal body (~120px), visible glow
- Strong emotion (max score > 0.5): Large body (~160px), bright glow extending well
  beyond the body, the orb fills more of its container and feels energized

The glow radius is ~1.5–2.5× the body radius. Glow opacity maps to intensity:
strong emotions have a noticeable ambient light, weak emotions have almost none.

### 2.4 Internal Motion: Arousal

The orb is never still. Its internal fluid flows at a speed that maps to **arousal**
(emotional activation energy, not valence).

- Low arousal (calm, reflective, gloomy): Slow, languid internal flow. The shape
  morphs on a 3-4 second cycle. Feels meditative.
- Neutral arousal: Medium flow, ~2 second shape cycle. Resting state.
- High arousal (panic, enthusiasm, anger): Fast internal churning, shape morphing on
  a 0.8-1.5 second cycle. Feels agitated or energized depending on the color.

The shape distortion amplitude also increases with arousal — at low arousal the
blob is nearly spherical, at high arousal it's more dramatically amorphous.

### 2.5 Surface Character: Emotional Complexity

When a single emotion dominates, the orb's color is smooth and unified — one
coherent color flowing through the body.

When multiple emotions compete (high entropy in the score distribution), the orb
develops **visible internal currents of different colors**. You can see distinct
hues swirling together. This communicates "something complex is happening" without
needing any label. The visual difference between "clearly angry" and "a complicated
mix of frustration, guilt, and sadness" is immediately apparent:

- Low complexity (one dominant): Smooth, unified color, coherent internal flow
- Medium complexity (2-3 active): Visible color gradients within the body, gentle
  streaking of secondary hue
- High complexity (many competing): Multiple distinct color currents, turbulent
  mixing, the orb looks like it's processing something difficult

### 2.6 Particle Emission: High Arousal Indicator

Above an arousal threshold (~0.6), the orb emits small luminous particles from its
surface. They drift outward and upward, fading as they go.

- 0.6 arousal: 2-3 particles per second, slow drift, barely noticeable
- 0.8 arousal: 8-10 particles per second, medium speed
- 0.95+ arousal: 15+ per second, fast emission, unmistakable visual signal

Particle color matches the orb's current core color. Particles are small (2-4px),
soft-edged, and have a short life (~1-2 seconds before fading to transparent).

This serves as a **pre-attentive alarm** — you notice particle emission in peripheral
vision without looking directly at the orb. It says "something is activated" before
you even check what.

---

## 3. The Dual-Orb System

### 3.1 Layout: Side by Side

Two orbs in a shared dark container, vertically aligned:

```
┌──────────────────────────┐
│                          │
│   ┌────────────────┐     │
│   │   YOU (read)   │     │
│   │   [orb: 100px] │     │
│   └────────────────┘     │
│                          │
│   ┌────────────────┐     │
│   │   MODEL        │     │
│   │   [orb: 120px] │     │
│   └────────────────┘     │
│                          │
│   ┌────────────────┐     │
│   │  VALENCE STRIP │     │
│   └────────────────┘     │
│                          │
│   ┌────────────────┐     │
│   │   TIMELINE     │     │
│   └────────────────┘     │
│                          │
└──────────────────────────┘
```

The model orb is slightly larger — it's the primary subject. The user-read orb is
secondary. Stacked vertically lets you see both without horizontal eye movement
during a chat conversation.

### 3.2 The Thermostat Made Visible

When the user-read orb is high-arousal (fast, bright, warm/hot) and the model orb
is simultaneously low-arousal (slow, calm, cool), the **contrast between the two
orbs tells the thermostat story** at a glance. No annotation needed.

A thin connecting element between the orbs — a gradient bar or flowing line —
shows the "temperature gradient" between them. When they're in opposition it's a
vivid gradient. When they're aligned it's a smooth single color.

---

## 4. Information Architecture: Three Tiers

### Tier 1 — Glance (0 cognitive effort)

What you see without trying:

- The orb's color → emotion category
- The orb's size/brightness → intensity
- The orb's motion speed → arousal level
- Particle emission → high arousal alert
- The contrast between two orbs → thermostat dynamic

**No text required.** The orb alone communicates. A first-time user gets the gist
in under 2 seconds: "one orb is dark and churning, the other is calm and teal —
something tense is being regulated."

### Tier 2 — Look (brief attention, ~3 seconds)

When you glance at the area around the orb:

- **Dominant emotion name** — small text below the orb, e.g., "curious" or "afraid."
  Single word. Updates smoothly (cross-fade, not snap).
- **Valence strip** — a horizontal gradient bar from negative (dark violet) through
  neutral (gray) to positive (warm gold), with a dot marker showing the current
  position. This is a self-teaching legend — the gradient IS the explanation.
- **Secondary emotion** — if a second emotion is significantly active, show it in
  smaller text: "curious · calm" means curiosity is dominant with calm secondary.

### Tier 3 — Study (deliberate investigation)

Available on hover, click, or in a collapsible detail panel:

- **Top 5 emotions** as a small horizontal bar chart
- **Valence and arousal** as numeric values
- **Complexity score** (entropy)
- **Raw cosine similarity** for all 20 emotions
- **Probe layer** being used
- **Token position** where the reading was taken

### Tier 4 — Research (toggle-able panel)

Full researcher mode, probably a separate tab or expandable section:

- Complete 20-emotion score table
- Emotion vector geometry visualization (2D PCA plot with current position)
- Timeline with full score history across conversation
- Export functionality (JSON, CSV)
- Comparison mode (overlay two conversations' emotional arcs)

---

## 5. The Valence Strip: Integrated Legend

The valence strip replaces a traditional legend. It sits below the orbs and serves
three purposes simultaneously:

1. **Shows the current valence position** (the dot marker)
2. **Teaches the color mapping** (the gradient from negative to positive IS the
   palette, so users see which colors map where)
3. **Provides spatial anchoring** (you can see "I'm here on the spectrum" at a
   glance)

**Design:**
- Full width of the orb container
- Height: ~24px
- Background: smooth gradient matching the OKLCH palette from desperation (dark red,
  left) through neutral (gray, center) to joy (warm gold, right)
- Marker: small bright vertical bar or dot with a glow, positioned at the current
  valence. Animated with spring physics (smooth tracking).
- Labels: very subtle, low-opacity text at the extremes and center:
  "negative — neutral — positive" or specific emotion anchors

---

## 6. The Conversation Timeline

### Purpose

Show the emotional arc of the entire conversation. Each message (user or model) is
a segment. After several messages, patterns emerge — escalation, de-escalation,
emotional shifts.

### Design

A horizontal strip below the valence strip, full width.

**Two bands** stacked (8-10px each):
- Top band: user emotion color at each turn
- Bottom band: model emotion color at each turn

Each segment is proportional in width. A thin gap or tick mark separates segments.
The most recent message is on the right.

When thermostat regulation is happening, you see complementary colors in the two
bands at the same position — visually obvious pattern of opposition.

**Hover on a segment** shows a tooltip with the specific emotions and scores for
that turn.

---

## 7. Animation Physics

### Spring System

Every animated property uses spring physics, not linear interpolation. Springs
create natural motion with overshoot and settle-back, matching physical intuition.

Each property has its own spring with tuned stiffness/damping:

```
Property          Stiffness   Damping   Character
──────────────────────────────────────────────────
Color hue         80          12        Medium transition — visible but not jarring
Color lightness   100         14        Slightly faster than hue
Color chroma      100         14        Tracks with lightness
Body radius       150         16        Fast response — orb "reacts" to intensity quickly
Flow speed        60          10        Slow change — breathing doesn't jump
Distortion amp    80          12        Medium — shape responds at conversation pace
Glow intensity    80          12        Tracks with radius
Turbulence/cplx   40          8         Very slow — complexity builds gradually
```

**Why different speeds:** When someone sends a distressing message, the orb should
respond quickly in *size* (immediate physical reaction), but the *color* should
shift at conversation speed (emotional tone change), and *complexity* should build
slowly (conflicting signals accumulate over time). This creates a natural cascade
of visual change rather than everything snapping at once.

### Blob Shape Animation

The blob boundary is defined by N control points (60-80) arranged radially. Each
point's distance from the center is:

```
r(θ, t) = base_radius
         + distort_amplitude × noise(θ × freq1 + t × flow_speed)
         + distort_amplitude × 0.6 × noise(θ × freq2 - t × flow_speed × 0.7)
         + complexity × distort_amplitude × 0.4 × noise(θ × freq3 + t × flow_speed × 1.5)
```

Three noise octaves at different frequencies and speeds create organic,
non-repeating motion. The `flow_speed` parameter (mapped from arousal) controls
how fast the shape morphs.

At low arousal, the shape barely changes — nearly a perfect circle that
imperceptibly breathes. At high arousal, the shape is dramatically amorphous and
rapidly evolving.

### Transition Behavior

When the emotion state changes between messages (the big moment):

1. **Frame 0-200ms:** Radius responds immediately (spring with high stiffness).
   The orb "reacts" — physically expands or contracts.
2. **Frame 200-800ms:** Color begins shifting. The new hue bleeds in from the core,
   spreading outward. If the previous color was very different, you briefly see
   both colors in the fluid before the new one dominates.
3. **Frame 800-2000ms:** Flow speed adjusts. The breathing gradually quickens or
   slows.
4. **Frame 2000-4000ms:** Complexity settles. If the new state has different
   complexity, the turbulence/multi-color currents gradually appear or dissolve.

This cascade creates a **living response** rather than a mechanical update.

---

## 8. Technical Architecture

### Phase 2 (Gradio Demo): Canvas 2D

For the Gradio integration, we use a self-contained HTML/JS Canvas 2D renderer
embedded in a `gr.HTML` component. Zero external dependencies.

**Why Canvas 2D, not WebGL:**
- Works inside Gradio's iframe sandbox
- No dependency loading or build step
- Sufficient for the blob shape, gradients, and particles
- Anyone can inspect and understand the rendering code

**Components:**
- `OrbRenderer` class — owns a canvas, handles the spring physics, noise
  generation, blob rendering, glow, and particles
- `ValenceStrip` — renders the gradient strip with marker
- `Timeline` — renders the conversation history bands
- `InfoPanel` — renders Tier 2 and Tier 3 information

**Data flow:**
```
Python (emotion_scope.probe) → JSON emotion state → JavaScript → OrbRenderer
```

The Gradio app sends the emotion state as a JSON object to the HTML component
via Gradio's JavaScript interop. The JS renders it.

### Phase 3 (Standalone React Component): R3F + Shaders

For the embeddable npm package (`@emotion-scope/orb`):

```
@react-three/fiber        — 3D scene and render loop
@react-three/drei         — MeshDistortMaterial (organic surface distortion),
                             Float (gentle hover), Sparkles (particles)
@react-three/postprocessing — Bloom (volumetric glow)
@react-spring/three       — Spring physics for all animated values
```

**Why upgrade to R3F for the standalone:**
- Real 3D lighting and depth
- MeshDistortMaterial gives us the blob surface with Perlin noise distortion
  controllable via props — exactly our fluid surface effect
- Bloom postprocessing creates physically correct glow
- React Spring gives us spring physics per-property with configurable stiffness
- The component accepts props like `<EmotionOrb valence={0.3} arousal={-0.2} />`
  and handles all animation internally
- People can `npm install @emotion-scope/orb` and drop it into their app

### What React Bits Gives Us

React Bits is a component collection, not a rendering engine. We don't use their
components directly (they don't accept emotion data as props). However:

- Their **Galaxy background** is a nice ambient container effect we could put behind
  the orbs for visual polish
- Their **design language** (dark themes, glow treatments, subtle borders) informs
  our container styling
- Their **AI Blob** and **Gradient Blob** pro components validate that the blob
  aesthetic resonates with the React community
- Their source code (when available) shows clean patterns for WebGL in React

We may use their Galaxy or Aurora as the background of the orb container in the
Phase 3 standalone component.

---

## 9. Color Blending Algorithm

When multiple emotions are active, the orb shows their blend. Here is the exact
algorithm:

```python
def blend_emotion_colors(scores: dict, palette: dict) -> OKLCHColor:
    """
    Blend active emotion colors weighted by their scores.
    Only positive scores contribute. If complexity is high,
    return multiple colors for multi-current rendering.
    """
    active = [(name, score) for name, score in scores.items() if score > 0.05]
    active.sort(key=lambda x: -x[1])  # strongest first

    if not active:
        return NEUTRAL_COLOR

    total = sum(s for _, s in active)

    # Weighted blend in OKLCH space
    L = sum(palette[n].L * s for n, s in active) / total
    C = sum(palette[n].C * s for n, s in active) / total

    # Hue blending requires circular interpolation
    # Use vector average in cartesian form
    hx = sum(math.cos(math.radians(palette[n].H)) * s for n, s in active) / total
    hy = sum(math.sin(math.radians(palette[n].H)) * s for n, s in active) / total
    H = math.degrees(math.atan2(hy, hx)) % 360

    return OKLCHColor(L, C, H)
```

**Circular hue interpolation** is critical — you can't just average hue degrees
because averaging 10° and 350° would give 180° (green) instead of 0° (red).
Vector averaging in cartesian coordinates handles this correctly.

For the **multi-current rendering** (visible different-colored streams in the orb),
when complexity > 0.3, the top 2-3 emotions are rendered as separate colored
patches within the blob using offset radial gradients with noise-driven positions.

---

## 10. Open Questions for Iteration

After the first implementation, we need real data to answer these:

1. **Score dynamic range** — what do actual emotion scores look like during a
   conversation? If they're all between 0.0 and 0.3, the visual mapping needs
   different normalization than if they span -0.2 to 0.7.

2. **How fast does state change between messages?** — if it's a subtle drift, the
   spring physics can be slow. If it's dramatic swings, we need faster springs
   or the orb will always be "catching up."

3. **Is the color palette perceptually distinct enough?** — Can you tell "curiosity
   teal" from "calm teal" at a glance, or do we need more hue separation between
   adjacent emotions?

4. **Does the orb feel too abstract?** — Some users might prefer a more figurative
   representation (face, emoji-adjacent, body language). The orb is aesthetically
   sophisticated but potentially less intuitive than a face. Worth A/B testing.

5. **Sound** — an ambient tone mapped to model state (low pitch = negative, high
   = positive, rich harmonics = high arousal). Default off. Experimental.

---

## 11. Implementation Priorities

This is the order for Claude Code. Each step builds on the last. No shortcuts.

**Step 1:** Build the lava lamp blob renderer as a standalone HTML file.
Get the shape, color, glow, and particles working with manual slider controls.
This is the foundation. Spend time making it beautiful.

**Step 2:** Build the valence strip and emotion label system. Integrate
below the orb. Confirm the information tiers work — glance, look, study.

**Step 3:** Build the timeline component. Static data first (hardcoded
conversation), then dynamic.

**Step 4:** Wire it to the Gradio chat interface. The `demo.py` receives
emotion state from the probe and passes it to the JS renderer. Single orb
(model state) first.

**Step 5:** Add the second orb (user read) once speaker separation is
implemented. Add the connecting gradient element.

**Step 6:** Add the collapsible detail panel (Tier 3 information).

**Step 7:** Polish. Tune spring constants against real conversation data.
Adjust the palette if needed. Add the researcher panel (Tier 4).

---

*This document is the source of truth for all EmotionScope visualization work.
Code that contradicts this document is wrong. If a design decision needs to change,
update this document first, then update the code.*
