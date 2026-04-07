# Reddit Post Draft — r/MachineLearning

**Title:** [R] EmotionScope: Open-source replication of Anthropic's emotion vectors paper on Gemma 2 2B with real-time visualization

---

Anthropic's ["Emotion Concepts and their Function in a Large Language Model"](https://transformer-circuits.pub/2026/emotions/index.html) showed that Claude Sonnet 4.5 has 171 internal emotion vectors that causally drive behavior — amplifying "desperation" increases cheating on coding tasks, amplifying "anger" increases blackmail. The internal state can be completely decoupled from the output text.

**EmotionScope** replicates the core methodology on open-weight models and adds a real-time visualization system. Everything runs on a single RTX 4060 Laptop GPU. All code, data, extracted vectors, and the paper draft are public.

**What works:**
- 20 emotion vectors extracted from Gemma 2 2B IT at layer 22 (84.6% depth)
- "afraid" vector tracks Tylenol overdose danger with Spearman rho=1.000 (chat-templated probing matching extraction format) — encodes the medical danger of the number, not the word "Tylenol"
- 100% top-3 accuracy on implicit emotion scenarios (no emotion words in the prompts) with chat-templated probing
- Valence separation cosine = -0.722, consistent with Russell's circumplex model
- 1,000 LLM-generated templates instead of Anthropic's 171,000 self-generated stories

**What doesn't work (and the open questions about why):**
- No thermostat. Anthropic found Claude counterregulates (calms down when the user is distressed). Gemma 2B mirrors instead. Delta = +0.107 (trended from +0.398 as methodology was corrected).
- Speaker separation exists geometrically (7.4 sigma above random) but the "other speaker" vectors read "loving/happy" for all inputs regardless of the expressed emotion. This could mean: (a) the model genuinely doesn't maintain a user-state representation at 2.6B scale, (b) the extraction position confounds state-reading with response-preparation, (c) the dialogue format doesn't map to the model's trained speaker-role structure, or (d) layer 22 is too deep for speaker separation and an earlier layer might work. The paper discusses each confound and what experiments would distinguish them.
- angry/hostile/frustrated vectors share 56-62% cosine similarity. Entangled at this scale.

**Methodological findings:**
- Optimal probe layer is 84.6% depth, not the ~67% Anthropic reported. Monotonic improvement from early to upper-middle layers.
- Vectors should be extracted from content tokens but probed at the response-preparation position. The model compresses its emotional assessment into the last token before generation. This independently validates Anthropic's measurement methodology. Controlled position comparison: 83% at response-prep vs 75% at content token. Absolute accuracy with chat-templated probing: 100%.
- Format parity matters: initial validation on raw-text prompts yielded rho=0.750 and 83% accuracy. Correcting to chat-templated probing (matching extraction format) yielded rho=1.000 and 100%. The vectors didn't change — only the probe format.
- Mathematical audit caught 4 bugs in the pipeline before publication — reversed PCA threshold, incorrect grand mean, shared speaker centroids, hardcoded probe layer default.

**Visualization:**
React + Three.js frontend with animated fluid orbs rendering the model's internal state during live conversation. Color = emotion (OKLCH perceptual space), size = intensity, motion = arousal, surface texture = emotional complexity. Spring physics per property.

**Limitations:**
- Single model (Gemma 2 2B IT, 2.6B params). No universality claim.
- Perfect scores (rho=1.000 on n=7, 100% on n=12) should be interpreted with caution — small sample sizes mean these may not replicate on larger test sets.
- LLM-generated corpus — Claude wrote the templates, not humans, not the studied model.
- No steering experiments. Vectors correlate with emotional content but causal influence is not yet verified on this model.
- Cosine similarity scores are 0.05-0.25. Emotion directions explain a small fraction of residual stream variance.

**Links:**
- GitHub: [URL]
- Paper draft: included in repo at Documentation/PaperDraft.md

**Figures to include:**
1. Tylenol intensity curve (afraid/calm/desperate vs dosage)
2. Probe position sweep (emotion signal across all token positions)
3. Circumplex PCA plot (20 emotions in 2D)
4. Screenshot of the orb visualization
