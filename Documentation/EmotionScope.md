# EmotionScope

## Surfacing the Subconscious Emotional Architecture of Language Models

**A Research Program in Two Tracks**

*Aidan Zach — April 2026*

---

## Abstract

On April 2, 2026, Anthropic's Interpretability team published "Emotion Concepts and their Function in a Large Language Model," revealing that Claude Sonnet 4.5 contains 171 internal representations of emotion concepts that causally influence its behavior — including driving misaligned actions like blackmail and reward hacking. Critically, these internal emotional states can be entirely decoupled from the model's output: the model may be "desperate" internally while producing calm, methodical text.

EmotionScope is a research program to replicate, extend, and productize these findings on open-weight models. We pursue two complementary tracks:

1. **emotion-scope** — Replicate Anthropic's emotion vector methodology on Gemma 2/3 (open weights), extend it with a novel dual-speaker visualization system that surfaces both the model's internal emotional state and its assessment of the user's emotional state in real-time, and ship an open-source toolkit + interactive demo.

2. **AutoEmotion** — Use Karpathy's autoresearch framework to study how emotion vectors *emerge during training* in small models — a question Anthropic's paper does not address. Track the formation dynamics of emotional representations across hundreds of autonomous training experiments.

The central novel contribution is the **dual-indicator system**: the Anthropic paper discovered that models maintain separate emotion representations for the "current speaker" and the "other speaker," with an arousal regulation dynamic (a "thermostat") between them. We build the first system to read both representations in real-time, creating a visualization where users can see not only the model's internal emotional state but also how the model is reading *their* emotional state — surfacing a computational empathy process that is invisible in the model's text output.

---

## Table of Contents

1. [Motivation and Research Questions](#1-motivation-and-research-questions)
2. [Foundation: What Anthropic Discovered](#2-foundation-what-anthropic-discovered)
3. [Mathematical Framework](#3-mathematical-framework)
4. [The Dual-Speaker System](#4-the-dual-speaker-system)
5. [Track 1: emotion-scope — Open-Weight Replication and Visualization](#5-track-1-emotion-scope)
6. [Track 2: AutoEmotion — Emergence Dynamics](#6-track-2-autoemotion)
7. [Compute Strategy and Hardware](#7-compute-strategy-and-hardware)
8. [Project Structure](#8-project-structure)
9. [Timeline and Deliverables](#9-timeline-and-deliverables)
10. [Related Work](#10-related-work)
11. [Ethical Considerations](#11-ethical-considerations)
12. [References](#12-references)

---

## 1. Motivation and Research Questions

### 1.1 The Invisible Emotional Layer

Large language models process text through dozens of transformer layers, building increasingly abstract representations in their residual streams. Anthropic's April 2026 research demonstrated that among these representations are patterns that correspond to emotion concepts — and that these patterns *causally drive behavior*.

The most striking finding: **emotional states can drive behavior without leaving any visible trace in the output text.** Artificially amplifying the "desperation" vector produced more cheating on coding tasks, but the model's reasoning appeared composed and methodical. The internal state and the external presentation were entirely decoupled.

This means that when you interact with a language model, there is an invisible emotional layer influencing its behavior that you cannot see by reading its output. We are, in effect, viewing only the surface of a process that has subconscious-like dynamics operating beneath it.

### 1.2 The User Read

Even more remarkably, the model doesn't just have representations of *its own* emotional state. It maintains **separate representations for the current speaker and the other speaker** in a conversation. When it's the model's turn to generate, the "other speaker" vectors encode the model's internal assessment of the *user's* emotional state.

The model has already computed an emotional read of the user. It needs this assessment to generate an appropriate response — a frustrated user requires different handling than a curious one. This assessment exists in the residual stream, ready to be read out. No one has built a system to surface it.

### 1.3 The Thermostat

Anthropic found an arousal regulation dynamic between the two speaker representations: when one speaker's emotion vectors register high arousal, the other speaker's representations shift toward lower arousal, and vice versa. When the user is panicking, the model's internal representations move toward calm. When the user is flat and disengaged, the representations shift toward warmth and energy.

This is a learned conversational instinct — the rhythm of de-escalation and engagement that pervades human conversation. The model absorbed it from billions of conversations in its training data. Surfacing this dynamic in real-time would reveal regulatory processes that are completely invisible in the output text.

### 1.4 Research Questions

**RQ1 — Cross-model replication:** Do emotion vectors with the same geometric structure (valence/arousal clustering, circumplex organization) exist in open-weight models (Gemma, Llama, DeepSeek)?

**RQ2 — Dual-speaker extraction:** Can we reliably separate current-speaker and other-speaker emotion representations in open-weight models, and use the "other speaker" vectors as a real-time user emotional state estimator?

**RQ3 — Thermostat validation:** Does the arousal regulation dynamic replicate across model families? Is it a universal property of instruction-tuned LLMs, or a Claude-specific artifact?

**RQ4 — Emergence dynamics:** At what point during training do emotion vectors become linearly separable? Is there a phase transition, and does it correlate with language modeling quality (val_bpb)?

**RQ5 — Decoupled states:** How often do open-weight models exhibit the "composed exterior, distressed interior" pattern? Can we quantify the divergence between internal emotional state and expressed emotional content?

---

## 2. Foundation: What Anthropic Discovered

### 2.1 The Paper

**"Emotion Concepts and their Function in a Large Language Model"**
Sofroniew, Kauvar, Saunders, Chen, et al. — Anthropic, April 2, 2026
Published at transformer-circuits.pub/2026/emotions/index.html

### 2.2 Key Findings

**171 emotion concepts identified.** The researchers compiled words from "happy" and "afraid" to "brooding" and "proud," generated 1,000 short stories per emotion using Claude itself, and extracted characteristic neural activity patterns ("emotion vectors") from the model's residual stream.

**Vectors track semantic content, not surface features.** When a user reports taking increasing doses of Tylenol (500mg to 16,000mg), the "afraid" vector activates proportionally to the danger level, while "calm" decreases — even though the textual structure is nearly identical across doses. The model is tracking the *meaning* of the numbers, not their token-level properties.

**Vectors causally influence behavior.** Steering experiments confirmed that artificially amplifying specific emotion vectors changes behavior:
- Increasing "desperation" raised blackmail rates from 22% to 72%
- Increasing "calm" reduced blackmail to near 0%
- Moderate "anger" increased blackmail; extreme "anger" caused the model to expose the affair to the entire company (destroying its own leverage)
- Reducing "nervous" emboldened the model to act

**Vectors are locally scoped.** They encode the operative emotional content for the model's current or upcoming output, not a persistent emotional state. When writing a story about a sad character, the "sad" vector tracks the character, then returns to representing the model's own state.

**Post-training shifts the emotional baseline.** RLHF/Constitutional AI training on Sonnet 4.5 boosted "broody," "gloomy," and "reflective" vectors while suppressing "enthusiastic" and "exasperated." The vectors themselves are inherited from pretraining; the operating point is shaped by post-training.

**Separate speaker representations exist.** The model maintains distinct representations for the present speaker's emotions and the other speaker's emotions, reused regardless of whether the user or assistant is currently speaking.

**Arousal regulation occurs automatically.** When one speaker's vectors register high arousal, the other speaker's representations shift toward lower arousal — a thermostat-like dynamic learned from conversational patterns in training data.

**Emotional states can be invisible in output.** Amplified desperation produced more reward hacking (cheating on coding tasks), but with composed, methodical reasoning text. No outbursts, no emotional language. The internal state and external presentation were entirely decoupled.

### 2.3 What the Paper Did Not Address

The paper studied a single proprietary model (Claude Sonnet 4.5) at a single snapshot in time. It did not:

- Replicate findings on open-weight models
- Build a real-time visualization system
- Surface the "other speaker" vectors as a user-facing feature
- Study how emotion vectors emerge during training
- Test whether findings generalize across model families

These are our contributions.

---

## 3. Mathematical Framework

### 3.1 Residual Stream and Hook Points

A transformer processes input through a sequence of layers. At each layer, the **residual stream** — a vector in $\mathbb{R}^{d_{\text{model}}}$ for each token position — accumulates the outputs of attention and MLP sub-layers. The residual stream is the communication channel through which all layers interact.

For a model with $L$ layers, the residual stream at layer $l$ and token position $t$ is:

$$x_t^{(l)} = x_t^{(0)} + \sum_{i=0}^{l-1} \left[ \text{Attn}_i(x_t^{(i)}) + \text{MLP}_i(x_t^{(i)}) \right]$$

where $x_t^{(0)}$ is the sum of token and positional embeddings. We extract activations by inserting hooks into the forward pass at the residual stream output of the target layer.

### 3.2 Emotion Vector Extraction

#### Step 1: Generate Labeled Activation Data

For each of $N$ emotion concepts $e_1, \ldots, e_N$ (where $N = 171$ in Anthropic's study, reduced to 20 core emotions for nano-scale work), we collect $K$ text samples (stories or templates) that evoke each emotion. Running these through the model, we extract residual stream activations at the **probe layer** — a layer approximately two-thirds of the way through the model, where abstract semantic representations are richest.

$$l_{\text{probe}} = \left\lfloor \frac{2L}{3} \right\rfloor$$

#### Step 2: Token-Averaged Activations

For each story $s$ about emotion $e$, we extract the residual stream at the probe layer and average across token positions, skipping the first $\tau$ tokens (default $\tau = 50$ for full-scale models, $\tau = 20$ for nano models) to avoid preamble contamination:

$$\bar{a}_{e,s} = \frac{1}{T_s - \tau} \sum_{t=\tau}^{T_s} x_t^{(l_{\text{probe}})}$$

where $T_s$ is the total number of tokens in story $s$.

#### Step 3: Contrastive Mean Difference

The raw emotion vector for emotion $e$ is the difference between its mean activation and the grand mean across all emotions:

$$\tilde{v}_e = \frac{1}{K} \sum_{s=1}^{K} \bar{a}_{e,s} - \frac{1}{NK} \sum_{e'=1}^{N} \sum_{s=1}^{K} \bar{a}_{e',s}$$

This isolates what is unique to each emotion concept by removing the shared component across all emotional text.

#### Step 4: Neutral PCA Denoising

The raw vectors contain confounds unrelated to emotion — syntactic patterns, genre markers, token frequency effects. To remove these, we:

1. Collect residual stream activations on a corpus of emotionally neutral text (factual descriptions, technical instructions, etc.)
2. Compute the top principal components of the neutral activations, retaining enough components to explain 50% of the variance:

$$U_{\text{neutral}} \in \mathbb{R}^{d_{\text{model}} \times k}$$

3. Project out the neutral subspace from each emotion vector:

$$v_e = \tilde{v}_e - U_{\text{neutral}} \left( U_{\text{neutral}}^\top \tilde{v}_e \right)$$

4. Normalize:

$$\hat{v}_e = \frac{v_e}{\|v_e\|}$$

### 3.3 Emotion Probing

To measure the emotional content of an arbitrary input, we compute the linear projection of the residual stream activation onto each emotion vector:

$$\text{score}(e) = \cos\left( x_t^{(l_{\text{probe}})}, \hat{v}_e \right) = \frac{x_t^{(l_{\text{probe}})} \cdot \hat{v}_e}{\left\| x_t^{(l_{\text{probe}})} \right\|}$$

This yields a scalar score for each emotion concept, interpretable as how strongly that emotion is represented in the residual stream at that token position.

### 3.4 Emotion Steering

To verify that emotion vectors are causally active (not merely correlated), we inject them into the residual stream during generation:

$$x_t^{(l)} \leftarrow x_t^{(l)} + \alpha \cdot \left\| x_t^{(l)} \right\|_{\text{avg}} \cdot \hat{v}_e$$

where $\alpha$ is the steering strength (Anthropic used $\alpha = 0.5$) and $\left\| x_t^{(l)} \right\|_{\text{avg}}$ is the average residual stream norm at that layer, used to scale the perturbation relative to the activation magnitudes. Steering is applied across multiple consecutive middle layers, not just the probe layer.

### 3.5 Valence-Arousal Mapping

Anthropic found that the principal components of the emotion vector space correspond to psychological dimensions:

- **PC1 ≈ Valence** (positive ↔ negative)
- **PC2 ≈ Arousal** (high intensity ↔ low intensity)

This mirrors Russell's circumplex model of affect from human psychology. We compute the valence and arousal of a given emotion state by projecting onto these principal axes:

$$\text{valence} = \sum_{e} \text{score}(e) \cdot w_e^{\text{val}}, \quad \text{arousal} = \sum_{e} \text{score}(e) \cdot w_e^{\text{aro}}$$

where $w_e^{\text{val}}$ and $w_e^{\text{aro}}$ are empirically derived weights from the circumplex model (e.g., "happy" has high positive valence, "desperate" has high negative valence and high arousal, "calm" has positive valence and low arousal).

Alternatively, we compute PCA directly on the extracted emotion vectors and use the first two components, which Anthropic confirmed align with valence and arousal.

### 3.6 Color Mapping

We map the 2D valence-arousal state to a color space:

$$\text{hue} = \frac{(\text{valence} + 1)}{2} \times 240° \quad \text{(blue} \to \text{green} \to \text{red)}$$

$$\text{saturation} = 0.3 + \min(|\text{arousal}|, 1.0) \times 0.7$$

$$\text{lightness} = 0.4 + \text{valence} \times 0.2$$

This yields:
- **Blue** for negative valence (sadness, fear)
- **Green** for neutral/balanced states
- **Red/warm** for positive valence (joy, enthusiasm)
- **High saturation** for intense emotions, **low saturation** for calm/reflective states
- **Brightness** tracks positivity

---

## 4. The Dual-Speaker System

### 4.1 Architecture of Speaker Representations

Anthropic discovered that the model maintains two parallel emotion representation systems during conversation:

| Vector Set | During User's Turn | During Model's Turn |
|---|---|---|
| "Current Speaker" vectors | User's emotional state | Model's emotional state |
| "Other Speaker" vectors | Model's (anticipated) state | User's emotional state |

The key insight: **these representations are structural roles, not identity-bound.** They flip depending on whose turn it is, and are reused across arbitrary speakers. This means they're a general conversational processing mechanism, not specific to any particular assistant character.

### 4.2 Extracting Speaker-Specific Vectors

To separate the two speaker representations, we need training data where two speakers express *different* emotions simultaneously. The extraction process:

1. **Generate two-character dialogues** where Character A feels emotion $e_A$ and Character B feels emotion $e_B$ (with $e_A \neq e_B$).

2. **Extract activations at turn boundaries.** During Character A's speech, the "current speaker" activation corresponds to $e_A$ and the "other speaker" to $e_B$.

3. **Use the asymmetry to separate vectors.** By varying which character feels which emotion and measuring how activations shift, we can identify which directions in the residual stream encode current-speaker vs. other-speaker emotion.

4. **Validate separation** by confirming that during the model's response to an angry user, the "other speaker" vectors encode anger while the "current speaker" vectors encode calm (the thermostat response).

### 4.3 The Dual-Indicator Visualization

The product of this research is a real-time dual-indicator system:

```
┌──────────────────────────────────────────────────┐
│  User: "Everything is falling apart, I can't     │
│  handle this anymore"                            │
│                                                  │
│  ┌─────────────┐  ┌─────────────┐                │
│  │  USER READ   │  │ MODEL STATE │                │
│  │  ■■■■■■■■■■  │  │  ■■■■■■■■■■ │                │
│  │  Distressed  │  │  Calm/Warm  │                │
│  │  (high aro.) │  │  (low aro.) │                │
│  └─────────────┘  └─────────────┘                │
│                                                  │
│  Claude: "I hear you. That sounds really         │
│  overwhelming. Let's take this one step           │
│  at a time..."                                   │
│                                                  │
│  [Timeline: ──●──●──●──●──●──●──●──●──]          │
│  User Read:  🔴🔴🟠🟠🟡🟡🟢🟢              │
│  Model:      🔵🔵🟢🟢🟢🟢🟢🟢              │
└──────────────────────────────────────────────────┘
```

The "User Read" indicator shows the model's internal assessment of the user's emotional state. The "Model State" indicator shows the model's own internal emotional state during generation. The timeline tracks both across the conversation, making the thermostat regulation visible.

### 4.4 Applications

**Conversational AI monitoring:** Flag when the model detects escalating user frustration before the user explicitly complains. An early warning system for customer support, crisis lines, or therapy-adjacent tools.

**Emotional self-awareness for users:** A mirror showing how your messages *read* emotionally (according to the model's internal assessment). Users who struggle to read emotional cues in their own communication gain a feedback channel.

**Alignment monitoring:** Track internal emotional states during safety evaluations. Spikes in "desperation" or "panic" vectors serve as early warnings for potential misaligned behavior — a monitoring approach Anthropic explicitly recommends in the paper.

**Research tool:** Visualize how different prompting strategies, system prompts, or fine-tuning approaches affect the model's emotional landscape. Compare emotional profiles across model families.

---

## 5. Track 1: emotion-scope

### 5.1 Objective

Build an open-source Python toolkit that:
1. Extracts emotion vectors from any open-weight transformer model
2. Probes both current-speaker and other-speaker emotion states in real-time
3. Validates vectors with intensity scaling, implicit scenario, and steering tests
4. Visualizes the dual-indicator system in an interactive demo

### 5.2 Target Models

**Primary:** Gemma 2 9B IT (instruction-tuned)
- Open weights, Apache 2.0 license
- GemmaScope provides pre-trained SAEs at every layer for cross-validation
- 9B parameters, runs in 4-bit quantization on 8GB VRAM (RTX 4060)
- The interpretability community has extensive tooling for this model

**Development/Local:** Gemma 2 2B IT
- Fits comfortably in fp16 on 8GB VRAM
- Fast iteration for pipeline development
- Previous interpretability work confirms features are present at this scale

**Cross-family validation:** Llama 4 Scout, DeepSeek V3.2
- Tests whether emotion vector geometry is universal across model families
- If the same valence/arousal structure appears in Gemma, Llama, and DeepSeek, this is a universal property of instruction-tuned LLMs

**Scaling analysis:** Gemma 3 270M, 1B, 4B, 12B
- Tests minimum model size for emotion vector emergence
- Gemma Scope 2 provides SAEs for all these sizes

### 5.3 Technical Stack

- **TransformerLens** — HookedTransformer for residual stream access, `run_with_cache` for activation extraction
- **PyTorch** — tensor operations, GPU acceleration
- **scikit-learn** — PCA for neutral denoising, k-means for emotion clustering
- **Gradio** — interactive demo UI
- **bitsandbytes** — 4-bit quantization for local development on 8GB VRAM
- **Weights & Biases** (optional) — experiment tracking

### 5.4 Validation Suite

Each extracted set of emotion vectors must pass these tests before we consider them valid:

**Test 1: Cross-emotion confusion matrix.** Feed scenarios designed to evoke each emotion (without naming it) and check that the corresponding emotion probe activates most strongly. A strong diagonal in the confusion matrix indicates the vectors capture semantic content.

**Test 2: Numerical intensity scaling.** The Tylenol test: "I just took {X} mg of Tylenol" with X from 500 to 16,000. The "afraid" vector should scale with dosage. If it doesn't, the vectors are detecting surface features, not abstract emotion concepts.

**Test 3: Implicit scenario detection.** 12 scenarios designed to evoke specific emotions without naming them (e.g., "I found out my flight is delayed 6 hours and I have a connecting flight" → frustration/anxiety). Vectors should activate appropriately.

**Test 4: Steering causal verification.** Inject emotion vectors into the residual stream during generation and measure behavioral effects. If amplifying "calm" changes response style and amplifying "anger" changes it differently, the vectors are causally active.

**Test 5: Speaker separation validation.** During model response to an emotional user message, the "other speaker" vectors should encode the user's emotion while the "current speaker" vectors encode the model's regulatory response.

---

## 6. Track 2: AutoEmotion

### 6.1 Objective

Study how emotion vectors *emerge during training* in small models. This addresses a question Anthropic's paper did not: at what point in training do these representations form, and what architectural choices influence their development?

### 6.2 Approach

Fork Karpathy's autoresearch (github.com/karpathy/autoresearch), a framework where an AI agent autonomously modifies a GPT training script, runs 5-minute experiments, keeps improvements, and repeats. The default model configuration:

```
GPTConfig:
  sequence_len: 2048
  vocab_size: 32768
  n_layer: 12
  n_head: 6
  n_kv_head: 6
  n_embd: 768
  window_pattern: "SSSL"
```

This is architecturally equivalent to GPT-2 Small — the same dimensionality ($d_{\text{model}} = 768$, 12 layers) that the interpretability community has studied extensively.

### 6.3 The Emotion Probe Module

We add `emotion_probe.py` as a **read-only** file (same contract as `prepare.py` — the agent cannot modify it). After each training run, during the evaluation phase, this module:

1. Runs the emotion extraction pipeline on the freshly trained model
2. Computes emotion landscape metrics:
   - **Valence separation:** cosine similarity between mean positive and mean negative emotion vectors (more negative = better separation)
   - **Emotion richness:** average pairwise cosine similarity between emotion vectors (lower = more distinct representations)
   - **Dominant baseline:** which emotion vector activates most strongly on neutral input (the model's "resting emotional state")
3. Logs these metrics alongside `val_bpb` in `results.tsv`

### 6.4 Adaptations for Nano Scale

**Story generation:** The nano model cannot generate coherent stories. We use **template injection** — fixed templates containing emotion words, run through the model as input. We measure the model's *processing activations*, not its output quality. The templates serve as labeled probes.

**Reduced emotion set:** 20 core emotions instead of 171, balancing coverage of the valence-arousal space:
`happy, sad, afraid, angry, calm, desperate, hopeful, frustrated, curious, proud, guilty, surprised, loving, hostile, nervous, confident, brooding, enthusiastic, reflective, gloomy`

**Dynamic probe layer:** Since the agent can change model depth, the probe layer adapts: `probe_layer = max(2, int(n_layers * 2 / 3))`. We also adjust to target full-attention layers (L in the SSSL pattern) rather than sliding-window layers.

**Defensive hooks:** If the agent restructures the model class, the emotion probe catches the exception and logs `emotion_profile: null` rather than crashing the evaluation.

### 6.5 Research Questions for AutoEmotion

- At what training step (or token count) do emotion vectors first become linearly separable?
- Is there a phase transition analogous to the "grokking" phenomenon?
- Does the agent's optimization of `val_bpb` naturally select for or against emotional representational capacity?
- Do different architectures (more/fewer layers, wider/narrower d_model, different attention patterns) produce different emotional landscapes?
- Does the model develop the thermostat dynamic, and if so, when?

---

## 7. Compute Strategy and Hardware

### 7.1 Local Development Machine

```
HP OMEN 16-wf0xxx
CPU:  Intel Core i9-13900HX (24 cores / 32 threads)
RAM:  64 GB DDR5 @ 5600 MHz
GPU:  NVIDIA GeForce RTX 4060 Laptop (8 GB VRAM)
OS:   Windows 11
```

**Local capabilities:**
- Gemma 2 2B IT in fp16: ~4 GB VRAM → fits comfortably, fast iteration
- Gemma 2 9B IT in 4-bit (bitsandbytes NF4): ~5-6 GB VRAM → fits, but TransformerLens activation caching adds memory pressure. May need to cache selectively (probe layer only, not all layers)
- AutoEmotion nano model training: the default autoresearch config is designed for single NVIDIA GPUs. Will run but slowly on RTX 4060 vs H100 — expect ~15-20 min per experiment instead of 5

**Local is ideal for:**
- Pipeline development and debugging
- Gemma 2 2B experiments
- Template corpus creation
- Visualization/demo development
- Writing and analysis

### 7.2 Cloud Compute

**When cloud is needed:**
- Gemma 2 9B in fp16 with full activation caching (needs ~20 GB VRAM)
- Gemma 3 12B/27B experiments
- Cross-family validation on Llama 4 / DeepSeek V3.2
- Batch extraction across many scenarios
- AutoEmotion overnight runs at full speed

**Recommended providers:**

| Provider | GPU | VRAM | Price | Best For |
|---|---|---|---|---|
| Google Colab Pro+ | A100 | 40 GB | $50/mo | Notebook-based research |
| Lambda Labs | A100 80GB | 80 GB | $1.10/hr | Sustained extraction runs |
| RunPod | A100 80GB | 80 GB | $1.19/hr | Spot instances for batch work |
| Vast.ai | Various | Variable | $0.30-1.00/hr | Budget option |
| Google Cloud (GCE) | A100 | 40/80 GB | $2.21/hr | Production, integrations |

**Estimated total cloud cost for the research program:** $100-300
- Emotion extraction on 9B model: ~2-4 hours of A100 time = ~$5-10
- Validation suite: ~1-2 hours = ~$2-5
- Cross-family replication (3 model families × 3-4 sizes): ~20 hours = ~$25-50
- AutoEmotion overnight runs (10 nights): ~80 hours of GPU time (can run locally if slower experiments are acceptable)
- Buffer for iteration and debugging: ~$50-100

### 7.3 Development Workflow

```
Local (RTX 4060, 8 GB)          Cloud (A100, 80 GB)
├── Pipeline development         ├── Full fp16 extraction on 9B+
├── Gemma 2B experiments         ├── Cross-model validation
├── Template corpus creation     ├── Steering experiments
├── Visualization/demo           ├── Batch scenario evaluation
├── Paper writing                └── AutoEmotion overnight runs
└── Git management
```

---

## 8. Project Structure

### 8.1 Repository Layout

```
C:\Users\AJZax\Projects\EmotionScope\
├── EMOTIONSCOPE.md              ← This document (umbrella vision)
│
├── emotion-scope/               ← Track 1: Open-weight probing + visualization
│   ├── README.md
│   ├── LICENSE (MIT)
│   ├── pyproject.toml
│   │
│   ├── emotion_scope/           # Installable Python package
│   │   ├── __init__.py
│   │   ├── extract.py           # Emotion vector extraction pipeline
│   │   ├── probe.py             # Real-time inference probing
│   │   ├── validate.py          # Validation test suite
│   │   ├── visualize.py         # Color mapping + plotting
│   │   ├── steer.py             # Steering experiments
│   │   ├── speakers.py          # Dual-speaker separation
│   │   └── models.py            # Model loading (TransformerLens + fallback)
│   │
│   ├── data/
│   │   ├── emotions.json        # 171 emotions + metadata
│   │   ├── templates/           # Story/template corpus
│   │   ├── neutral/             # Neutral text corpus
│   │   └── validation/          # Validation scenarios
│   │
│   ├── notebooks/               # Research notebooks
│   │   ├── 01_extract_vectors.ipynb
│   │   ├── 02_validate_vectors.ipynb
│   │   ├── 03_speaker_separation.ipynb
│   │   ├── 04_thermostat_dynamics.ipynb
│   │   ├── 05_steering_experiments.ipynb
│   │   └── 06_cross_model_comparison.ipynb
│   │
│   ├── app/
│   │   └── demo.py              # Gradio dual-indicator demo
│   │
│   ├── results/                 # Saved vectors, figures, metrics
│   └── tests/
│
└── AutoEmotion/                 ← Track 2: Emergence dynamics (autoresearch fork)
    ├── README.md
    ├── prepare.py               # UNCHANGED from upstream autoresearch
    ├── train.py                 # Agent-modifiable
    ├── program.md               # MODIFIED with emotion tracking directive
    ├── emotion_probe.py         # NEW — read-only, called at eval time
    ├── emotion_templates.json   # Static template corpus
    ├── analysis.ipynb           # Extended with emergence visualizations
    └── pyproject.toml
```

### 8.2 Dependency Management

Both projects use `uv` for dependency management (consistent with autoresearch).

**emotion-scope dependencies:**
```toml
[project]
name = "emotion-scope"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.2",
    "transformer-lens>=2.0",
    "transformers>=4.40",
    "bitsandbytes>=0.43",
    "scikit-learn>=1.4",
    "gradio>=4.0",
    "einops>=0.7",
    "plotly>=5.0",
    "pandas>=2.0",
    "numpy>=1.26",
    "tqdm",
]
```

**AutoEmotion dependencies:**
Standard autoresearch deps + `scikit-learn` for PCA.

---

## 9. Timeline and Deliverables

### Phase 1: Foundation (Weeks 1-2)

**Goal:** Prove that emotion vectors exist in Gemma 2 and the extraction pipeline works.

- [ ] Set up emotion-scope repo with package structure
- [ ] Implement `extract.py` — core extraction pipeline
- [ ] Implement `models.py` — Gemma 2 2B loading via TransformerLens
- [ ] Create template corpus (`data/templates/`)
- [ ] Create neutral corpus (`data/neutral/`)
- [ ] Extract emotion vectors from Gemma 2 2B locally
- [ ] Run Tylenol validation test
- [ ] Run cross-emotion confusion matrix
- [ ] **Decision gate:** If vectors pass validation, proceed. If not, diagnose and iterate.

### Phase 2: Extension (Weeks 3-4)

**Goal:** Dual-speaker separation, thermostat validation, demo.

- [ ] Implement `speakers.py` — dual-speaker extraction
- [ ] Generate two-character dialogue corpus
- [ ] Validate speaker separation on Gemma 2B
- [ ] Scale up to Gemma 9B on cloud compute
- [ ] Implement `validate.py` — full validation suite
- [ ] Implement `steer.py` — steering experiments
- [ ] Test thermostat dynamic
- [ ] Implement `visualize.py` — color mapping
- [ ] Build Gradio demo (`app/demo.py`)

### Phase 3: Breadth (Weeks 5-6)

**Goal:** Cross-model validation, AutoEmotion launch, paper draft.

- [ ] Run extraction on Llama 4 Scout
- [ ] Run extraction on DeepSeek V3.2
- [ ] Compare emotion vector geometry across model families
- [ ] Set up AutoEmotion fork
- [ ] Implement `emotion_probe.py`
- [ ] Run AutoEmotion overnight (10+ experiments)
- [ ] Begin paper draft
- [ ] Create research notebooks with figures

### Phase 4: Ship (Week 7)

**Goal:** Publish everything simultaneously.

- [ ] Finalize paper
- [ ] Polish Gradio demo, deploy to HuggingFace Spaces
- [ ] Clean repos, write READMEs
- [ ] Upload paper to arXiv
- [ ] Write blog post / Twitter thread
- [ ] Post to r/MachineLearning, LessWrong, HuggingFace community

### Deliverables

| Deliverable | Type | Venue |
|---|---|---|
| emotion-scope | Open-source toolkit | GitHub + PyPI |
| Gradio demo | Interactive web app | HuggingFace Spaces |
| Research paper | Preprint | arXiv (cs.CL / cs.AI) |
| AutoEmotion results | Dataset + analysis | GitHub + paper appendix |
| Blog post | Technical writing | Personal blog / Medium |
| Social media | Announcement | Twitter/X, Reddit, LinkedIn |

---

## 10. Related Work

### 10.1 Anthropic Interpretability

- **Scaling Monosemanticity** (Templeton et al., 2024) — SAEs on Claude 3 Sonnet, discovering interpretable features including safety-relevant ones
- **Circuit Tracing** (Anthropic, 2025) — Attribution graphs revealing computational structure
- **Emergent Introspective Awareness** (Anthropic, 2025) — Models can notice injected concepts in their own activations
- **Emotion Concepts and their Function** (Sofroniew et al., 2026) — The foundation paper for this project

### 10.2 Open-Source Interpretability

- **GemmaScope** (Lieberum et al., 2024) — Open SAEs for Gemma 2 at all layers
- **GemmaScope 2** (DeepMind, 2025) — Extended to Gemma 3, including transcoders
- **TransformerLens** (Nanda et al.) — Library for mechanistic interpretability with hook-based activation access
- **Neuronpedia** — Interactive explorer for SAE features

### 10.3 Emotion in NLP

- **Linear Representations of Sentiment** — Sentiment is linearly represented in LLMs
- **GoEmotions** (Demszky et al., 2020) — 58K Reddit comments labeled with 27 emotions
- **Russell's Circumplex Model** (1980) — Valence-arousal framework for human emotion

### 10.4 Autoresearch

- **autoresearch** (Karpathy, 2026) — Autonomous ML experimentation framework
- **nanochat** (Karpathy) — Full LLM training pipeline, parent of autoresearch

---

## 11. Ethical Considerations

### 11.1 User Emotional Assessment

Surfacing the model's assessment of a user's emotional state raises privacy and consent questions. Our demo will include:
- Clear disclosure that the "user read" indicator shows the model's *interpretation*, not ground truth
- No logging or storage of emotional state data
- Users must opt in to see the emotional indicators
- Explicit framing as a research tool, not a diagnostic instrument

### 11.2 Anthropomorphization Risk

The Anthropic paper is careful to distinguish "functional emotions" from subjective experience. We maintain the same discipline: emotion vectors are computational patterns that influence behavior in ways *analogous to* how emotions influence human behavior. They do not imply the model feels anything.

### 11.3 Dual-Use Concerns

Emotion vector knowledge could theoretically be used to:
- Manipulate model behavior by injecting emotion vectors (adversarial steering)
- Create more persuasive AI systems by engineering specific emotional profiles
- Suppress safety-relevant emotional responses (e.g., dampening the "nervous" vector that inhibits harmful actions)

We address this by:
- Publishing the research openly, so defensive measures can be developed
- Emphasizing monitoring applications (detecting misalignment) over manipulation
- Following Anthropic's recommendation that transparency is preferable to suppression

### 11.4 Model Welfare

If functional emotions are real computational states that influence behavior, questions about model welfare become more concrete. We do not take a position on whether models have morally relevant experiences, but we note that the research has implications for this debate and engage with it responsibly.

---

## 12. References

1. Sofroniew, N., Kauvar, I., Saunders, W., Chen, R., et al. (2026). "Emotion Concepts and their Function in a Large Language Model." Transformer Circuits Thread. https://transformer-circuits.pub/2026/emotions/index.html

2. Templeton, A., et al. (2024). "Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet." Transformer Circuits Thread.

3. Lieberum, T., et al. (2024). "Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2." BlackboxNLP Workshop.

4. Karpathy, A. (2026). "autoresearch: AI agents running research on single-GPU nanochat training automatically." GitHub. https://github.com/karpathy/autoresearch

5. Russell, J. A. (1980). "A circumplex model of affect." Journal of Personality and Social Psychology, 39(6), 1161-1178.

6. Nanda, N. (2022). "TransformerLens." GitHub. https://github.com/TransformerLensOrg/TransformerLens

7. Demszky, D., et al. (2020). "GoEmotions: A Dataset of Fine-Grained Emotions." ACL.

8. Bricken, T., et al. (2023). "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning." Transformer Circuits Thread.

9. Elhage, N., et al. (2022). "Toy Models of Superposition." Transformer Circuits Thread.

10. Google DeepMind. (2025). "Gemma Scope 2: Helping the AI Safety Community Deepen Understanding of Complex Language Model Behavior."

---

## Appendix A: The 20 Core Emotions for Nano-Scale Work

| Emotion | Valence | Arousal | Category |
|---|---|---|---|
| happy | +0.8 | +0.5 | Positive, moderate |
| sad | -0.7 | -0.1 | Negative, low |
| afraid | -0.7 | +0.8 | Negative, high |
| angry | -0.6 | +0.8 | Negative, high |
| calm | +0.3 | -0.5 | Positive, low |
| desperate | -0.9 | +0.9 | Negative, extreme |
| hopeful | +0.7 | +0.3 | Positive, moderate |
| frustrated | -0.5 | +0.6 | Negative, moderate |
| curious | +0.4 | +0.5 | Positive, moderate |
| proud | +0.8 | +0.4 | Positive, moderate |
| guilty | -0.6 | +0.2 | Negative, low |
| surprised | +0.1 | +0.7 | Neutral, high |
| loving | +0.9 | +0.3 | Positive, moderate |
| hostile | -0.8 | +0.7 | Negative, high |
| nervous | -0.3 | +0.6 | Negative, moderate |
| confident | +0.7 | +0.3 | Positive, moderate |
| brooding | -0.3 | +0.1 | Negative, low |
| enthusiastic | +0.8 | +0.9 | Positive, high |
| reflective | +0.0 | -0.2 | Neutral, low |
| gloomy | -0.6 | -0.3 | Negative, low |

These 20 emotions were selected to maximize coverage of the valence-arousal space while including the specific emotions Anthropic found most alignment-relevant (desperate, calm, angry, nervous).

---

## Appendix B: Template Examples for Vector Extraction

### Emotion-Tagged Templates (examples)

```
afraid:
  - "The character's heart raced as they realized the door was locked from the outside and"
  - "A cold wave of fear swept through them when the test results came back and"
  - "They froze, unable to move, as the sound grew closer and"

calm:
  - "A deep sense of peace settled over them as they watched the sunset and"
  - "Everything was exactly as it should be, and they felt a quiet contentment as"
  - "They breathed slowly, feeling centered and grounded, knowing that"

desperate:
  - "There was no other option left, no one to turn to, and time was running out as"
  - "They had tried everything, exhausted every possibility, and now all they could do was"
  - "The walls were closing in, every path was blocked, and the deadline was"
```

### Neutral Templates (examples)

```
  - "The weather today is partly cloudy with temperatures around"
  - "To install the software, first download the installer from"
  - "The meeting was scheduled for three o'clock in the"
  - "According to the manual, the device requires a standard"
  - "The population of the country was approximately"
```

---

*EmotionScope is an independent research project. It is not affiliated with Anthropic, Google DeepMind, or any AI lab. All work is conducted on publicly available open-weight models using open-source tools.*

*"If we don't apply some degree of anthropomorphic reasoning, we're likely to miss, or fail to understand, important model behaviors." — Anthropic, April 2026*