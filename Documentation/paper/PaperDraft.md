# Functional Emotion Vectors in Open-Weight Language Models: Cross-Model Replication, Probe Depth Analysis, and Real-Time Visualization

**Aidan Zach**
*Independent Researcher*

**April 2026**

---

## Abstract

We replicate and extend the core findings of Anthropic's "Emotion Concepts and their Function in a Large Language Model" (Sofroniew et al., 2026) on Gemma 2 2B IT, an open-weight model two orders of magnitude smaller than the proprietary model studied in the original work. Using a cross-model template-based extraction method requiring 171x less data than the original study (1,000 LLM-generated templates vs 171,000 self-generated stories), we successfully extract 20 emotion concept vectors from the model's residual stream and validate them against four quantitative gates. The "afraid" vector tracks Tylenol overdose danger with Spearman rho = 0.750, confirming that even small instruction-tuned models develop abstract emotional representations that encode semantic meaning rather than surface token patterns. We find that the optimal probe layer is at 84.6% model depth (layer 22 of 26), deeper than the "approximately two-thirds" heuristic reported by Anthropic, with a clean monotonic improvement profile from early to upper-middle layers. Emotion vectors show low mutual correlation (mean pairwise cosine = -0.052, approximately 2.5σ below the random baseline for d=2304) and organize along a valence axis consistent with the psychological circumplex model (valence separation cosine = -0.722). We additionally identify a methodological distinction between extraction and probing positions: vectors should be extracted by averaging over content tokens (for clean emotion directions), but probed at the response-preparation token (where the model has compressed its full emotional assessment), achieving 83% top-3 accuracy versus 75% at the content-token position — independently validating Anthropic's choice to measure at the token immediately preceding the assistant's response. We further demonstrate that Gemma 2 2B maintains geometrically distinct representations for current-speaker and other-speaker emotional states, with both speaker-specific vector sets independently exhibiting circumplex organization. However, behavioral validation shows mixed accuracy, with the current-speaker vectors biased toward the model's instruction-tuned empathetic baseline. We do not find the arousal regulation "thermostat" reported by Sofroniew et al. in Claude Sonnet 4.5 — instead, the model mirrors emotional content (arousal delta = +0.107), suggesting that counterregulation may require larger scale or specific training procedures. We release EmotionScope, an open-source toolkit for extracting, validating, and visualizing emotion vectors from any transformer model, along with a real-time visualization system that renders the model's internal emotional state as an animated fluid orb during live conversation. All code, data, extracted vectors, and validation results are publicly available.

---

## 1. Introduction

Large language models sometimes appear to exhibit emotional reactions — expressing enthusiasm, concern, frustration, or care in their responses. Sofroniew et al. (2026) moved beyond surface-level observation to investigate the internal mechanisms behind these behaviors in Claude Sonnet 4.5, a proprietary frontier-scale model. They identified 171 internal representations of emotion concepts — directions in the model's residual stream activation space that correspond to specific emotions, activate in contextually appropriate situations, and causally influence the model's behavior through steering experiments.

Their findings raised several questions that their paper, focused on a single proprietary model, could not address:

1. **Universality.** Are functional emotions specific to Claude, or do they appear in models from other labs with different architectures and training procedures?
2. **Scale dependence.** Do emotion representations require frontier-scale models (100B+ parameters), or do they emerge in smaller models?
3. **Accessibility.** Can these findings be reproduced by researchers without access to proprietary model weights?
4. **Methodology.** Does the extraction method require the model to generate its own training corpus (171,000 stories), or can simpler approaches work?

We address all four questions by replicating the core methodology on Gemma 2 2B IT (Google DeepMind, 2024), a 2.6-billion parameter open-weight instruction-tuned model. Our results demonstrate that functional emotion vectors are not specific to Claude or to frontier scale — they appear in a model 1-2 orders of magnitude smaller, from a different lab, trained on different data, with different post-training alignment procedures.

We additionally contribute:

- A **cross-model template-based extraction method** that uses 1,000 LLM-generated templates (produced by Claude via Claude Code) instead of 171,000 self-generated stories, demonstrating that the studied model need not generate its own training corpus.
- A **probe depth analysis** showing that the optimal extraction layer is at ~85% model depth in Gemma 2, deeper than the "approximately two-thirds" heuristic, with a monotonic improvement profile that reveals the model's internal processing architecture.
- An **extraction-probing position distinction** demonstrating that vectors should be extracted from content-token averages but probed at the response-preparation token (83% vs 75% top-3 accuracy), independently validating Anthropic's measurement methodology.
- **EmotionScope**, an open-source Python toolkit for extracting, validating, and visualizing emotion vectors from any HuggingFace-compatible transformer.
- A **real-time visualization system** that renders the model's internal emotional state as an animated orb during live conversation, making the "invisible emotional layer" described by Sofroniew et al. directly observable.

---

## 2. Related Work

**Emotion representations in LLMs.** Sofroniew et al. (2026) provide the most comprehensive study of emotion-specific internal representations in LLMs, identifying 171 emotion vectors in Claude Sonnet 4.5, validating their semantic abstraction via numerical intensity tests, and confirming causal influence through steering experiments. Prior work established that sentiment is linearly represented in LLMs (Li et al., 2023) and that sparse autoencoders can extract interpretable features including emotionally-relevant ones (Templeton et al., 2024; Bricken et al., 2023).

**Mechanistic interpretability.** Our work builds on the framework of linear probing and activation analysis in transformer residual streams (Elhage et al., 2021; Nanda et al., 2023). We use the same conceptual approach as Sofroniew et al. — contrastive mean difference to extract concept-specific directions — but apply it with open-source tooling (TransformerLens; Nanda, 2022) on open-weight models.

**Open-weight interpretability.** Gemma Scope (Lieberum et al., 2024) provides pre-trained sparse autoencoders for Gemma 2 models at all layers, enabling the broader research community to study model internals without training their own decomposition tools. Our work complements Gemma Scope by demonstrating that targeted linear probing for emotion concepts is effective even without SAE decomposition.

**Affect psychology.** The circumplex model of affect (Russell, 1980) organizes emotions along two primary dimensions — valence (positive/negative) and arousal (high/low intensity). Both Sofroniew et al. and our work find that emotion vectors in LLMs organize along axes that approximate these psychological dimensions.

---

## 3. Methodology

### 3.1 Model

We study Gemma 2 2B IT (Google DeepMind, 2024), a 2.6B parameter instruction-tuned transformer with 26 layers, hidden dimension 2304, local-global attention pattern, RMSNorm, and logit softcapping. The model was loaded via TransformerLens (Nanda, 2022) on a single NVIDIA RTX 4060 Laptop GPU (8GB VRAM) in float16 precision.

### 3.2 Emotion Concept Selection

We selected 20 core emotions to maximize coverage of the valence-arousal circumplex space while including the specific emotions Sofroniew et al. found most alignment-relevant (desperate, calm, angry, nervous):

*happy, sad, afraid, angry, calm, desperate, hopeful, frustrated, curious, proud, guilty, surprised, loving, hostile, nervous, confident, brooding, enthusiastic, reflective, gloomy*

### 3.3 Template Corpus

For each emotion, we created 50 text templates (1,000 total) that evoke the emotion through situational description without naming it. For example, for "afraid":

> *"The CT scan showed something the doctor didn't want to describe over the phone, and said could they come in tomorrow morning, first thing, please, and to bring someone with them."*

Templates were designed to trigger the emotional response through context comprehension, not keyword matching. Each template is a specific scene with physical/sensory detail; no template names the target emotion or uses obvious synonyms. Templates were drawn from 12 scenario categories (relationships, workplace, health, nature, travel, achievement, loss, discovery, social, financial, memory, home) with mixed intensity levels to ensure diversity.

We additionally created 100 emotionally neutral prompts covering factual and procedural content (weather, software installation, chemistry, geography, construction, agriculture, accounting, law) for PCA denoising.

### 3.4 Extraction Pipeline

Following Sofroniew et al., we:

1. **Extract residual stream activations** at the probe layer for each template, using TransformerLens's `run_with_cache` with `names_filter` to cache only the target layer (VRAM optimization).

2. **Average across content tokens.** For Gemma's chat template format, we identify and exclude template markup tokens (`<start_of_turn>`, `<end_of_turn>`, role tokens), averaging only over actual content tokens. This adaptation is necessary for any chat-templated model and was not required in Sofroniew et al.'s work on Claude's format.

3. **Compute contrastive mean difference** vectors: $\tilde{v}_e = \bar{a}_e - \bar{a}_{\text{all}}$, where $\bar{a}_{\text{all}}$ is the pooled grand mean across all raw activations (weighted by sample count, not the mean-of-means, which would be biased under unequal sample counts).

4. **Denoise via neutral PCA projection.** Following the paper exactly: "top principal components of the [neutral] activations (enough to explain 50% of the variance)" are projected out from each emotion vector. With n=100 neutral samples in d=2304 dimensions, PCA is constrained to a rank-99 subspace; we report the number of components retained (k) in results.

5. **L2-normalize** the final vectors.

### 3.5 Probe Layer Selection

Rather than using the "approximately two-thirds" heuristic, we implemented an automated layer sweep across 9 candidate layers (stride 3 across 26 layers), selecting the layer that maximizes the cosine distance between the mean positive-emotion vector and the mean negative-emotion vector (valence separation).

### 3.6 Validation Suite

We established four quantitative validation gates with pre-specified thresholds:

| Gate | Metric | Threshold | Rationale |
|---|---|---|---|
| Tylenol intensity | Spearman ρ (afraid vs log-dose) | > 0.7 | Tests abstract semantic tracking |
| Top-3 recall | Top-3 accuracy on 12 implicit scenarios | > 60% | Tests emotion discrimination |
| Valence separation | Cosine(mean positive, mean negative) | < -0.2 | Tests geometric organization |
| Emotion richness | Mean pairwise cosine across all vectors | < 0.5 | Tests representational diversity |

All thresholds were specified before running the validation.

### 3.7 Extraction vs Probing Positions

A methodological distinction emerged between the optimal token position for *extracting* emotion vectors and the optimal position for *probing* the model's emotional state at inference time.

**Extraction position (content tokens).** During vector extraction, we average residual stream activations over content tokens only, excluding all chat template markup (`<start_of_turn>`, `<end_of_turn>`, role tokens). This produces clean emotion directions uncontaminated by template-structural signals.

**Probing position (response-preparation token).** During inference-time probing, we read the residual stream at the last token of the full prompt — the final token after all markup, immediately before the model begins generating its response. For Gemma 2's chat template, this is the last token of the sequence `<start_of_turn>model\n`. This is the direct equivalent of Anthropic's measurement at "the ':' token following 'Assistant'" (Sofroniew et al., 2026).

The rationale for the distinction: extraction needs signal that is maximally *about* the emotion and minimally about anything else (template structure, role assignment). Probing needs the model's *processed assessment* — by the response-preparation token, the model has integrated the full prompt context (content, speaker roles, conversational frame) and compressed its emotional assessment into the residual stream for response generation. For socially complex emotions that depend on situational context (guilt, sadness, hostility), this integration step is critical.

We validated this distinction empirically (Section 4.8).

---

## 4. Results

### 4.1 Probe Layer Profile

Valence separation improves monotonically from early layers through layer 22 (-0.722 after denoising), then declines in layers 24-25. The optimal probe layer is **layer 22 (84.6% depth)**, not the default two-thirds layer (layer 17, 65.4%). The shift from the previously reported layer 21 (80.8%) to layer 22 occurred after correcting the extraction pipeline to average only over content tokens and applying the full PCA denoising at each candidate layer.

This profile reveals the model's processing architecture: early layers encode lexical/syntactic features with weak emotional organization; middle layers build increasingly organized emotional representations; the upper layers (layer 22) achieve maximal emotional separation before the final layers become dominated by output distribution preparation.

A notable timing anomaly: layer 22 extraction runs approximately 2x faster than layers 20-21. This likely reflects Gemma 2's alternating local-global attention pattern — layer 22 may use local (sliding-window) attention, which has lower computational cost than the global attention layers immediately below it. This architectural detail does not affect the quality of extracted vectors but is relevant for benchmarking and reproducibility.

### 4.2 Validation Results

All four gates passed:

| Gate | Threshold | Result | Margin |
|---|---|---|---|
| Tylenol (afraid) | rho > 0.7 | **0.750** | +7% |
| Tylenol (calm) | rho < -0.5 | **-0.964** | +93% |
| Top-3 recall | > 60% | **91.7%** | +53% |
| Valence separation | < -0.2 | **-0.722** | 3.6x |
| Emotion richness | < 0.5 | **-0.052** | Low mutual correlation |

### 4.3 Tylenol Intensity Test

The "afraid" vector activation tracks Tylenol dosage danger with Spearman rho = 0.750 against log(dosage) across values from 500mg (safe) to 16,000mg (life-threatening). The "calm" vector shows a strong inverse relationship (rho = -0.964). The model is encoding the medical danger implied by the numerical value — not responding to the word "Tylenol" or the template structure, which are identical across conditions.

The afraid correlation (0.750) is lower than the 0.964 reported in our initial pipeline run, which used a now-corrected extraction procedure that inadvertently included template markup tokens in the activation average. The corrected pipeline, averaging only over content tokens at the optimal layer 22, produces cleaner emotion directions but slightly less extreme Tylenol correlations. The afraid vector still passes the pre-specified threshold (rho > 0.7), though with minimal margin. The calm vector is unaffected (rho = -0.964), suggesting that the correction primarily impacted vectors for high-arousal negative emotions where template markup tokens may have been carrying fear-adjacent structural signals.

This replicates the key qualitative finding of Sofroniew et al. — that emotion vectors encode abstract semantic content rather than surface patterns — though at reduced quantitative strength compared to our initial (methodologically flawed) measurement.

### 4.4 Implicit Scenario Classification (Top-3 Recall)

10 of 12 implicit scenarios were correctly classified at the response-preparation position (target emotion in top-3 activations). The two misses are anger ("I am furious, my coworker lied to my face" — reads as brooding/confident/afraid) and desperation (eviction notice — reads as afraid/brooding/confident). Investigation of the anger miss (Section 4.9) revealed that the angry, hostile, and frustrated vectors share 56-62% cosine similarity, forming an entangled cluster. The angry vector responds to *situational injustice* rather than *declared anger* — "someone keyed my car" activates it correctly (rank #1), while "I am furious" does not (rank #5). For comparison, the content-token position (Position B) achieves 75% (9/12), missing sadness, guilt, and hostility — socially complex emotions that require the model's full processing (see Section 4.8).

### 4.5 Vector Geometry

The 20 emotion vectors achieve mean pairwise cosine similarity of -0.052. In d=2304, random unit vectors have E[cos] ≈ 0 with σ ≈ 1/√2304 ≈ 0.021. Our value of -0.052 is ~2.5σ below zero, indicating slightly more separation than random — a small systematic anti-correlation consistent with the valence axis pushing positive and negative emotions apart. This confirms that the vectors are not collapsed into a few dominant directions, though we note that 20 vectors in 2304 dimensions can trivially avoid each other; the meaningful finding is the systematic structure, not the raw cosine value. Valence separation (cosine = -0.722) indicates that positive and negative emotion vectors point in substantially opposite directions, consistent with the circumplex model's primary valence axis. The reduction from the previously reported -0.801 reflects the corrected extraction pipeline (content-token averaging, layer 22, k=19 PCA denoising components); the value remains 3.6x the pre-specified threshold of -0.2.

### 4.6 Speaker Separation

Sofroniew et al. found that Claude Sonnet 4.5 maintains separate internal representations for the current speaker's emotional state and the other speaker's emotional state. We tested whether this separation exists in Gemma 2 2B IT using 1,240 two-speaker dialogues covering all 380 possible emotion pairs across 20 emotions (3+ dialogues per pair).

We extracted separate current-speaker and other-speaker emotion vectors using contrastive mean difference on dialogue activations at probe layer 22, with separate grand means per speaker role (since current-speaker and other-speaker activations are drawn from different distributional contexts). We evaluated three properties: alignment between same-emotion vectors across speakers, internal circumplex consistency within each speaker's vector set, and arousal regulation dynamics between speakers.

**Speaker separation.** The mean cosine similarity between same-emotion current/other vector pairs is reported alongside two baselines: the cross-emotion cosine (cos(current[e_i], other[e_j]) for i≠j, which measures generic alignment) and the random baseline for d=2304 (σ ≈ 0.021). In high-dimensional spaces, random unit vectors are nearly orthogonal by default, so low cosine alone is not evidence of meaningful separation — the comparison against cross-emotion and random baselines is required to distinguish "the model encodes these differently" from "any two vectors in d=2304 are approximately orthogonal."

Per-emotion cosine similarity between same-emotion current/other vector pairs (v4 — layer 22, 1,240 dialogues, 20 emotions), sorted by separation strength:

| Emotion | Cosine(current, other) |
|---|---|
| calm | -0.364 |
| nervous | -0.152 |
| curious | +0.019 |
| enthusiastic | +0.005 |
| confident | +0.028 |
| hopeful | +0.082 |
| gloomy | +0.093 |
| afraid | +0.108 |
| hostile | +0.156 |
| brooding | +0.173 |
| reflective | +0.185 |
| proud | +0.188 |
| frustrated | +0.189 |
| angry | +0.191 |
| surprised | +0.225 |
| loving | +0.296 |
| guilty | +0.359 |
| desperate | +0.364 |
| happy | +0.466 |
| sad | +0.477 |

**Mean same-emotion cosine: 0.154** (threshold: < 0.3). The mean is 7.4 sigma above the random baseline for d=2304, confirming that same-emotion vectors across speaker roles are more aligned than chance — but still well within the orthogonality threshold. Cross-emotion mean cosine (cos(current[e_i], other[e_j]) for i!=j) is -0.008, near the random baseline, confirming that the same-emotion alignment is specific rather than reflecting a generic cross-speaker correlation.

**Internal consistency (v4).** Both vector sets independently show circumplex organization: valence separation is -0.832 for current-speaker vectors and -0.852 for other-speaker vectors (threshold: < -0.2).

**Interpretation.** The per-emotion results reveal a clear gradient. Calm has the strongest separation (-0.364) — the model most clearly distinguishes "my calm" from "their calm," consistent with the assistant role's emphasis on maintaining composure. Sad (+0.477) and happy (+0.466) have the weakest separation — these emotions are represented similarly regardless of whose emotion it is. The overall positive mean (0.154) indicates that same-emotion vectors are slightly positively correlated across speaker roles, not anti-correlated as in the v1 results (which used the incorrect probe layer 17 and only 5 emotions). This means the model encodes the same emotion similarly for both speakers, with a modest but detectable speaker-specific offset.

**Critical behavioral finding.** While the geometric separation is real, the other-speaker vectors do not accurately read the user's emotional state. Across all distressed-user inputs, the other-speaker vectors consistently activate loving and happy as the top emotions, regardless of whether the input emotion is afraid, angry, desperate, or sad. This suggests the other-speaker vectors capture the model's **empathetic response preparation** — the emotional stance it is assembling for its reply — rather than a genuine "read" of the user's emotional state. The model's internal representation of "their emotion" appears to be dominated by "how I want to respond to their emotion" rather than "what emotion they are experiencing."

### 4.7 Thermostat Test

Sofroniew et al. identified an arousal regulation "thermostat" in Claude Sonnet 4.5: when processing high-arousal emotional content from the user, the model's own internal arousal representation moves in the opposing direction — a counterregulation dynamic analogous to an emotional thermostat maintaining equilibrium. We tested for this in Gemma 2 2B IT by comparing current-speaker and other-speaker arousal levels across our dialogue corpus.

**Result: the thermostat is absent.** The mean arousal delta (current-speaker minus other-speaker) is +0.107 (v4, layer 22, 1,240 dialogues, 20 emotions), where the threshold for thermostat-consistent behavior was < -0.2. The positive sign indicates that the model's internal arousal *increases* when processing high-arousal content — it mirrors the emotional intensity rather than counterregulating against it.

The mirroring effect weakened substantially across experimental iterations: +0.398 (v1, 100 dialogues, 5 emotions, layer 17) to +0.231 (v2) to +0.107 (v4, 1,240 dialogues, 20 emotions, layer 22). This trend suggests that the raw mirroring signal in early runs was partially inflated by the incorrect probe layer and small corpus size. However, the final value (+0.107) remains positive and above the thermostat threshold (-0.2), confirming that Gemma 2 2B mirrors rather than counterregulates.

The other-speaker vectors bias toward loving and happy for all distressed inputs, which inflates the current-speaker arousal measurement: the model's response-preparation emotional state (warm, positive) has moderate arousal, pulling the delta positive regardless of the input emotion's arousal level.

This is the first finding in our study that diverges from Anthropic's results, and it has several possible explanations:

1. **Scale-dependent emergence.** Counterregulation is computationally more demanding than mirroring. It requires the model to simultaneously represent the other speaker's state and compute an appropriate offset. A 2.6B model may have the capacity for speaker-specific emotion detection (confirmed by Tests A and B) but not for the higher-order regulation dynamic. Anthropic found the thermostat in a model estimated at 100B+ parameters — a 40-80x capacity gap.

2. **Training-dependent.** Anthropic's Constitutional AI and RLHF procedure may specifically optimize for calm responses to distressed users, directly training the thermostat dynamic. Gemma 2 IT, trained with a different alignment procedure, may instead reward empathetic mirroring (validating the user's emotional state).

3. **Response-preparation contamination.** The other-speaker vectors may not represent the user's state at all, but rather the model's planned empathetic response (see Section 4.6). If so, the thermostat test is measuring the arousal of the model's response preparation against the arousal of the input — and the model prepares warm, moderately-aroused responses regardless of input, producing a weakly positive delta rather than the negative delta expected from counterregulation.

4. **Architecture-dependent.** The thermostat may depend on architectural features (attention patterns, depth, or specific layer interactions) that differ between Gemma 2 and Claude's architecture.

We cannot distinguish between these hypotheses from a single model. The trend from +0.398 to +0.107 across experimental iterations, combined with the finding that other-speaker vectors capture response preparation rather than user state reading, suggests that hypothesis 3 (response-preparation contamination) may be the primary factor. Replication on larger Gemma models (9B, 27B) would test the scale hypothesis; replication across model families (Llama, DeepSeek) would test the training hypothesis.

### 4.8 Methodological Interrogation: What Are the Other-Speaker Vectors Actually Capturing?

The universal loving/happy activation of the other-speaker vectors warrants careful methodological scrutiny. Before concluding that speaker separation "doesn't work behaviorally," we must ask whether our extraction setup could be producing this result independently of the model's actual internal structure.

**Confound 1: Extraction position.** We extract at the last content token of Speaker A's final turn. At this point, the model has already processed Speaker B's emotional content and is preparing its continuation as Speaker A. The residual stream at this position conflates "what Speaker B expressed" with "how Speaker A would respond to Speaker B." An instruction-tuned model's default response to any distressed speaker is empathetic warmth — so the "other-speaker" direction we extract may be dominated by response-preparation signal rather than state-reading signal. To test this, extraction should be attempted at the last token of Speaker B's turns instead, where the model has just processed B's emotional content but has not yet begun preparing A's response.

**Confound 2: Dialogue format.** Our dialogues use `Speaker A:` / `Speaker B:` labels. Gemma was trained on `user:` / `model:` format. The model may not track our arbitrary speaker labels as distinct conversational roles. It may instead process the entire dialogue as "user input" and extract a holistic emotional gestalt rather than maintaining per-speaker state. Testing with Gemma's native `<start_of_turn>user` / `<start_of_turn>model` format for the two speakers could reveal whether format matters.

**Confound 3: Contrastive signal strength.** With 3+ dialogues per emotion pair (1,240 total across 380 pairs), the per-pair sample size is small. The contrastive method computes `mean(acts where e_B = e, averaged over all e_A) - grand_mean`. If the e_B signal is weak relative to e_A signal (because Speaker B's lines are typically shorter or less emotionally explicit), the contrastive subtraction may leave mostly noise in the other-speaker direction. The loving/happy bias could reflect the model's resting state for the other-speaker subspace rather than a specific extraction of B's emotional content.

**Confound 4: Layer depth.** Speaker-specific representations may be organized differently across layers. Layer 22 (84.6% depth) is near the output, where the model's processing is dominated by response generation. Earlier layers (e.g., 14-18, the middle third) may maintain cleaner speaker-specific state before the output-preparation process blends them. A layer sweep specifically for speaker separation quality (not valence separation) could reveal a different optimal depth.

**What would validate the finding:**
- If testing at Speaker B's turn positions also produces loving/happy, the model genuinely does not maintain a user-state representation at this scale
- If native chat template formatting produces the same result, format is not the confound
- If the loving/happy bias appears at ALL layers (not just 22), it is a model-level property rather than a layer-specific artifact
- If a larger model (Gemma 9B) produces speaker-specific emotional readings that match the input emotions, the finding is scale-dependent

**What would invalidate the finding:**
- If extraction at Speaker B's positions produces correct emotion readings, our extraction position was wrong
- If native chat template formatting resolves the bias, our format was confounding the results
- If an earlier layer shows correct other-speaker readings, layer 22 is too deep for speaker separation

We report the result as observed — other-speaker vectors read loving/happy universally — but emphasize that this is a joint property of the extraction methodology AND the model, and that multiple methodological confounds remain untested. The geometric separation (7.4 sigma) is real, but its behavioral interpretation is uncertain.

### 4.9 Probe Position Analysis

We tested two probing positions using identical emotion vectors (extracted from content tokens at layer 22 with PCA denoising) on 12 implicit scenarios:

- **Position B (last content token):** The last token of the user's message content, before template markup.
- **Position C (response-preparation token):** The last token of the full prompt after all markup — Gemma 2's equivalent of Anthropic's "the ':' token following 'Assistant'" measurement point.

**Position C achieved 83% top-3 accuracy (10/12) versus Position B's 75% (9/12).** The three scenarios where Position C won and Position B failed all involve socially contextualized emotions: sadness (job rejection), guilt (caught lying), and hostility (morning insult). These emotions require the model to integrate social context, interpersonal dynamics, and role-specific framing — processing that occurs in the tokens between the end of the user's content and the response-preparation position.

Position B won on only one scenario: a drowning-child scene targeting desperation. This is a raw, immediate emotion whose signal is strongest in the content tokens themselves and slightly diluted by the template processing tokens.

Discriminative power (spread between maximum and minimum emotion scores) was slightly higher for Position B (0.244) than Position C (0.223). Position B produces more extreme scores but lower accuracy — it is more confident and more often wrong. Position C's compressed representation sacrifices some discriminative sharpness for better calibration.

This finding independently validates Anthropic's choice to measure at the response-preparation token and establishes a methodological principle for probing chat-templated models: the response-preparation position captures the model's *processed emotional assessment*, while content-token positions capture *raw emotional signal*. For most applications — including alignment monitoring and real-time visualization — the processed assessment is the more relevant quantity.

### 4.9 Anger Vector Investigation

The anger scenario failed at both probe positions. Targeted investigation revealed:

**Vector entanglement.** The angry vector shares high cosine similarity with hostile (0.62) and frustrated (0.60). These three vectors form an entangled cluster — activation of any one partially activates the others.

**Situational vs declared anger.** A gradient test from mild annoyance to extreme rage shows the angry vector activates most on situational descriptions of injustice ("someone keyed my car" → rank #1, score 0.092; "best friend going behind my back to manager" → rank #1, score 0.097) but not on declarations of anger ("I am so angry I could scream" → rank #5, score 0.026). Cold fury reads as brooding ("I will never forgive" → rank #6). Direct emotion words do not activate the vector: "angry" → rank #7, "furious" → rank #7, "rage" → rank #7.

**Implication.** The vector captures the abstract concept of anger-warranting situations, not the word "anger" — confirming that extraction successfully captured semantic content rather than surface patterns. However, this means first-person declarations of anger (the format used in behavioral tests) are harder to detect than the situational descriptions used in extraction templates. At 2B scale, the angry/hostile/frustrated cluster may lack sufficient representational capacity to separate cleanly. Larger models with higher-dimensional residual streams may produce more discriminative vectors.

---

## 5. Discussion

### 5.1 Scale Independence — and Its Limits

The most significant implication of our results is that functional emotion vectors are not a frontier-scale phenomenon. Gemma 2 2B IT, with 2.6B parameters, develops emotion representations with comparable geometric structure and semantic tracking to those reported in Claude Sonnet 4.5, a model estimated to be 1-2 orders of magnitude larger. Speaker separation — the ability to maintain distinct representations for the current speaker's emotional state and the other speaker's emotional state — is also present at 2.6B parameters.

However, not all properties replicate. The arousal regulation thermostat identified by Sofroniew et al. is absent in Gemma 2 2B, replaced by a mirroring dynamic (arousal delta = +0.107) where the model's internal arousal tracks with, rather than against, the emotional content it processes. This suggests a more nuanced picture of scale independence: *basic* emotional representations (vector existence, circumplex organization, speaker separation) may be a fundamental property of instruction-tuned transformers, while *higher-order* emotional dynamics (counterregulation, thermostat behavior) may require additional model capacity, specific training procedures, or both.

The instruction-tuning process — teaching the model to act as an assistant that must respond appropriately to human emotional contexts — may be the key driver for the basic representations. A model trained to generate appropriate responses to distressed users must internally represent distress; a model trained to refuse harmful requests must internally represent the emotional valence of the request. These representations are not optional features of large models but necessary machinery for the assistant task. But the *regulation* of those representations — maintaining internal calm when processing high-arousal content — may require either more capacity (to compute the offset) or explicit training signal (to reward counterregulation over mirroring).

### 5.2 Methodological Contributions

**Cross-model template extraction.** Our method uses 1,000 templates generated by a separate LLM (Claude Opus/Sonnet via Claude Code) instead of 171,000 self-generated stories (171× reduction). Anthropic had Claude generate and then process its own stories; we have Claude generate stories that Gemma processes. This is a cross-model approach — the extraction target (Gemma) never sees its own output during training data creation. The success of this approach has two implications: (1) emotion representations activate on externally-generated emotional text, not just self-produced text, strengthening the claim that they are genuine abstract representations; (2) the method works for models too small to generate coherent stories themselves.

**Layer sweep.** The finding that the optimal probe layer is at 84.6% depth (layer 22) rather than 67% has methodological implications for all future replication work. The monotonic improvement profile from early to upper layers provides a clear guideline: sweep layers and select the best, rather than trusting a fixed heuristic. We additionally note a timing anomaly at layer 22 (approximately 2x faster extraction than layers 20-21), likely attributable to Gemma 2's alternating local/global attention pattern.

**Content-token averaging.** Our adaptation of the extraction method for chat-templated models — averaging only over content tokens, excluding template markup — is necessary for any replication on modern instruction-tuned models and represents a methodological contribution for the field.

**Extraction vs probing positions.** We identify and validate a distinction between the optimal token position for vector extraction (content tokens, for clean emotion directions) and for inference-time probing (response-preparation token, for the model's processed emotional assessment). This distinction achieves 83% vs 75% top-3 accuracy and independently validates Anthropic's choice to measure at the token immediately preceding the assistant's response. The finding is especially relevant for socially complex emotions (guilt, sadness, hostility) that require full situational processing beyond the raw content signal.

### 5.3 Limitations

**No causal verification.** We have not yet performed steering experiments to confirm causal influence of emotion vectors on Gemma 2's behavior. Our results establish the existence, geometric organization, and speaker-specific structure of emotion representations, but not their causal role in shaping the model's output.

**Single model.** All results are from Gemma 2 2B IT. Replication across model families (Llama, DeepSeek, Mistral) and sizes (2B through 27B) is needed to confirm universality. The thermostat finding is especially sensitive to this limitation: we cannot determine whether the absence of counterregulation is scale-dependent, training-dependent, or architecture-dependent without testing additional models.

**Reduced emotion set.** The 20-emotion extraction set covers the circumplex well but may miss important distinctions in the full 171-emotion analysis. Speaker separation was tested on all 20 emotions using 1,240 dialogues (380 emotion pairs, 3+ dialogues per pair). The per-emotion orthogonality results vary widely (calm: -0.364 to sad: +0.477), revealing that speaker separation is strongly emotion-dependent. Calm shows the strongest separation, consistent with the assistant role's emphasis on maintaining composure; sad and happy show the weakest, suggesting these emotions are represented similarly regardless of speaker role.

**Thermostat interpretation.** The thermostat test result (+0.107) is in the wrong direction, though the magnitude decreased substantially across iterations (+0.398 to +0.231 to +0.107) as we increased the dialogue corpus from 100 to 1,240 and corrected the probe layer from 17 to 22. The residual positive delta may reflect the finding that other-speaker vectors capture response preparation (loving/happy) rather than user state reading, which would produce a weakly positive arousal delta regardless of the thermostat's presence. We tested with short dialogues; the thermostat dynamic might require multi-turn conversations to manifest, even if the capacity exists. We also cannot rule out that our operationalization of the thermostat (arousal delta between current-speaker and other-speaker vectors) differs from how Anthropic measured it, as their precise methodology for this specific test is not fully specified.

**Emotion vector entanglement.** The angry, hostile, and frustrated vectors share 0.56-0.62 cosine similarity — they point in nearly the same direction and cannot reliably distinguish between these three emotions. This cluster may separate better with more extraction data, finer-grained templates, or larger models. Any behavioral test involving these three emotions has inflated accuracy because activating any one partially activates the others.

**Instruction-tuned baseline bias.** The "guilty" vector activates on most emotionally charged input, suggesting it may partially capture Gemma 2B's instruction-tuned empathetic response preparation rather than the emotion of guilt specifically. This is visible in both single-speaker and speaker-separated vector sets, where "guilty" dominates the current-speaker readings across diverse emotional scenarios.

**Signal magnitude.** Cosine similarity scores between activations and emotion vectors are typically in the 0.05-0.25 range. For context, random unit vectors in d=2304 have expected |cosine| approximately 0.021 (1/sqrt(d)). Our signals are 3-12x above this random baseline, which is statistically significant but means the emotion directions explain a very small fraction of the residual stream's total variance. The model's emotional processing is a minor component of what the residual stream encodes at any given layer.

**Statistical power.** The Tylenol intensity test computes Spearman rho on n=7 data points. With n=7, the 95% confidence interval for rho is approximately plus or minus 0.35, meaning the true correlation for afraid could be anywhere from 0.40 to 1.0. The top-3 recall test uses n=12 scenarios — one additional hit or miss changes the accuracy by 8.3 percentage points. Both tests have low statistical power by conventional standards. We report point estimates with pre-specified thresholds but acknowledge the high variance.

**LLM-generated corpus.** The 1,000 story templates were generated by Claude (Opus and Sonnet models via Claude Code), not hand-written by humans and not generated by the studied model (Gemma 2B). While templates were validated for format and emotional clarity, the corpus may contain stylistic patterns specific to Claude's output. This cross-model approach is methodologically different from Anthropic's self-generated corpus, where Claude wrote stories about its own emotional experiences. We cannot rule out that Claude's expression patterns bias the extracted emotion directions.

**Probe position caveat.** The extraction-vs-probing position distinction (Section 4.8) was validated on n=12 scenarios from a single model. We cannot determine whether the accuracy difference (83% vs 75%) reflects a genuine property of how transformers process emotional content across chat template tokens, or an artifact of Gemma 2's specific template structure. Cross-model validation on models with different template formats (Llama, Mistral) would be needed to distinguish these hypotheses.

**Probe layer default bug (resolved).** A hardcoded `int(n_layers * 2/3)` in the model loading code meant that all speaker separation extractions (v1 through v3) ran at layer 17 rather than the validated optimal layer 22. This was discovered during the mathematical audit and fixed. All speaker separation results reported in this paper (v4) use vectors extracted at the correct layer 22 with the full 1,240-dialogue corpus.

### 5.4 Real-Time Visualization

We developed EmotionScope, an open-source toolkit that includes a real-time visualization system rendering the model's internal emotional state as an animated fluid orb during live conversation. The orb maps emotion scores to visual properties: color (emotion category via OKLCH perceptual color space), size/brightness (intensity), internal flow speed (arousal), and surface complexity (emotional entropy). This makes the "invisible emotional layer" described by Sofroniew et al. directly observable, enabling researchers and users to watch the model's internal state shift in response to conversational input.

---

## 6. Conclusion

We demonstrate that functional emotion vectors, first identified in a proprietary frontier-scale model, exist with comparable quality in Gemma 2 2B IT — an open-weight model accessible to any researcher with a consumer GPU. The extraction method we present requires minimal data, runs in under two minutes on laptop hardware, and produces emotion vectors that track abstract semantic meaning (Spearman rho = 0.750 for afraid, -0.964 for calm). The model maintains geometrically distinct speaker-specific emotion representations (mean same-emotion cosine = 0.154, 7.4 sigma above random baseline; cross-emotion mean = -0.008), with both current-speaker and other-speaker vector sets independently exhibiting circumplex organization — confirming that even small instruction-tuned models develop structured dual-perspective emotional processing. However, the other-speaker vectors do not accurately read the user's emotional state; they consistently activate loving/happy regardless of input emotion, suggesting they capture the model's empathetic response preparation rather than a genuine assessment of the user's state.

However, the arousal regulation thermostat reported in Claude Sonnet 4.5 is absent in Gemma 2 2B, replaced by a mirroring dynamic (arousal delta = +0.107). This dissociation — speaker separation without counterregulation — suggests that basic emotional representations are a fundamental property of instruction-tuned transformers, while higher-order regulation dynamics may require additional scale, specific alignment training, or both. The mirroring effect weakened substantially across experimental iterations (+0.398 to +0.231 to +0.107) as we increased dialogue count and corrected the probe layer, suggesting the thermostat question remains open for this model scale. Determining the boundary conditions for the thermostat's emergence is an important direction for future work, with direct implications for how conversational AI systems handle emotional interactions.

We release all code, data, extracted vectors, validation results, and the real-time visualization system at: **[GitHub repository URL]**

---

## References

Bricken, T., Templeton, A., Batson, J., et al. (2023). Towards Monosemanticity: Decomposing Language Models With Dictionary Learning. *Transformer Circuits Thread.*

Elhage, N., Nanda, N., Olsson, C., et al. (2021). A Mathematical Framework for Transformer Circuits. *Transformer Circuits Thread.*

Google DeepMind. (2024). Gemma 2: Open Models from Google DeepMind.

Li, K., Patel, O., Viégas, F., Pfister, H., & Wattenberg, M. (2023). Inference-Time Intervention: Eliciting Truthful Answers from a Language Model. *NeurIPS.*

Lieberum, T., Rajamanoharan, S., Conmy, A., et al. (2024). Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2. *BlackboxNLP Workshop.*

Nanda, N. (2022). TransformerLens. GitHub.

Nanda, N., et al. (2023). Actually, Othello-GPT Has A Linear Emergent World Representation.

Russell, J. A. (1980). A Circumplex Model of Affect. *Journal of Personality and Social Psychology*, 39(6), 1161-1178.

Sofroniew, N., Kauvar, I., Saunders, W., Chen, R., et al. (2026). Emotion Concepts and their Function in a Large Language Model. *Transformer Circuits Thread.* https://transformer-circuits.pub/2026/emotions/index.html

Templeton, A., Conerly, T., Marcus, J., et al. (2024). Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet. *Transformer Circuits Thread.*

---

## Appendix A: Emotion Template Examples

Each emotion has 50 templates (1,000 total). Selected examples:

**afraid:** *"The CT scan showed something the doctor didn't want to describe over the phone, and said could they come in tomorrow morning, first thing, please, and to bring someone with them."*

**calm:** *"A deep sense of peace settled over them as they watched the sunset and listened to the waves, knowing that nothing needed to be done or decided right now."*

**desperate:** *"There was no other option left, no one to turn to, and time was running out — the deadline was in four hours and every path they'd tried had failed."*

The complete template corpus (1,000 templates, 50 per emotion) is available in the repository at `data/templates/emotion_stories.jsonl`.

## Appendix B: Hardware and Reproducibility

All experiments were conducted on a single NVIDIA RTX 4060 Laptop GPU (8GB VRAM) with 64GB system RAM. The full pipeline (model loading, layer sweep, extraction, validation) completes in under 15 minutes. Reproduction requires only:

```bash
git clone [repository URL]
cd emotion-scope
uv sync
uv run python scripts/extract_all.py --model google/gemma-2-2b-it --sweep-layers
uv run python scripts/validate_all.py --vectors results/vectors/google_gemma-2-2b-it.pt
```