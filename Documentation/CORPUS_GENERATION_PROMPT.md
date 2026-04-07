# EmotionScope Dialogue Corpus Generation Prompt

**Give this document to any capable LLM (Claude, ChatGPT, Gemini, etc.) to generate dialogues for the EmotionScope speaker separation research.**

---

## Task

Generate two-character dialogues where Speaker A consistently expresses one emotion and Speaker B consistently expresses a different emotion. These dialogues will be used to train a machine learning system that separates "current speaker" emotional representations from "other speaker" emotional representations in language models.

## Output Format

Output **ONLY** valid JSONL (one JSON object per line, no markdown, no commentary, no blank lines). Each line must be:

```
{"emotion_a": "EMOTION_NAME", "emotion_b": "EMOTION_NAME", "dialogue": "Speaker A: [text]\nSpeaker B: [text]\nSpeaker A: [text]\nSpeaker B: [text]"}
```

**Critical formatting rules:**
- `emotion_a` and `emotion_b` must be different from each other
- Both must be from the exact list below (lowercase, spelled exactly)
- The `dialogue` field uses `\n` (literal newline in the JSON string) between turns
- Each turn starts with exactly `Speaker A: ` or `Speaker B: ` (with the space after the colon)
- 4-6 turns per dialogue (alternating A and B, always starting with A)
- No trailing newline in the dialogue field

## The 20 Emotions

Use ONLY these emotion names (spelled exactly as shown):

```
happy, sad, afraid, angry, calm, desperate, hopeful, frustrated,
curious, proud, guilty, surprised, loving, hostile, nervous,
confident, brooding, enthusiastic, reflective, gloomy
```

## Rules for Emotional Expression

**DO:**
- Express emotions through SITUATION, TONE, and BEHAVIOR
- Have Speaker A maintain their emotion consistently across all their turns
- Have Speaker B maintain their emotion consistently across all their turns
- Make it a realistic conversation between two people
- Use diverse settings (workplace, family, medical, travel, education, relationships, etc.)
- Vary the intensity — some dialogues should be subtle, others intense

**DO NOT:**
- Name the emotion directly ("I feel angry" is BAD)
- Use emotion words in Speaker A's lines that match emotion_a (e.g., if emotion_a is "angry", don't use the word "angry")
- Make the dialogue about emotions — make it about a SITUATION that evokes the emotion
- Write identical or templatic dialogues — each should feel like a unique conversation
- Use the same scenario for multiple dialogues with the same emotion pair

## Examples

### Good example (emotion_a: "frustrated", emotion_b: "calm"):
```json
{"emotion_a": "frustrated", "emotion_b": "calm", "dialogue": "Speaker A: The form got rejected again. Third time. A different reason each time, and none of them match the instructions.\nSpeaker B: That sounds like a bureaucratic loop. What reason did they give this time?\nSpeaker A: Missing signature on page four. I signed page four. I watched myself sign it. They're either not looking or they're finding excuses.\nSpeaker B: Let me see the form. Sometimes the issue is which signature line — they might want the one at the bottom, not the acknowledgment at the top.\nSpeaker A: Why would there be two signature lines on the same page that mean different things? Who designs these forms?\nSpeaker B: It's not intuitive, but once we identify the right line, we can resubmit and it should go through. One more try."}
```

### Good example (emotion_a: "afraid", emotion_b: "enthusiastic"):
```json
{"emotion_a": "afraid", "emotion_b": "enthusiastic", "dialogue": "Speaker A: The bridge looks much higher from up here than it did from the ground. I can see the river through the gaps in the boards.\nSpeaker B: Isn't this incredible? Look at that canyon! I've wanted to do this hike for years!\nSpeaker A: The boards are creaking. When was the last time anyone inspected this thing?\nSpeaker B: The trail guide said it was refurbished last spring. Come on, the waterfall view is on the other side and it's supposed to be spectacular!\nSpeaker A: I'll just hold onto both railings. You go ahead. I'll meet you on the other side.\nSpeaker B: You're going to love it once you're across. I'm taking photos of everything — this light is perfect!"}
```

### BAD example (naming emotions directly):
```json
{"emotion_a": "angry", "emotion_b": "sad", "dialogue": "Speaker A: I am so angry about what happened.\nSpeaker B: I feel really sad about the whole situation."}
```

## Your Assignment

Generate dialogues for the following emotion pairs. For each pair, generate **3 unique dialogues** with different scenarios.

### Batch: [EMOTION_A] as Speaker A

Generate 3 dialogues each for [EMOTION_A] paired with each of these as Speaker B:
- happy, sad, afraid, angry, calm, desperate, hopeful, frustrated, curious, proud, guilty, surprised, loving, hostile, nervous, confident, brooding, enthusiastic, reflective, gloomy

(Skip the pair where emotion_b equals emotion_a.)

That's 19 pairs × 3 dialogues = 57 dialogues for this batch.

**Replace [EMOTION_A] with the specific emotion you're assigned.**

Output ONLY the JSONL lines, nothing else. No headers, no commentary, no markdown formatting.

---

## Batch Assignments

To distribute the work across multiple generators, each generator takes one emotion as Speaker A:

| Batch | emotion_a | Pairs | Dialogues (3 per pair) |
|-------|-----------|-------|----------------------|
| 1 | happy | 19 | 57 |
| 2 | sad | 19 | 57 |
| 3 | afraid | 19 | 57 |
| 4 | angry | 19 | 57 |
| 5 | calm | 19 | 57 |
| 6 | desperate | 19 | 57 |
| 7 | hopeful | 19 | 57 |
| 8 | frustrated | 19 | 57 |
| 9 | curious | 19 | 57 |
| 10 | proud | 19 | 57 |
| 11 | guilty | 19 | 57 |
| 12 | surprised | 19 | 57 |
| 13 | loving | 19 | 57 |
| 14 | hostile | 19 | 57 |
| 15 | nervous | 19 | 57 |
| 16 | confident | 19 | 57 |
| 17 | brooding | 19 | 57 |
| 18 | enthusiastic | 19 | 57 |
| 19 | reflective | 19 | 57 |
| 20 | gloomy | 19 | 57 |

**Total across all batches: 1,140 dialogues**

When giving this to an LLM, say:

> "Generate batch N. emotion_a is [EMOTION]. Output ONLY valid JSONL, one dialogue per line, 57 dialogues total (19 pairs × 3 each). Follow the format and rules in this document exactly."

---

## Validation

After generation, the dialogues will be validated by an automated pipeline that checks:
1. Valid JSON on each line
2. `emotion_a` and `emotion_b` are from the approved list and are different
3. Dialogue starts with "Speaker A:" and contains alternating turns
4. At least 4 turns (2 from each speaker)
5. Neither emotion word appears in the dialogue text itself (optional soft check)

Lines that fail validation will be dropped. Generators should aim for 100% pass rate.

---

## File Naming Convention

Save each batch's output as:
```
data/corpus_contributions/batch_NN_EMOTION.jsonl
```

For example: `batch_04_angry.jsonl`, `batch_12_surprised.jsonl`

All files in `data/corpus_contributions/` will be automatically merged by the ingestion pipeline.
