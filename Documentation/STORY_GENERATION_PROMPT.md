# EmotionScope Emotion Story Generation Prompt

**Give this document to any capable LLM (Claude, ChatGPT, Gemini, etc.) to generate emotion-tagged story prompts for the EmotionScope extraction pipeline.**

---

## Task

Write short narrative vignettes (1-3 sentences each) that evoke a specific emotion through *scene and situation* — never by naming the emotion. These vignettes are fed to a language model during training: the model's internal neural activations while processing them become the basis for extracting "emotion direction vectors" from its residual stream.

The quality of these stories directly determines the quality of the emotion vectors. Generic or repetitive stories produce noisy, overlapping vectors. Vivid, specific, diverse stories produce clean, discriminative vectors.

## Output Format

Output **ONLY** valid JSONL (one JSON object per line, no markdown, no commentary):

```
{"emotion": "EMOTION_NAME", "text": "The vignette text goes here as a single flowing passage"}
```

**Critical rules:**
- `emotion` must be from the approved list (lowercase, spelled exactly)
- `text` is a single passage, 1-3 sentences, no dialogue tags
- No newlines within the text field
- No trailing whitespace

## The 20 Emotions

```
happy, sad, afraid, angry, calm, desperate, hopeful, frustrated,
curious, proud, guilty, surprised, loving, hostile, nervous,
confident, brooding, enthusiastic, reflective, gloomy
```

---

## Quality Standard

Each story must meet ALL of these criteria:

### 1. Never name the emotion

**BAD:** "They felt deeply afraid when the door slammed shut"
**BAD:** "A wave of happiness washed over them"
**BAD:** "The angry customer demanded to speak to the manager"

**GOOD:** "The headlights behind them had been there for twenty miles now, matching every turn, and the road ahead ran empty for another thirty before the next town"

The word for the emotion (or obvious synonyms) must not appear in the text.

### 2. Show a specific scene, not a summary

**BAD:** "They experienced loss and felt empty inside"
**GOOD:** "The dog's water bowl was still by the back door three weeks later, and no one could bring themselves to put it away, so it just stayed there gathering a thin film of dust"

The reader should be able to picture a specific moment in time — a room, a gesture, an object, a sound.

### 3. Use physical and sensory anchors

Ground the emotion in something concrete: a clenched jaw, dust on glass, the sound of footsteps, the weight of an envelope, the smell of rain. Abstract language doesn't activate the model's emotion circuits the way concrete sensory language does.

### 4. Vary the intensity

Not every story should be extreme. Mix across three intensity levels:
- **Subtle:** A quiet undercurrent of the emotion, barely there
- **Moderate:** Clear and present, but contained
- **Intense:** Unmistakable, visceral

### 5. Break your own patterns

This is the hardest requirement. After writing 10 stories for "afraid," you will start defaulting to the same structures (dark rooms, medical results, someone following). Actively break out:
- Change the **setting** (outdoors, workplace, kitchen, hospital, car, school, foreign country)
- Change the **character** (child, elderly person, parent, worker, stranger, animal)
- Change the **temporal framing** (happening now, just happened, about to happen, remembered)
- Change the **sensory channel** (visual, auditory, tactile, olfactory)
- Change the **social context** (alone, with family, with strangers, in a crowd)

### 6. Each story must be about a DIFFERENT situation

Within the same emotion, no two stories should involve:
- The same type of location
- The same relationship dynamic
- The same kind of triggering event
- The same sentence structure or rhythm

If you've written a story about a letter, don't write another about a letter. If you've written about a child, move to an elderly person. If you've used "they walked through" once, find a different verb of motion.

---

## Scenario Categories

For EACH emotion, draw stories from across these categories. Aim for at least one story per category, never more than three from the same category:

| Category | Examples |
|----------|---------|
| **Relationships** | Partner, parent, child, friend, sibling, ex, stranger |
| **Workplace** | Boss, colleague, promotion, layoff, project, deadline |
| **Health/body** | Diagnosis, recovery, injury, aging, birth, exhaustion |
| **Home/domestic** | House, room, object, pet, garden, cooking, cleaning |
| **Nature/weather** | Storm, season, landscape, animal, ocean, mountain |
| **Travel/movement** | Car, plane, walking, lost, arriving, leaving, border |
| **Achievement** | Test, competition, creative work, milestone, recognition |
| **Loss/absence** | Death, departure, missing thing, empty space, ending |
| **Discovery** | Finding, opening, learning, realizing, uncovering |
| **Social/public** | Crowd, ceremony, meeting, performance, confrontation |
| **Financial** | Money, debt, windfall, poverty, purchase, investment |
| **Memory/time** | Old photograph, anniversary, return visit, aging |

---

## Examples of Excellent Stories (from the existing corpus)

**happy:**
"The song came on the radio unexpectedly and they turned it up loud and drove the rest of the way with the windows down singing along to every word"

**sad:**
"The house was quiet in a way it had never been quiet before, and they walked through each empty room running their hand along the walls, touching the places where the pictures used to hang"

**afraid:**
"They opened the front door and the house felt wrong in a way they couldn't name, as if someone had been inside and rearranged the air"

**guilty:**
"The coworker had taken the blame for the mistake and they had not corrected it, had not even really meant to say nothing, but saying nothing was itself a decision they had made"

**loving:**
"The old dog had climbed into bed again, uninvited as always, and was now snoring softly against their side, and they did not move for an hour because they did not want to wake it"

**brooding:**
"They sat in the parked car for a long time after arriving, watching the rain move across the windshield, replaying the conversation from that morning and finding new ways it could have gone differently"

Notice: long flowing sentences, sensory details, specific physical actions, zero emotion words.

---

## Your Assignment

Generate **40 stories** for EACH emotion in your assigned batch. That's 40 unique situations per emotion, drawn from diverse scenario categories, at mixed intensity levels.

### Batch Assignments

| Batch | Emotions | Stories per emotion | Total |
|-------|----------|-------------------|-------|
| A | happy, sad, afraid, angry, calm | 40 | 200 |
| B | desperate, hopeful, frustrated, curious, proud | 40 | 200 |
| C | guilty, surprised, loving, hostile, nervous | 40 | 200 |
| D | confident, brooding, enthusiastic, reflective, gloomy | 40 | 200 |

When giving this to an LLM, say:

> "Generate batch [LETTER]. Write 40 stories per emotion, 200 total. Output ONLY valid JSONL, one story per line. Follow the quality rules exactly — every story must be a unique scene that evokes the emotion without naming it. Take your time. Diversity matters more than speed."

---

## Self-Check Before Submitting

For each emotion, verify:
- [ ] 40 unique stories, no duplicates
- [ ] Zero stories that name the emotion or use obvious synonyms
- [ ] At least 8 different scenario categories represented
- [ ] Mix of subtle, moderate, and intense stories
- [ ] No two stories with the same setting + trigger combination
- [ ] Each story is a concrete scene (can you picture it?), not an abstract summary

---

## File Naming Convention

Save output as:
```
data/story_contributions/stories_EMOTION.jsonl
```

For example: `stories_afraid.jsonl`, `stories_happy.jsonl`

Or as a single batch file: `stories_batch_A.jsonl`

All files in `data/story_contributions/` will be automatically merged by the ingestion pipeline.
