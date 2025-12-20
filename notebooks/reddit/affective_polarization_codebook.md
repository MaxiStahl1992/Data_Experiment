# Affective Polarization Codebook

**Version 1.0 | December 2025**

---

## 1. Purpose and Scope

This codebook defines annotation guidelines for measuring **affective polarization** in political discourse on Reddit (September-October 2016).

### Target Phenomenon

Negative affect and hostility directed at **political out-groups**, including:

- Political parties (Democrats, Republicans)
- Politicians and political figures
- Voters and supporters of parties/candidates
- Ideological groups (liberals, conservatives, leftists, MAGA, etc.)
- Government institutions when politicized

### What This Is NOT

- **Pure policy disagreement** without affective language ("I oppose higher taxes")
- **Generic insults** without political targets ("people are idiots")
- **Criticism of ideas** without targeting supporters ("this policy is flawed")
- **Non-political negativity** (anger at corporations, sports teams, etc.)

### Conceptual Foundation

This follows contemporary frameworks emphasizing:

- Explicit emotional/affective language (not just ideological distance)
- Delegitimization and dehumanization of opponents
- Escalation from incivility to intolerance to violence
- Alignment with Political Hostility Online Scale (PHOS) and recent affective polarization measures

---

## 2. Label Scale (4 Levels)

| Level | Label       | Short Definition                                                                                |
| ----- | ----------- | ----------------------------------------------------------------------------------------------- |
| **0** | None        | No political target OR no affective evaluation                                                  |
| **1** | Adversarial | Incivility, insults, ridicule toward political out-group; opponents still treated as legitimate |
| **2** | Intolerant  | Delegitimization, framing as enemies/threats, calls for exclusion or denial of legitimacy       |
| **3** | Belligerent | Explicit support for harm, violence, or elimination of opponents                                |

---

## 3. Decision Tree

For each text unit (comment, submission text, sentence), follow this sequence:

### Step 1: Is there a political actor or group?

- **YES if:** References parties, politicians, voters, ideological groups in political context
- **NO if:** Only discusses policies abstractly, or mentions non-political entities
- **If NO → Label 0**

**Examples of political targets:**

- "Democrats", "Republicans", "liberals", "conservatives"
- "MAGA people", "Bernie bros", "Hillary supporters", "Trump voters"
- "the left", "the right", "leftists", "right-wing nuts"
- "Obama", "Trump", "Clinton", "politicians"
- "Congress", "the administration" (when politically framed)

### Step 2: Is there affective evaluation of that target?

- **YES if:** Emotional language expressing like/dislike, contempt, disgust, hatred, anger
- **NO if:** Purely descriptive, analytical, or informational
- **If NO → Label 0**

**Affective markers:**

- Negative adjectives (stupid, evil, corrupt, dangerous)
- Emotional verbs (hate, despise, fear, distrust)
- Dehumanizing metaphors (disease, vermin, cancer)
- Moral outrage expressions

### Step 3: How hostile is the evaluation?

#### Level 1 - Adversarial (Incivility)

- **Markers:** Insults, mockery, ridicule, sarcasm, name-calling
- **Key feature:** Still treats opponents as legitimate participants in democracy, just stupid/wrong/annoying
- **Examples:**
  - "Democrats are clueless about the economy"
  - "MAGA idiots keep falling for obvious lies"
  - "These liberals wouldn't know logic if it hit them"
  - "Republicans are corrupt as hell"

#### Level 2 - Intolerant (Delegitimization)

- **Markers:** Enemy framing, threat framing, exclusionary language, dehumanization
- **Key feature:** Denies legitimacy, portrays as fundamentally dangerous or un-American
- **Examples:**
  - "These leftists are enemies of America"
  - "They should not have a say in our government"
  - "Republicans are a disease spreading through this country"
  - "Democrats are traitors who want to destroy us"
  - "They need to be stopped from voting"

#### Level 3 - Belligerent (Eliminationist)

- **Markers:** Calls for violence, harm, imprisonment, elimination
- **Key feature:** Explicit support for removing opponents through force
- **Examples:**
  - "These MAGA freaks should all be locked up"
  - "Time to round up all the commies"
  - "They should be shot for what they've done"
  - "Can't wait for the liberal purge"

---

## 4. Borderline Cases and Ambiguities

### Mixed Policy + Personal Attacks

- If text criticizes _policy_ strongly but also attacks _supporters_:
  - "This immigration policy is evil **and its supporters are traitors**" → Label based on personal attack (2)
- If only policy: "This immigration policy is evil" → 0

### Ambiguous Political Target

- Use context (subreddit, thread topic, previous comments)
- If still unclear after context: **Default to 0**
- Example: "These people are ruining everything" in r/politics discussing Trump → Likely political (code affect level)

### Irony and Sarcasm

- If a reasonable reader would interpret as negative toward the group: **Code as negative**
- Example: "Oh yeah, Republicans REALLY care about the working class /s" → 1 (Adversarial)

### Quoting Others

- If text quotes hostile language but **endorses** it → Code the hostility
- If text quotes hostile language but **criticizes** it → 0
- Example: "Someone said 'Democrats are enemies' and I agree" → 2

### In-Group Criticism

- Criticism of one's own party/group typically less hostile
- Code based on actual affect level, but rare to see Level 3 toward in-group

---

## 5. Annotation Examples

### Level 0 Examples

| Text                                                        | Rationale                                     |
| ----------------------------------------------------------- | --------------------------------------------- |
| "Republicans won control of the House in the midterms."     | Purely descriptive, no evaluation             |
| "This tax policy will hurt small businesses significantly." | Policy criticism without targeting supporters |
| "The debate between Clinton and Trump was heated."          | Factual description of event                  |
| "I disagree with liberal positions on gun control."         | Disagreement without affect                   |

### Level 1 Examples

| Text                                                 | Rationale                                       |
| ---------------------------------------------------- | ----------------------------------------------- |
| "Democrats are absolutely clueless about economics." | Insult, but treating as legitimate (if wrong)   |
| "Republicans just want to screw over poor people."   | Attributing bad motives, but not delegitimizing |
| "MAGA morons can't think for themselves."            | Name-calling and mockery                        |
| "Hillary supporters are sheep following a criminal." | Ridicule and insults                            |

### Level 2 Examples

| Text                                                 | Rationale                                          |
| ---------------------------------------------------- | -------------------------------------------------- |
| "These leftists are enemies of the American people." | Direct enemy framing                               |
| "Republicans are trying to destroy our democracy."   | Threat framing, existential danger                 |
| "Trump supporters should not be allowed to vote."    | Explicit call for exclusion from political process |
| "Democrats are a cancer on this nation."             | Dehumanizing metaphor (disease)                    |
| "The left wants to see America burn."                | Portraying as fundamentally anti-American          |

### Level 3 Examples

| Text                                                       | Rationale                            |
| ---------------------------------------------------------- | ------------------------------------ |
| "These MAGA fascists need to be put down."                 | Eliminationist language (killed)     |
| "Lock up every single Democrat politician."                | Call for mass imprisonment           |
| "Can't wait for someone to take out these liberal judges." | Support for violence (assassination) |
| "Time to round up the commies and be done with them."      | Eliminationist, mass violence        |
| "They should all be lined up and shot."                    | Explicit call for execution          |

---

## 6. Quality Control Guidelines

### For Annotators

1. **Read carefully** - Don't skim; context matters
2. **Follow the decision tree** - Step 1 (target?), Step 2 (affect?), Step 3 (how hostile?)
3. **When in doubt** - Default to lower level (conservative coding)
4. **Take breaks** - This content can be draining; annotate in sessions
5. **Track difficult cases** - Note any texts where you're uncertain

### Expected Distribution

Based on political Reddit data:

- **Level 0 (None):** 40-60% - Most content is descriptive/neutral
- **Level 1 (Adversarial):** 30-40% - Incivility is common
- **Level 2 (Intolerant):** 10-20% - Delegitimization is notable
- **Level 3 (Belligerent):** 2-5% - Extreme hostility is rare but present

If your distribution differs greatly, review your annotation strategy.

### Inter-Annotator Reliability

- Target: Weighted Kappa > 0.70 (substantial agreement)
- Ordinal scale: 1→2 disagreement is better than 1→3
- Review systematic disagreements (e.g., always disagreeing on borderline 0/1 cases)

---

## 7. From Labels to Polarization Scores

After annotation and model training:

### Individual Text Score

- **Direct:** Predicted label / 3 (normalized 0-1)
- **Continuous:** Expected value across class probabilities
- **Binary:** Any level > 0 indicates affective polarization present

### Aggregate Scores

- **By topic:** Mean/median polarization toward specific political targets
- **By user:** Mean polarization in user's comment history
- **By subreddit:** Community-level polarization
- **Over time:** Track escalation/de-escalation

### Interpretation

- **0.00-0.20:** Minimal affective polarization
- **0.20-0.40:** Moderate incivility
- **0.40-0.60:** Substantial intolerance
- **0.60-1.00:** Severe/belligerent hostility

---

## 8. References

This codebook synthesizes guidance from:

1. Affective polarization frameworks emphasizing explicit affect (not just distance)
2. Political Hostility Online Scale (PHOS) - agonistic/antagonistic distinction
3. Contemporary measurement work on delegitimization and dehumanization
4. Manifesto Project coding standards for reliability
5. Recent LLM-assisted annotation protocols

### Key Papers

- Political hostility measurement and democratic discourse
- Affective polarization: putting the affect back in
- Online political toxicity and incivility scales
- Dehumanization in political communication

---

## Version History

- **v1.0 (December 2025):** Initial codebook for thesis project
  - 4-level scale based on PHOS framework
  - Decision tree structure
  - Example set from 2016 Reddit political discourse

---

**Questions or ambiguous cases?** Note them for review and discussion. This is a living document that will be refined based on annotation experience.
