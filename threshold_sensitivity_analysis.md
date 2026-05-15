# ManTacAi — Threshold Sensitivity Analysis

> This document catalogs every tunable threshold in the ManTacAi system, its current value,
> design justification, failure modes at extreme values, and proposed calibration methodology.
> All thresholds were initially set as expert-informed prior bounds and require empirical
> validation against labeled datasets as future work.

## System Thresholds

| # | Threshold | Value | Location | Justification | If Too Low | If Too High | Proposed Calibration |
|---|-----------|-------|----------|---------------|------------|-------------|----------------------|
| 1 | **DARVO Synergy** | 0.15 | `scoring.py:174-176` | Component presence floor — determines whether a D/A/R component is "present" enough to count toward synergy | False DARVO on benign disagreements where a single component barely registers | Misses partial DARVO patterns where one component is subtle but real | Grid search over [0.05, 0.10, 0.15, 0.20, 0.25] on labeled DARVO corpus |
| 2 | **Semantic Echo** | 0.75 | `semantic_engine.py` | Cosine similarity threshold for detecting victim parroting attacker's language | False echo detection on unrelated messages that share common vocabulary | Misses genuine linguistic mirroring where phrasing differs slightly | Grid search over [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90] on paired message dataset |
| 3 | **Contrastive Danger** | 0.50 | `semantic_engine.py:18`, `scoring.py:105` | Contrastive score (danger_sim − safe_sim) threshold; positive = closer to danger centroids | Over-triggers on neutral phrases that happen to be slightly closer to danger concepts than safe | Misses genuinely threatening language that uses indirect/euphemistic phrasing | ROC curve analysis on danger-labeled vs safe-labeled messages |
| 4 | **Context Reduction** | 0.85 | `context_scoring.py:150,187` | Determines whether contextual prediction "reduces" isolated prediction (contextual < 85% of isolated) | Too sensitive — even minor noise in context causes role flip to reactor | Too strict — only extreme context shifts trigger reactor classification | Precision-recall tradeoff on labeled initiator/reactor pairs |
| 5 | **Circuit Breaker** | 0.85 | `context_engine.py:107` | Risk threshold for immediate EXPLOSION state transition | Premature EXPLOSION on moderate-risk messages; state machine becomes too sensitive | Only catches near-certain threats; subtle escalation patterns slip through | Sensitivity analysis on conversation transcripts with known escalation events |
| 6 | **Safe Reset (Tension)** | 20 msgs + 1 hour | `context_engine.py:167` | Volume AND time gate to transition from TENSION/EXPLOSION → NORMAL | Premature de-escalation after brief neutral exchange; cycling too fast | System stays in high-alert indefinitely after one incident; never resets | Ablation study varying messages [10, 15, 20, 30, 50] × time [30m, 1h, 2h, 4h] |
| 7 | **Safe Reset (Honeymoon)** | 50 msgs + 1 hour | `context_engine.py:175` | Stricter gate for exiting HONEYMOON (love-bombing is deceptive) | Exits honeymoon too easily; misses sustained love-bombing manipulation | Never exits honeymoon; system gets stuck in post-explosion analysis | Same ablation as #6 but with honeymoon-labeled conversation data |
| 8 | **Dampening Decay Rate** | 3.0 | `context_scoring.py:162,193` | Exponential decay constant controlling how quickly dampening decays with message distance | Dampening decays too quickly — distant reactions still get strong dampening | Dampening persists too long — reactions many messages later are still heavily suppressed | Ablation study over [1.0, 2.0, 3.0, 5.0, 10.0] on labeled conversation sequences |
| 9 | **Semantic Veto** | 0.20 | `heuristics.py:253` | Contrastive score floor that aborts accountability defense dampening | Over-vetoes: even mildly proximate danger concepts block legitimate accountability phrases | Under-vetoes: dangerous messages with accountability keywords slip through dampening | Precision-recall tradeoff on accountability+threat mixed-intent messages |
| 10 | **Emergency Confidence Gate** | 0.75 | `main.py` | Model confidence threshold for flagging urgent_emergency as genuine | False emergencies: mundane urgency ("Call the plumber!") triggers helpline modal | Misses genuine emergencies where model confidence is moderate | Precision at fixed recall (target: 95% recall for genuine emergencies) |
| 11 | **Spike Weight** | 0.7 | `main.py` | Weight for single worst-moment risk in per-speaker aggregation formula | Pattern over-inflates risk: sustained low-grade abuse scores disproportionately | Single spike too dominant: one outlier message defines entire speaker risk | Ablation study over [0.5, 0.6, 0.7, 0.8, 0.9] on multi-message conversations |
| 12 | **Pattern Weight** | 0.3 | `main.py` | Weight for mean+std behavioral pattern in per-speaker aggregation formula | Only spikes detected: sustained gaslighting over 30 messages scores same as 1 insult | Sustained low-severity chatter inflates risk inappropriately | Ablation study over [0.1, 0.2, 0.3, 0.4, 0.5] (inverse of spike weight) |
| 13 | **Compound Synergy Trigger** | 0.25 | `scoring.py` | Minimum softmax probability for a tactic to count as "co-occurring" in compound manipulation | False compound detection on mild classification ambiguity between similar tactics | Misses legitimate compound abuse where secondary tactic has moderate probability | Grid search over [0.20, 0.25, 0.30, 0.35] on compound-tactic labeled messages |

## Per-Speaker Risk Aggregation Formula

```
per_speaker_risk = max(0.7 × max_risk, 0.3 × mean_risk + std_risk)
conversation_risk = max(per_speaker_risk across all speakers)
```

**Design Principle:** In forensic abuse contexts, a single extreme event (spike) is more actionable than a statistical pattern. Courts prioritize "he said he would kill me on March 3rd" over "his average hostility over 6 months was 0.45." The 0.7/0.3 split encodes this forensic actionability bias.

**Guard:** Speakers with ≤ 2 messages bypass the formula entirely and use `max_risk` directly, preventing single-message threats from being degraded by the 0.7 multiplier.

## Compound Manipulation Synergy Boost

```
high_severity_tactics = [t for t in predictions if prob > 0.25 AND weight ≥ 0.8]
if |high_severity_tactics| ≥ 3: risk *= 1.5  (near-unreachable with softmax)
if |high_severity_tactics| ≥ 2: risk *= 1.2  (primary active branch)
```

**Design Principle:** Consistent with how DARVO synergy is already computed — compound abuse should receive a targeted multiplier. The 3+ branch is written for completeness but is mathematically near-unreachable because softmax outputs sum to 1.0, making three tactics above 0.25 require ≥ 0.75 allocated to just three classes.

## Ablation Study Plan (Priority 4 — Deferred)

**Objective:** Determine whether `emotion-english-distilroberta-base` is the optimal starting point for manipulation detection vs. a generic sequence classifier.

| Model | Base | Expected Advantage |
|-------|------|--------------------|
| A (current) | `j-hartmann/emotion-english-distilroberta-base` | Pre-learned emotional weight patterns transfer to manipulation tactics |
| B (ablation) | `distilbert-base-uncased` | No pretrained emotional bias; clean slate for manipulation-specific patterns |

**Protocol:**
1. Train identical 18-class classification heads (same architecture, data, hyperparameters, 4 epochs)
2. Compare: per-class F1 (weighted), compound-tactic accuracy, confusion matrix overlap
3. Report results as design justification in thesis

**Status:** Deferred until GPU compute is available.
