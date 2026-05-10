# Neuroscience → Algorithm Mapping

Each neuroscience principle has a concrete implementation with code references.

## 1. Oscillatory Phase-Locking (Theta/Gamma)

| Neuroscience | Algorithm | Code |
|---|---|---|
| Place cells fire at progressively earlier theta phases (phase precession) | `Oscillator.phase_offset` encodes sequence position within a constellation | `anchor.py:18-29` |
| Theta organizes sequences, gamma organizes items (theta-gamma coupling) | `natural_frequency` = theta band (0.1-1.0), edges = gamma coupling | `anchor.py:22-24` |
| Arnold tongues: stable phase-locking when driving frequency ≈ natural frequency | `_resonance_score()` = `(1-w)×‖z_q·z_m‖/‖z_q‖‖z_m‖ + w×cos(Δφ)` | `retriever.py:173-194` |
| Phase-locked neurons fire synchronously → ensemble retrieval | `find_constellation()` groups phase-locked anchors | `graph.py` |
| Sharp-wave ripples (SWRs): compressed ~20× replay during rest | `_swr_replay()` with replay_count increment | `sleep.py:98-140` |

## 2. Memory Reconsolidation

| Neuroscience | Algorithm | Code |
|---|---|---|
| Retrieval destabilizes memory for 4-6 hours | `activate()` sets `stability *= 0.7` | `anchor.py:128-130` |
| Prediction error gates reconsolidation | `consolidate(prediction_error)` → `strengthen`/`update`/`novel` | `anchor.py:132-149` |
| Small error → strengthen, medium → update, large → new memory | Three-threshold gating with `AnchorPrediction` | `anchor.py:118-126` |
| Prediction generation from retrieved memories | `AnchorPrediction` fields: next_topic, emotional_tone, expected_duration, confidence | `anchor.py:86-99` |

## 3. Systems Consolidation (Hippocampal → Cortical Transfer)

| Neuroscience | Algorithm | Code |
|---|---|---|
| New memories depend on hippocampus, old on cortex | `hippocampal_dependency` decays with replay count | `sleep.py:150-183` |
| Sleep spindle-SWR coupling enables transfer | Exponential decay: `exp(-replay_factor × cycles / τ)` where τ=20 | `sleep.py:156-159` |
| Semanticization: episodic details fade, gist remains | Cortical anchors simplify text to first sentence when `stability > 0.8` | `sleep.py:165-168` |
| Multiple trace theory: each replay creates trace | `replay_count` per anchor drives consolidation | `sleep.py:155` |
| Episodic edges weaken, semantic edges strengthen | Temporal edges decay `×0.985`, topical/causal strengthen `×1.01` | `sleep.py:170-178` |

## 4. Schema Formation (mPFC Abstraction)

| Neuroscience | Algorithm | Code |
|---|---|---|
| mPFC extracts invariant structures across episodes | `_schema_extraction()` groups by tag overlap, finds common template | `sleep.py:213-275` |
| Assimilation: new info encoded via existing schema | `Schema.match()` checks fit; anchors reference `schema_ref` | `graph.py` (Schema class) |
| Accommodation: schema-incongruent → new/updated schema | MIN_SIMILARITY=0.7 gates schema formation | `sleep.py:253-255` |
| Schemas resist forgetting more than episodes | Schemas stored separately in `graph.schemas` dict | `graph.py` |

## 5. Emotional Modulation

| Neuroscience | Algorithm | Code |
|---|---|---|
| Amygdala enhances encoding of emotional events | `emotional_valence` boosts importance during SWR replay: `+30% per |valence|` | `sleep.py:134` |
| Noradrenaline/cortisol modulation | Higher |valence| → higher replay priority (`×0.4` weight) | `sleep.py:107` |
| Emotional stripping during sleep (adaptive) | `emotional_valence *= 0.75` per cycle; importance preserved | `sleep.py:190-208` |
| Wisdom: retain lesson without emotional burden | `importance = max(importance×0.97, |old_valence|×0.25 + 0.15)` | `sleep.py:203-207` |

## 6. Adaptive Forgetting (Active Pruning + Savings)

| Neuroscience | Algorithm | Code |
|---|---|---|
| Ebbinghaus forgetting curve (exponential + power-law) | `retention_score` combines recency, frequency, stability, importance | `anchor.py:138-145` |
| Interference: similar memories compete | Contradiction detection penalizes weaker anchor in pair (`−0.2`) | `sleep.py:332-342` |
| Savings effect: relearning faster than original | `GhostAnchor` with partial embedding residue; `revival_count` | `anchor.py:54-75` |
| Active inhibition of competing memories | `get_prune_candidates(threshold)` filters by `retention_score` | `graph.py` |
| Stale ghost cleanup (traces that never revive) | Ghosts >30 days with 0 revivals removed | `sleep.py:358-365` |

## 7. Hebbian Plasticity

| Neuroscience | Algorithm | Code |
|---|---|---|
| Cells that fire together, wire together | `_hebbian_update()`: co-activated edges strengthen `+0.03` | `sleep.py:420-429` |
| Long-term depression: dormant connections weaken | Dormant edges decay `0.02 × log(1 + hours/24)` | `sleep.py:428-429` |
| Synaptic homeostasis: global downscaling during sleep | Mean weight target 0.3; all weights scaled proportionally | `sleep.py:433-458` |
| Dynamic range preservation | Partial scaling: `weight *= (0.5 + 0.5 × scale)` | `sleep.py:453` |

## 8. Spreading Activation

| Neuroscience | Algorithm | Code |
|---|---|---|
| Semantic priming spreads activation through association network | `spread_activation(seeds, steps=3, decay=0.5)` | `graph.py` |
| Activation decays with graph distance | Each step multiplies activation by decay factor | `graph.py` |
| Top-k activated nodes form constellation boundary | `find_constellation()` expands until activation drops below threshold | `graph.py` |

## 9. Predictive Coding (Free Energy Principle)

| Neuroscience | Algorithm | Code |
|---|---|---|
| Brain minimizes prediction error (free energy) | `predictive_retrieve()` returns `(Constellation, action)` | `resonance.py` |
| Higher cortex predicts lower; only errors propagate | `AnchorPrediction` compared to actual; error drives `consolidate()` | `anchor.py:132-149` |
| Model update minimizes future surprise | Novel experiences create anchors with `surprise=1.0` | `anchor.py` |
| Precision-weighted prediction errors | `confidence` field in `AnchorPrediction` weights error impact | `anchor.py:94` |

## Summary Table

| # | Principle | Brain Region | Key Structure | Core Method |
|---|---|---|---|---|
| 1 | Phase-locking | Hippocampus | Theta oscillations | `Oscillator` + phasor resonance |
| 2 | Reconsolidation | Hippocampus | Protein synthesis window | `consolidate(prediction_error)` |
| 3 | Consolidation | HPC→Cortex | Sleep spindles + SWRs | `hippocampal_dependency` decay |
| 4 | Schema | mPFC | Pattern separation | `Schema` + tag-based clustering |
| 5 | Emotion | Amygdala | Noradrenergic modulation | `emotional_valence` gating |
| 6 | Forgetting | Whole-brain | Synaptic renormalization | `GhostAnchor` + `retention_score` |
| 7 | Hebbian | Synapses | LTP/LTD | Edge weight `strengthen`/`weaken` |
| 8 | Priming | Association cortex | Spreading activation | `spread_activation()` |
| 9 | Prediction | Cortical hierarchy | Prediction error | `AnchorPrediction` + `consolidate()` |
