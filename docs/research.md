# Star Graph Memory: Deep Research

## 1. Oscillatory Resonance — Beyond Spreading Activation

### Neuroscience Basis
Hippocampal place cells fire in sequence during navigation. During rest/sleep, these sequences replay in compressed time (~20×) via sharp-wave ripples (SWRs). Critically, place cell firing is modulated by theta oscillations (4-12 Hz): as a rat moves through a place field, the neuron fires at progressively earlier theta phases — **phase precession**. This means temporal sequences are encoded as phase relationships within a single theta cycle.

Gamma oscillations (30-100 Hz) are coupled to specific theta phases (theta-gamma coupling), creating a hierarchical timing system: theta organizes sequences, gamma organizes individual items within a sequence.

### Computational Principle
Memory retrieval shouldn't be graph traversal with fixed weights. It should be **oscillatory resonance**: each anchor has a characteristic frequency and phase. When context provides a driving oscillation, anchors whose natural frequencies match will phase-lock and fire together. This naturally handles:
- Sequence retrieval (phase precession encodes order)
- Hierarchical structure (theta-gamma nesting)
- Simultaneous activation of related memories (phase synchrony)

### Algorithm: Phase-Locked Resonance

```
Each anchor has:
  - natural_frequency: float (0.1..1.0, akin to preferred theta phase)
  - phase_offset: float (0..2π)
  - coupling_strength: float (how strongly it couples to the driving oscillation)

Resonance(context_embedding, graph):
  driving_freq, driving_phase = derive_oscillation(context_embedding)
  
  resonance_map = {}
  for each anchor in graph:
    freq_diff = |anchor.natural_frequency - driving_freq|
    phase_diff = |anchor.phase_offset - driving_phase|
    
    # Arnold tongue: resonance occurs when frequencies are close
    # and phases align within a critical window
    if freq_diff < CRITICAL_FREQ_WINDOW:
      coupling = anchor.coupling_strength * exp(-freq_diff² / σ²)
      resonance_map[anchor.id] = coupling * cos(phase_diff)
  
  # Activated anchors form an oscillatory assembly
  # Phase-locked anchors fire synchronously → constellation retrieval
  synchronized = find_phase_locked_groups(resonance_map)
  return [extract_constellation(g) for g in synchronized]
```

### Why This Is Better
- **Sequence encoding**: Phase precession naturally orders events. Theta phase within a constellation encodes temporal order without explicit edge types.
- **Graceful degradation**: Partial phase locking still works. No hard threshold.
- **Interference patterns**: Two similar contexts can create beating patterns — constructive interference amplifies the right constellation, destructive suppresses others.

---

## 2. Memory Reconsolidation — Retrieval as Rewriting

### Neuroscience Basis
Memories are not read-only. When a memory is retrieved (reactivated), it enters a labile state lasting 4-6 hours. During this window, the memory can be:
- Strengthened (additional encoding)
- Weakened (extinction learning)
- Updated (new information incorporated)
- Distorted (misinformation effect)

The key signal governing reconsolidation is **prediction error**: the difference between what the memory predicts and what actually occurs. Large prediction error triggers reconsolidation. No prediction error → memory stays stable. Moderate error → memory update. Extreme error → new memory formation.

This means **every retrieval is also a potential rewrite**.

### Computational Principle
When an anchor is activated (retrieved), it should:
1. Generate a prediction about what will be experienced
2. Compare prediction to actual experience (prediction error)
3. Based on error magnitude:
   - ε < threshold_strengthen: reinforce the anchor (increase stability)
   - threshold_strengthen < ε < threshold_update: update the anchor (modify text, adjust vector)
   - ε > threshold_update: create new anchor, weakened link to old

### Algorithm: Reconsolidation Gate

```
reconsolidate(anchor, new_experience, prediction_error):
  # Phase 1: Destabilization
  anchor.vector.stability *= 0.5  # becomes labile
  
  if prediction_error < STRENGTHEN_THRESHOLD:
    # Confirmation: memory was accurate
    anchor.vector.importance = min(1.0, anchor.vector.importance + 0.1)
    anchor.vector.stability = min(1.0, anchor.vector.stability + 0.3)
    
  elif prediction_error < UPDATE_THRESHOLD:
    # Update: incorporate new information
    anchor.text = merge_text(anchor.text, new_experience)
    anchor.vector.surprise = (anchor.vector.surprise + prediction_error) / 2
    anchor.embedding = recompute_embedding(anchor.text)
    # Update edges: old associations may weaken, new ones form
    adjust_edges(anchor, new_experience)
    
  else:
    # Novelty: create new anchor, don't overwrite old one
    new_anchor = Anchor.create(new_experience)
    new_anchor.vector.surprise = 1.0  # highly surprising
    # Link old and new with a "revised" edge
    graph.add_edge(anchor.id, new_anchor.id, weight=0.3, edge_type="revision")
```

---

## 3. Schema Formation — Abstraction Across Episodes

### Neuroscience Basis
The medial prefrontal cortex (mPFC) extracts common structures across related memories, forming **schemas** — abstract frameworks that capture the invariant elements across episodes. Once a schema exists:
- New information consistent with the schema is encoded rapidly (assimilation)
- Schema-incongruent information triggers prediction error and either schema update (accommodation) or separate encoding
- Schemas are more resistant to forgetting than individual episodes

Schemas emerge gradually. Initial learning is hippocampal-dependent and episodic. Over time and sleep, common patterns are extracted to mPFC. The hippocampus then preferentially encodes schema-relevant new information.

### Computational Principle
A **Schema** is a higher-order structure that abstracts the common elements from multiple constellations. It has:
- A template: the invariant structure
- Slots: variable elements that differ across instances
- Defaults: typical slot values when no specific info is available

Schemas guide encoding: new anchors that match a schema's template are encoded in terms of that schema (filling slots), rather than as entirely new structures.

### Algorithm: Schema Extraction

```
Schema:
  template: str          # abstract pattern description
  slots: dict[str, Any]  # variable components
  instances: list[str]   # anchor IDs that instantiate this schema
  confidence: float      # 0..1, how well-established

extract_schemas(constellation):
  # After multiple sleep cycles, look for patterns
  # across anchors in the same constellation
  
  # Cluster anchors by structural similarity
  clusters = cluster_by_structure(constellation.anchors)
  
  schemas = []
  for cluster in clusters:
    if len(cluster) >= MIN_SCHEMA_INSTANCES:
      # Extract common template
      template = extract_common_template(cluster)
      # Identify variable slots
      slots = identify_variable_slots(cluster, template)
      # Create schema
      schema = Schema(template=template, slots=slots, 
                      instances=[a.id for a in cluster],
                      confidence=len(cluster) / MIN_SCHEMA_INSTANCES)
      schemas.append(schema)
  
  return schemas

# Schema-guided encoding
encode_with_schema(new_experience, schemas, graph):
  for schema in schemas:
    match_result = match_schema(schema, new_experience)
    if match_result.confidence > MATCH_THRESHOLD:
      # Assimilate: encode in terms of schema
      anchor = Anchor.create(
        text=fill_slots(schema.template, match_result.slot_values),
        tags=schema.tags + ["schema_instance"]
      )
      anchor.schema_ref = schema.id
      return anchor, "assimilated"
  
  # No matching schema → accommodate: potentially new schema
  return Anchor.create(new_experience), "new_pattern"
```

---

## 4. Predictive Coding — Unifying Memory and Perception

### Neuroscience Basis
The brain is fundamentally a prediction machine. Cortical hierarchy: higher levels predict lower-level activity. What propagates upward is not raw sensory data but **prediction errors**. Memory is not a separate system — it's the brain's internal model that generates predictions.

The free energy principle: organisms minimize surprise by either (a) updating their internal model to match sensory input (perception/learning), or (b) acting to make sensory input match their predictions (action).

Memory retrieval, in this view, is the brain **generating a prediction** about the current situation. Memory formation is **updating the model** when predictions fail.

### Computational Principle
Every anchor in the star graph encodes a mini-prediction model. When encountering new context:
1. Activated constellation generates a prediction
2. Prediction error is computed against actual context
3. Error drives model update (memory formation/reconsolidation) OR attention shift (seek better model)

### Algorithm: Prediction Error Minimization

```
class PredictiveAnchor:
    text: str
    # Each anchor predicts: what topic, what emotional tone, what next action
    predicts: {
      "next_topic": embedding,
      "emotional_tone": float,    # -1..+1
      "expected_duration": float,  # in minutes
    }
    
    def prediction_error(self, actual_context) -> float:
        errors = {
          "topic": cosine_distance(self.predicts["next_topic"], actual_context.embedding),
          "tone": abs(self.predicts["emotional_tone"] - actual_context.emotional_tone),
        }
        return weighted_sum(errors)  # composite prediction error

# During retrieval
predictive_retrieve(context, graph):
    # Step 1: Find constellation that best predicts current context
    constellations = resonator.resonate(context)
    
    best_constellation = None
    min_error = float('inf')
    
    for c in constellations:
        # Generate prediction from constellation
        prediction = aggregate_predictions(c.anchors)
        error = compute_prediction_error(prediction, context)
        
        if error < min_error:
            min_error = error
            best_constellation = c
    
    # Step 2: Use prediction error to decide action
    if min_error < CONFIRMATION_THRESHOLD:
        # Good prediction → retrieve and strengthen
        return best_constellation, "confirm"
    elif min_error < SURPRISE_THRESHOLD:
        # Moderate error → retrieve but note update needed
        return best_constellation, "update"
    else:
        # High error → this is new, need to encode
        return None, "novel"
```

---

## 5. Emotional Modulation — What Makes Memories Stick

### Neuroscience Basis
Amygdala activation during encoding enhances hippocampal consolidation. Noradrenaline and cortisol modulate synaptic plasticity: emotional arousal triggers the release of these neuromodulators, which strengthen the encoding of emotionally salient events.

However, during sleep, emotional charge is progressively stripped from memories while the informational content is retained. This is adaptive: you remember that the snake was dangerous without re-experiencing the full fear response.

### Computational Principle
- `emotional_valence` in the anchor vector acts as a learning rate multiplier during encoding
- High |valence| → faster encoding, higher initial importance
- During sleep, emotional_valence decays toward 0 while importance is preserved
- This creates "wisdom": retaining the lesson without the emotional burden

### Algorithm: Emotional Gating

```
encode_with_emotion(anchor, emotional_intensity):
    # Emotional boost during encoding
    boost = 1.0 + abs(emotional_intensity) * EMOTIONAL_MULTIPLIER
    anchor.vector.importance *= boost
    anchor.vector.emotional_valence = clamp(emotional_intensity, -1, 1)
    anchor.vector.surprise *= boost
    # Higher emotional events are more likely to be replayed during sleep

sleep_emotional_strip(anchor):
    # During sleep, decouple emotion from information
    if anchor.vector.stability > 0.5:  # consolidated memory
        anchor.vector.emotional_valence *= DECAY_RATE  # e.g., 0.7 per sleep cycle
        # But keep the importance: "this mattered"
        anchor.vector.importance = max(
            anchor.vector.importance * 0.95,  # slight decay
            abs(anchor.vector.emotional_valence) * 0.3 + 0.2  # floor from emotion
        )
```

---

## 6. Forgetting as Adaptive Optimization

### Neuroscience Basis
Forgetting is not a failure — it's an active, adaptive process. The brain has limited capacity and must optimize what it retains. The Ebbinghaus forgetting curve shows exponential decay with a power-law component: rapid initial forgetting followed by a slow tail of retained memories.

Key principles:
- **Interference**: similar memories compete and interfere. The brain actively inhibits competing memories.
- **Savings**: even "forgotten" memories leave traces. Relearning is faster than original learning.
- **Adaptive forgetting**: memories that are frequently used, emotionally salient, or connected to many other memories are more resistant to decay.

### Computational Principle
Our system already has `retention_score`. But we need:
1. **Interference-based pruning**: when two anchors are contradictory, the weaker one should be actively suppressed
2. **Savings traces**: when an anchor is pruned, leave a lightweight "ghost" that can be revived
3. **Adaptive decay rates**: different types of memory decay at different rates

### Algorithm: Adaptive Forgetting

```
class GhostAnchor:
    """A pruned anchor's residual trace — enables savings."""
    id: str
    residue: list[float]  # partial embedding
    pruned_at: float
    revival_count: int = 0

adaptive_prune(graph, threshold):
    for anchor in graph.anchors:
        score = anchor.retention_score
        
        # Check for interference
        for other in graph.anchors:
            if other.id != anchor.id and are_contradictory(anchor, other):
                # Whichever has lower retention_score gets extra penalty
                if anchor.retention_score < other.retention_score:
                    score *= INTERFERENCE_PENALTY  # e.g., 0.7
        
        if score < threshold:
            # Don't fully delete — create a ghost
            ghost = GhostAnchor(
                id=anchor.id,
                residue=anchor.embedding[:DIM//4],  # partial vector
                pruned_at=time.time()
            )
            graph.ghosts[anchor.id] = ghost
            graph.remove_anchor(anchor.id)

# Revival: when a ghost resonates with new experience
try_revive(ghost, new_anchor, resonance_strength):
    if resonance_strength > REVIVAL_THRESHOLD:
        ghost.revival_count += 1
        # Revival is faster than new learning (savings effect)
        revived = Anchor(
            id=ghost.id,
            text=new_anchor.text,
            vector=AnchorVector(
                importance=0.3 + 0.1 * ghost.revival_count,
                frequency=0.1,
                recency=1.0,
                stability=0.2,
                surprise=0.8
            )
        )
        return revived
    return None
```

---

## 7. Systems Consolidation — From Hippocampal to Cortical

### Neuroscience Basis
New memories depend on the hippocampus for retrieval. Over time (days to years), through repeated reactivation during sleep, memories become increasingly independent of the hippocampus and are stored directly in neocortical connections. This is **systems consolidation**.

Key characteristics:
- **Multiple trace theory**: each reactivation creates an additional memory trace. Old memories have many traces, making them more robust.
- **Semanticization**: episodic details fade; the gist/semantic core remains.
- **Sleep spindle – slow oscillation coupling**: Slow oscillations (<1 Hz) coordinate hippocampal sharp-wave ripples with cortical sleep spindles, enabling the transfer.

### Computational Principle
- New anchors have high "hippocampal dependency" (they need the graph structure to be retrieved)
- As they're repeatedly replayed during sleep, their "cortical" representations strengthen
- Eventually, anchors can be retrieved directly (by embedding similarity) without needing graph traversal
- Long-term, anchors lose episodic detail (the specific conversation) but retain semantic core

### Algorithm: Systems Consolidation

```
systems_consolidation(graph, sleep_cycle):
    for anchor in graph.anchors:
        # Replay count drives consolidation
        replay_count = anchor.vector.frequency * sleep_cycle
        anchor.vector.stability = 1.0 - exp(-replay_count / TAU_CONSOLIDATION)
        
        # As stability increases:
        # 1. Reduce dependency on specific edges
        if anchor.vector.stability > 0.7:
            # Weaken specific episodic edges
            for neighbor in graph.neighbors(anchor.id):
                edge = graph.edges[graph._key(anchor.id, neighbor)]
                if edge.edge_type == "temporal":
                    edge.weaken(0.01)  # episodic links fade
                elif edge.edge_type == "topical":
                    edge.strengthen(0.02)  # semantic links strengthen
        
        # 2. Semanticization: simplify text toward gist
        if anchor.vector.stability > 0.8:
            anchor.text = extract_gist(anchor.text)  # shorter, more abstract
        
        # 3. Add direct embedding index for cortical-like retrieval
        if anchor.vector.stability > 0.5 and anchor.embedding:
            graph.cortical_index.add(anchor.embedding, anchor.id)
```

---

## Summary: The Complete Architecture (v0.2)

```
                    ┌─────────────────────────┐
                    │   New Experience / Query  │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │   RESONANCE ENGINE        │
                    │   - Oscillatory phase-    │
                    │     locking               │
                    │   - Prediction generation │
                    │   - Error computation     │
                    └───────────┬─────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
     ┌────────▼──────┐  ┌──────▼──────┐  ┌───────▼──────┐
     │  CONFIRM       │  │  UPDATE      │  │  NOVEL        │
     │  Strengthen    │  │  Reconsolid. │  │  New Anchor   │
     └────────┬──────┘  └──────┬──────┘  └───────┬──────┘
              │                 │                 │
              └─────────────────┼─────────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │   STAR GRAPH              │
                    │   ┌───────────────────┐   │
                    │   │ Anchors + Vectors │   │
                    │   │ Edges (typed)     │   │
                    │   │ Constellations    │   │
                    │   │ Schemas           │   │
                    │   │ Ghosts            │   │
                    │   │ Cortical Index    │   │
                    │   └───────────────────┘   │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │   SLEEP ENGINE             │
                    │   1. SWR Replay            │
                    │   2. Systems Consolidation │
                    │   3. Emotional Stripping   │
                    │   4. Schema Extraction     │
                    │   5. Merge Similar         │
                    │   6. Adaptive Prune+Ghosts │
                    │   7. Bridge Constellations │
                    │   8. Hebbian Update        │
                    └───────────────────────────┘
```
