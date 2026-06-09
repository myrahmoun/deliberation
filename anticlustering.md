
Build a group formation and recommendation system for a large-scale deliberation platform (**Frankly**, at `app.frankly.org`). The target scale is **~10,000 participants** with **groups of 8**.

---

## Core Problem: Anti-Clustering for Group Formation

Groups should be formed to maximize **viewpoint diversity** — the opposite of clustering. This is referred to as **anti-clustering**.

### Key Design Decisions

- Anti-clustering is done **purely based on distances in the embedding space** (not raw features, which are encoded in the embedding itself).
    
- Use Carter's existing embeddings directly — no need to re-implement embedding logic.
    
- Group quality is modeled as a **quadratic function of dispersion**:
    
    ```
    quality(G) = a·D(G)² + b·D(G) + c
    ```
    
    where:
    
    - `D(G)` = dispersion of group `G` (e.g., mean distance of points to centroid)
    - `a` is **negative** (inverted-U relationship — too little or too much disagreement is bad)
    - `a`, `b`, `c` are **learnable parameters** fit via post-deliberation survey data (inspired by [Fung 2014](https://doi.org/10.1080/10584609.2014.969466))

### Optimization Goal

Maximize average group quality across all groups:

```
maximize: mean over groups G of [a·D(G)² + b·D(G) + c]
```

---

## Implementation Tasks

### Task 1 — Use Carter's Embeddings

- Integrate with Carter's existing embedding pipeline.
- No changes needed to the embedding itself.

### Task 2 — Anti-Clustering Algorithm (Synthetic Data Baseline)

- Implement the anti-clustering algorithm using the quadratic quality function above.
- Test on **synthetic data**.
- Performance target: **10,000 data points processed in ≤ 30 seconds**.
- Output metric: **runtime as a function of number of data points** (scaling curve).

### Task 3 — Full Pipeline on Real Data

- Combine Tasks 1 & 2: run the anti-clustering algorithm on **Remesh data** using Carter's embeddings.

### Task 4 (Future) — Parameter Fitting Infrastructure

- Build infrastructure to fit `a`, `b`, `c` from post-deliberation survey responses.
- Modeled after the methodology in Fung (2014).

---

## Bridging Statement Recommendations (Bailey's Proposal)

Once groups are formed, surface external viewpoints to help groups broaden their discussion.

### Proposed Approach

1. Find a **representative slate of ~6 bridging comments** from the full deliberation (via Bridge).
2. Show the slate to all participants.
3. Give each participant a **personalized nudge** toward the statements most distant from their own viewpoint.

### Design Considerations (from Carter & Bailey)

|Concern|Proposed Resolution|
|---|---|
|Single statement gives incomplete picture|Show a slate of ~6 statements|
|Slate covers multiple topics → fragmented discussion|Topic-based clustering; route topic-relevant slate to each group|
|Personalized nudges pull group in different directions|3-min individual exploration → 3-min group vote to pick one statement|
|Low relevance if group topic isn't controlled|Let group collaboratively choose from the slate|

### Alternative Considered

- Show a **2D visualization of statements in embedding space**, color-coded by topic, with personalized recommendations overlaid — more exploratory, less prescriptive.

---

## References

- Archon Fung (2014) — ["The right amount of disagreement"](https://doi.org/10.1080/10584609.2014.969466) — empirical basis for the quadratic quality model; disagreement measured as SD of ideal points in 1D.
