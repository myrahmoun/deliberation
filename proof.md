# Correctness of the Pairwise-Exchange Anti-Clustering Algorithm

This document proves the correctness of `form_groups` in
[`algorithm.py`](algorithm.py), together with the incremental update helpers in
[`helpers.py`](helpers.py).

> At every iteration the algorithm maintains a valid partition into
> equal-sized groups; every accepted swap *strictly increases* the true objective; and the algorithm
> terminates. Hence it returns a valid partition whose objective is no worse than
> the random initial partition, and which is a **local optimum with respect to the
> single-pair swap neighborhood** once no improving swap remains.

---

## 1. Notation and the objective

Let $X = \{x_0, \dots, x_{n-1}\} \subset \mathbb{R}^d$ be the embeddings
(`embeddings`), with $n$ divisible by the group size $m$ (`group_size`), giving
$K = n/m$ groups (`n_groups`). The line
[`algorithm.py:45`](algorithm.py#L45) enforces divisibility.

A **partition** $P = (G_1, \dots, G_K)$ assigns each point to exactly one group,
with $|G_k| = m$ for all $k$.

For a group $G$ with points $\{p_1, \dots, p_m\}$, define its **centroid** and
**dispersion** (`group_dispersion`, [`helpers.py:107`](helpers.py#L107)):

$$
c(G) = \frac{1}{m}\sum_{i=1}^{m} p_i,
\qquad
D(G) = \frac{1}{m}\sum_{i=1}^{m} \lVert p_i - c(G) \rVert_2 .
$$

Given a quality function $q : \mathbb{R} \to \mathbb{R}$ (`quality_fn`, e.g.
$q(D) = aD^2 + bD + c$), the **objective** maximized is mean group quality:

$$
Q(P) = \frac{1}{K}\sum_{k=1}^{K} q\bigl(D(G_k)\bigr).
$$

---

## 2. The swap neighborhood

A **swap** picks two distinct groups $g_1 \neq g_2$ and one point from each,
$p_1 \in G_{g_1}$ at slot $i$ and $p_2 \in G_{g_2}$ at slot $j$, and exchanges
them. The selection at [`algorithm.py:69-74`](algorithm.py#L69-L74) uses
`rng.choice(..., replace=False)`, guaranteeing $g_1 \neq g_2$ and hence
$p_1 \neq p_2$.

**Lemma 1 (Partition invariance).** A swap maps a valid partition to a valid
partition.

*Proof.* A swap removes $p_1$ from $G_{g_1}$ and inserts $p_2$, and vice versa.
Each group's size is unchanged ($m$ in, $m$ out), every point remains in exactly
one group, and no other group is touched. The initial partition
([`algorithm.py:57-63`](algorithm.py#L57-L63)) is a permutation of
$\{0,\dots,n-1\}$ cut into $K$ contiguous blocks of size $m$, hence valid.
By induction over accepted swaps (lines
[90-93](algorithm.py#L90-L93)), every partition the algorithm produces is
valid. $\qquad\blacksquare$

---

## 3. The incremental updates are exact

Let $G_{g_1}$ have centroid $c_1$ and contain $x := p_1$ at slot $i$. After
swapping out $x$ and swapping in $y := p_2$, write the resulting group as
$G_{g_1}' = (G_{g_1} \setminus \{x\}) \cup \{y\}$ with centroid $c_1'$.

**Lemma 2 (Centroid update).** `updated_centroid`
([`helpers.py:120`](helpers.py#L120)) computes $c_1'$ exactly:

$$
c_1' = c_1 + \frac{y - x}{m}.
$$

*Proof.* The group's coordinate sum is $S_1 = m\,c_1$. Replacing $x$ by $y$ gives
sum $S_1 - x + y$, so

$$
c_1' = \frac{S_1 - x + y}{m} = c_1 + \frac{y - x}{m},
$$

which is exactly `old_centroid + (added - removed) / group_size`. $\qquad\blacksquare$

**Lemma 3 (Dispersion update).** `dispersion_after_swap`
([`helpers.py:129`](helpers.py#L129)) computes $D(G_{g_1}')$ exactly.

*Proof.* By definition $D(G_{g_1}') = \frac{1}{m}\sum_{p \in G_{g_1}'} \lVert p -
c_1' \rVert$. The members of $G_{g_1}'$ are the $m-1$ unchanged points of
$G_{g_1}$ plus the new point $y$. The function receives `group_embeddings` =
the *old* member array (still holding $x$ at slot $i$) and the new centroid
$c_1'$. It computes

- `dists` $= \lVert p - c_1' \rVert$ for **every** old member $p$
  ([`helpers.py:136`](helpers.py#L136)) — correct for the $m-1$ retained points,
  since dispersion is measured against the *new* centroid;
- then overwrites slot $i$ with $\lVert y - c_1' \rVert$
  ([`helpers.py:137`](helpers.py#L137)), replacing the stale $\lVert x - c_1'
  \rVert$ by the new point's contribution.

The resulting set of $m$ distances is exactly $\{\lVert p - c_1'\rVert : p \in
G_{g_1}'\}$, and its mean is $D(G_{g_1}')$. $\qquad\blacksquare$

> **Remark.** Because dispersion is a *mean distance* (not a variance), there is
> no closed-form shortcut: moving the centroid changes all $m$ distances. The
> helper therefore performs an honest $O(md)$ recomputation, which is exact, not
> an approximation. The only saving over a from-scratch call is reusing the
> already-summed centroid via Lemma 2.

**Corollary (Cache consistency).** Maintain the invariant: at the start of every
iteration, `centroids[k]` $= c(G_k)$ and `dispersions[k]` $= D(G_k)$ for all $k$.

*Proof.* True initially, since both are computed from scratch
([`algorithm.py:65-66`](algorithm.py#L65-L66)). On a **rejected** swap, state is
untouched, so the invariant persists. On an **accepted** swap, the partition
changes only for $g_1, g_2$, and lines
[94-97](algorithm.py#L94-L97) write exactly the values
$c_1', c_2', D(G_{g_1}'), D(G_{g_2}')$ that Lemmas 2–3 proved equal to the true
new centroids and dispersions. All other groups are unchanged. The invariant is
preserved. $\qquad\blacksquare$

---

## 4. The acceptance test reflects the true objective

**Lemma 4 (Correct $\Delta$).** Let $P$ be the current partition and $P'$ the
partition after a candidate swap. Then $\Delta(Q)$
([`algorithm.py:84-87`](algorithm.py#L84-L87)) satisfies

$$
\Delta Q = K\,(Q(P') - Q(P)).
$$

*Proof.* Only groups $g_1, g_2$ change, so

$$
Q(P') - Q(P) = \frac{1}{K}\bigl[q(D(G_{g_1}')) + q(D(G_{g_2}')) - q(D(G_{g_1})) - q(D(G_{g_2}))\bigr].
$$

By the cache-consistency corollary, `dispersions[g1]` equals $D(G_{g_1})$ and
`dispersions[g2]` equals $D(G_{g_2})$; by Lemma 3, `new_d1` equals $D(G_{g_1}')$ and
`new_d2` equals $D(G_{g_2}')$. Substituting into the code's expression gives exactly
the bracketed term, i.e. $K\,(Q(P') - Q(P))$. $\qquad\blacksquare$

Since $K > 0$, the test $\Delta Q > 0$ ([`algorithm.py:89`](algorithm.py#L89)) is
**equivalent** to $Q(P') > Q(P)$. The algorithm accepts a swap **iff** it
strictly increases the true mean-quality objective.

---

## 5. Monotonicity, termination, and local optimality

**Theorem (restated and proved).**

1. *(Validity)* Every partition visited is valid — Lemma 1.

2. *(Monotone improvement)* The sequence of objective values $Q(P_0), Q(P_1),
   \dots$ is non-decreasing, and strictly increases on each accepted swap. By
   Lemma 4, a swap is applied only when $Q(P') > Q(P)$; rejected swaps leave $P$
   unchanged. Hence $Q$ never decreases, and the returned partition satisfies
   $Q(P_{\text{final}}) \ge Q(P_0)$.

3. *(Termination)* The loop executes a fixed `n_iter` iterations
   ([`algorithm.py:68`](algorithm.py#L68)), each doing $O(md)$ work, then returns.
   Termination is unconditional.

4. *(Local optimality)* Call $P$ a **swap-local optimum** if no single-pair swap
   strictly increases $Q$. If the search reaches such a $P$, every subsequent
   candidate is rejected and $P$ is returned unchanged. Conversely, the only
   stationary points of the dynamics are swap-local optima. Thus, modulo the
   finite iteration budget, the output is a local optimum over the swap
   neighborhood. $\qquad\blacksquare$

---

## 6. Scope and limitations

- **No global-optimality claim.** Hill climbing over the swap neighborhood can
  halt at a local optimum that is not globally optimal; finding the global
  optimum of balanced anti-clustering is NP-hard in general. The proof guarantees
  *validity, exactness of the incremental bookkeeping, and monotone improvement*,
  not the global maximum.

- **Budget vs. convergence.** With a finite `n_iter` the run may stop before
  reaching a swap-local optimum. Convergence to a local optimum is guaranteed
  only in the limit of an unbounded budget (or once a full neighborhood sweep
  yields no accepted swap). The default `n_iter = 50n` is a heuristic, not a
  convergence certificate.

- **Objective-agnostic.** Nothing above uses the specific form of $q$; any
  `quality_fn` inherits the same guarantees. For the default quadratic
  $q(D) = aD^2 + bD + c$ with $a < 0$, $q$ is concave with a unique maximizer at
  $D^\star = -b/(2a)$ (`optimal_dispersion`, [`helpers.py:163`](helpers.py#L163)),
  so improving swaps drive each group's dispersion toward $D^\star$.