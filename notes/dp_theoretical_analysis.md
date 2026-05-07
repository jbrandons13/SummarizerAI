# DP Theoretical Analysis: Matching and Alignment

## Section 2.1: DP recurrence formal recap

The matching process is formulated as finding a sequence of scene indices $P = (j_1, j_2, \dots, j_N)$ that maximizes the total score:

$$J(P) = \sum_{i=1}^N S(i, j_i) - \sum_{i=2}^N T(j_{i-1}, j_i)$$

where $S(i, j)$ is the similarity score between sentence $i$ and scene $j$, and $T(k, j)$ is the transition cost. This is solved using Dynamic Programming with the recurrence:

$$V(i, j) = S(i, j) + \max_k [V(i-1, k) - T(k, j)]$$

The transition cost $T(k, j)$ is defined as:
- **Forward Jump:** $jp \cdot \Delta t$ if $\Delta t > 0$
- **Backward Jump:** $jp \cdot |\Delta t| + bp$ if $\Delta t < 0$
- **Scene Reuse:** $-rb$ if $k = j$

where $\Delta t$ is the normalized time difference between scene $k$ and scene $j$. Our parameters are $jp=0.01$ (jump penalty), $rb=0.01$ (reuse bonus), and $bp=0.5$ (backward penalty).

## Section 2.2: When DP differs from Greedy (sufficient condition)

Greedy matching selects $j^*_i = \arg\max_j S(i, j)$ for each sentence independently. DP and Greedy coincide if the greedy path is also the global optimum.

**Sufficient Condition for Coincidence:**
Greedy and DP paths will coincide if for every sentence $i$, the score gap between the greedy choice and any alternative is large enough to outweigh any possible transition advantage. Specifically:

$$S(i, j^*_i) - S(i, j_{alt}) > [T(j_{prev}, j^*_i) + T(j^*_i, j_{next})] - [T(j_{prev}, j_{alt}) + T(j_{alt}, j_{next})]$$

for all $j_{alt}$.

**Proof Sketch:**
If the per-sentence score penalty of choosing $j_{alt}$ over $j^*_i$ is strictly greater than the transition cost saving gained by using $j_{alt}$ (relative to both the previous and next nodes), then any path containing $j_{alt}$ at step $i$ will have a lower total objective than the path containing $j^*_i$. Since this holds for all $i$, the greedy path is the unique global optimum.

## Section 2.3: When does DP help (heuristic argument)

We characterize three regimes of matching behavior:

### Regime A: High Score Separation (Greedy Dominant)
The score gap between the best and second-best scenes is large for every sentence. The semantic signal is so strong that transition costs are negligible. Greedy and DP produce identical results.
*Observed in:* SigLIP track on most videos, where visual embeddings are highly discriminative.

### Regime B: Ordered Ambiguity (Greedy Fortuitous)
The score gaps are small, but the per-sentence greedy picks happen to be temporally ordered. Even without a temporal prior, Greedy produces a smooth path. DP has no "errors" to correct and yields a similar result.
*Observed in:* Simple linear reviews where the narration follows the exact filming order.

### Regime C: Disordered Ambiguity (DP Active)
The score gaps are small AND per-sentence picks are temporally disordered (non-monotonic). Greedy produces "jittery" summaries with frequent backward jumps. DP can sacrifice a small amount of similarity score to enforce a forward-moving sequence.
*Observed in:* Caption track on complex videos where textual descriptions are generic and apply to multiple scenes.

## Section 2.4: Scene-attractor failure mode

The "Scene-Attractor" problem occurs when a single scene $j^*$ has high similarity scores across a long sequence of sentences.

If there exists $j^*$ such that $S(i, j^*)$ is the highest score for sentences $i \in [m, n]$, and the score gap $\delta S_i = S(i, j^*) - \max_{j \neq j^*} S(i, j)$ satisfies:

$$\delta S_i > -(rb + T(j^*, j_{alt}))$$

Equivalently, DP stays at $j^*$ even when the alternative $j_{alt}$ has a higher per-sentence score than $j^*$, as long as the alternative's advantage is less than $rb + T(j^*, j_{alt})$. Because the backward penalty $bp = 0.5$ dominates the transition cost when $j_{alt}$ lies earlier in the video, this creates a substantial moat: a backward alternative must be more than approximately $0.5 + rb$ better in score to dislodge DP from $j^*$. With typical Caption-track score gaps below this threshold, the algorithm remains locked.

Then DP will stay "locked" on $j^*$. Because $T(j^*, j^*) = -rb$ (a bonus), switching to a new scene $j_{alt}$ incurs a penalty of both the score loss and the transition cost. If the semantic scores are biased (e.g., a "generic" keyframe that matches many sentences), DP becomes a "greedy-with-reuse" algorithm rather than a "greedy-with-ordering" algorithm. This explains why `review_7` on the Caption track shows extreme looping.

## Section 2.5: Implications for DP design

The current DP implementation relies purely on temporal metadata to enforce coherence. This analysis suggests several improvements:

1. **Explicit Diversity Penalty:** Introducing a penalty $\lambda \cdot \mathbb{1}[k=j]$ instead of a bonus would force the algorithm to seek new scenes when the similarity gap is small.
2. **Visual-Smoothness Transitions:** Transition costs should be based on visual feature similarity between scenes $k$ and $j$ (e.g., optical flow or feature distance) rather than just timestamp differences. This would favor "cut-sequences" that look natural.
3. **Windowed Constraints:** Restricting the search space for $j_{i}$ based on $j_{i-1}$ (e.g., $j_i > j_{i-1}$) could prevent backward jumps more effectively than a soft penalty, though it risks non-feasibility if no good matches exist downstream.

These findings motivate a shift from "Temporal DP" to "Contextual DP" in future iterations.
