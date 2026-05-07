# TASK: Fix Sign Error in DP Theoretical Analysis Section 2.4

## Context

The current `notes/dp_theoretical_analysis.md` has a mathematical error in
Section 2.4 (Scene-attractor failure mode). The condition stated for DP
"locking" onto an attractor scene is:

> $\delta S_i > -rb + T(j^*, j_{alt})$

This formula has a sign error. Walking through the DP recurrence carefully
gives a different result.

## Correct Derivation

DP recurrence (from Section 2.1):
$$V(i, j) = S(i, j) + \max_k [V(i-1, k) - T(k, j)]$$

Transition cost convention:
- $T(j^*, j^*) = -rb$ (reuse: negative cost = bonus)
- $T(j^*, j_{alt}) > 0$ (any move: positive cost)

Suppose at step $i-1$ DP was at $j^*$. Compare staying at $j^*$ vs moving to $j_{alt}$:

- **Stay**: contribution = $S(i, j^*) - T(j^*, j^*) = S(i, j^*) + rb$
- **Move**: contribution = $S(i, j_{alt}) - T(j^*, j_{alt})$

DP stays at $j^*$ iff:

$$S(i, j^*) + rb > S(i, j_{alt}) - T(j^*, j_{alt})$$

Rearrange:

$$S(i, j^*) - S(i, j_{alt}) > -rb - T(j^*, j_{alt})$$

$$\delta S_i > -(rb + T(j^*, j_{alt}))$$

This is the **correct stay condition**.

## Why the Original Formula Was Wrong

The original wrote $\delta S_i > -rb + T$, which expands to $\delta S_i > T - rb$
(approximately $T$ since $rb$ is tiny).

For backward jumps where $T \approx 0.5$, this would say DP stays only when
$j^*$ is at least 0.49 better than $j_{alt}$.

The correct condition $\delta S_i > -(rb + T)$ for the same case gives
$\delta S_i > -0.51$, meaning DP stays even when $j_{alt}$ is up to 0.51
**better than** $j^*$.

The two have opposite practical meaning. The corrected version is consistent
with the prose explanation already in Section 2.4 ("DP interprets the reuse
bonus and low-cost self-transition as the optimal path... switching to a new
scene incurs a penalty of both the score loss and the transition cost"),
which is correct.

## Required Edit

In `notes/dp_theoretical_analysis.md`, Section 2.4 ("Scene-attractor failure
mode"), replace the inequality:

> $\delta S_i > -rb + T(j^*, j_{alt})$

with:

> $\delta S_i > -(rb + T(j^*, j_{alt}))$

Also, after the corrected inequality, add 2-3 sentences of clarifying
explanation for the reader:

> Equivalently, DP stays at $j^*$ even when the alternative $j_{alt}$ has
> a higher per-sentence score than $j^*$, as long as the alternative's
> advantage is less than $rb + T(j^*, j_{alt})$. Because the backward
> penalty $bp = 0.5$ dominates the transition cost when $j_{alt}$ lies
> earlier in the video, this creates a substantial moat: a backward
> alternative must be more than approximately $0.5 + rb$ better in score
> to dislodge DP from $j^*$. With typical Caption-track score gaps below
> this threshold, the algorithm remains locked.

## Critical Rules

- Do NOT change any other section of the document.
- Do NOT change Section 2.2 (Sufficient Condition) — that derivation is correct.
- Do NOT modify the prose conclusion or the regime characterization.
- The fix is purely the inequality sign and the added clarifying paragraph.

## Output

Updated `notes/dp_theoretical_analysis.md` with the corrected inequality and
added explanation. Confirm completion with a one-line summary.
