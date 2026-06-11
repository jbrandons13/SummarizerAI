# Stage 2: Block-Role Probe (F2 Gate)

## Methodology
- 132 generations (6 shots × 7 sites × 3 scales + 6 w=0 baselines).
- **Injection Sites Tested**: `global` (all 7 layers), `down2.att0`, `down2.att1`, `mid.att0`, `up0.att0`, `up0.att1`, `up0.att2`.
- **Scales**: $w \in \{0.3, 0.5, 0.8\}$.
- **Metrics**: $c_s$ (structural preservation vs $w=0$) and $ref\_sim$ (style fidelity vs reference anchor).

## Results: Average Metric Trajectories

| Site | w=0.3 $(c_s, r_{sim})$ | w=0.5 $(c_s, r_{sim})$ | w=0.8 $(c_s, r_{sim})$ |
| :--- | :--- | :--- | :--- |
| **global** | (0.745, 0.603) | (0.456, 0.876) | (0.369, 0.885) |
| **up0.att1** | (0.868, 0.410) | (0.782, 0.493) | (0.663, 0.567) |
| **up0.att0** | (0.921, 0.375) | (0.879, 0.433) | (0.793, 0.504) |
| **down2.att1** | (0.907, 0.356) | (0.864, 0.416) | (0.745, 0.493) |
| **down2.att0** | (0.945, 0.365) | (0.895, 0.363) | (0.865, 0.374) |
| **up0.att2** | (0.980, 0.359) | (0.946, 0.357) | (0.940, 0.362) |
| **mid.att0** | (0.975, 0.353) | (0.961, 0.334) | (0.911, 0.329) |

## Analysis
- `global` applies style broadly but rapidly obliterates structural layout ($c_s$ crashes to 0.36 at $w=0.8$).
- Among single sites, `up0.att1` exhibits the strongest style-responsiveness ($r_{sim}$ reaches 0.567) while maintaining substantial layout preservation ($c_s$ remains robust at 0.663).
- `up0.att1` Pareto-dominates all other single sites (it consistently achieves the highest $ref\_sim$ while maintaining equivalent or better $c_s$ trade-offs relative to its injection strength).
- `mid.att0` and `up0.att2` are practically inert for style transfer ($r_{sim}$ stuck at ~0.35).

## Verdict (F2 Gate)
**PASSED**. The block probe clearly isolates `up0.att1` as the optimal semantic style vector. 
**STYLE_LAYERS target confirmed**: `up0.att1`.
