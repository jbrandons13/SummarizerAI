from pipeline.facet.scoring_wrap import ScoringWrap
scorer = ScoringWrap()
emb_new = scorer.embed_dino("tmp_anchor_geo_001.png")
emb_cached = scorer.embed_dino("runs/G0_A0_geology/images/A0/w0.00/shot_001.png")
c_sim = float((emb_new * emb_cached).sum().item())
print(f"Anchor equivalence DINO similarity: {c_sim}")
