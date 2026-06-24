import pandas as pd
df = pd.read_csv("predictive_daca_offline.csv")
out = df[["video", "shot", "d_s", "true_w", "pred_w_b", "content_b", "breach_b"]].copy()
out.columns = ["video", "shot", "d_s", "true_w*", "predicted_w*", "content_at_pred", "breach_flag"]
out.to_csv("predictive_daca_offline.csv", index=False)
