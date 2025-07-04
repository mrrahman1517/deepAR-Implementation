#!/usr/bin/env python
"""
Validate a trained DeepAR model on the same CSV it was trained on.
Metrics:
    • sMAPE  – symmetric MAPE (official M4)
    • MASE   – mean absolute scaled error   (official M4)
Example:
    python -m deep_ar_client.validate \
        --checkpoint model.pt \
        --data data/Hourly-train.csv
"""
import argparse, torch, numpy as np
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

from deep_ar.model import DeepAR
from deep_ar_client.m4 import load_m4_csv
from deep_ar.data import StandardScaler

# --- forecasting helper -------------------------------------------------
@torch.no_grad()
def forecast(model, ctx, horizon, n_samples=100):
    model.eval()
    samples = []
    for _ in range(n_samples):
        past = ctx.clone()
        preds = []
        for _ in range(horizon):
            params = model(past)
            mu, sig_r = params[:, -1].chunk(2, -1)
            sig = F.softplus(sig_r)
            y = torch.normal(mu, sig)
            preds.append(y)
            past = torch.cat([past, y.unsqueeze(1)], 1)
        samples.append(torch.cat(preds, 1))
    return torch.stack(samples)          # (S,1,H)

# --- M4 metrics ----------------------------------------------------------
def smape(y_true, y_pred):
    return 200 * np.mean(np.abs(y_pred - y_true) /
                         (np.abs(y_true) + np.abs(y_pred) + 1e-8))

def mase(y_true, y_pred, insample, seasonality=1):
    """
    seasonality = 1 for hourly/daily; 12 for monthly; 4 for quarterly
    """
    d = np.abs(np.diff(insample, n=seasonality)).mean()
    return np.mean(np.abs(y_pred - y_true)) / (d + 1e-8)

# ------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--samples", type=int, default=100)
    p.add_argument("--plot", action="store_true",
               help="Show matplotlib plots for the first 5 series")
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = DeepAR(input_dim=1)
    model.load_state_dict(ckpt["model_state"])

    raw_series = load_m4_csv(args.data)
    scaler = StandardScaler()
    stats = ckpt["scaler_stats"]
    ctx_len = ckpt["context_length"]
    horizon = args.horizon

    all_truth = []
    all_pred = []

    smape_list, mase_list = [], []
    print(f"Validating on {len(raw_series)} series …")
    for i in tqdm(range(len(raw_series))):
        ts = raw_series[i]
        if len(ts) < ctx_len + horizon:
            continue  # skip very short series

        insample  = ts[: -horizon]        # everything before forecast window
        truth     = ts[-horizon:]         # ground-truth future
        stat      = stats[i]
        ts_scaled = (ts - stat.mean) / stat.std

        ctx = torch.tensor(ts_scaled[-(ctx_len + horizon):-horizon],
                           dtype=torch.float32).view(1, -1, 1)

        samples   = forecast(model, ctx, horizon, args.samples)
        mean_pred = samples.mean(0).squeeze().numpy() * stat.std + stat.mean

        smape_list.append(smape(truth, mean_pred))
        mase_list .append(mase (truth, mean_pred, insample))

        all_truth.append(truth)          # NEW
        all_pred.append(mean_pred)

    import pandas as pd
    df = pd.DataFrame({
        "series_id"  : range(len(all_pred)),
        "truth"      : all_truth,
        "prediction" : all_pred,
    })
    df.to_csv("validation_predictions.csv", index=False)
    print("Saved per-series predictions → validation_predictions.csv")
    # ---------

    if args.plot:
        import matplotlib.pyplot as plt
        for sid in range(min(5, len(all_pred))):      # first 5 series
            t = np.arange(args.horizon)
            plt.figure()
            plt.plot(t, all_truth[sid],  label="actual")
            plt.plot(t, all_pred[sid],   label="forecast")
            plt.title(f"Series {sid}")
            plt.legend()
            plt.show()

    print("\n— Validation summary —")
    print(f"sMAPE  : {np.mean(smape_list):7.3f}")
    print(f"MASE   : {np.mean(mase_list):7.3f}")
    print(f"Series : {len(smape_list)} evaluated")

if __name__ == "__main__":
    main()
