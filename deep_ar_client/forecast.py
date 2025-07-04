#!/usr/bin/env python
"""
Enhanced forecasting script for the deep_ar_client.

Features
--------
1. **Batch inference**: forecast many series in parallel.
2. **Vectorised ancestral sampling**: speed‑up over python‑level loops.
3. **Fast / deterministic mode** (`--deterministic`): uses μ without sampling.
4. Forecast *all* series in a CSV (`--all-series`) or subset by index list.
"""

import argparse, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

from deep_ar.model import DeepAR
from deep_ar_client.m4 import load_m4_csv
from deep_ar.data import StandardScaler

def sample_forecast(model, ctx, horizon, n_samples=100, deterministic=False):
    model.eval()
    B = ctx.size(0)
    device = ctx.device

    if deterministic:
        samples = torch.empty(1, B, horizon, device=device)
        past = ctx.clone()
        for t in range(horizon):
            params = model(past)
            mu, _ = params[:, -1].chunk(2, -1)
            samples[0, :, t] = mu.squeeze(-1)
            past = torch.cat([past, mu.unsqueeze(1)], 1)
        return samples

    S = n_samples
    samples = torch.empty(S, B, horizon, device=device)
    for s in range(S):
        past = ctx.clone()
        for t in range(horizon):
            params = model(past)
            mu, sig_r = params[:, -1].chunk(2, -1)
            sig = F.softplus(sig_r)
            y = torch.normal(mu, sig)
            samples[s, :, t] = y.squeeze(-1)
            past = torch.cat([past, y.unsqueeze(1)], 1)
    return samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--indices", nargs="*", type=int,
                   help="Series indices; omit for single --index or --all-series")
    p.add_argument("--index", type=int, default=0,
                   help="Single series index (ignored if --indices or --all-series)")
    p.add_argument("--all-series", action="store_true",
                   help="Forecast every series in the CSV")
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--samples", type=int, default=100)
    p.add_argument("--deterministic", action="store_true",
                   help="Skip sampling, use mean (fast)")
    p.add_argument("--batch-size", type=int, default=256,
                   help="Batch size for parallel inference")
    p.add_argument("--save-csv", default=None,
                   help="Write forecasts to CSV file")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = DeepAR(input_dim=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    stats = ckpt["scaler_stats"]
    ctx_len = ckpt["context_length"]
    horizon = args.horizon

    series = load_m4_csv(args.data)
    if args.all_series:
        idx_list = list(range(len(series)))
    elif args.indices:
        idx_list = args.indices
    else:
        idx_list = [args.index]

    print(f"Forecasting {len(idx_list)} series "
          f"({'deterministic' if args.deterministic else str(args.samples)+' samples'})")

    scaler = StandardScaler()

    out_dict = {"series_id": [], **{f"t+{i+1}": [] for i in range(horizon)}}

    for batch_start in tqdm(range(0, len(idx_list), args.batch_size)):
        batch_idx = idx_list[batch_start: batch_start + args.batch_size]
        batch_ctx = []
        batch_stat = []

        for sid in batch_idx:
            ts = series[sid]
            st = stats[sid]
            if len(ts) < ctx_len:
                raise ValueError(f"Series {sid} shorter than context length")

            ts_scaled = (ts - st.mean) / st.std
            ctx = torch.tensor(ts_scaled[-ctx_len:], dtype=torch.float32
                               ).view(1, -1, 1)
            batch_ctx.append(ctx)
            batch_stat.append(st)

        ctx_tensor = torch.cat(batch_ctx, 0).to(device)

        samples = sample_forecast(
            model, ctx_tensor, horizon,
            n_samples=args.samples, deterministic=args.deterministic
        )

        mean_pred = samples.mean(0).cpu().numpy()

        for local_i, sid in enumerate(batch_idx):
            st = batch_stat[local_i]
            pred_denorm = mean_pred[local_i] * st.std + st.mean
            out_dict["series_id"].append(sid)
            for h in range(horizon):
                out_dict[f"t+{h+1}"].append(pred_denorm[h])

    import pandas as pd
    df = pd.DataFrame(out_dict)
    if args.save_csv:
        Path(args.save_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.save_csv, index=False)
        print("Saved forecasts →", args.save_csv)
    else:
        print(df.head())


if __name__ == "__main__":
    main()
