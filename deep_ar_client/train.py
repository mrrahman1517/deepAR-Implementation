#!/usr/bin/env python
"""Train DeepAR on an M4 CSV.

Example:
    python train.py --data Hourly-train.csv --epochs 10 --checkpoint model.pt
"""

import argparse, torch, numpy as np
from torch.utils.data import DataLoader
from pathlib import Path

from deep_ar.data import StandardScaler, SlidingWindowDataset
from deep_ar.model import DeepAR
from deep_ar.trainer import train_epoch
from deep_ar_client.m4 import load_m4_csv

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to *-train.csv from M4")
    p.add_argument("--context-length", type=int, default=48)
    p.add_argument("--prediction-length", type=int, default=24)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=40)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--checkpoint", default="deep_ar_model.pt")
    args = p.parse_args()

    print("Loading dataset …")
    raw_series = load_m4_csv(args.data)
    scaler = StandardScaler()
    scaled, stats = [], []
    for ts in raw_series:
        ts_s, st = scaler.fit_transform(ts)
        scaled.append(ts_s)
        stats.append(st)

    dataset = SlidingWindowDataset(scaled, args.context_length, args.prediction_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DeepAR(input_dim=1, rnn_hidden=args.hidden, rnn_layers=args.layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"Training on {len(dataset)} windows ({len(raw_series)} series)…")
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, loader, opt, device)
        print(f"[{epoch:03d}/{args.epochs}] nll={loss:.4f}")

    torch.save({
        "model_state": model.state_dict(),
        "scaler_stats": stats,
        "context_length": args.context_length
    }, args.checkpoint)
    print(f"Saved checkpoint → {args.checkpoint}")

if __name__ == "__main__":
    main()
