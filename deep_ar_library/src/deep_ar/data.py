from dataclasses import dataclass
from torch.utils.data import Dataset
import torch, numpy as np

@dataclass
class SeriesStats:
    mean: float
    std: float

class StandardScaler:
    def fit_transform(self, series: np.ndarray) -> tuple[np.ndarray, SeriesStats]:
        m, s = series.mean(), series.std() + 1e-8
        return (series - m) / s, SeriesStats(m, s)

    def inverse(self, series_norm: torch.Tensor, stats: SeriesStats) -> torch.Tensor:
        return series_norm * stats.std + stats.mean

class SlidingWindowDataset(Dataset):
    def __init__(self, series_list, context_length=48, prediction_length=24):
        self.ctx = context_length
        self.pred = prediction_length
        self.items = []

        for ts in series_list:
            for t in range(self.ctx, len(ts) - self.pred):
                past = ts[t - self.ctx:t]
                fut = ts[t:t + self.pred]
                self.items.append((past, fut))

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        past, fut = self.items[i]
        past = torch.tensor(past, dtype=torch.float32).unsqueeze(-1)
        fut = torch.tensor(fut, dtype=torch.float32).unsqueeze(-1)
        return past, fut
