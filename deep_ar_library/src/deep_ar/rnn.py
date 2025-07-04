import torch.nn as nn, torch

class RNNEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 40, layers: int = 2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, layers, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, h0=None):
        return self.lstm(x, h0 or self._init_hidden(x.size(0)))[0]

    def _init_hidden(self, B):
        H, L = self.lstm.hidden_size, self.lstm.num_layers
        device = next(self.parameters()).device
        return (torch.zeros(L, B, H, device=device), torch.zeros(L, B, H, device=device))
