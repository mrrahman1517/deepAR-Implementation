import torch.nn as nn, torch
from deep_ar.rnn import RNNEncoder
from deep_ar.likelihoods import GaussianLikelihood, Likelihood

class DeepAR(nn.Module):
    def __init__(self, input_dim:int, likelihood:Likelihood|None=None,
                 rnn_hidden=40, rnn_layers=2):
        super().__init__()
        self.encoder = RNNEncoder(input_dim, rnn_hidden, rnn_layers)
        self.lik = likelihood or GaussianLikelihood()
        self.proj = nn.Linear(rnn_hidden, self.lik.n_params)

    def forward(self, x): return self.proj(self.encoder(x))

    def loss(self, x, y): return self.lik.loss(self(x), y)
