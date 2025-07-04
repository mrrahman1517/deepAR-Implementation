import torch, torch.nn as nn, torch.nn.functional as F
from abc import ABC, abstractmethod

class Likelihood(ABC, nn.Module):
    @property
    @abstractmethod
    def n_params(self): ...

    @abstractmethod
    def loss(self, params: torch.Tensor, target: torch.Tensor): ...

class GaussianLikelihood(Likelihood):
    def __init__(self, target_dim=1): super().__init__(); self._n = 2*target_dim
    @property
    def n_params(self): return self._n

    def loss(self, params, target):
        mu, sigma_raw = params.chunk(2, -1)
        sigma = F.softplus(sigma_raw)+1e-5
        ll = -0.5*(torch.log(2*torch.pi*sigma**2) + ((target-mu)**2)/sigma**2)
        return -ll.mean()
