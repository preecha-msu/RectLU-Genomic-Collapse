import torch
import torch.nn as nn

class RectLU(nn.Module):
    def __init__(self, L=0.0, W=2.0):
        super().__init__()
        self.L = float(L)
        self.W = float(W)

    def forward(self, x):
        L, W = self.L, self.W
        z = torch.zeros_like(x)
        q = x * (x - L) / W
        lin = x - L - 0.5 * W
        return torch.where(x < L, z, torch.where(x < L + W, q, lin))

class SmoothRectLU(nn.Module):
    def __init__(self, L=0.0, W=2.0):
        super().__init__()
        self.L = float(L)
        self.W = float(W)

    def forward(self, x):
        L, W = self.L, self.W
        z = torch.zeros_like(x)
        q = ((x - L) ** 2) / (2.0 * W)
        lin = x - L - 0.5 * W
        return torch.where(x < L, z, torch.where(x < L + W, q, lin))

class ASRectLU(nn.Module):
    def __init__(self, init_L=0.0, init_W=2.0):
        super().__init__()
        self.L = nn.Parameter(torch.tensor(float(init_L)))
        self.W = nn.Parameter(torch.tensor(float(init_W)))

    def forward(self, x):
        W_pos = torch.clamp(self.W, min=1e-3)
        L = self.L
        z = torch.zeros_like(x)
        q = ((x - L) ** 2) / (2.0 * W_pos)
        lin = x - L - 0.5 * W_pos
        return torch.where(x < L, z, torch.where(x < L + W_pos, q, lin))

class SigmaASRectLU(nn.Module):
    def __init__(self, kappa=0.5, gamma=0.0, eps=1e-5):
        super().__init__()
        self.kappa = float(kappa)
        self.gamma = float(gamma)
        self.eps = float(eps)

    def forward(self, x):
        var = torch.var(x, unbiased=False)
        sigma = torch.sqrt(var + self.eps).detach()
        W = torch.clamp(self.kappa * sigma, min=1e-3)
        L = -self.gamma * sigma
        z = torch.zeros_like(x)
        q = ((x - L) ** 2) / (2.0 * W)
        lin = x - L - 0.5 * W
        return torch.where(x < L, z, torch.where(x < L + W, q, lin))

def get_activation(name):
    mapping = {
        'ReLU': lambda: nn.ReLU(),
        'LeakyReLU': lambda: nn.LeakyReLU(0.01),
        'PReLU': lambda: nn.PReLU(init=0.25),
        'SELU': lambda: nn.SELU(),
        'RectLU': lambda: RectLU(L=0.0, W=2.0),
        'ELU': lambda: nn.ELU(alpha=1.0),
        'SmeLU': lambda: SmoothRectLU(L=-0.5, W=1.0),
        'SmoothRectLU_W2': lambda: SmoothRectLU(L=0.0, W=2.0),
        'SmoothRectLU_W05': lambda: SmoothRectLU(L=0.0, W=0.5),
        'ASRectLU': lambda: ASRectLU(init_L=0.0, init_W=2.0),
        'SigmaASRectLU': lambda: SigmaASRectLU(kappa=0.5, gamma=0.0),
        'GELU': lambda: nn.GELU(),
        'SiLU': lambda: nn.SiLU(),
        'Mish': lambda: nn.Mish(),
        'Softplus': lambda: nn.Softplus(),
    }
    return mapping[name]()
