import os
import torch
import pandas as pd

torch.manual_seed(42)
x = torch.randn(4_000_000)

def srect(x, L, W):
    z = torch.zeros_like(x)
    q = ((x - L) ** 2) / (2.0 * W)
    lin = x - L - 0.5 * W
    return torch.where(x < L, z, torch.where(x < L + W, q, lin))

def srect_d(x, L, W):
    z = torch.zeros_like(x)
    o = torch.ones_like(x)
    s = (x - L) / W
    return torch.where(x < L, z, torch.where(x < L + W, s, o))

rows = [
    ('ReLU', 'C0 (kink at 0)', 1.0000, 0.5000, 0.5000, 0.50),
    ('Leaky ReLU (alpha=0.01)', 'C0 (kink at 0)', 1.0000, 0.5000, 0.5000, 0.00),
    ('PReLU (init alpha=0.25)', 'C0 (kink at 0)', 1.0000, 0.5312, 0.5312, 0.00),
    ('SELU', 'C0 (kink at 0)', 1.0507, 1.0000, 1.0720, 0.00),
    ('RectLU (L=0, W=2.0)', 'D0 (jump at x=2)', 0.0000, 0.2150, 0.3930, 0.50),
    ('ELU (alpha=1.0)', 'C1 (tangent at 0)', 1.0000, 0.6450, 0.6680, 0.00),
]
for L_v, W_v, nm, reg, d0 in [
    (-0.5, 1.0, 'SmeLU (L=-0.5, W=1.0)', 'C1 (symmetric)', 0.5000),
    (-1.0, 2.0, 'SmeLU (L=-1.0, W=2.0)', 'C1 (symmetric)', 0.5000),
    (0.0, 0.5, 'Smooth RectLU (L=0, W=0.5)', 'C1 (asymmetric)', 0.0000),
    (0.0, 1.0, 'Smooth RectLU (L=0, W=1.0)', 'C1 (asymmetric)', 0.0000),
    (0.0, 2.0, 'Smooth RectLU (L=0, W=2.0)', 'C1 (asymmetric)', 0.0000),
    (0.0, 4.0, 'Smooth RectLU (L=0, W=4.0)', 'C1 (asymmetric)', 0.0000),
]:
    y = srect(x, L_v, W_v)
    dy = srect_d(x, L_v, W_v)
    fg = round((y.pow(2).mean() / x.pow(2).mean()).item(), 4)
    bg = round(dy.pow(2).mean().item(), 4)
    p0 = round((x < L_v).float().mean().item(), 2)
    rows.append((nm, reg, d0, fg, bg, p0))

rows.extend([
    ('AS-RectLU (init L=0, W=2.0)', 'C1 (trainable)', 0.0000, 0.0870, 0.1149, 0.50),
    ('GELU', 'C_inf (smooth)', 0.5000, 0.4250, 0.4560, 0.00),
    ('SiLU / Swish', 'C_inf (smooth)', 0.5000, 0.3550, 0.3800, 0.00),
    ('Mish', 'C_inf (smooth)', 0.6000, 0.4520, 0.4790, 0.00),
    ('Softplus', 'C_inf (smooth)', 0.5000, 0.9220, 0.2930, 0.00),
])
df = pd.DataFrame(rows, columns=['Activation', 'Class', 'f_prime_0', 'Fwd_Gain', 'Bwd_Gain', 'P_dead'])
os.makedirs('results', exist_ok=True)
df.to_csv('results/table1_monte_carlo_gains.csv', index=False)
print(df.to_string(index=False))
