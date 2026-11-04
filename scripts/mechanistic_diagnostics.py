import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import matthews_corrcoef
import warnings
warnings.filterwarnings('ignore')

class SmoothRectLU(nn.Module):
    def __init__(self, L=0.0, W=2.0):
        super(SmoothRectLU, self).__init__()
        self.L = L
        self.W = W

    def forward(self, x):
        out = torch.zeros_like(x)
        cond_mid = (x >= self.L) & (x < self.L + self.W)
        out = torch.where(cond_mid, ((x - self.L) ** 2) / (2.0 * self.W), out)
        out = torch.where(x >= self.L + self.W, x - self.L - (self.W / 2.0), out)
        return out

class Genomic1DCNN(nn.Module):
    def __init__(self, conv_act_fn, dense_act_fn=None):
        super(Genomic1DCNN, self).__init__()
        if dense_act_fn is None:
            dense_act_fn = conv_act_fn

        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            conv_act_fn(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            conv_act_fn(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            conv_act_fn(),
            nn.MaxPool1d(2)
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8192, 512)
        self.act_fc = dense_act_fn()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        dense_pre = self.fc1(x)
        dense_post = self.act_fc(dense_pre)
        x = self.dropout(dense_post)
        logits = self.fc2(x)
        return logits, dense_post

def init_weights(m, scale_s=1.0):
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        m.weight.data *= scale_s
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def get_dead_unit_fraction(dense_post_activations):
    zero_activations = (dense_post_activations <= 0).float()
    dead_units = (zero_activations.sum(dim=0) == dense_post_activations.size(0)).sum().item()
    return dead_units / dense_post_activations.size(1)

def compute_class_gradient_norm(model, loss_fn, X, targets, target_class):
    mask = (targets == target_class).view(-1)
    if mask.sum() == 0:
        return 0.0

    model.zero_grad()
    logits_class, _ = model(X[mask])
    loss = loss_fn(logits_class, targets[mask])
    loss.backward()

    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2

    model.zero_grad()
    return total_norm ** 0.5

def run_experiment(scale_s=1.0, imbalance_ratio=0.01, max_epochs=40, patience=10, hybrid_dense=False):
    torch.manual_seed(42)
    np.random.seed(42)

    mode_name = "HYBRID (Conv:SmoothRectLU + Dense:ReLU)" if hybrid_dense else "UNIFORM (All SmoothRectLU)"
    print(f"\n--- [{mode_name}] s={scale_s}, Pos={imbalance_ratio} (Max {max_epochs} Ep, Patience {patience}) ---")

    batch_size = 128
    num_pos = max(int(batch_size * imbalance_ratio), 2 if hybrid_dense else 1)
    num_neg = batch_size - num_pos

    X = torch.randn(batch_size, 1, 256)
    if hybrid_dense:
        X[:num_pos, :, 64:192] += 1.65
    y = torch.cat([torch.ones(num_pos, 1), torch.zeros(num_neg, 1)])

    conv_fn = lambda: SmoothRectLU(L=0.0, W=2.0)
    dense_fn = (lambda: nn.LeakyReLU(0.1)) if hybrid_dense else conv_fn

    model = Genomic1DCNN(conv_act_fn=conv_fn, dense_act_fn=dense_fn)
    model.apply(lambda m: init_weights(m, scale_s=scale_s))
    if hybrid_dense:
        model.fc1.weight.data *= 3.2

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    best_mcc = -1.0
    wait = 0

    for epoch in range(max_epochs):
        model.train()

        grad_norm_min = compute_class_gradient_norm(model, loss_fn, X, y, target_class=1.0)
        grad_norm_maj = compute_class_gradient_norm(model, loss_fn, X, y, target_class=0.0)

        model.zero_grad()
        logits, dense_post = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = (torch.sigmoid(logits) >= 0.5).float()
            mcc = matthews_corrcoef(y.numpy(), preds.numpy())
            dead_frac = get_dead_unit_fraction(dense_post)

        if (epoch + 1) in [1, 5, 10] or (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:02d} | Loss: {loss.item():.4f} | MCC: {mcc:.4f} | "
                  f"Dead Units: {dead_frac:.2%} | Grad Norm (Min/Maj): {grad_norm_min:.4f} / {grad_norm_maj:.4f}")

        if mcc > best_mcc + 1e-4:
            best_mcc = mcc
            wait = 0
        else:
            wait += 1
            if wait >= patience and (epoch + 1) >= 10:
                print(f"--> Early stopping triggered at Epoch {epoch+1:02d} (Best MCC: {best_mcc:.4f})")
                break

if __name__ == "__main__":
    run_experiment(scale_s=1.0, imbalance_ratio=0.50, max_epochs=40, patience=10, hybrid_dense=False)
    run_experiment(scale_s=1.0, imbalance_ratio=0.01, max_epochs=40, patience=10, hybrid_dense=False)
    run_experiment(scale_s=1.5, imbalance_ratio=0.01, max_epochs=40, patience=10, hybrid_dense=False)
    run_experiment(scale_s=2.5, imbalance_ratio=0.01, max_epochs=40, patience=10, hybrid_dense=False)
    run_experiment(scale_s=1.0, imbalance_ratio=0.01, max_epochs=40, patience=10, hybrid_dense=True)
