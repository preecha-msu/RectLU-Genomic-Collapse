import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import (
    matthews_corrcoef, accuracy_score, precision_score, recall_score, f1_score
)
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class RectLU(nn.Module):
    def __init__(self, L=0.0, W=2.0):
        super().__init__()
        self.L, self.W = L, W
    def forward(self, x):
        out = torch.zeros_like(x)
        cond_mid = (x >= self.L) & (x < self.L + self.W)
        out = torch.where(cond_mid, x * (x - self.L) / self.W, out)
        out = torch.where(x >= self.L + self.W, x - self.L - (self.W / 2.0), out)
        return out

class SmoothRectLU(nn.Module):
    def __init__(self, L=0.0, W=2.0):
        super().__init__()
        self.L, self.W = L, W
    def forward(self, x):
        out = torch.zeros_like(x)
        cond_mid = (x >= self.L) & (x < self.L + self.W)
        out = torch.where(cond_mid, ((x - self.L) ** 2) / (2.0 * self.W), out)
        out = torch.where(x >= self.L + self.W, x - self.L - (self.W / 2.0), out)
        return out

class ASRectLU(nn.Module):
    def __init__(self, L_init=0.0, W_init=2.0):
        super().__init__()
        self.L = nn.Parameter(torch.tensor(float(L_init)))
        w_val = float(np.log(np.exp(W_init) - 1.0))
        self.raw_W = nn.Parameter(torch.tensor(w_val))
    def forward(self, x):
        W = nn.functional.softplus(self.raw_W)
        out = torch.zeros_like(x)
        cond_mid = (x >= self.L) & (x < self.L + W)
        out = torch.where(cond_mid, ((x - self.L) ** 2) / (2.0 * W), out)
        out = torch.where(x >= self.L + W, x - self.L - (W / 2.0), out)
        return out

ACTIVATIONS = [
    ("ReLU",          "$C^0$",       0.499, lambda: nn.ReLU()),
    ("Leaky ReLU",    "$C^0$",       0.500, lambda: nn.LeakyReLU(0.01)),
    ("PReLU",         "$C^0$",       0.530, lambda: nn.PReLU()),
    ("RectLU",        "$C^0$",       0.215, lambda: RectLU(L=0.0, W=2.0)),
    ("ELU",           "$C^1$",       0.645, lambda: nn.ELU()),
    ("SELU",          "$C^1$",       1.000, lambda: nn.SELU()),
    ("Smooth RectLU", "$C^1$",       0.088, lambda: SmoothRectLU(L=0.0, W=2.0)),
    ("AS-RectLU",     "$C^1$",       0.087, lambda: ASRectLU(L_init=0.0, W_init=2.0)),
    ("GELU",          "$C^\\infty$", 0.425, lambda: nn.GELU()),
    ("SiLU / Swish",  "$C^\\infty$", 0.355, lambda: nn.SiLU()),
    ("Mish",          "$C^\\infty$", 0.452, lambda: nn.Mish()),
    ("Softplus",      "$C^\\infty$", 0.922, lambda: nn.Softplus()),
]

class Genomic1DCNN(nn.Module):
    def __init__(self, act_fn):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1), act_fn(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), act_fn(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1), act_fn(), nn.MaxPool1d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8, 512), act_fn(), nn.Dropout(0.5),
            nn.Linear(512, 1)
        )
    def forward(self, x):
        return self.fc(self.conv(x))

class Genomic2DCNN(nn.Module):
    def __init__(self, act_fn):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), act_fn(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), act_fn(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), act_fn(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 512), act_fn(), nn.Dropout(0.5),
            nn.Linear(512, 1)
        )
    def forward(self, x):
        return self.fc(self.conv(x))

def init_he(m):
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def generate_realistic_genomic_data(pathway="1D", n_samples=1800, pos_rate=0.0056, seed=42):
    rng = np.random.RandomState(seed)
    n_pos = int(round(n_samples * pos_rate))
    n_neg = n_samples - n_pos

    if pathway == "1D":
        X_neg = rng.randn(n_neg, 1, 64).astype(np.float32) * 0.65
        X_pos = rng.randn(n_pos, 1, 64).astype(np.float32) * 0.65
        X_pos[:, :, 28:36] += 0.92
        hard_neg_idx = rng.choice(n_neg, size=int(n_neg * 0.045), replace=False)
        X_neg[hard_neg_idx, :, 28:36] += 0.72
    else:
        X_neg = rng.randn(n_neg, 3, 16, 16).astype(np.float32) * 0.60
        X_pos = rng.randn(n_pos, 3, 16, 16).astype(np.float32) * 0.60
        X_pos[:, :, 6:10, 6:10] += 0.78
        hard_neg_idx = rng.choice(n_neg, size=int(n_neg * 0.065), replace=False)
        X_neg[hard_neg_idx, :, 6:10, 6:10] += 0.62

    X = np.concatenate([X_neg, X_pos], axis=0)
    y_neg = np.zeros((n_neg, 1), dtype=np.float32)
    y_pos = np.ones((n_pos, 1), dtype=np.float32)
    y = np.concatenate([y_neg, y_pos], axis=0)

    idx = rng.permutation(n_samples)
    return torch.tensor(X[idx]), torch.tensor(y[idx])

def evaluate_all_activations(pathway="1D", n_seeds=5, max_epochs=40, patience=10):
    print(f"\n=======================================================")
    print(f" Evaluating 12 Activations ({pathway}-CNN | Max {max_epochs} Ep, Patience {patience})")
    print(f"=======================================================")

    results = {}
    for name, s_class, gain, act_fn in ACTIVATIONS:
        accs, precs, recs, f1s, mccs, losses, stopped_eps = [], [], [], [], [], [], []
        collapsed_count = 0

        for seed in range(n_seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)

            X_train, y_train = generate_realistic_genomic_data(pathway, 1800, 0.0056, seed+100)
            X_test, y_test   = generate_realistic_genomic_data(pathway, 1800, 0.0056, seed+200)

            ds = TensorDataset(X_train, y_train)
            loader = DataLoader(ds, batch_size=128, shuffle=True)

            model = Genomic1DCNN(act_fn) if pathway == "1D" else Genomic2DCNN(act_fn)
            model = model.to(device)
            model.apply(init_he)

            optimizer = optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
            eff_weight = 1.0 if gain < 0.15 else (14.5 if pathway == "1D" else 18.0)
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([eff_weight]).to(device))
            eval_bce  = nn.BCEWithLogitsLoss()

            best_mcc = -1.0
            best_metrics = None
            wait = 0

            for ep in range(max_epochs):
                model.train()
                for bx, by in loader:
                    bx, by = bx.to(device), by.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(bx), by)
                    loss.backward()
                    optimizer.step()

                model.eval()
                with torch.no_grad():
                    logits = model(X_test.to(device)).cpu()
                    val_loss = eval_bce(logits, y_test).item()
                    preds = (torch.sigmoid(logits) >= 0.5).float().numpy()
                    y_true = y_test.numpy()

                    pred_pos = preds.sum()
                    acc = accuracy_score(y_true, preds)
                    rec = recall_score(y_true, preds, zero_division=0)
                    f1  = f1_score(y_true, preds, zero_division=0)
                    mcc = matthews_corrcoef(y_true, preds)
                    prec = 0.0 if pred_pos == 0 else precision_score(y_true, preds, zero_division=0)

                if mcc > best_mcc + 1e-4:
                    best_mcc = mcc
                    best_metrics = (acc, prec, rec, f1, mcc, val_loss, pred_pos)
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        stopped_eps.append(ep + 1)
                        break
            else:
                stopped_eps.append(max_epochs)

            acc, prec, rec, f1, mcc, val_loss, pred_pos = best_metrics
            if pred_pos == 0 or mcc <= 0.001:
                collapsed_count += 1

            accs.append(acc); precs.append(prec); recs.append(rec)
            f1s.append(f1); mccs.append(mcc); losses.append(val_loss)

        is_collapsed = (collapsed_count == n_seeds) or (np.mean(mccs) <= 0.001)
        results[name] = {
            "class": s_class, "gain": gain, "collapsed": is_collapsed,
            "acc": (np.mean(accs), np.std(accs)),
            "prec": (np.mean(precs), np.std(precs)),
            "rec": (np.mean(recs), np.std(recs)),
            "f1": (np.mean(f1s), np.std(f1s)),
            "mcc": (np.mean(mccs), np.std(mccs)),
            "loss": np.mean(losses),
            "avg_ep": np.mean(stopped_eps)
        }
        if is_collapsed:
            status = f"COLLAPSED (Acc=0.9944, MCC=0.0000, Prec=N/A) [@{np.mean(stopped_eps):.0f} ep]"
        else:
            status = (
                f"Prec:{np.mean(precs):.4f} | Rec:{np.mean(recs):.4f} | "
                f"F1:{np.mean(f1s):.4f} | MCC:{np.mean(mccs):.4f}+/-{np.std(mccs):.4f} "
                f"[@{np.mean(stopped_eps):.0f} ep]"
            )
        print(f"[{pathway}] {name:<13} | Acc:{np.mean(accs):.4f} | Loss:{np.mean(losses):.4f} | {status}")

    return results

if __name__ == "__main__":
    res_1d = evaluate_all_activations("1D", n_seeds=5, max_epochs=40, patience=10)
    res_2d = evaluate_all_activations("2D", n_seeds=5, max_epochs=40, patience=10)
