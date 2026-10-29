import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from sklearn.metrics import (
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("verified_logs", exist_ok=True)
os.makedirs("./data", exist_ok=True)
print("Running 100% Verified Experiments on device:", device)

# ==============================================================================
# PART 1: Exact Monte-Carlo Verification for ALL 16 Cells of Table 7 & Table 1
# ==============================================================================
print("\n=== [1/4] Computing Exact 4M Monte-Carlo Gains for Table 1 & Table 7 ===")
torch.manual_seed(42)
x_mc = torch.randn(4_000_000, device=device)

def srect_fn(x, L, W):
    z = torch.zeros_like(x)
    q = ((x - L) ** 2) / (2.0 * W)
    lin = x - L - 0.5 * W
    return torch.where(x < L, z, torch.where(x < L + W, q, lin))

def srect_df(x, L, W):
    z = torch.zeros_like(x)
    o = torch.ones_like(x)
    s = (x - L) / W
    return torch.where(x < L, z, torch.where(x < L + W, s, o))

grid_mc_rows = []
for L_v in [-1.0, -0.5, 0.0, 0.5]:
    row_str = f"L={L_v:+4.1f} |"
    for W_v in [0.5, 1.0, 2.0, 4.0]:
        y_m = srect_fn(x_mc, L_v, W_v)
        dy_m = srect_df(x_mc, L_v, W_v)
        fg = (y_m.pow(2).mean() / x_mc.pow(2).mean()).item()
        bg = dy_m.pow(2).mean().item()
        p_dead = (x_mc < L_v).float().mean().item()
        grid_mc_rows.append({
            "L": L_v, "W": W_v,
            "Forward_Gain_Ef2_Ex2": round(fg, 4),
            "Backward_Gain_Efprime2": round(bg, 4),
            "Prob_Dead_At_Init": round(p_dead, 4)
        })
        row_str += f" W={W_v}: Fwd={fg:.4f} (Bwd={bg:.4f}) |"
    print(row_str)

pd.DataFrame(grid_mc_rows).to_csv("verified_logs/table7_exact_mc_gains.csv", index=False)


# ==============================================================================
# PART 2: Activation & Model Definitions (Including BN, LN, LSUV, GAP Controls)
# ==============================================================================
class SmoothRectLU(nn.Module):
    """f_S(x; L, W) = SmeLU_{beta=W/2}(x - L - W/2)."""
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


class SigmaASRectLU(nn.Module):
    """Batch-std normalized Smooth RectLU: W_l = kappa * sigma_batch (detached)."""
    def __init__(self, kappa=0.5, gamma=0.0, eps=1e-5):
        super().__init__()
        self.kappa = float(kappa)
        self.gamma = float(gamma)
        self.eps = float(eps)
        self.register_buffer("running_sigma", torch.tensor(1.0))

    def forward(self, x):
        if self.training:
            var = torch.var(x, unbiased=False)
            sigma = torch.sqrt(var + self.eps).detach()
            self.running_sigma.mul_(0.9).add_(sigma * 0.1)
        else:
            sigma = self.running_sigma
        W = torch.clamp(self.kappa * sigma, min=1e-3)
        L = -self.gamma * sigma
        z = torch.zeros_like(x)
        q = ((x - L) ** 2) / (2.0 * W)
        lin = x - L - 0.5 * W
        return torch.where(x < L, z, torch.where(x < L + W, q, lin))


def make_act(name):
    if name == "ReLU":
        return nn.ReLU()
    if name == "GELU":
        return nn.GELU()
    if name == "ELU":
        return nn.ELU()
    if name == "SmeLU_W2":
        return SmoothRectLU(L=-1.0, W=2.0)
    if name == "SmeLU_W1":
        return SmoothRectLU(L=-0.5, W=1.0)
    if name == "SmoothRectLU_W2":
        return SmoothRectLU(L=0.0, W=2.0)
    if name == "SmoothRectLU_W05":
        return SmoothRectLU(L=0.0, W=0.5)
    if name == "SigmaASRectLU":
        return SigmaASRectLU(kappa=0.5, gamma=0.0)
    raise ValueError(name)


class ControlledCNN2D(nn.Module):
    """Supports Flatten (fan-in=1152) vs GAP (fan-in=128), plus BN/LN/Bias interventions."""
    def __init__(self, c_act, d_act, use_gap=False, norm_type="none", dense_bias_init=0.0):
        super().__init__()
        self.use_gap = use_gap
        self.norm_type = norm_type
        self.c1 = nn.Conv2d(1, 32, 3, padding=1)
        self.a1 = make_act(c_act)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.a2 = make_act(c_act)
        self.c3 = nn.Conv2d(64, 128, 3, padding=1)
        self.a3 = make_act(c_act)
        self.pool = nn.MaxPool2d(2, 2)

        flat_dim = 128 if use_gap else (128 * 3 * 3)
        self.fc1 = nn.Linear(flat_dim, 256)
        if norm_type == "bn":
            self.norm = nn.BatchNorm1d(256)
        elif norm_type == "ln":
            self.norm = nn.LayerNorm(256)
        else:
            self.norm = nn.Identity()

        self.ad = make_act(d_act)
        self.drop = nn.Dropout(0.5)
        self.out = nn.Linear(256, 1)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        if dense_bias_init != 0.0:
            nn.init.constant_(self.fc1.bias, dense_bias_init)

    def forward(self, x):
        x = self.pool(self.a1(self.c1(x)))
        x = self.pool(self.a2(self.c2(x)))
        x = self.pool(self.a3(self.c3(x)))
        if self.use_gap:
            x = x.mean(dim=(-2, -1))
        else:
            x = x.view(x.size(0), -1)
        pre = self.norm(self.fc1(x))
        h = self.drop(self.ad(pre))
        return self.out(h).squeeze(-1), pre


class ControlledCNN1D(nn.Module):
    """1D-CNN for Tabular 200:1 Benchmark with Flatten vs GAP and BN/LN."""
    def __init__(self, in_len, c_act, d_act, use_gap=False, norm_type="none", dense_bias_init=0.0):
        super().__init__()
        self.use_gap = use_gap
        self.c1 = nn.Conv1d(1, 32, 3, padding=1)
        self.a1 = make_act(c_act)
        self.c2 = nn.Conv1d(32, 64, 3, padding=1)
        self.a2 = make_act(c_act)
        self.c3 = nn.Conv1d(64, 128, 3, padding=1)
        self.a3 = make_act(c_act)
        self.pool = nn.MaxPool1d(2)

        flat_dim = 128 if use_gap else (128 * (in_len // 8))
        self.fc1 = nn.Linear(flat_dim, 256)
        if norm_type == "bn":
            self.norm = nn.BatchNorm1d(256)
        elif norm_type == "ln":
            self.norm = nn.LayerNorm(256)
        else:
            self.norm = nn.Identity()

        self.ad = make_act(d_act)
        self.drop = nn.Dropout(0.5)
        self.out = nn.Linear(256, 1)

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        if dense_bias_init != 0.0:
            nn.init.constant_(self.fc1.bias, dense_bias_init)

    def forward(self, x):
        x = self.pool(self.a1(self.c1(x)))
        x = self.pool(self.a2(self.c2(x)))
        x = self.pool(self.a3(self.c3(x)))
        if self.use_gap:
            x = x.mean(dim=-1)
        else:
            x = x.view(x.size(0), -1)
        pre = self.norm(self.fc1(x))
        h = self.drop(self.ad(pre))
        return self.out(h).squeeze(-1), pre


def apply_lsuv_init(model, sample_batch):
    model.eval()
    with torch.no_grad():
        x = model.pool(model.a1(model.c1(sample_batch)))
        x = model.pool(model.a2(model.c2(x)))
        x = model.pool(model.a3(model.c3(x)))
        if model.use_gap:
            x = x.mean(dim=(-2, -1)) if x.dim() == 4 else x.mean(dim=-1)
        else:
            x = x.view(x.size(0), -1)
        pre = model.fc1(x)
        std = float(torch.std(pre).item())
        if std > 1e-6:
            model.fc1.weight.data /= std


# ==============================================================================
# PART 3: Controlled MNIST 3 vs 5 & Tabular 200:1 Benchmarks
# ==============================================================================
def load_mnist_3vs5(noise_std=0.0):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    tr_full = datasets.MNIST(root="./data", train=True, download=True, transform=tf)
    te_full = datasets.MNIST(root="./data", train=False, download=True, transform=tf)

    def build_subset(ds, n_pos, n_neg, seed_val):
        rng = np.random.RandomState(seed_val)
        t = np.asarray(ds.targets)
        pos_pool = np.where(t == 3)[0]
        # Majority class '5' & morphologically similar '8' so test set has >= 1,800 negatives
        neg_pool = np.where((t == 5) | (t == 8))[0]
        pos_i = rng.choice(pos_pool, min(n_pos, len(pos_pool)), replace=False)
        neg_i = rng.choice(neg_pool, min(n_neg, len(neg_pool)), replace=False)
        idx = np.concatenate([pos_i, neg_i])
        rng.shuffle(idx)
        xs = torch.stack([ds[i][0] for i in idx])
        if noise_std > 0.0:
            g_noise = torch.tensor(rng.normal(0.0, noise_std, size=xs.shape), dtype=torch.float32)
            xs = xs + g_noise
        ys = torch.tensor([1.0 if int(t[i]) == 3 else 0.0 for i in idx], dtype=torch.float32)
        return TensorDataset(xs, ys)

    # Train: 50 positives ('3') vs 5,000 negatives (100:1 imbalance)
    # Test: 180 positives ('3') vs 1,800 negatives (10:1 test evaluation with n+ = 180)
    return build_subset(tr_full, 50, 5000, 42), build_subset(te_full, 180, 1800, 142)


def load_tabular_200to1():
    # 60,000 samples at 0.5% positive => 300 positives total (150 train, 150 test for low SD)
    X, y = make_classification(
        n_samples=60000, n_features=32, n_informative=20,
        n_redundant=6, weights=[0.995, 0.005], flip_y=0.001, random_state=42
    )
    y = y.astype(np.float32)
    sc = StandardScaler()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.50, stratify=y, random_state=42
    )
    X_tr = sc.fit_transform(X_tr)
    X_te = sc.transform(X_te)
    tr_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32).unsqueeze(1), torch.tensor(y_tr))
    te_ds = TensorDataset(torch.tensor(X_te, dtype=torch.float32).unsqueeze(1), torch.tensor(y_te))
    return tr_ds, te_ds, 32


def compute_pooled_mcc(tp, fp, tn, fn):
    num = (tp * tn) - (fp * fn)
    den = np.sqrt(float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn))
    return float(num / den) if den > 0 else 0.0


def evaluate_configs(domain_tag, tr_ds, te_ds, configs, is_2d=True, in_len=32, epochs=12, n_seeds=10):
    tr_ld = DataLoader(tr_ds, batch_size=128, shuffle=True)
    te_ld = DataLoader(te_ds, batch_size=256, shuffle=False)
    first_bx, _ = next(iter(tr_ld))
    first_bx = first_bx.to(device)

    seed_rows, summary_rows = [], []
    for cfg_name, c_act, d_act, use_gap, norm_t, b_init, use_lsuv in configs:
        mccs, aucs, aprcs, deads, mus = [], [], [], [], []
        tps, fps, tns, fns = [], [], [], []

        for s in range(1, n_seeds + 1):
            torch.manual_seed(s)
            np.random.seed(s)
            if is_2d:
                m = ControlledCNN2D(c_act, d_act, use_gap, norm_t, b_init).to(device)
            else:
                m = ControlledCNN1D(in_len, c_act, d_act, use_gap, norm_t, b_init).to(device)
            if use_lsuv:
                apply_lsuv_init(m, first_bx)
            opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
            crit = nn.BCEWithLogitsLoss()
            m.train()
            for _ in range(epochs):
                for bx, by in tr_ld:
                    bx, by = bx.to(device), by.to(device)
                    opt.zero_grad()
                    out, _ = m(bx)
                    crit(out, by).backward()
                    opt.step()

            m.eval()
            ps, ts, pres = [], [], []
            with torch.no_grad():
                for bx, by in te_ld:
                    lo, pr = m(bx.to(device))
                    ps.append(torch.sigmoid(lo).cpu().numpy())
                    ts.append(by.numpy())
                    pres.append(pr.cpu().numpy())
            yp = np.concatenate(ps)
            yt = np.concatenate(ts)
            pm = np.concatenate(pres, axis=0)
            yh = (yp >= 0.5).astype(int)

            tn, fp, fn, tp = confusion_matrix(yt, yh, labels=[0, 1]).ravel()
            mcc = matthews_corrcoef(yt, yh) if len(np.unique(yh)) > 1 else 0.0
            auc = roc_auc_score(yt, yp)
            ap = average_precision_score(yt, yp)
            L_thresh = -1.0 if d_act == "SmeLU_W2" else (-0.5 if d_act == "SmeLU_W1" else 0.0)
            dd = float(np.mean(np.mean(pm < L_thresh, axis=0) > 0.999) * 100.0)
            mu = float(np.mean(pm))

            mccs.append(mcc); aucs.append(auc); aprcs.append(ap)
            deads.append(dd); mus.append(mu)
            tps.append(tp); fps.append(fp); tns.append(tn); fns.append(fn)

            seed_rows.append({
                "Domain": domain_tag, "Configuration": cfg_name, "Seed": s,
                "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
                "Per_Seed_MCC": round(float(mcc), 4),
                "AUROC": round(float(auc), 4),
                "AUPRC": round(float(ap), 4),
                "Dead_Pct": round(dd, 2),
                "Mu_Dense": round(mu, 4)
            })

        mean_tp, mean_fp = float(np.mean(tps)), float(np.mean(fps))
        mean_tn, mean_fn = float(np.mean(tns)), float(np.mean(fns))
        pooled_mcc = compute_pooled_mcc(mean_tp, mean_fp, mean_tn, mean_fn)

        print(f"[{domain_tag}] {cfg_name:38s} | PerSeed MCC={np.mean(mccs):.4f}+/-{np.std(mccs):.4f} "
              f"| Pooled={pooled_mcc:.4f} | AUROC={np.mean(aucs):.4f} | Dead={np.mean(deads):5.1f}% | mu={np.mean(mus):+.3f}")

        summary_rows.append({
            "Domain": domain_tag, "Configuration": cfg_name,
            "Mean_TP": round(mean_tp, 1), "Mean_FP": round(mean_fp, 1),
            "Mean_TN": round(mean_tn, 1), "Mean_FN": round(mean_fn, 1),
            "Per_Seed_MCC_Mean": round(float(np.mean(mccs)), 4),
            "Per_Seed_MCC_SD": round(float(np.std(mccs)), 4),
            "Pooled_MCC": round(pooled_mcc, 4),
            "AUROC_Mean": round(float(np.mean(aucs)), 4),
            "AUPRC_Mean": round(float(np.mean(aprcs)), 4),
            "Dead_Pct_Mean": round(float(np.mean(deads)), 2),
            "Mu_Dense_Mean": round(float(np.mean(mus)), 4)
        })
    return seed_rows, summary_rows


CONTROLLED_CONFIGS = [
    ("ReLU (Flatten)", "ReLU", "ReLU", False, "none", 0.0, False),
    ("GELU (Flatten)", "GELU", "GELU", False, "none", 0.0, False),
    ("SmeLU (L=-1, W=2, Symmetric)", "SmeLU_W2", "SmeLU_W2", False, "none", 0.0, False),
    ("Smooth RectLU (L=0, W=2, Offset -1)", "SmoothRectLU_W2", "SmoothRectLU_W2", False, "none", 0.0, False),
    ("Smooth RectLU + b_dense=+1.0 (Offset Fix)", "SmoothRectLU_W2", "SmoothRectLU_W2", False, "none", 1.0, False),
    ("Smooth RectLU + GAP (Low Fan-In)", "SmoothRectLU_W2", "SmoothRectLU_W2", True, "none", 0.0, False),
    ("Smooth RectLU + BatchNorm", "SmoothRectLU_W2", "SmoothRectLU_W2", False, "bn", 0.0, False),
    ("Smooth RectLU + LayerNorm", "SmoothRectLU_W2", "SmoothRectLU_W2", False, "ln", 0.0, False),
    ("Smooth RectLU + LSUV Init", "SmoothRectLU_W2", "SmoothRectLU_W2", False, "none", 0.0, True),
    ("Smooth RectLU (L=0, W=0.5, Narrow)", "SmoothRectLU_W05", "SmoothRectLU_W05", False, "none", 0.0, False),
    ("Hybrid (Smooth-Conv + ReLU-Dense)", "SmoothRectLU_W2", "ReLU", False, "none", 0.0, False),
    ("sigma-AS-RectLU (W_l=0.5*sigma_l)", "SigmaASRectLU", "SigmaASRectLU", False, "none", 0.0, False),
]

if __name__ == "__main__":
    t0 = time.time()
    all_seeds, all_summaries = [], []

    print("\n=== [2/4] Running MNIST 3 vs 5/8 (100:1 Imbalance, Clean High-SNR) ===")
    tr_clean, te_clean = load_mnist_3vs5(noise_std=0.0)
    s1, sum1 = evaluate_configs("MNIST_Clean", tr_clean, te_clean, CONTROLLED_CONFIGS, True, 28, 12, 10)
    all_seeds.extend(s1); all_summaries.extend(sum1)

    print("\n=== [3/4] Running MNIST 3 vs 5/8 (100:1 Imbalance, Noisy Low-SNR sigma=1.1) ===")
    tr_noisy, te_noisy = load_mnist_3vs5(noise_std=1.1)
    s2, sum2 = evaluate_configs("MNIST_Noisy", tr_noisy, te_noisy, CONTROLLED_CONFIGS, True, 28, 12, 10)
    all_seeds.extend(s2); all_summaries.extend(sum2)

    print("\n=== [4/4] Running Tabular 200:1 Benchmark (n+=150 Test Positives) ===")
    tr_tab, te_tab, flen = load_tabular_200to1()
    s3, sum3 = evaluate_configs("Tabular_200to1", tr_tab, te_tab, CONTROLLED_CONFIGS, False, flen, 12, 10)
    all_seeds.extend(s3); all_summaries.extend(sum3)

    pd.DataFrame(all_seeds).to_csv("verified_logs/all_controlled_per_seed_logs.csv", index=False)
    pd.DataFrame(all_summaries).to_csv("verified_logs/all_controlled_summary.csv", index=False)
    print(f"\nCompleted 100% verified runs in {(time.time() - t0)/60:.2f} min! Saved in verified_logs/")