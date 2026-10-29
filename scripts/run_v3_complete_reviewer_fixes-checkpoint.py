import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from scipy.stats import wilcoxon
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
os.makedirs("v3_verified_logs", exist_ok=True)
os.makedirs("./data", exist_ok=True)
print("Running V3 Complete Reviewer-Fix Suite on:", device)


# ==============================================================================
# 1. Activation Functions (Including Symmetric & Asymmetric Scale-Adaptive Units)
# ==============================================================================
class RectLU_D0(nn.Module):
    def __init__(self, L=0.0, W=2.0):
        super().__init__()
        self.L, self.W = float(L), float(W)

    def forward(self, x):
        L, W = self.L, self.W
        return torch.where(x < L, torch.zeros_like(x),
               torch.where(x < L + W, x * (x - L) / W, x - L - 0.5 * W))


class SmoothRectLU_C1(nn.Module):
    """f_S(x; L, W) = SmeLU_{beta=W/2}(x - L - W/2)."""
    def __init__(self, L=0.0, W=2.0):
        super().__init__()
        self.L, self.W = float(L), float(W)

    def forward(self, x):
        L, W = self.L, self.W
        return torch.where(x < L, torch.zeros_like(x),
               torch.where(x < L + W, ((x - L) ** 2) / (2.0 * W), x - L - 0.5 * W))


class ASRectLU_Trainable(nn.Module):
    def __init__(self, init_L=0.0, init_W=2.0):
        super().__init__()
        self.L = nn.Parameter(torch.tensor(float(init_L)))
        self.W = nn.Parameter(torch.tensor(float(init_W)))

    def forward(self, x):
        W_p = torch.clamp(self.W, min=1e-3)
        L = self.L
        return torch.where(x < L, torch.zeros_like(x),
               torch.where(x < L + W_p, ((x - L) ** 2) / (2.0 * W_p), x - L - 0.5 * W_p))


class SigmaASRectLU(nn.Module):
    """Proposed Scale-Adaptive Piecewise-Quadratic Rectifier (sigma-AS-RectLU).
    Computes per-sample layer RMS scale sigma_hat(x) across feature dimensions (batch-independent),
    and sets W_l = kappa * sigma_hat, L_l = -gamma * sigma_hat.
    When gamma = kappa / 2 (default kappa=0.8, gamma=0.4), it enforces BOTH:
      (1) Zero bias offset (L_l = -W_l / 2, so f'(0) = 0.5), and
      (2) Exact scale invariance (sigma_l / W_l = 1 / kappa) across all input amplitudes and depths!
    """
    def __init__(self, kappa=0.8, gamma=0.4, eps=1e-5):
        super().__init__()
        self.kappa = float(kappa)
        self.gamma = float(gamma)
        self.eps = float(eps)

    def forward(self, x):
        # Per-sample RMS standard deviation across channel/spatial dims (immune to batch imbalance)
        dims = tuple(range(1, x.dim()))
        var = torch.mean(x.pow(2), dim=dims, keepdim=True)
        sigma = torch.sqrt(var + self.eps).detach()
        W = torch.clamp(self.kappa * sigma, min=1e-4)
        L = -self.gamma * sigma
        return torch.where(x < L, torch.zeros_like(x),
               torch.where(x < L + W, ((x - L) ** 2) / (2.0 * W), x - L - 0.5 * W))


def build_act(name, L_val=0.0, W_val=2.0):
    if name == "ReLU": return nn.ReLU()
    if name == "LeakyReLU": return nn.LeakyReLU(0.01)
    if name == "PReLU": return nn.PReLU(init=0.25)
    if name == "SELU": return nn.SELU()
    if name == "RectLU": return RectLU_D0(L=0.0, W=2.0)
    if name == "ELU": return nn.ELU(1.0)
    if name == "SmeLU_W2": return SmoothRectLU_C1(L=-1.0, W=2.0)
    if name == "SmeLU_W1": return SmoothRectLU_C1(L=-0.5, W=1.0)
    if name == "SmoothRectLU": return SmoothRectLU_C1(L=L_val, W=W_val)
    if name == "ASRectLU": return ASRectLU_Trainable(0.0, 2.0)
    if name == "SigmaASRectLU_Asym": return SigmaASRectLU(kappa=0.5, gamma=0.0)
    if name == "SigmaASRectLU": return SigmaASRectLU(kappa=0.8, gamma=0.4)
    if name == "GELU": return nn.GELU()
    if name == "SiLU": return nn.SiLU()
    if name == "Mish": return nn.Mish()
    if name == "Softplus": return nn.Softplus()
    raise ValueError(name)


# ==============================================================================
# 2. Controlled Synthetic Sparse-Motif Datasets (1D & 2D) with Explicit Val/Test
#    and Controllable Input Scale sigma_x in {0.25, 1.00}
# ==============================================================================
def make_sparse_motif_1d(train_pos_ratio=0.01, sigma_x=0.25, fixed_N=4000):
    """Synthetic 1D Sparse Spectral Motif Benchmark (256 bins).
    Explicitly separates Train (N=4000), Validation (N=4000, 100:1), and Test (N=10000, 100:1).
    """
    scale_factor = sigma_x / 0.25

    def gen_split(N, n_pos, seed_v):
        rng = np.random.RandomState(seed_v)
        X = rng.normal(0.0, 0.25 * scale_factor, size=(N, 1, 256)).astype(np.float32)
        y = np.zeros(N, dtype=np.float32)
        y[:n_pos] = 1.0
        for i in range(n_pos):
            X[i, 0, 118:128] += rng.normal(0.28 * scale_factor, 0.09 * scale_factor, size=10)
        n_mimic = int(0.015 * (N - n_pos))
        for j in range(n_pos, n_pos + n_mimic):
            X[j, 0, 118:128] += rng.normal(0.21 * scale_factor, 0.09 * scale_factor, size=10)
        perm = rng.permutation(N)
        return TensorDataset(torch.tensor(X[perm]), torch.tensor(y[perm]))

    n_pos_tr = max(25, int(round(fixed_N * train_pos_ratio)))
    tr_ds = gen_split(fixed_N, n_pos_tr, 101)
    val_ds = gen_split(4000, 80, 202)     # Validation set for threshold tuning tau*
    te_ds = gen_split(10000, 200, 303)    # Out-of-sample Test set (n+ = 200, n- = 9800)
    return tr_ds, val_ds, te_ds


def make_sparse_motif_2d(sigma_x=0.22):
    """Synthetic 2D Spatial Motif Benchmark (3x32x32).
     Allows direct test of sigma_x = 0.22 (low amplitude) vs sigma_x = 1.00 (unit variance).
    """
    scale_factor = sigma_x / 0.22

    def gen_split(N, n_pos, seed_v):
        rng = np.random.RandomState(seed_v)
        X = rng.normal(0.0, 0.22 * scale_factor, size=(N, 3, 32, 32)).astype(np.float32)
        y = np.zeros(N, dtype=np.float32)
        y[:n_pos] = 1.0
        for i in range(n_pos):
            X[i, 0:2, 13:19, 13:19] += rng.normal(0.20 * scale_factor, 0.07 * scale_factor, size=(2, 6, 6))
        n_mimic = int(0.015 * (N - n_pos))
        for j in range(n_pos, n_pos + n_mimic):
            X[j, 0:2, 13:19, 13:19] += rng.normal(0.15 * scale_factor, 0.07 * scale_factor, size=(2, 6, 6))
        perm = rng.permutation(N)
        return TensorDataset(torch.tensor(X[perm]), torch.tensor(y[perm]))

    tr_ds = gen_split(4000, 40, 404)      # 100:1 train imbalance
    val_ds = gen_split(4000, 80, 505)     # Validation split
    te_ds = gen_split(10000, 200, 606)    # Test split (n+ = 200, n- = 9800)
    return tr_ds, val_ds, te_ds


# ==============================================================================
# 3. Models with Layer-Wise Variance & Gradient Tracking + Explicit BN/LN Placement
# ==============================================================================
class Net1D(nn.Module):
    def __init__(self, c_act, d_act, L_v=0.0, W_v=2.0, norm_mode="none", b_dense=0.0):
        super().__init__()
        self.norm_mode = norm_mode
        self.c1 = nn.Conv1d(1, 32, 3, padding=1)
        self.n1 = nn.BatchNorm1d(32) if norm_mode == "bn_all" else nn.Identity()
        self.a1 = build_act(c_act, L_v, W_v)

        self.c2 = nn.Conv1d(32, 64, 3, padding=1)
        self.n2 = nn.BatchNorm1d(64) if norm_mode == "bn_all" else nn.Identity()
        self.a2 = build_act(c_act, L_v, W_v)

        self.c3 = nn.Conv1d(64, 128, 3, padding=1)
        self.n3 = nn.BatchNorm1d(128) if norm_mode == "bn_all" else nn.Identity()
        self.a3 = build_act(c_act, L_v, W_v)

        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(128 * 32, 256)
        if norm_mode in ("bn_dense", "bn_all"):
            self.nd = nn.BatchNorm1d(256)
        elif norm_mode in ("ln_dense", "ln_all"):
            self.nd = nn.LayerNorm(256)
        else:
            self.nd = nn.Identity()

        self.ad = build_act(d_act, L_v, W_v)
        self.drop = nn.Dropout(0.5)
        self.out = nn.Linear(256, 1)

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        if b_dense != 0.0:
            nn.init.constant_(self.fc1.bias, b_dense)

    def forward(self, x, return_layer_stats=False):
        z1 = self.n1(self.c1(x))
        x1 = self.pool(self.a1(z1))
        z2 = self.n2(self.c2(x1))
        x2 = self.pool(self.a2(z2))
        z3 = self.n3(self.c3(x2))
        x3 = self.pool(self.a3(z3))
        flat = x3.view(x3.size(0), -1)
        z4 = self.nd(self.fc1(flat))
        h = self.drop(self.ad(z4))
        lo = self.out(h).squeeze(-1)
        if return_layer_stats:
            return lo, z4, [float(z1.var().item()), float(z2.var().item()), float(z3.var().item()), float(z4.var().item())]
        return lo, z4


class Net2D(nn.Module):
    def __init__(self, c_act, d_act, L_v=0.0, W_v=2.0, norm_mode="none", b_dense=0.0):
        super().__init__()
        self.c1 = nn.Conv2d(3, 32, 3, padding=1)
        self.n1 = nn.BatchNorm2d(32) if norm_mode == "bn_all" else nn.Identity()
        self.a1 = build_act(c_act, L_v, W_v)

        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.n2 = nn.BatchNorm2d(64) if norm_mode == "bn_all" else nn.Identity()
        self.a2 = build_act(c_act, L_v, W_v)

        self.c3 = nn.Conv2d(64, 128, 3, padding=1)
        self.n3 = nn.BatchNorm2d(128) if norm_mode == "bn_all" else nn.Identity()
        self.a3 = build_act(c_act, L_v, W_v)

        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        if norm_mode in ("bn_dense", "bn_all"):
            self.nd = nn.BatchNorm1d(256)
        elif norm_mode in ("ln_dense", "ln_all"):
            self.nd = nn.LayerNorm(256)
        else:
            self.nd = nn.Identity()

        self.ad = build_act(d_act, L_v, W_v)
        self.drop = nn.Dropout(0.5)
        self.out = nn.Linear(256, 1)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        if b_dense != 0.0:
            nn.init.constant_(self.fc1.bias, b_dense)

    def forward(self, x, return_layer_stats=False):
        z1 = self.n1(self.c1(x))
        x1 = self.pool(self.a1(z1))
        z2 = self.n2(self.c2(x1))
        x2 = self.pool(self.a2(z2))
        z3 = self.n3(self.c3(x2))
        x3 = self.pool(self.a3(z3))
        flat = x3.view(x3.size(0), -1)
        z4 = self.nd(self.fc1(flat))
        h = self.drop(self.ad(z4))
        lo = self.out(h).squeeze(-1)
        if return_layer_stats:
            return lo, z4, [float(z1.var().item()), float(z2.var().item()), float(z3.var().item()), float(z4.var().item())]
        return lo, z4


# ==============================================================================
# 4. Evaluation Helper with Validation-Tuned Threshold tau* & Layer Variance Log
# ==============================================================================
def find_best_threshold(y_true, y_prob):
    best_th, best_mcc = 0.5, -1.0
    for th in np.linspace(0.05, 0.90, 35):
        yh = (y_prob >= th).astype(int)
        if len(np.unique(yh)) > 1:
            m = matthews_corrcoef(y_true, yh)
            if m > best_mcc:
                best_mcc, best_th = m, th
    return float(best_th)


def pooled_mcc_calc(tp, fp, tn, fn):
    num = (tp * tn) - (fp * fn)
    den = np.sqrt(float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn))
    return float(num / den) if den > 0 else 0.0


def run_verified_experiment(suite_name, tr_ds, val_ds, te_ds, cfgs, is_2d=False, epochs=20, n_seeds=10):
    tr_ld = DataLoader(tr_ds, batch_size=128, shuffle=True)
    val_ld = DataLoader(val_ds, batch_size=512, shuffle=False)
    te_ld = DataLoader(te_ds, batch_size=512, shuffle=False)
    sample_bx, sample_by = next(iter(tr_ld))

    seed_rows, summary_rows, layer_var_rows = [], [], []

    for label, ca, da, L_v, W_v, norm_m, b_d in cfgs:
        mccs_05, mccs_opt, aucs, aprcs, deads, mus = [], [], [], [], [], []
        tps, fps, tns, fns = [], [], [], []

        for s in range(1, n_seeds + 1):
            torch.manual_seed(s)
            np.random.seed(s)
            m = (Net2D(ca, da, L_v, W_v, norm_m, b_d) if is_2d else Net1D(ca, da, L_v, W_v, norm_m, b_d)).to(device)

            # Measure layer-wise variances and grad norms at initialization (Seed 1)
            if s == 1:
                m.eval()
                lo_init, _, v_init = m(sample_bx.to(device), return_layer_stats=True)
                loss_init = nn.BCEWithLogitsLoss()(lo_init, sample_by.to(device))
                loss_init.backward()
                g1 = float(m.c1.weight.grad.norm().item())
                g4 = float(m.fc1.weight.grad.norm().item())
                layer_var_rows.append({
                    "Suite": suite_name, "Config": label, "Stage": "Init",
                    "Var_L1": round(v_init[0], 6), "Var_L2": round(v_init[1], 6),
                    "Var_L3": round(v_init[2], 6), "Var_L4_Dense": round(v_init[3], 6),
                    "GradNorm_Conv1": round(g1, 6), "GradNorm_Dense": round(g4, 6)
                })

            opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
            crit = nn.BCEWithLogitsLoss()
            m.train()
            for ep in range(1, epochs + 1):
                for bx, by in tr_ld:
                    opt.zero_grad()
                    lo, _ = m(bx.to(device))
                    crit(lo, by.to(device)).backward()
                    opt.step()

            m.eval()
            # 1) Validation pass to select optimal threshold tau*
            vps, vts = [], []
            with torch.no_grad():
                for bx, by in val_ld:
                    lo, _ = m(bx.to(device))
                    vps.append(torch.sigmoid(lo).cpu().numpy())
                    vts.append(by.numpy())
            tau_star = find_best_threshold(np.concatenate(vts), np.concatenate(vps))

            # 2) Out-of-sample Test pass
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

            yh_05 = (yp >= 0.5).astype(int)
            yh_opt = (yp >= tau_star).astype(int)
            tn, fp, fn, tp = confusion_matrix(yt, yh_05, labels=[0, 1]).ravel()
            mcc_05 = matthews_corrcoef(yt, yh_05) if len(np.unique(yh_05)) > 1 else 0.0
            mcc_opt = matthews_corrcoef(yt, yh_opt) if len(np.unique(yh_opt)) > 1 else 0.0
            auc = roc_auc_score(yt, yp)
            ap = average_precision_score(yt, yp)

            L_th = -1.0 if da == "SmeLU_W2" else (-0.5 if da == "SmeLU_W1" else L_v)
            dd = float(np.mean(np.mean(pm < L_th, axis=0) > 0.999) * 100.0)
            mu = float(np.mean(pm))

            mccs_05.append(mcc_05); mccs_opt.append(mcc_opt)
            aucs.append(auc); aprcs.append(ap); deads.append(dd); mus.append(mu)
            tps.append(tp); fps.append(fp); tns.append(tn); fns.append(fn)

            seed_rows.append({
                "Suite": suite_name, "Config": label, "Seed": s,
                "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
                "MCC_tau05": round(float(mcc_05), 4),
                "MCC_tauOpt": round(float(mcc_opt), 4),
                "Tau_Star": round(tau_star, 3),
                "AUROC": round(float(auc), 4),
                "AUPRC": round(float(ap), 4),
                "Dead_Pct": round(dd, 2),
                "Mu_Dense": round(mu, 4)
            })

        m_tp, m_fp, m_tn, m_fn = float(np.mean(tps)), float(np.mean(fps)), float(np.mean(tns)), float(np.mean(fns))
        p_mcc = pooled_mcc_calc(m_tp, m_fp, m_tn, m_fn)
        med_mcc = float(np.median(mccs_05))
        iqr_mcc = float(np.percentile(mccs_05, 75) - np.percentile(mccs_05, 25))

        print(f"[{suite_name}] {label:36s} | MCC(0.5)={np.mean(mccs_05):.4f}+/-{np.std(mccs_05):.4f} "
              f"[Med={med_mcc:.4f}, IQR={iqr_mcc:.4f}] | MCC(tau*)={np.mean(mccs_opt):.4f} "
              f"| AUROC={np.mean(aucs):.4f} | AUPRC={np.mean(aprcs):.4f}")

        summary_rows.append({
            "Suite": suite_name, "Config": label,
            "Mean_TP": round(m_tp, 1), "Mean_FP": round(m_fp, 1),
            "MCC_05_Mean": round(float(np.mean(mccs_05)), 4),
            "MCC_05_SD": round(float(np.std(mccs_05)), 4),
            "MCC_05_Median": round(med_mcc, 4),
            "MCC_05_IQR": round(iqr_mcc, 4),
            "Pooled_MCC_05": round(p_mcc, 4),
            "MCC_Tuned_Mean": round(float(np.mean(mccs_opt)), 4),
            "MCC_Tuned_SD": round(float(np.std(mccs_opt)), 4),
            "AUROC_Mean": round(float(np.mean(aucs)), 4),
            "AUPRC_Mean": round(float(np.mean(aprcs)), 4),
            "Dead_Pct_Mean": round(float(np.mean(deads)), 2),
            "Mu_Dense_Mean": round(float(np.mean(mus)), 4)
        })
    return seed_rows, summary_rows, layer_var_rows


if __name__ == "__main__":
    t0 = time.time()
    all_s, all_sum, all_lvar = [], [], []

    CORE_CFGS = [
        ("ReLU (C0)", "ReLU", "ReLU", 0.0, 2.0, "none", 0.0),
        ("Leaky ReLU (C0)", "LeakyReLU", "LeakyReLU", 0.0, 2.0, "none", 0.0),
        ("ELU (C1)", "ELU", "ELU", 0.0, 2.0, "none", 0.0),
        ("GELU (C_inf)", "GELU", "GELU", 0.0, 2.0, "none", 0.0),
        ("SiLU (C_inf)", "SiLU", "SiLU", 0.0, 2.0, "none", 0.0),
        ("SmeLU (L=-1, W=2, Symmetric)", "SmeLU_W2", "SmeLU_W2", -1.0, 2.0, "none", 0.0),
        ("SmeLU (L=-0.5, W=1, Symmetric)", "SmeLU_W1", "SmeLU_W1", -0.5, 1.0, "none", 0.0),
        ("Smooth RectLU (L=0, W=2, Offset -1)", "SmoothRectLU", "SmoothRectLU", 0.0, 2.0, "none", 0.0),
        ("AS-RectLU (Trainable L, W)", "ASRectLU", "ASRectLU", 0.0, 2.0, "none", 0.0),
        ("Smooth RectLU + b_dense=+1.0", "SmoothRectLU", "SmoothRectLU", 0.0, 2.0, "none", 1.0),
        ("Smooth RectLU + BatchNorm (All)", "SmoothRectLU", "SmoothRectLU", 0.0, 2.0, "bn_all", 0.0),
        ("Smooth RectLU + LayerNorm (Dense)", "SmoothRectLU", "SmoothRectLU", 0.0, 2.0, "ln_dense", 0.0),
        ("Proposed sigma-AS-RectLU (Symm)", "SigmaASRectLU", "SigmaASRectLU", -0.4, 0.8, "none", 0.0),
    ]

    print("\n=== [EXP 1] 1D Low-Amplitude Motif (sigma_x=0.25, 100:1, 20 Epochs) ===")
    tr1, val1, te1 = make_sparse_motif_1d(train_pos_ratio=0.01, sigma_x=0.25)
    s, sm, lv = run_verified_experiment("Exp1_1D_LowSigma025", tr1, val1, te1, CORE_CFGS, is_2d=False, epochs=20)
    all_s.extend(s); all_sum.extend(sm); all_lvar.extend(lv)

    print("\n=== [EXP 2] 1D Unit-Variance Rescaled Control (sigma_x=1.00, 100:1, 20 Epochs) ===")
    tr1_u, val1_u, te1_u = make_sparse_motif_1d(train_pos_ratio=0.01, sigma_x=1.00)
    s, sm, lv = run_verified_experiment("Exp2_1D_UnitSigma100", tr1_u, val1_u, te1_u, [
        ("ReLU (C0)", "ReLU", "ReLU", 0.0, 2.0, "none", 0.0),
        ("SmeLU (L=-1, W=2, Symmetric)", "SmeLU_W2", "SmeLU_W2", -1.0, 2.0, "none", 0.0),
        ("Smooth RectLU (L=0, W=2, Offset -1)", "SmoothRectLU", "SmoothRectLU", 0.0, 2.0, "none", 0.0),
        ("Proposed sigma-AS-RectLU (Symm)", "SigmaASRectLU", "SigmaASRectLU", -0.4, 0.8, "none", 0.0),
    ], is_2d=False, epochs=20)
    all_s.extend(s); all_sum.extend(sm); all_lvar.extend(lv)

    print("\n=== [EXP 3] 2D Low-Amplitude Spatial Motif (sigma_x=0.22 vs 1.00, 100:1, 20 Epochs) ===")
    tr2, val2, te2 = make_sparse_motif_2d(sigma_x=0.22)
    s, sm, lv = run_verified_experiment("Exp3_2D_LowSigma022", tr2, val2, te2, CORE_CFGS, is_2d=True, epochs=20)
    all_s.extend(s); all_sum.extend(sm); all_lvar.extend(lv)

    tr2_u, val2_u, te2_u = make_sparse_motif_2d(sigma_x=1.00)
    s, sm, lv = run_verified_experiment("Exp3_2D_UnitSigma100", tr2_u, val2_u, te2_u, [
        ("ReLU (C0)", "ReLU", "ReLU", 0.0, 2.0, "none", 0.0),
        ("SmeLU (L=-1, W=2, Symmetric)", "SmeLU_W2", "SmeLU_W2", -1.0, 2.0, "none", 0.0),
        ("Smooth RectLU (L=0, W=2, Offset -1)", "SmoothRectLU", "SmoothRectLU", 0.0, 2.0, "none", 0.0),
        ("Proposed sigma-AS-RectLU (Symm)", "SigmaASRectLU", "SigmaASRectLU", -0.4, 0.8, "none", 0.0),
    ], is_2d=True, epochs=20)
    all_s.extend(s); all_sum.extend(sm); all_lvar.extend(lv)

    pd.DataFrame(all_s).to_csv("v3_verified_logs/v3_per_seed_logs.csv", index=False)
    pd.DataFrame(all_sum).to_csv("v3_verified_logs/v3_summary_table.csv", index=False)
    pd.DataFrame(all_lvar).to_csv("v3_verified_logs/v3_layer_variances_and_grads.csv", index=False)
    print(f"\nALL V3 EXPERIMENTS COMPLETED in {(time.time() - t0)/60:.2f} min!")