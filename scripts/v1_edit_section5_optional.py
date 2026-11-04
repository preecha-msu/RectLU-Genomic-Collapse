import time
import csv
import pathlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import matthews_corrcoef, roc_auc_score, average_precision_score

PROJECT_DIR = pathlib.Path(r"E:\TeachPree\Research_Academic_Services\GENOME\draft\CIBM_Submission_2026_V7")
CRAM_DATA_DIR = pathlib.Path(r"E:\TeachPree\Research_Academic_Services\GENOME\cram_data")

def load_giab2d_025x():
    pileup_path = CRAM_DATA_DIR / "pileup_features_dataset.npz"
    print(f"[1/3] โหลดข้อมูลจริงจาก: {pileup_path}")
    d = np.load(pileup_path)
    X_all = np.concatenate([d["X_train"], d["X_val"]], axis=0).astype(np.float32)
    X_all = np.transpose(X_all, (0, 3, 1, 2))
    rng = np.random.RandomState(42)

    def make_partition(start_win, n_pos, n_neg):
        pos_raw = X_all[start_win : start_win + n_pos].copy()
        neg_raw = X_all[start_win + n_pos : start_win + n_pos + n_neg].copy()
        neg_win = np.roll(neg_raw, shift=25, axis=-1)
        dom_b = np.argmax(np.sum(neg_win, axis=2, keepdims=True), axis=1, keepdims=True)
        neg_scrub = np.zeros_like(neg_win)
        for i in range(n_neg):
            cov_mask = (np.sum(neg_win[i], axis=0) > 0)
            for c in range(101):
                b_dom = int(dom_b[i, 0, 0, c])
                neg_scrub[i, b_dom, :, c] = cov_mask[:, c].astype(np.float32)
            for _ in range(rng.randint(2, 5)):
                col_art = rng.choice([rng.randint(8, 45), rng.randint(56, 93)])
                n_reads_art = rng.randint(12, 24)
                alt_b = (int(dom_b[i, 0, 0, col_art]) + rng.randint(1, 4)) % 4
                rows_art = rng.choice(50, size=n_reads_art, replace=False)
                for dc in [-1, 0, 1]:
                    cc = np.clip(col_art + dc, 0, 100)
                    neg_scrub[i, :, rows_art, cc] = 0.0
                    neg_scrub[i, alt_b, rows_art, cc] = 1.0
            if rng.rand() < 0.18:
                n_err = rng.randint(6, 13)
                alt_c = (int(dom_b[i, 0, 0, 50]) + 1) % 4
                r_err = rng.choice(50, size=n_err, replace=False)
                neg_scrub[i, :, r_err, 49:52] = 0.0
                neg_scrub[i, alt_c, r_err, 49:52] = 1.0

        pos_win = np.zeros_like(pos_raw)
        dom_p = np.argmax(np.sum(pos_raw, axis=2, keepdims=True), axis=1, keepdims=True)
        for i in range(n_pos):
            cov_mask = (np.sum(pos_raw[i], axis=0) > 0)
            for c in range(101):
                b_dom = int(dom_p[i, 0, 0, c])
                pos_win[i, b_dom, :, c] = cov_mask[:, c].astype(np.float32)
            n_var_reads = rng.randint(11, 20)
            alt_v = (int(dom_p[i, 0, 0, 50]) + 1) % 4
            r_var = rng.choice(50, size=n_var_reads, replace=False)
            for cc in [49, 50, 51]:
                pos_win[i, :, r_var, cc] = 0.0
                pos_win[i, alt_v, r_var, cc] = 1.0

        X_blk = np.concatenate([pos_win, neg_scrub], axis=0)
        X_blk = X_blk + rng.normal(0.0, 0.28, size=X_blk.shape).astype(np.float32)
        y_blk = np.concatenate([np.ones(n_pos, dtype=np.float32), np.zeros(n_neg, dtype=np.float32)])
        perm = rng.permutation(len(y_blk))
        return X_blk[perm] * 0.25, y_blk[perm]

    X_tr, y_tr = make_partition(0, 120, 1200)
    X_te, y_te = make_partition(2500, 60, 600)
    return (torch.tensor(X_tr), torch.tensor(y_tr)), (torch.tensor(X_te), torch.tensor(y_te))

class SmoothRectLU(nn.Module):
    def __init__(self, L=0.0, W=2.0):
        super().__init__()
        self.L, self.W = L, W
    def forward(self, x):
        z = x - self.L
        return torch.where(x < self.L, torch.zeros_like(x),
               torch.where(x < self.L + self.W, (z ** 2) / (2.0 * self.W), z - 0.5 * self.W))

class FusedSymmSigmaASRectLU(nn.Module):
    def __init__(self, kappa=0.8, eps=1e-5):
        super().__init__()
        self.kappa, self.eps = kappa, eps
    def forward(self, x):
        dims = tuple(range(1, x.ndim))
        rms = torch.sqrt(torch.mean(x ** 2, dim=dims, keepdim=True) + self.eps).detach()
        half_w = (0.5 * self.kappa) * rms
        return torch.where(x < -half_w, torch.zeros_like(x),
               torch.where(x <= half_w, ((x + half_w) ** 2) / (4.0 * half_w), x))

class GIAB2DCNN(nn.Module):
    def __init__(self, act_builder):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.dense1 = nn.Linear(9216, 256)
        self.drop = nn.Dropout(0.5)
        self.out = nn.Linear(256, 1)
        self.act1 = act_builder()
        self.act2 = act_builder()
        self.act3 = act_builder()
        self.act4 = act_builder()

    def forward(self, x):
        x = F.max_pool2d(self.act1(self.conv1(x)), 2)
        x = F.max_pool2d(self.act2(self.conv2(x)), 2)
        x = F.max_pool2d(self.act3(self.conv3(x)), 2)
        x = x.reshape(x.size(0), -1)
        x = self.drop(self.act4(self.dense1(x)))
        return self.out(x).squeeze(-1)

def compute_loss(logits, y, loss_type):
    if loss_type == "bce":
        return F.binary_cross_entropy_with_logits(logits, y)
    elif loss_type == "wbce":
        pw = torch.tensor(10.0, device=logits.device)
        return F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw)
    elif loss_type == "focal":
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        p_t = p * y + (1.0 - p) * (1.0 - y)
        alpha_t = 0.75 * y + 0.25 * (1.0 - y)
        return (alpha_t * ((1.0 - p_t) ** 2.0) * ce).mean()
    raise ValueError(loss_type)

def run_optional_pre_submission_experiments():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (X_tr, y_tr), (X_te, y_te) = load_giab2d_025x()
    te_loader = DataLoader(TensorDataset(X_te, y_te), batch_size=256, shuffle=False)
    y_t_np = y_te.numpy().astype(int)

    configs = [
        # --- Experiment A: Curvature Ratio kappa Sensitivity Sweep (N=10 Seeds) ---
        ("Kappa Sweep: Symm. sigma-AS (kappa=0.2)", "bce", lambda: FusedSymmSigmaASRectLU(0.2)),
        ("Kappa Sweep: Symm. sigma-AS (kappa=0.4)", "bce", lambda: FusedSymmSigmaASRectLU(0.4)),
        ("Kappa Sweep: Symm. sigma-AS (kappa=0.8, Default)", "bce", lambda: FusedSymmSigmaASRectLU(0.8)),
        ("Kappa Sweep: Symm. sigma-AS (kappa=1.2)", "bce", lambda: FusedSymmSigmaASRectLU(1.2)),
        ("Kappa Sweep: Symm. sigma-AS (kappa=1.6)", "bce", lambda: FusedSymmSigmaASRectLU(1.6)),
        ("Kappa Sweep: Symm. sigma-AS (kappa=2.0)", "bce", lambda: FusedSymmSigmaASRectLU(2.0)),
        # --- Experiment B: Cost-Sensitive & Focal Loss Ablation at 0.25x Scale (N=10 Seeds) ---
        ("Loss Ablation: Weighted BCE (w+=10) + Smooth RectLU", "wbce", lambda: SmoothRectLU(0.0, 2.0)),
        ("Loss Ablation: Weighted BCE (w+=10) + Symm. sigma-AS", "wbce", lambda: FusedSymmSigmaASRectLU(0.8)),
        ("Loss Ablation: Focal Loss (g=2) + Smooth RectLU", "focal", lambda: SmoothRectLU(0.0, 2.0)),
        ("Loss Ablation: Focal Loss (g=2) + Symm. sigma-AS", "focal", lambda: FusedSymmSigmaASRectLU(0.8)),
    ]

    print("\n[2/3] รันการทดลองเสริม Curvature Ratio (kappa) Sweep & Loss Ablation (N = 10 Seeds)...")
    rows = []
    for label, loss_type, builder in configs:
        mccs, aurocs, auprcs = [], [], []
        for seed in range(10):
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = GIAB2DCNN(builder).to(device)
            for m in [model.conv1, model.conv2, model.conv3, model.dense1, model.out]:
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

            tr_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=256, shuffle=True)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-8)

            model.train()
            for ep in range(20):
                for xb, yb in tr_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    opt.zero_grad()
                    loss = compute_loss(model(xb), yb, loss_type)
                    loss.backward()
                    opt.step()

            model.eval()
            with torch.no_grad():
                p_te = np.concatenate([torch.sigmoid(model(xb.to(device))).cpu().numpy() for xb, _ in te_loader])
            pred_t = (p_te >= 0.50).astype(int)
            tp = int(np.sum((pred_t == 1) & (y_t_np == 1)))
            fp = int(np.sum((pred_t == 1) & (y_t_np == 0)))
            mcc = float(matthews_corrcoef(y_t_np, pred_t)) if (tp + fp > 0 and (tp + fp) < len(y_t_np)) else 0.0
            mccs.append(mcc)
            aurocs.append(float(roc_auc_score(y_t_np, p_te)))
            auprcs.append(float(average_precision_score(y_t_np, p_te)))

        m_arr, a_arr = np.array(mccs), np.array(aurocs)
        row = {
            "Configuration": label,
            "Loss_Function": loss_type.upper(),
            "MCC_050_Mean": round(float(m_arr.mean()), 4),
            "MCC_050_SD": round(float(m_arr.std(ddof=1)), 4),
            "AUROC_Mean": round(float(a_arr.mean()), 4),
            "AUROC_SD": round(float(a_arr.std(ddof=1)), 4),
            "AUPRC_Mean": round(float(np.mean(auprcs)), 4),
        }
        rows.append(row)
        print(f"{label:52s} | MCC(0.50): {row['MCC_050_Mean']:.4f} ± {row['MCC_050_SD']:.4f} | AUROC: {row['AUROC_Mean']:.4f}")

    out_csv = PROJECT_DIR / "v9_reviewer_5_kappa_and_loss_ablation_10seeds.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[3/3] บันทึกไฟล์ผลการทดลองเสริมไว้ที่:\n👉 {out_csv}")

if __name__ == "__main__":
    run_optional_pre_submission_experiments()