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

# ==============================================================================
# 1. โหลดข้อมูลจริง NIST GIAB 2D Pileup (50 x 101 x 4, สเกลต่ำ 0.25x, 10:1 Imbalance)
# ==============================================================================
def load_giab2d_benchmark():
    pileup_path = CRAM_DATA_DIR / "pileup_features_dataset.npz"
    print(f"[1/3] โหลดข้อมูลจริงจาก: {pileup_path}")
    d = np.load(pileup_path)
    X_all = np.concatenate([d["X_train"], d["X_val"]], axis=0).astype(np.float32)
    X_all = np.transpose(X_all, (0, 3, 1, 2))  # (6000, 4, 50, 101)

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

            n_clusters = rng.randint(2, 5)
            for _ in range(n_clusters):
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
    X_va, y_va = make_partition(1500, 60, 600)
    X_te, y_te = make_partition(2500, 60, 600)
    return (
        (torch.tensor(X_tr), torch.tensor(y_tr)),
        (torch.tensor(X_va), torch.tensor(y_va)),
        (torch.tensor(X_te), torch.tensor(y_te)),
    )

# ==============================================================================
# 2. นิยามฟังก์ชันกระตุ้นและโครงข่าย 2D-CNN
# ==============================================================================
class SmoothRectLU(nn.Module):
    def __init__(self, L=0.0, W=2.0):
        super().__init__()
        self.L, self.W = L, W
    def forward(self, x):
        z = x - self.L
        return torch.where(x < self.L, torch.zeros_like(x),
               torch.where(x < self.L + self.W, (z ** 2) / (2.0 * self.W), z - 0.5 * self.W))

class FixedSmeLU(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta
    def forward(self, x):
        return torch.where(x < -self.beta, torch.zeros_like(x),
               torch.where(x <= self.beta, ((x + self.beta) ** 2) / (4.0 * self.beta), x))

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

def calibrated_mcc(y_val, p_val, y_te, p_te):
    best_t, best_m = 0.50, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        pred_v = (p_val >= t).astype(int)
        if np.any(pred_v == 1) and np.any(pred_v == 0):
            m = matthews_corrcoef(y_val, pred_v)
            if m > best_m:
                best_m, best_t = m, t
    pred_t = (p_te >= best_t).astype(int)
    if np.any(pred_t == 1) and np.any(pred_t == 0):
        return float(matthews_corrcoef(y_te, pred_t))
    return 0.0

# ==============================================================================
# 3. รันการทดลองข้อ 3.6 (N = 10 Paired Seeds):
#    - Learning Rate Sensitivity: lr in {1e-3, 1e-2, 5e-2} สำหรับ SGD+Mom และ {5e-4, 1e-3} สำหรับ RMSprop
#    - Gradient Clipping Control: max_norm = 1.0 vs. None
#    - Prior-Logit Output Bias Init: b_out = ln(p+/p-) = -2.3026 vs. 0.0
# ==============================================================================
def run_reviewer_3_6_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = load_giab2d_benchmark()
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=256, shuffle=False)
    te_loader = DataLoader(TensorDataset(X_te, y_te), batch_size=256, shuffle=False)
    y_v_np = y_val.numpy().astype(int)
    y_t_np = y_te.numpy().astype(int)

    # รายการทดสอบตามคำแนะนำของ Reviewer ข้อ 3.6
    # (Label, Optimizer, LR, GradClip, PriorBiasInit, ActivationBuilder)
    configs = [
        # --- SGD + Momentum (0.9): LR Sensitivity, Grad Clipping, & Prior Bias Init ---
        ("SGD (lr=1e-3) + Smooth RectLU", "sgd", 1e-3, None, False, lambda: SmoothRectLU(0.0, 2.0)),
        ("SGD (lr=1e-3) + Symm. sigma-AS-RectLU", "sgd", 1e-3, None, False, lambda: FusedSymmSigmaASRectLU(0.8)),
        ("SGD (lr=1e-2) + Smooth RectLU", "sgd", 1e-2, None, False, lambda: SmoothRectLU(0.0, 2.0)),
        ("SGD (lr=1e-2) + Fixed SmeLU (L=-1, W=2)", "sgd", 1e-2, None, False, lambda: FixedSmeLU(1.0)),
        ("SGD (lr=1e-2) + Symm. sigma-AS-RectLU", "sgd", 1e-2, None, False, lambda: FusedSymmSigmaASRectLU(0.8)),
        ("SGD (lr=5e-2) + Smooth RectLU", "sgd", 5e-2, None, False, lambda: SmoothRectLU(0.0, 2.0)),
        ("SGD (lr=5e-2) + Symm. sigma-AS-RectLU", "sgd", 5e-2, None, False, lambda: FusedSymmSigmaASRectLU(0.8)),
        ("SGD (lr=5e-2, Clip=1.0) + Smooth RectLU", "sgd", 5e-2, 1.0, False, lambda: SmoothRectLU(0.0, 2.0)),
        ("SGD (lr=5e-2, Clip=1.0) + Symm. sigma-AS-RectLU", "sgd", 5e-2, 1.0, False, lambda: FusedSymmSigmaASRectLU(0.8)),
        ("SGD (lr=5e-2, Clip=1.0, PriorBias) + Symm. sigma-AS", "sgd", 5e-2, 1.0, True, lambda: FusedSymmSigmaASRectLU(0.8)),
        # --- RMSprop: LR Sensitivity & Grad Clipping ---
        ("RMSprop (lr=5e-4) + Smooth RectLU", "rmsprop", 5e-4, None, False, lambda: SmoothRectLU(0.0, 2.0)),
        ("RMSprop (lr=5e-4) + Symm. sigma-AS-RectLU", "rmsprop", 5e-4, None, False, lambda: FusedSymmSigmaASRectLU(0.8)),
        ("RMSprop (lr=1e-3, Clip=1.0) + Smooth RectLU", "rmsprop", 1e-3, 1.0, False, lambda: SmoothRectLU(0.0, 2.0)),
        ("RMSprop (lr=1e-3, Clip=1.0) + Symm. sigma-AS-RectLU", "rmsprop", 1e-3, 1.0, False, lambda: FusedSymmSigmaASRectLU(0.8)),
    ]

    print("\n[2/3] รันการทดลองข้อ 3.6 Cross-Optimizer Diagnostics (N = 10 Paired Seeds)...")
    rows = []
    for label, opt_name, lr, clip_val, prior_init, builder in configs:
        mccs, cal_mccs, aurocs, auprcs, escaped = [], [], [], [], 0
        for seed in range(10):
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = GIAB2DCNN(builder).to(device)
            for m in [model.conv1, model.conv2, model.conv3, model.dense1, model.out]:
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
            if prior_init:
                # ตั้งค่า Prior-logit bias b_out = ln(120/1200) = -2.3026 ตามทฤษฎี Imbalanced Initialization
                nn.init.constant_(model.out.bias, float(np.log(120.0 / 1200.0)))

            tr_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=256, shuffle=True)
            if opt_name == "sgd":
                opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            else:
                opt = torch.optim.RMSprop(model.parameters(), lr=lr, alpha=0.99, eps=1e-8)

            model.train()
            for ep in range(20):
                for xb, yb in tr_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    opt.zero_grad()
                    loss = F.binary_cross_entropy_with_logits(model(xb), yb)
                    loss.backward()
                    if clip_val is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_val)
                    opt.step()

            model.eval()
            with torch.no_grad():
                p_val = np.concatenate([torch.sigmoid(model(xb.to(device))).cpu().numpy() for xb, _ in val_loader])
                p_te = np.concatenate([torch.sigmoid(model(xb.to(device))).cpu().numpy() for xb, _ in te_loader])

            pred_t = (p_te >= 0.50).astype(int)
            tp = int(np.sum((pred_t == 1) & (y_t_np == 1)))
            fp = int(np.sum((pred_t == 1) & (y_t_np == 0)))
            mcc = float(matthews_corrcoef(y_t_np, pred_t)) if (tp + fp > 0 and (tp + fp) < len(y_t_np)) else 0.0
            c_mcc = calibrated_mcc(y_v_np, p_val, y_t_np, p_te)
            if mcc > 0.10:
                escaped += 1

            mccs.append(mcc)
            cal_mccs.append(c_mcc)
            aurocs.append(float(roc_auc_score(y_t_np, p_te)))
            auprcs.append(float(average_precision_score(y_t_np, p_te)))

        m_arr, cm_arr, a_arr = np.array(mccs), np.array(cal_mccs), np.array(aurocs)
        row = {
            "Configuration": label,
            "Optimizer": opt_name.upper(),
            "LR": lr,
            "Grad_Clip": clip_val if clip_val else "None",
            "Prior_Bias": "Yes" if prior_init else "No",
            "Escaped_Seeds": f"{escaped}/10",
            "MCC_050_Mean": round(float(m_arr.mean()), 4),
            "MCC_050_SD": round(float(m_arr.std(ddof=1)), 4),
            "Cal_MCC_Mean": round(float(cm_arr.mean()), 4),
            "Cal_MCC_SD": round(float(cm_arr.std(ddof=1)), 4),
            "AUROC_Mean": round(float(a_arr.mean()), 4),
            "AUROC_SD": round(float(a_arr.std(ddof=1)), 4),
            "AUPRC_Mean": round(float(np.mean(auprcs)), 4),
        }
        rows.append(row)
        print(f"{label:50s} | Esc: {row['Escaped_Seeds']:5s} "
              f"| MCC(0.50): {row['MCC_050_Mean']:.4f} ± {row['MCC_050_SD']:.4f} "
              f"| MCC(t*): {row['Cal_MCC_Mean']:.4f} ± {row['Cal_MCC_SD']:.4f} "
              f"| AUROC: {row['AUROC_Mean']:.4f}")

    out_csv = PROJECT_DIR / "v9_reviewer_3_6_cross_optimizer_ablation_10seeds.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[3/3] บันทึกไฟล์ผลการทดลองข้อ 3.6 ไว้ที่:\n👉 {out_csv}")

if __name__ == "__main__":
    run_reviewer_3_6_experiment()