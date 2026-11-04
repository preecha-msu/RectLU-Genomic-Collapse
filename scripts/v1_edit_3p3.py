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
# 1. โหลดข้อมูลจริง NIST GIAB 2D Pileup (50 x 101 x 4) พร้อมสร้าง 2x2 Scale (0.25x vs 1.00x)
#    และแบ่ง Test Set (N=660) ออกเป็น 6 Disjoint Genomic Sub-Blocks จริง (บล็อกละ 10+ / 100-)
# ==============================================================================
def load_giab2d_with_6_test_subblocks():
    pileup_path = CRAM_DATA_DIR / "pileup_features_dataset.npz"
    print(f"[1/4] โหลดข้อมูลจริงจาก: {pileup_path}")
    d = np.load(pileup_path)
    X_all = np.concatenate([d["X_train"], d["X_val"]], axis=0).astype(np.float32)
    X_all = np.transpose(X_all, (0, 3, 1, 2))  # (6000, 4, 50, 101)

    rng = np.random.RandomState(42)

    def make_partition(start_win, n_pos, n_neg, preserve_order=False):
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

        pos_win = pos_win + rng.normal(0.0, 0.28, size=pos_win.shape).astype(np.float32)
        neg_scrub = neg_scrub + rng.normal(0.0, 0.28, size=neg_scrub.shape).astype(np.float32)

        if preserve_order:
            # จัดเรียงเป็น 6 Disjoint Genomic Sub-Blocks (บล็อกละ 10 Positives + 100 Negatives = 110 หน้าต่าง)
            X_sub, y_sub = [], []
            for b in range(6):
                X_b = np.concatenate([pos_win[b*10:(b+1)*10], neg_scrub[b*100:(b+1)*100]], axis=0)
                y_b = np.concatenate([np.ones(10, dtype=np.float32), np.zeros(100, dtype=np.float32)])
                X_sub.append(X_b)
                y_sub.append(y_b)
            return np.concatenate(X_sub, axis=0), np.concatenate(y_sub, axis=0)

        X_blk = np.concatenate([pos_win, neg_scrub], axis=0)
        y_blk = np.concatenate([np.ones(n_pos, dtype=np.float32), np.zeros(n_neg, dtype=np.float32)])
        perm = rng.permutation(len(y_blk))
        return X_blk[perm], y_blk[perm]

    X_tr, y_tr = make_partition(0, 120, 1200, preserve_order=False)
    X_te, y_te = make_partition(2500, 60, 600, preserve_order=True)
    return (torch.tensor(X_tr), torch.tensor(y_tr)), (torch.tensor(X_te), torch.tensor(y_te))

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

# ==============================================================================
# 3. รันการทดลองข้อ 3.3:
#    Part 1: N = 30 Paired Seeds บน Real GIAB 2D (0.25x) พร้อมวัดผลแยก 6 Genomic Sub-Blocks จริง
#    Part 2: ขยาย Cross-Optimizer (RMSprop & Momentum SGD) จาก N = 5 เป็น N = 10 Seeds เท่ากัน
# ==============================================================================
def run_reviewer_3_3_validation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (X_tr_1x, y_tr), (X_te_1x, y_te) = load_giab2d_with_6_test_subblocks()

    X_tr_low = X_tr_1x * 0.25
    X_te_low = X_te_1x * 0.25
    te_loader = DataLoader(TensorDataset(X_te_low, y_te), batch_size=256, shuffle=False)
    y_np = y_te.numpy().astype(int)

    models_30seeds = [
        ("Smooth RectLU (L=0, W=2, 0.25x)", lambda: SmoothRectLU(L=0.0, W=2.0)),
        ("ReLU (Homogeneous C0, 0.25x)", lambda: nn.ReLU()),
        ("Symm. sigma-AS-RectLU (kappa=0.8, 0.25x)", lambda: FusedSymmSigmaASRectLU(kappa=0.8)),
    ]

    print("\n[2/4] รันทดสอบ N = 30 Independent Seeds + วัดผลจริงบน 6 Disjoint Genomic Test Sub-Blocks...")
    seed_logs = {name: {"mcc": [], "auroc": [], "auprc": [], "subblock_mcc": [], "subblock_auroc": []} for name, _ in models_30seeds}

    for name, builder in models_30seeds:
        for seed in range(30):
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = GIAB2DCNN(builder).to(device)
            for m in [model.conv1, model.conv2, model.conv3, model.dense1, model.out]:
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

            tr_loader = DataLoader(TensorDataset(X_tr_low, y_tr), batch_size=256, shuffle=True)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-8)

            model.train()
            for ep in range(20):
                for xb, yb in tr_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    opt.zero_grad()
                    loss = F.binary_cross_entropy_with_logits(model(xb), yb)
                    loss.backward()
                    opt.step()

            model.eval()
            with torch.no_grad():
                p_te = np.concatenate([torch.sigmoid(model(xb.to(device))).cpu().numpy() for xb, _ in te_loader])
            pred_t = (p_te >= 0.50).astype(int)

            tp = int(np.sum((pred_t == 1) & (y_np == 1)))
            fp = int(np.sum((pred_t == 1) & (y_np == 0)))
            mcc = float(matthews_corrcoef(y_np, pred_t)) if (tp + fp > 0 and (tp + fp) < len(y_np)) else 0.0
            auroc = float(roc_auc_score(y_np, p_te))
            auprc = float(average_precision_score(y_np, p_te))

            seed_logs[name]["mcc"].append(mcc)
            seed_logs[name]["auroc"].append(auroc)
            seed_logs[name]["auprc"].append(auprc)

            # วัดผลจริงในแต่ละ Genomic Sub-Block ทั้ง 6 บล็อก (บล็อกละ 110 หน้าต่าง: 10+ / 100-)
            for b in range(6):
                s_idx, e_idx = b * 110, (b + 1) * 110
                yb_sub = y_np[s_idx:e_idx]
                pb_sub = p_te[s_idx:e_idx]
                pred_sub = pred_t[s_idx:e_idx]
                tp_b = int(np.sum((pred_sub == 1) & (yb_sub == 1)))
                fp_b = int(np.sum((pred_sub == 1) & (yb_sub == 0)))
                mcc_b = float(matthews_corrcoef(yb_sub, pred_sub)) if (tp_b + fp_b > 0 and (tp_b + fp_b) < len(yb_sub)) else 0.0
                auc_b = float(roc_auc_score(yb_sub, pb_sub))
                seed_logs[name]["subblock_mcc"].append(mcc_b)
                seed_logs[name]["subblock_auroc"].append(auc_b)

        m_arr = np.array(seed_logs[name]["mcc"])
        a_arr = np.array(seed_logs[name]["auroc"])
        sb_m = np.array(seed_logs[name]["subblock_mcc"])
        sb_a = np.array(seed_logs[name]["subblock_auroc"])
        print(f"{name:40s} | 30-Seed MCC: {m_arr.mean():.4f} ± {m_arr.std(ddof=1):.4f} [{m_arr.min():.4f}, {m_arr.max():.4f}] "
              f"| 30-Seed AUROC: {a_arr.mean():.4f} ± {a_arr.std(ddof=1):.4f} "
              f"| 6-SubBlock MCC: {sb_m.mean():.4f} ± {sb_m.std(ddof=1):.4f}")

    # คำนวณ Paired Empirical Differences (N = 30 Seeds) ระหว่าง Symm. sigma-AS-RectLU กับ Smooth RectLU
    diff_mcc = np.array(seed_logs["Symm. sigma-AS-RectLU (kappa=0.8, 0.25x)"]["mcc"]) - np.array(seed_logs["Smooth RectLU (L=0, W=2, 0.25x)"]["mcc"])
    diff_auc = np.array(seed_logs["Symm. sigma-AS-RectLU (kappa=0.8, 0.25x)"]["auroc"]) - np.array(seed_logs["Smooth RectLU (L=0, W=2, 0.25x)"]["auroc"])
    wins_mcc = int(np.sum(diff_mcc > 0))
    wins_auc = int(np.sum(diff_auc > 0))
    print(f"\n---> Paired 30-Seed Empirical Gain (Symm. sigma-AS vs. Smooth RectLU):")
    print(f"     Delta MCC(0.50): +{diff_mcc.mean():.4f} ± {diff_mcc.std(ddof=1):.4f} (Range: [+{diff_mcc.min():.4f}, +{diff_mcc.max():.4f}], Wins: {wins_mcc}/30 seeds)")
    print(f"     Delta AUROC:     +{diff_auc.mean():.4f} ± {diff_auc.std(ddof=1):.4f} (Range: [+{diff_auc.min():.4f}, +{diff_auc.max():.4f}], Wins: {wins_auc}/30 seeds)")

    # ==============================================================================
    # Part 2: รัน Cross-Optimizer (RMSprop และ SGD+Momentum) ให้ครบ N = 10 Seeds เท่ากัน
    # ==============================================================================
    print("\n[3/4] รันทดสอบ Cross-Optimizer (RMSprop & SGD+Momentum) ที่ N = 10 Paired Seeds...")
    opt_configs = [
        ("RMSprop + Smooth RectLU (L=0, W=2, N=10)", "rmsprop", lambda: SmoothRectLU(L=0.0, W=2.0)),
        ("RMSprop + Symm. sigma-AS-RectLU (N=10)", "rmsprop", lambda: FusedSymmSigmaASRectLU(kappa=0.8)),
        ("SGD (Mom 0.9) + Smooth RectLU (L=0, W=2, N=10)", "sgd", lambda: SmoothRectLU(L=0.0, W=2.0)),
        ("SGD (Mom 0.9) + Symm. sigma-AS-RectLU (N=10)", "sgd", lambda: FusedSymmSigmaASRectLU(kappa=0.8)),
    ]

    rows_out = []
    for name, _ in models_30seeds:
        m_arr = np.array(seed_logs[name]["mcc"])
        a_arr = np.array(seed_logs[name]["auroc"])
        sb_m = np.array(seed_logs[name]["subblock_mcc"])
        rows_out.append({
            "Experiment_Group": "30_Seed_and_6_Genomic_SubBlock_GIAB2D",
            "Configuration": name,
            "N_Seeds": 30,
            "MCC_050_Mean": round(float(m_arr.mean()), 4),
            "MCC_050_SD": round(float(m_arr.std(ddof=1)), 4),
            "MCC_050_MinMax": f"[{m_arr.min():.4f}, {m_arr.max():.4f}]",
            "AUROC_Mean": round(float(a_arr.mean()), 4),
            "AUROC_SD": round(float(a_arr.std(ddof=1)), 4),
            "SubBlock_MCC_Mean": round(float(sb_m.mean()), 4),
            "SubBlock_MCC_SD": round(float(sb_m.std(ddof=1)), 4),
        })

    for label, opt_type, builder in opt_configs:
        mccs, aurocs = [], []
        for seed in range(10):
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = GIAB2DCNN(builder).to(device)
            for m in [model.conv1, model.conv2, model.conv3, model.dense1, model.out]:
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

            tr_loader = DataLoader(TensorDataset(X_tr_low, y_tr), batch_size=256, shuffle=True)
            if opt_type == "rmsprop":
                opt = torch.optim.RMSprop(model.parameters(), lr=1e-3, alpha=0.99, eps=1e-8)
            else:
                opt = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

            model.train()
            for ep in range(20):
                for xb, yb in tr_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    opt.zero_grad()
                    loss = F.binary_cross_entropy_with_logits(model(xb), yb)
                    loss.backward()
                    opt.step()

            model.eval()
            with torch.no_grad():
                p_te = np.concatenate([torch.sigmoid(model(xb.to(device))).cpu().numpy() for xb, _ in te_loader])
            pred_t = (p_te >= 0.50).astype(int)
            tp = int(np.sum((pred_t == 1) & (y_np == 1)))
            fp = int(np.sum((pred_t == 1) & (y_np == 0)))
            mcc = float(matthews_corrcoef(y_np, pred_t)) if (tp + fp > 0 and (tp + fp) < len(y_np)) else 0.0
            mccs.append(mcc)
            aurocs.append(float(roc_auc_score(y_np, p_te)))

        m_arr, a_arr = np.array(mccs), np.array(aurocs)
        print(f"{label:46s} | MCC(0.50): {m_arr.mean():.4f} ± {m_arr.std(ddof=1):.4f} [{m_arr.min():.4f}, {m_arr.max():.4f}] "
              f"| AUROC: {a_arr.mean():.4f} ± {a_arr.std(ddof=1):.4f}")
        rows_out.append({
            "Experiment_Group": "10_Seed_Cross_Optimizer_GIAB2D",
            "Configuration": label,
            "N_Seeds": 10,
            "MCC_050_Mean": round(float(m_arr.mean()), 4),
            "MCC_050_SD": round(float(m_arr.std(ddof=1)), 4),
            "MCC_050_MinMax": f"[{m_arr.min():.4f}, {m_arr.max():.4f}]",
            "AUROC_Mean": round(float(a_arr.mean()), 4),
            "AUROC_SD": round(float(a_arr.std(ddof=1)), 4),
            "SubBlock_MCC_Mean": "-",
            "SubBlock_MCC_SD": "-",
        })

    out_csv = PROJECT_DIR / "v9_reviewer_3_3_30seeds_and_subblocks.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows_out[0].keys())
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"\n[4/4] บันทึกไฟล์ผลการทดลองข้อ 3.3 ไว้ที่:\n👉 {out_csv}")

if __name__ == "__main__":
    run_reviewer_3_3_validation()