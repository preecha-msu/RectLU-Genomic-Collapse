import time
import csv
import pathlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import matthews_corrcoef, roc_auc_score, average_precision_score

# ==============================================================================
# การทดลองตอบ Reviewer ข้อ 3.4 (Experiment A): Causal q_4 Variance Intervention
# ทดสอบบน Smooth RectLU (L=0, W=2) โดยบังคับค่า q_4 = Var(x_4) ที่ชั้น Dense1:
#   Condition 1: Unclamped Smooth RectLU (L=0, W=2) [Baseline]
#   Condition 2: Clamped q_4 = 0.55 (บังคับให้อยู่ในย่าน Trapped Regime [0.46, 0.75])
#   Condition 3: Clamped q_4 = 3.70 (บังคับให้อยู่ในย่าน Escaped Regime [3.54, 3.90])
# ==============================================================================

class SmoothRectLU(nn.Module):
    def __init__(self, L=0.0, W=2.0):
        super().__init__()
        self.L = L
        self.W = W

    def forward(self, x):
        z = x - self.L
        quad = (z ** 2) / (2.0 * self.W)
        lin = z - 0.5 * self.W
        return torch.where(x < self.L, torch.zeros_like(x),
               torch.where(x < self.L + self.W, quad, lin))

class CausalIntervention2DCNN(nn.Module):
    def __init__(self, in_channels, flat_dim, clamp_q4=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.dense1 = nn.Linear(flat_dim, 256)
        self.drop = nn.Dropout(0.5)
        self.out = nn.Linear(256, 1)
        self.act = SmoothRectLU(L=0.0, W=2.0)
        self.clamp_q4 = clamp_q4

    def forward(self, x, return_q4=False):
        a1 = F.max_pool2d(self.act(self.conv1(x)), 2)
        a2 = F.max_pool2d(self.act(self.conv2(a1)), 2)
        a3 = F.max_pool2d(self.act(self.conv3(a2)), 2)
        flat = a3.reshape(a3.size(0), -1)
        x4 = self.dense1(flat)

        # Direct Causal Intervention on fourth-layer pre-activation variance q_4
        if self.clamp_q4 is not None:
            var_x4 = torch.var(x4, dim=1, keepdim=True, unbiased=False)
            x4 = x4 * torch.sqrt(self.clamp_q4 / (var_x4 + 1e-8))

        q4_val = float(torch.var(x4, unbiased=False).detach().item())
        a4 = self.act(x4)
        logits = self.out(self.drop(a4)).squeeze(-1)
        if return_q4:
            return logits, q4_val
        return logits

def load_real_giab_or_motif():
    """ค้นหาไฟล์ข้อมูลจริง pileup_features_dataset.npz ก่อนเพื่อให้ตรงกับ Table 9 100%"""
    candidates = [
        pathlib.Path("data/pileup_features_dataset.npz"),
        pathlib.Path(r"E:\TeachPree\Research_Academic_Services\GENOME\data\pileup_features_dataset.npz"),
        pathlib.Path(r"E:\TeachPree\Research_Academic_Services\GENOME\draft\CIBM_Submission_2026_V7\data\pileup_features_dataset.npz"),
    ]
    for p in candidates:
        if p.exists():
            print(f"โหลดข้อมูลจริงจาก: {p}")
            data = np.load(p)
            X = data[list(data.keys())[0]].astype(np.float32)
            # จัดรูปเป็น (N, C, H, W) และคูณสเกลต่ำ 0.25x ตามโปรโตคอล Table 9 Part A
            if X.ndim == 4 and X.shape[-1] == 4:
                X = np.transpose(X, (0, 3, 1, 2))
            X = X * 0.25
            y_path = p.parent / "y_labels_dataset.npz"
            if y_path.exists():
                y_data = np.load(y_path)
                y = y_data[list(y_data.keys())[0]].astype(np.float32)
            else:
                y = data[list(data.keys())[1]].astype(np.float32)
            # แบ่ง Disjoint Genomic Blocks: 1320 Train / 660 Val / 660 Test ตาม Table 2 & 9
            X_tr, y_tr = torch.tensor(X[:1320]), torch.tensor(y[:1320])
            X_te, y_te = torch.tensor(X[1980:2640]), torch.tensor(y[1980:2640])
            in_ch = X.shape[1]
            flat_dim = 128 * (X.shape[2] // 8) * (X.shape[3] // 8)
            return (X_tr, y_tr), (X_te, y_te), in_ch, flat_dim, "Real_GIAB_2D_0.25x"

    # กรณีรันนอกโฟลเดอร์ data ให้ใช้ Pathway B 2D Spatial Motif Generator (sigma_x = 0.22, mimic spikes 1.5%)
    print("สร้างชุดข้อมูล Pathway B 2D Spatial Motif (sigma_x = 0.22, 100:1 imbalance, with 1.5% mimic spikes)")
    rng = np.random.RandomState(42)
    def make_split(n_total, pos_frac):
        n_pos = int(n_total * pos_frac)
        n_neg = n_total - n_pos
        X = rng.normal(0.0, 0.22, size=(n_total, 3, 32, 32)).astype(np.float32)
        y = np.zeros(n_total, dtype=np.float32)
        y[:n_pos] = 1.0
        for i in range(n_pos):
            r, c = rng.randint(4, 22), rng.randint(4, 22)
            X[i, 0:2, r:r+6, c:c+6] += 0.92 * 0.22
        # 1.5% majority negative mimic spikes ตาม Section 4.1
        n_mimic = int(0.015 * n_neg)
        for i in range(n_pos, n_pos + n_mimic):
            r, c = rng.randint(4, 22), rng.randint(4, 22)
            X[i, 0:2, r:r+6, c:c+6] += 0.84 * 0.22
        perm = rng.permutation(n_total)
        return torch.tensor(X[perm]), torch.tensor(y[perm])
    return make_split(4000, 0.01), make_split(10000, 0.02), 3, 2048, "PathwayB_2D_Motif_0.22"

def run_causal_q4_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (X_tr, y_tr), (X_te, y_te), in_ch, flat_dim, dataset_tag = load_real_giab_or_motif()
    te_loader = DataLoader(TensorDataset(X_te, y_te), batch_size=256, shuffle=False)

    interventions = [
        ("Unclamped Smooth RectLU (L=0, W=2)", None),
        ("Clamped Inside Ramp (q_4 = 0.55)", 0.55),
        ("Clamped Past Ramp (q_4 = 3.70)", 3.70),
    ]

    summary_rows = []
    print(f"\n=== รันทดสอบข้อ 3.4 Causal q_4 Intervention บน {dataset_tag} (N=10 Seeds) ===")
    for label, clamp_val in interventions:
        grad_w1_ep0, grad_w4_ep0, q4_ep20 = [], [], []
        mccs, aurocs, auprcs, tps, fps, fns = [], [], [], [], [], []
        escaped_count = 0

        for seed in range(10):
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = CausalIntervention2DCNN(in_ch, flat_dim, clamp_q4=clamp_val).to(device)
            for m in [model.conv1, model.conv2, model.conv3, model.dense1, model.out]:
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

            tr_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=256, shuffle=True)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-8)

            # วัด Gradient Norm จริงที่ Epoch 0 (Step แรกสุดก่อนปรับค่าน้ำหนัก)
            xb0, yb0 = next(iter(tr_loader))
            xb0, yb0 = xb0.to(device), yb0.to(device)
            opt.zero_grad()
            logits0, _ = model(xb0, return_q4=True)
            loss0 = F.binary_cross_entropy_with_logits(logits0, yb0)
            loss0.backward()
            grad_w1_ep0.append(float(model.conv1.weight.grad.norm(2).item()))
            grad_w4_ep0.append(float(model.dense1.weight.grad.norm(2).item()))

            # ฝึก 20 Epochs ตามโปรโตคอลหลัก
            model.train()
            for ep in range(20):
                for xb, yb in tr_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    opt.zero_grad()
                    loss = F.binary_cross_entropy_with_logits(model(xb), yb)
                    loss.backward()
                    opt.step()

            # วัดผลจริงบน Test Set (ไม่ใช้ Bootstrap)
            model.eval()
            probs_list, q4_batch_list = [], []
            with torch.no_grad():
                for xb, _ in te_loader:
                    lg, q4_b = model(xb.to(device), return_q4=True)
                    probs_list.append(torch.sigmoid(lg).cpu().numpy())
                    q4_batch_list.append(q4_b)

            probs = np.concatenate(probs_list)
            preds = (probs >= 0.50).astype(int)
            y_np = y_te.numpy().astype(int)

            tp = int(np.sum((preds == 1) & (y_np == 1)))
            fp = int(np.sum((preds == 1) & (y_np == 0)))
            fn = int(np.sum((preds == 0) & (y_np == 1)))
            mcc = float(matthews_corrcoef(y_np, preds)) if (tp + fp > 0) else 0.0
            if mcc > 0.10:
                escaped_count += 1

            mccs.append(mcc)
            aurocs.append(float(roc_auc_score(y_np, probs)))
            auprcs.append(float(average_precision_score(y_np, probs)))
            q4_ep20.append(float(np.mean(q4_batch_list)))
            tps.append(tp); fps.append(fp); fns.append(fn)

        row = {
            "Intervention": label,
            "Ep0_Grad_Conv1_Mean": np.mean(grad_w1_ep0),
            "Ep0_Grad_Conv1_SD": np.std(grad_w1_ep0, ddof=1),
            "Ep20_q4_Mean": np.mean(q4_ep20),
            "Escaped_Seeds": f"{escaped_count}/10",
            "MCC_050_Mean": np.mean(mccs),
            "MCC_050_SD": np.std(mccs, ddof=1),
            "AUROC_Mean": np.mean(aurocs),
            "AUROC_SD": np.std(aurocs, ddof=1),
            "AUPRC_Mean": np.mean(auprcs),
            "Mean_TP_FP_FN": f"{np.mean(tps):.1f}/{np.mean(fps):.1f}/{np.mean(fns):.1f}"
        }
        summary_rows.append(row)
        print(f"{label:36s} | Ep0 ||g_W1||: {row['Ep0_Grad_Conv1_Mean']:.2e} "
              f"| Escaped: {row['Escaped_Seeds']} "
              f"| MCC(0.50): {row['MCC_050_Mean']:.4f} ± {row['MCC_050_SD']:.4f} "
              f"| AUROC: {row['AUROC_Mean']:.4f} | TP/FP/FN: {row['Mean_TP_FP_FN']}")

    out_csv = pathlib.Path("v9_reviewer_3_4_causal_q4_intervention_10seeds.csv")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nบันทึกผลการทดลองข้อ 3.4 ลงไฟล์ {out_csv} เรียบร้อยแล้ว!")

if __name__ == "__main__":
    run_causal_q4_experiment()