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
# การทดลองตอบ Reviewer ข้อ 3.1: เปรียบเทียบทั้ง 5 โครงสร้างภายใต้ Seeds N=10 ชุดเดียวกัน
# 1. Fixed-width SmeLU (L = -1, W = 2)
# 2. LayerNorm (No-Affine) + SmeLU (beta = 0.4)
# 3. Per-Sample RMSNorm (No-Affine, No Inverse Rescale) + SmeLU (beta = 0.4)
# 4. Per-Sample RMSNorm + SmeLU (beta = 0.4) + Inverse Rescaling (Explicit 3-step)
# 5. Symmetric sigma-AS-RectLU (kappa = 0.8, Fused formulation Eq. 6)
# ==============================================================================

class FixedSmeLU(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta
    def forward(self, x):
        return torch.where(x < -self.beta, torch.zeros_like(x),
               torch.where(x <= self.beta, ((x + self.beta) ** 2) / (4.0 * self.beta), x))

class LayerNormSmeLU(nn.Module):
    def __init__(self, beta=0.4, eps=1e-5):
        super().__init__()
        self.beta = beta
        self.eps = eps
    def forward(self, x):
        dims = tuple(range(1, x.ndim))
        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, keepdim=True, unbiased=False)
        z = (x - mean) / torch.sqrt(var + self.eps)
        return torch.where(z < -self.beta, torch.zeros_like(z),
               torch.where(z <= self.beta, ((z + self.beta) ** 2) / (4.0 * self.beta), z))

class RMSNormSmeLUNoRescale(nn.Module):
    """รายการที่ 1 ของ Reviewer: RMSNorm + SmeLU (โดยไม่มี Inverse Rescaling)"""
    def __init__(self, beta=0.4, eps=1e-5):
        super().__init__()
        self.beta = beta
        self.eps = eps
    def forward(self, x):
        dims = tuple(range(1, x.ndim))
        rms = torch.sqrt(torch.mean(x ** 2, dim=dims, keepdim=True) + self.eps).detach()
        z = x / rms
        return torch.where(z < -self.beta, torch.zeros_like(z),
               torch.where(z <= self.beta, ((z + self.beta) ** 2) / (4.0 * self.beta), z))

class ExplicitRMSNormSmeLURescaled(nn.Module):
    """รายการที่ 2 ของ Reviewer: RMSNorm + SmeLU + Inverse Rescaling (แยก 3 ขั้นตอน)"""
    def __init__(self, beta=0.4, eps=1e-5):
        super().__init__()
        self.beta = beta
        self.eps = eps
    def forward(self, x):
        dims = tuple(range(1, x.ndim))
        rms = torch.sqrt(torch.mean(x ** 2, dim=dims, keepdim=True) + self.eps).detach()
        z = x / rms
        smelu_z = torch.where(z < -self.beta, torch.zeros_like(z),
                  torch.where(z <= self.beta, ((z + self.beta) ** 2) / (4.0 * self.beta), z))
        return smelu_z * rms

class FusedSymmSigmaASRectLU(nn.Module):
    """รายการที่ 3 ของ Reviewer: Symmetric sigma-AS-RectLU (สมการที่ 6 แบบยุบรวม)"""
    def __init__(self, kappa=0.8, eps=1e-5):
        super().__init__()
        self.kappa = kappa
        self.eps = eps
    def forward(self, x):
        dims = tuple(range(1, x.ndim))
        rms = torch.sqrt(torch.mean(x ** 2, dim=dims, keepdim=True) + self.eps).detach()
        half_w = (0.5 * self.kappa) * rms
        return torch.where(x < -half_w, torch.zeros_like(x),
               torch.where(x <= half_w, ((x + half_w) ** 2) / (4.0 * half_w), x))

class PathwayB2DCNN(nn.Module):
    def __init__(self, act_builder):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.dense1 = nn.Linear(2048, 256)
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
        x = x.view(x.size(0), -1)
        x = self.drop(self.act4(self.dense1(x)))
        return self.out(x).squeeze(-1)

def generate_pathway_b_2d(sigma_x=0.22, seed=42):
    rng = np.random.RandomState(seed)
    def make_split(n_total, pos_frac):
        n_pos = int(n_total * pos_frac)
        X = rng.normal(0.0, sigma_x, size=(n_total, 3, 32, 32)).astype(np.float32)
        y = np.zeros(n_total, dtype=np.float32)
        y[:n_pos] = 1.0
        for i in range(n_pos):
            r, c = rng.randint(4, 22), rng.randint(4, 22)
            X[i, 0:2, r:r+6, c:c+6] += 1.12 * sigma_x
        perm = rng.permutation(n_total)
        return torch.tensor(X[perm]), torch.tensor(y[perm])
    return make_split(4000, 0.01), make_split(4000, 0.02), make_split(10000, 0.02)

def run_reviewer_3_1_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = generate_pathway_b_2d(sigma_x=0.22, seed=42)
    te_loader = DataLoader(TensorDataset(X_te, y_te), batch_size=256, shuffle=False)

    configs = [
        ("Fixed-width SmeLU (L=-1, W=2)", lambda: FixedSmeLU(beta=1.0)),
        ("LayerNorm (No-Affine) + SmeLU", lambda: LayerNormSmeLU(beta=0.4)),
        ("RMSNorm (No-Rescale) + SmeLU", lambda: RMSNormSmeLUNoRescale(beta=0.4)),
        ("Explicit RMSNorm + SmeLU + InvRescale", lambda: ExplicitRMSNormSmeLURescaled(beta=0.4)),
        ("Symm. sigma-AS-RectLU (Fused Eq. 6)", lambda: FusedSymmSigmaASRectLU(kappa=0.8)),
    ]

    rows = []
    for name, builder in configs:
        mccs, aurocs, auprcs, times_ms = [], [], [], []
        for seed in range(10):
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = PathwayB2DCNN(builder).to(device)
            for m in [model.conv1, model.conv2, model.conv3, model.dense1, model.out]:
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

            opt = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-8)
            tr_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=256, shuffle=True)

            t0 = time.perf_counter()
            model.train()
            for ep in range(20):
                for xb, yb in tr_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    opt.zero_grad()
                    loss = F.binary_cross_entropy_with_logits(model(xb), yb)
                    loss.backward()
                    opt.step()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            ep_ms = (time.perf_counter() - t0) * 1000.0 / 20.0
            times_ms.append(ep_ms)

            model.eval()
            probs_list = []
            with torch.no_grad():
                for xb, _ in te_loader:
                    probs_list.append(torch.sigmoid(model(xb.to(device))).cpu().numpy())
            probs = np.concatenate(probs_list)
            preds = (probs >= 0.50).astype(int)
            y_np = y_te.numpy().astype(int)
            mcc = float(matthews_corrcoef(y_np, preds)) if np.any(preds) else 0.0
            mccs.append(mcc)
            aurocs.append(float(roc_auc_score(y_np, probs)))
            auprcs.append(float(average_precision_score(y_np, probs)))

        print(f"{name:38s} | MCC(0.50): {np.mean(mccs):.4f} ± {np.std(mccs, ddof=1):.4f} "
              f"| AUROC: {np.mean(aurocs):.4f} | Time: {np.mean(times_ms):.2f} ms/ep")
        rows.append({
            "Configuration": name,
            "MCC_Mean": np.mean(mccs), "MCC_SD": np.std(mccs, ddof=1),
            "AUROC_Mean": np.mean(aurocs), "AUROC_SD": np.std(aurocs, ddof=1),
            "AUPRC_Mean": np.mean(auprcs), "Time_ms_per_epoch": np.mean(times_ms)
        })

    out_csv = pathlib.Path("v9_reviewer_3_1_ablation_10seeds.csv")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nบันทึกผลการทดลองจริงลงไฟล์ {out_csv} เรียบร้อยแล้ว!")

if __name__ == "__main__":
    run_reviewer_3_1_experiment()