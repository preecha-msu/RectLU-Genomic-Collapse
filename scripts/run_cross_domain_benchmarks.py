import os
import time
import tarfile
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
os.makedirs("results", exist_ok=True)
os.makedirs("./data", exist_ok=True)
print("Using device:", device)


class SmoothRectLU(nn.Module):
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
    def __init__(self, kappa=0.5, gamma=0.0, eps=1e-5):
        super().__init__()
        self.kappa = float(kappa)
        self.gamma = float(gamma)
        self.eps = float(eps)

    def forward(self, x):
        var = torch.var(x, unbiased=False)
        sigma = torch.sqrt(var + self.eps).detach()
        W = torch.clamp(self.kappa * sigma, min=1e-3)
        L = -self.gamma * sigma
        z = torch.zeros_like(x)
        q = ((x - L) ** 2) / (2.0 * W)
        lin = x - L - 0.5 * W
        return torch.where(x < L, z, torch.where(x < L + W, q, lin))


def get_act(name):
    if name == "ReLU":
        return nn.ReLU()
    if name == "GELU":
        return nn.GELU()
    if name == "SmeLU":
        return SmoothRectLU(L=-0.5, W=1.0)
    if name == "SmoothRectLU_W2":
        return SmoothRectLU(L=0.0, W=2.0)
    if name == "SmoothRectLU_W05":
        return SmoothRectLU(L=0.0, W=0.5)
    if name == "SigmaASRectLU":
        return SigmaASRectLU(kappa=0.5, gamma=0.0)
    raise ValueError(name)


print("\n=== PART 1: Monte-Carlo Gain (N=4,000,000) ===")
torch.manual_seed(42)
x_mc = torch.randn(4_000_000, device=device)
mc_list = [
    (0.0, 0.5, "Smooth RectLU (L=0, W=0.5)"),
    (0.0, 1.0, "Smooth RectLU (L=0, W=1.0)"),
    (0.0, 2.0, "Smooth RectLU (L=0, W=2.0)"),
    (0.0, 4.0, "Smooth RectLU (L=0, W=4.0)"),
    (-0.5, 1.0, "SmeLU (L=-0.5, W=1.0)"),
    (-1.0, 2.0, "SmeLU (L=-1.0, W=2.0)")
]
for L_v, W_v, lbl in mc_list:
    fn = SmoothRectLU(L=L_v, W=W_v)
    y_mc = fn(x_mc)
    z_mc = torch.zeros_like(x_mc)
    o_mc = torch.ones_like(x_mc)
    s_mc = (x_mc - L_v) / W_v
    dy = torch.where(x_mc < L_v, z_mc, torch.where(x_mc < L_v + W_v, s_mc, o_mc))
    fg = (y_mc.pow(2).mean() / x_mc.pow(2).mean()).item()
    bg = dy.pow(2).mean().item()
    print(f"{lbl:28s} | FwdGain={fg:.4f} | BwdGain={bg:.4f}")


class CNN2D(nn.Module):
    def __init__(self, c_act, d_act):
        super().__init__()
        self.c1 = nn.Conv2d(3, 64, 3, padding=1)
        self.a1 = get_act(c_act)
        self.c2 = nn.Conv2d(64, 128, 3, padding=1)
        self.a2 = get_act(c_act)
        self.c3 = nn.Conv2d(128, 256, 3, padding=1)
        self.a3 = get_act(c_act)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.ad = get_act(d_act)
        self.drop = nn.Dropout(0.5)
        self.out = nn.Linear(512, 1)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.pool(self.a1(self.c1(x)))
        x = self.pool(self.a2(self.c2(x)))
        x = self.pool(self.a3(self.c3(x)))
        x = x.view(x.size(0), -1)
        pre = self.fc1(x)
        h = self.drop(self.ad(pre))
        return self.out(h).squeeze(-1), pre


class CNN1D(nn.Module):
    def __init__(self, in_len, c_act, d_act):
        super().__init__()
        self.c1 = nn.Conv1d(1, 64, 3, padding=1)
        self.a1 = get_act(c_act)
        self.c2 = nn.Conv1d(64, 128, 3, padding=1)
        self.a2 = get_act(c_act)
        self.c3 = nn.Conv1d(128, 256, 3, padding=1)
        self.a3 = get_act(c_act)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(256 * (in_len // 8), 512)
        self.ad = get_act(d_act)
        self.drop = nn.Dropout(0.5)
        self.out = nn.Linear(512, 1)
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.pool(self.a1(self.c1(x)))
        x = self.pool(self.a2(self.c2(x)))
        x = self.pool(self.a3(self.c3(x)))
        x = x.view(x.size(0), -1)
        pre = self.fc1(x)
        h = self.drop(self.ad(pre))
        return self.out(h).squeeze(-1), pre


def get_2d_vision_benchmark():
    ext = "./data/cifar-10-batches-py"
    tgz = "./data/cifar-10-python.tar.gz"
    if not os.path.exists(ext):
        # Remove partial file first so curl does not fail with code 33
        if os.path.exists(tgz):
            os.remove(tgz)
        url = "https://ossci-datasets.s3.amazonaws.com/cifar-10-python.tar.gz"
        print("Downloading clean CIFAR-10 from AWS S3 mirror...")
        cmd = f'curl.exe -L --retry 3 -o "{tgz}" "{url}"'
        if os.system(cmd) == 0 and os.path.exists(tgz) and os.path.getsize(tgz) > 160_000_000:
            with tarfile.open(tgz, "r:gz") as tar:
                tar.extractall(path="./data")

    if os.path.exists(ext):
        print("Loaded CIFAR-10 (100:1 Imbalance) successfully!")
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261))
        ])
        tr = datasets.CIFAR10(root="./data", train=True, download=False, transform=tf)
        te = datasets.CIFAR10(root="./data", train=False, download=False, transform=tf)
        dom_name = "CIFAR10_100to1"
    else:
        print("Using MNIST (100:1 Imbalance, resized 32x32x3) via fast AWS S3...")
        tf = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))
        ])
        tr = datasets.MNIST(root="./data", train=True, download=True, transform=tf)
        te = datasets.MNIST(root="./data", train=False, download=True, transform=tf)
        dom_name = "MNIST_100to1"

    def sub(ds, n_pos, n_neg):
        rng = np.random.RandomState(42)
        t = np.array(ds.targets)
        p_i = rng.choice(np.where(t == 0)[0], n_pos, replace=False)
        n_i = rng.choice(np.where(t == 1)[0], n_neg, replace=False)
        idx = np.concatenate([p_i, n_i])
        rng.shuffle(idx)
        xs = torch.stack([ds[i][0] for i in idx])
        ys = torch.tensor([1.0 if int(t[i]) == 0 else 0.0 for i in idx], dtype=torch.float32)
        return TensorDataset(xs, ys)

    return sub(tr, 50, 5000), sub(te, 100, 1000), dom_name


def get_tabular():
    X, y = make_classification(
        n_samples=40000, n_features=32, n_informative=20,
        n_redundant=6, weights=[0.995, 0.005], random_state=42
    )
    y = y.astype(np.float32)
    sc = StandardScaler()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    X_tr = sc.fit_transform(X_tr)
    X_te = sc.transform(X_te)
    tr_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32).unsqueeze(1), torch.tensor(y_tr))
    te_ds = TensorDataset(torch.tensor(X_te, dtype=torch.float32).unsqueeze(1), torch.tensor(y_te))
    return tr_ds, te_ds, 32


CFGS = [
    ("ReLU", "ReLU", "ReLU"),
    ("GELU", "GELU", "GELU"),
    ("SmeLU (L=-0.5, W=1.0)", "SmeLU", "SmeLU"),
    ("Smooth RectLU (L=0, W=2.0)", "SmoothRectLU_W2", "SmoothRectLU_W2"),
    ("Smooth RectLU (L=0, W=0.5)", "SmoothRectLU_W05", "SmoothRectLU_W05"),
    ("Hybrid (Smooth-Conv+ReLU-Dense)", "SmoothRectLU_W2", "ReLU"),
    ("Proposed sigma-AS-RectLU", "SigmaASRectLU", "SigmaASRectLU")
]


def run_exp(dom, tr_ds, te_ds, is_2d=True, in_len=32, epochs=20):
    rows = []
    tr_ld = DataLoader(tr_ds, batch_size=128, shuffle=True)
    te_ld = DataLoader(te_ds, batch_size=256, shuffle=False)
    for lbl, ca, da in CFGS:
        mccs, aucs, aprcs, deads, mus = [], [], [], [], []
        for s in range(1, 11):
            torch.manual_seed(s)
            np.random.seed(s)
            m = CNN2D(ca, da) if is_2d else CNN1D(in_len, ca, da)
            m = m.to(device)
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
            mcc = matthews_corrcoef(yt, yh) if len(np.unique(yh)) > 1 else 0.0
            auc = roc_auc_score(yt, yp)
            ap = average_precision_score(yt, yp)
            dd = np.mean(np.mean(pm < 0.0, axis=0) > 0.999) * 100.0
            mu = float(np.mean(pm))
            tn, fp, fn, tp = confusion_matrix(yt, yh, labels=[0, 1]).ravel()
            mccs.append(mcc)
            aucs.append(auc)
            aprcs.append(ap)
            deads.append(dd)
            mus.append(mu)
            rows.append({
                "Domain": dom, "Config": lbl, "Seed": s,
                "TP": tp, "FP": fp, "TN": tn, "FN": fn,
                "MCC": mcc, "AUROC": auc, "AUPRC": ap,
                "Dead_Pct": dd, "Mu_Dense": mu
            })
        print(f"[{dom}] {lbl:32s} | MCC={np.mean(mccs):.4f}+/-{np.std(mccs):.4f} "
              f"| AUROC={np.mean(aucs):.4f} | Dead={np.mean(deads):5.1f}% | mu={np.mean(mus):+.3f}")
    return rows


if __name__ == "__main__":
    t0 = time.time()
    all_r = []
    print("\n=== PART 2: Imbalanced 2D Vision (100:1, N=10 Seeds) ===")
    c_tr, c_te, dom_label = get_2d_vision_benchmark()
    all_r.extend(run_exp(dom_label, c_tr, c_te, True, 32, 20))

    print("\n=== PART 3: Imbalanced Tabular (200:1, N=10 Seeds) ===")
    t_tr, t_te, flen = get_tabular()
    all_r.extend(run_exp("Tabular_200to1", t_tr, t_te, False, flen, 15))

    pd.DataFrame(all_r).to_csv("results/cross_domain_per_seed_results.csv", index=False)
    print(f"\nDone in {(time.time() - t0)/60:.2f} min! Saved to results/cross_domain_per_seed_results.csv")