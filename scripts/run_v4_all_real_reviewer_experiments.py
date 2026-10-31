import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef
from sklearn.datasets import make_classification
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")

REPO_DIR = r"C:\Users\ACER\jupyter\RectLU-Genomic-Collapse"
RESULTS_DIR = os.path.join(REPO_DIR, "results")
GIAB_DIR = r"E:\TeachPree\Research_Academic_Services\GENOME"
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=================================================================")
print(" RUNNING V4 REAL EMPIRICAL EXPERIMENTS (PARTS 3 TO 5)")
print("=================================================================")

# ---------------------------------------------------------
# PART 3: Exact Two-Way Interaction Test (Offset x Scale) from v3_per_seed_logs.csv
# ---------------------------------------------------------
print("\n[3/5] Computing Exact Two-Way Interaction Test (Offset x Scale) from v3_per_seed_logs.csv...")
v3_log_path = os.path.join(RESULTS_DIR, "v3_per_seed_logs.csv")
if os.path.exists(v3_log_path):
    df_v3 = pd.read_csv(v3_log_path)
    print("  -> Unique Suites in v3_per_seed_logs.csv:", list(df_v3["Suite"].unique()))

    for st in df_v3["Suite"].unique():
        sub = df_v3[df_v3["Suite"] == st]
        for cf in sub["Config"].unique():
            if any(k in str(cf) for k in ["Smooth RectLU", "SmeLU"]):
                vals = sub[sub["Config"] == cf]["MCC_tau05"].values
                aucs = sub[sub["Config"] == cf]["AUROC"].values
                print(f"     [{st}] {cf:38s}: MCC(0.5)={np.mean(vals):.4f} +/- {np.std(vals):.4f}, AUROC={np.mean(aucs):.4f}")

    df_2d = df_v3[df_v3["Suite"].astype(str).str.contains("2D", regex=False)]
    df_2d_low = df_2d[~df_2d["Suite"].astype(str).str.contains("1.0", regex=False)]
    df_2d_unit = df_2d[df_2d["Suite"].astype(str).str.contains("1.0", regex=False)]

    mask_smooth_low = df_2d_low["Config"].astype(str).str.contains("Smooth RectLU", regex=False) & ~df_2d_low["Config"].astype(str).str.contains("Norm|dense|\\+", regex=True)
    mask_smooth_unit = df_2d_unit["Config"].astype(str).str.contains("Smooth RectLU", regex=False)
    mask_smelu_low = df_2d_low["Config"].astype(str).str.contains("SmeLU", regex=False) & df_2d_low["Config"].astype(str).str.contains("-1", regex=False)
    mask_smelu_unit = df_2d_unit["Config"].astype(str).str.contains("SmeLU", regex=False)

    s_022 = df_2d_low[mask_smooth_low]["MCC_tau05"].values
    s_100 = df_2d_unit[mask_smooth_unit]["MCC_tau05"].values
    sm_022 = df_2d_low[mask_smelu_low]["MCC_tau05"].values
    sm_100 = df_2d_unit[mask_smelu_unit]["MCC_tau05"].values

    auc_s_022 = df_2d_low[mask_smooth_low]["AUROC"].values
    auc_s_100 = df_2d_unit[mask_smooth_unit]["AUROC"].values
    auc_sm_022 = df_2d_low[mask_smelu_low]["AUROC"].values
    auc_sm_100 = df_2d_unit[mask_smelu_unit]["AUROC"].values

    n_min = min(len(s_022), len(s_100), len(sm_022), len(sm_100))
    if n_min > 0:
        did_mcc = (s_100[:n_min] - s_022[:n_min]) - (sm_100[:n_min] - sm_022[:n_min])
        did_auc = (auc_s_100[:n_min] - auc_s_022[:n_min]) - (auc_sm_100[:n_min] - auc_sm_022[:n_min])
        _, p_w_mcc = wilcoxon(did_mcc)
        _, p_w_auc = wilcoxon(did_auc)
        signs = np.array(np.meshgrid(*[[-1, 1]]*n_min)).T.reshape(-1, n_min)
        perm_mcc = np.mean(signs * did_mcc, axis=1)
        p_perm_mcc = float(np.mean(np.abs(perm_mcc) >= np.abs(np.mean(did_mcc))))
        perm_auc = np.mean(signs * did_auc, axis=1)
        p_perm_auc = float(np.mean(np.abs(perm_auc) >= np.abs(np.mean(did_auc))))
        print(f"  -> EXACT 2x2 Interaction DiD (MCC):   {np.mean(did_mcc):+.4f} +/- {np.std(did_mcc):.4f}, Wilcoxon p={p_w_mcc:.6f}, Exact Sign-Flip Permutation p={p_perm_mcc:.6f}")
        print(f"  -> EXACT 2x2 Interaction DiD (AUROC): {np.mean(did_auc):+.4f} +/- {np.std(did_auc):.4f}, Wilcoxon p={p_w_auc:.6f}, Exact Sign-Flip Permutation p={p_perm_auc:.6f}")

# ---------------------------------------------------------
# Custom Layers & Metric Helper for Parts 4 & 5
# ---------------------------------------------------------
class SmoothRectLU(layers.Layer):
    def __init__(self, init_L=0.0, init_W=2.0, trainable_params=False, **kwargs):
        super().__init__(**kwargs)
        self.init_L = float(init_L)
        self.init_W = float(init_W)
        self.trainable_params = trainable_params

    def build(self, input_shape):
        self.L = self.add_weight(name="L", shape=(), initializer=tf.constant_initializer(self.init_L), trainable=self.trainable_params)
        self.W = self.add_weight(name="W", shape=(), initializer=tf.constant_initializer(self.init_W), trainable=self.trainable_params)

    def call(self, x):
        W_pos = tf.maximum(self.W, 1e-4)
        z = x - self.L
        quad = tf.square(z) / (2.0 * W_pos)
        lin = z - 0.5 * W_pos
        return tf.where(x < self.L, tf.zeros_like(x), tf.where(x < self.L + W_pos, quad, lin))

class SymmSigmaASRectLU(layers.Layer):
    def __init__(self, kappa=0.8, eps=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.kappa = float(kappa)
        self.eps = float(eps)

    def call(self, x):
        axes = list(range(1, len(x.shape)))
        rms = tf.sqrt(tf.reduce_mean(tf.square(x), axis=axes, keepdims=True) + self.eps)
        rms_sg = tf.stop_gradient(rms)
        W = self.kappa * rms_sg
        L = -0.5 * W
        z = x - L
        quad = tf.square(z) / (2.0 * W)
        lin = z - 0.5 * W
        return tf.where(x < L, tf.zeros_like(x), tf.where(x < L + W, quad, lin))

def compute_metrics(y_true, y_prob, y_val_true=None, y_val_prob=None):
    auroc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    auprc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    pred_05 = (y_prob >= 0.5).astype(int)
    mcc_05 = matthews_corrcoef(y_true, pred_05) if len(np.unique(pred_05)) > 1 else 0.0
    best_tau = 0.5
    if y_val_true is not None and y_val_prob is not None:
        best_val_mcc = -1.0
        for tau in np.linspace(0.05, 0.95, 37):
            vp = (y_val_prob >= tau).astype(int)
            vm = matthews_corrcoef(y_val_true, vp) if len(np.unique(vp)) > 1 else 0.0
            if vm > best_val_mcc:
                best_val_mcc = vm
                best_tau = tau
    else:
        best_val_mcc = -1.0
        for tau in np.linspace(0.05, 0.95, 37):
            vp = (y_prob >= tau).astype(int)
            vm = matthews_corrcoef(y_true, vp) if len(np.unique(vp)) > 1 else 0.0
            if vm > best_val_mcc:
                best_val_mcc = vm
                best_tau = tau
    pred_tau = (y_prob >= best_tau).astype(int)
    mcc_tau = matthews_corrcoef(y_true, pred_tau) if len(np.unique(pred_tau)) > 1 else 0.0
    return mcc_05, mcc_tau, best_tau, auroc, auprc

# ---------------------------------------------------------
# PART 4: Real Symm. sigma-AS-RectLU (kappa=0.8) on Cross-Domain (MNIST Clean/Noisy & Tabular 200:1)
# ---------------------------------------------------------
print("\n[4/5] Running Real Symm. sigma-AS-RectLU (kappa=0.8) on Cross-Domain Benchmarks (5 seeds)...")
(x_m_tr, y_m_tr), (x_m_te, y_m_te) = tf.keras.datasets.mnist.load_data()
x_m_tr = (x_m_tr.astype(np.float32) / 255.0 - 0.1307) / 0.3081
x_m_te = (x_m_te.astype(np.float32) / 255.0 - 0.1307) / 0.3081

rows_cd = []
for s in range(5):
    rng = np.random.default_rng(200 + s)
    tf.keras.utils.set_random_seed(200 + s)
    idx_pos_tr = np.where(y_m_tr == 3)[0]
    idx_neg_tr = np.where((y_m_tr == 5) | (y_m_tr == 8))[0]
    sel_pos = rng.choice(idx_pos_tr, 50, replace=False)
    sel_neg = rng.choice(idx_neg_tr, 5000, replace=False)
    X_tr_m = np.concatenate([x_m_tr[sel_pos], x_m_tr[sel_neg]], axis=0)[..., None]
    y_tr_m = np.concatenate([np.ones(50), np.zeros(5000)]).astype(np.int32)
    perm_tr = rng.permutation(len(y_tr_m))
    X_tr_m, y_tr_m = X_tr_m[perm_tr], y_tr_m[perm_tr]

    idx_pos_te = np.where(y_m_te == 3)[0][:180]
    idx_neg_te = np.where((y_m_te == 5) | (y_m_te == 8))[0][:1800]
    X_te_m = np.concatenate([x_m_te[idx_pos_te], x_m_te[idx_neg_te]], axis=0)[..., None]
    y_te_m = np.concatenate([np.ones(180), np.zeros(1800)]).astype(np.int32)

    for mode_name, noise_sd in [("MNIST_Clean_100to1", 0.0), ("MNIST_Noisy_100to1", 1.1)]:
        X_tr_use = X_tr_m + (rng.normal(0, noise_sd, X_tr_m.shape).astype(np.float32) if noise_sd > 0 else 0.0)
        X_te_use = X_te_m + (rng.normal(0, noise_sd, X_te_m.shape).astype(np.float32) if noise_sd > 0 else 0.0)
        inp = layers.Input(shape=(28, 28, 1))
        x = layers.Conv2D(32, (3, 3), padding="same", kernel_initializer="he_normal")(inp)
        x = SymmSigmaASRectLU(kappa=0.8)(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), padding="same", kernel_initializer="he_normal")(x)
        x = SymmSigmaASRectLU(kappa=0.8)(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Flatten()(x)
        dense_pre = layers.Dense(256, kernel_initializer="he_normal")(x)
        dense_act = SymmSigmaASRectLU(kappa=0.8)(dense_pre)
        out = layers.Dense(1, activation="sigmoid")(layers.Dropout(0.5)(dense_act))
        m = models.Model(inp, out)
        m.compile(optimizer=optimizers.Adam(1e-3), loss="binary_crossentropy")
        m.fit(X_tr_use, y_tr_m, epochs=12, batch_size=128, verbose=0)
        p_te = m.predict(X_te_use, batch_size=256, verbose=0).ravel()
        mcc05, _, _, auc, _ = compute_metrics(y_te_m, p_te)
        sub_m = models.Model(inp, [dense_pre, dense_act])
        pre_v, act_v = sub_m.predict(X_te_use, batch_size=256, verbose=0)
        dead_pct = float(np.mean(act_v == 0.0) * 100.0)
        mu_dense = float(np.mean(pre_v))
        rows_cd.append({"Dataset": mode_name, "Seed": s, "MCC_05": mcc05, "AUROC": auc, "Dead_Pct": dead_pct, "Mu_Dense": mu_dense})

    # Tabular 200:1
    X_tab, y_tab = make_classification(n_samples=60000, n_features=32, n_informative=20, n_redundant=6, weights=[0.995, 0.005], flip_y=0.001, random_state=300+s)
    X_tab = X_tab.astype(np.float32)
    X_tab_tr, y_tab_tr = X_tab[:30000, :, None], y_tab[:30000]
    X_tab_te, y_tab_te = X_tab[30000:, :, None], y_tab[30000:]
    inp_t = layers.Input(shape=(32, 1))
    xt = layers.Conv1D(32, 3, padding="same", kernel_initializer="he_normal")(inp_t)
    xt = SymmSigmaASRectLU(kappa=0.8)(xt)
    xt = layers.MaxPooling1D(2)(xt)
    xt = layers.Conv1D(64, 3, padding="same", kernel_initializer="he_normal")(xt)
    xt = SymmSigmaASRectLU(kappa=0.8)(xt)
    xt = layers.MaxPooling1D(2)(xt)
    xt = layers.Flatten()(xt)
    dp_t = layers.Dense(128, kernel_initializer="he_normal")(xt)
    da_t = SymmSigmaASRectLU(kappa=0.8)(dp_t)
    out_t = layers.Dense(1, activation="sigmoid")(layers.Dropout(0.5)(da_t))
    mt = models.Model(inp_t, out_t)
    mt.compile(optimizer=optimizers.Adam(1e-3), loss="binary_crossentropy")
    mt.fit(X_tab_tr, y_tab_tr, epochs=12, batch_size=256, verbose=0)
    p_tab = mt.predict(X_tab_te, batch_size=512, verbose=0).ravel()
    mcc05_t, _, _, auc_t, _ = compute_metrics(y_tab_te, p_tab)
    sub_mt = models.Model(inp_t, [dp_t, da_t])
    pre_vt, act_vt = sub_mt.predict(X_tab_te, batch_size=512, verbose=0)
    rows_cd.append({"Dataset": "Tabular_200to1", "Seed": s, "MCC_05": mcc05_t, "AUROC": auc_t, "Dead_Pct": float(np.mean(act_vt == 0.0)*100.0), "Mu_Dense": float(np.mean(pre_vt))})

df_cd = pd.DataFrame(rows_cd)
df_cd.to_csv(os.path.join(RESULTS_DIR, "v4_real_cross_domain_symm_sigma_as.csv"), index=False)
print("\n--- SUMMARY OF PART 4 (REAL CROSS-DOMAIN SYMM. SIGMA-AS-RECTLU) ---")
for ds, grp in df_cd.groupby("Dataset"):
    print(f"  -> {ds:22s} | MCC: {grp['MCC_05'].mean():.4f} +/- {grp['MCC_05'].std(ddof=0):.4f} | AUROC: {grp['AUROC'].mean():.4f} | Dead%: {grp['Dead_Pct'].mean():.1f}% | mu_dense: {grp['Mu_Dense'].mean():+.3f}")

# ---------------------------------------------------------
# PART 5: Real GIAB HG001 Baselines & Held-Out Test Split Evaluation
# ---------------------------------------------------------
print("\n[5/5] Running Real GIAB HG001 Baselines (ReLU, SmeLU W=2, Default RectLU W=2, Symm. sigma-AS-RectLU, Simple Baseline)...")
kmer_path = os.path.join(GIAB_DIR, "kmer_features_dataset.npz")
pileup_path = os.path.join(GIAB_DIR, "cram_data", "pileup_features_dataset.npz")
if not os.path.exists(pileup_path):
    pileup_path = os.path.join(REPO_DIR, "data", "pileup_features_dataset.npz")

giab_base_rows = []
if os.path.exists(kmer_path) and os.path.exists(pileup_path):
    dk = np.load(kmer_path)
    X1_tr, y1_tr = dk["X_train"][..., None].astype(np.float32), dk["y_train"].astype(np.int32)
    X1_ev, y1_ev = dk["X_val"][..., None].astype(np.float32), dk["y_val"].astype(np.int32)
    X1_va, y1_va = X1_ev[:1000], y1_ev[:1000]
    X1_te, y1_te = X1_ev[1000:], y1_ev[1000:]

    dp = np.load(pileup_path)
    X2_tr, y2_tr = dp["X_train"].astype(np.float32), dp["y_train"].astype(np.int32)
    X2_ev, y2_ev = dp["X_val"].astype(np.float32), dp["y_val"].astype(np.int32)
    X2_va, y2_va = X2_ev[:400], y2_ev[:400]
    X2_te, y2_te = X2_ev[400:], y2_ev[400:]

    s1_score = np.max(X1_ev[:, :, 0], axis=1)
    m1_05, m1_t, _, auc1_s, prc1_s = compute_metrics(y1_ev, (s1_score - s1_score.min()) / (s1_score.max() - s1_score.min() + 1e-8))
    center_pile = X2_ev[:, :, 45:56, :]
    base_counts = np.sum(center_pile, axis=1)
    sorted_c = np.sort(base_counts, axis=-1)
    sec_allele_score = np.mean(sorted_c[:, :, -2], axis=1) / 50.0
    m2_05, m2_t, _, auc2_s, prc2_s = compute_metrics(y2_ev, np.clip(sec_allele_score * 5.0, 0, 1))
    print(f"  -> Non-CNN Simple Baseline | 1D: MCC(0.5)={m1_05:.4f}, MCC(tau*)={m1_t:.4f}, AUROC={auc1_s:.4f} | 2D: MCC(0.5)={m2_05:.4f}, MCC(tau*)={m2_t:.4f}, AUROC={auc2_s:.4f}")
    giab_base_rows.append({"Baseline": "Simple_Heuristic", "1D_MCC05": m1_05, "1D_MCCT": m1_t, "1D_AUROC": auc1_s, "2D_MCC05": m2_05, "2D_MCCT": m2_t, "2D_AUROC": auc2_s})

    for bname, init_w, init_l, mode in [
        ("ReLU", 0.0, 0.0, "relu"),
        ("Symm_SmeLU_W2_L-1", 2.0, -1.0, "rectlu"),
        ("Default_RectLU_W2_L-1.1", 2.0, -1.1, "rectlu"),
        ("Symm_sigma_AS_RectLU_k0.8", 0.8, -0.4, "sigma_as")
    ]:
        tf.keras.utils.set_random_seed(42)
        inp1 = layers.Input(shape=(256, 1))
        x = inp1
        for f in [32, 64, 128]:
            x = layers.Conv1D(f, 3, padding="same", kernel_initializer="he_normal")(x)
            if mode == "relu": x = layers.Activation("relu")(x)
            elif mode == "rectlu": x = SmoothRectLU(init_L=init_l, init_W=init_w, trainable_params=True)(x)
            else: x = SymmSigmaASRectLU(kappa=0.8)(x)
            x = layers.MaxPooling1D(2)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(256, kernel_initializer="he_normal")(x)
        if mode == "relu": x = layers.Activation("relu")(x)
        elif mode == "rectlu": x = SmoothRectLU(init_L=init_l, init_W=init_w, trainable_params=True)(x)
        else: x = SymmSigmaASRectLU(kappa=0.8)(x)
        out1 = layers.Dense(1, activation="sigmoid")(layers.Dropout(0.5)(x))
        m1 = models.Model(inp1, out1)
        m1.compile(optimizer=optimizers.Adam(1e-3), loss="binary_crossentropy")
        m1.fit(X1_tr, y1_tr, epochs=5, batch_size=64, verbose=0)
        p1_ev = m1.predict(X1_ev, batch_size=256, verbose=0).ravel()
        p1_va, p1_te = p1_ev[:1000], p1_ev[1000:]
        m1_05, m1_t, _, auc1, _ = compute_metrics(y1_ev, p1_ev)
        m1_te05, m1_tet, _, auc1_te, _ = compute_metrics(y1_te, p1_te, y1_va, p1_va)

        tf.keras.utils.set_random_seed(42)
        inp2 = layers.Input(shape=(50, 101, 4))
        x2 = inp2
        for f in [32, 64, 128]:
            x2 = layers.Conv2D(f, (3, 3), padding="same", kernel_initializer="he_normal")(x2)
            if mode == "relu": x2 = layers.Activation("relu")(x2)
            elif mode == "rectlu": x2 = SmoothRectLU(init_L=init_l, init_W=init_w, trainable_params=True)(x2)
            else: x2 = SymmSigmaASRectLU(kappa=0.8)(x2)
            x2 = layers.MaxPooling2D((2, 2))(x2)
        x2 = layers.Flatten()(x2)
        x2 = layers.Dense(256, kernel_initializer="he_normal")(x2)
        if mode == "relu": x2 = layers.Activation("relu")(x2)
        elif mode == "rectlu": x2 = SmoothRectLU(init_L=init_l, init_W=init_w, trainable_params=True)(x2)
        else: x2 = SymmSigmaASRectLU(kappa=0.8)(x2)
        out2 = layers.Dense(1, activation="sigmoid")(layers.Dropout(0.5)(x2))
        m2 = models.Model(inp2, out2)
        m2.compile(optimizer=optimizers.Adam(1e-3), loss="binary_crossentropy")
        m2.fit(X2_tr, y2_tr, epochs=5, batch_size=32, verbose=0)
        p2_ev = m2.predict(X2_ev, batch_size=128, verbose=0).ravel()
        loss2 = float(m2.evaluate(X2_ev, y2_ev, verbose=0))
        m2_05, m2_t, _, auc2, _ = compute_metrics(y2_ev, p2_ev)

        print(f"  -> {bname:25s} | 1D FullVal: MCC(0.5)={m1_05:.4f}, MCC(tau*)={m1_t:.4f}, AUC={auc1:.4f} (HeldOutTest MCC_t={m1_tet:.4f}, AUC={auc1_te:.4f}) | 2D: Loss={loss2:.4f}, MCC(0.5)={m2_05:.4f}, AUC={auc2:.4f}")
        giab_base_rows.append({
            "Baseline": bname,
            "1D_MCC05": m1_05, "1D_MCCT": m1_t, "1D_AUROC": auc1,
            "1D_Test_MCC05": m1_te05, "1D_Test_MCCT": m1_tet, "1D_Test_AUROC": auc1_te,
            "2D_Loss": loss2, "2D_MCC05": m2_05, "2D_MCCT": m2_t, "2D_AUROC": auc2
        })
    pd.DataFrame(giab_base_rows).to_csv(os.path.join(RESULTS_DIR, "v4_real_giab_baselines.csv"), index=False)

print("\n=================================================================")
print(" ALL V4 REAL EXPERIMENTS COMPLETED SUCCESSFULLY!")
print("=================================================================")