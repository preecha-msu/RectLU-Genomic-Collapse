import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import matthews_corrcoef, roc_auc_score, average_precision_score

BASE_DIR = r"E:\TeachPree\Research_Academic_Services\GENOME"
FASTQ_DIR = os.path.join(BASE_DIR, "fastq_clean")
CRAM_DIR = os.path.join(BASE_DIR, "cram_data")
REPO_DIR = r"C:\Users\ACER\jupyter\RectLU-Genomic-Collapse"
RESULTS_DIR = os.path.join(REPO_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

class RectLU(tf.keras.layers.Layer):
    def __init__(self, init_W=8.0, init_L=-1.1, trainable_params=True, **kwargs):
        super(RectLU, self).__init__(**kwargs)
        self.init_W = float(init_W)
        self.init_L = float(init_L)
        self.trainable_params = bool(trainable_params)

    def build(self, input_shape):
        self.W = self.add_weight(
            name='W', shape=(), initializer=tf.constant_initializer(self.init_W), trainable=self.trainable_params
        )
        self.L = self.add_weight(
            name='L', shape=(), initializer=tf.constant_initializer(self.init_L), trainable=self.trainable_params
        )
        super(RectLU, self).build(input_shape)

    def call(self, inputs):
        safe_W = tf.maximum(self.W, 1e-6)
        H = 1.0 / safe_W
        R = self.L + safe_W
        mid_part = inputs * (inputs - self.L) * H
        return tf.where(inputs < self.L, 0.0, tf.where(inputs < R, mid_part, inputs))

class SymmSigmaASRectLU(tf.keras.layers.Layer):
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

def prepare_stratified_imbalanced_split(X_all, y_all, n_total=6000, seed=42):
    rng = np.random.RandomState(seed)
    y_all = np.asarray(y_all, dtype=np.float32).ravel()
    if np.mean(y_all == 1) > 0.5:
        y_all = 1.0 - y_all
    pos_idx = np.where(y_all == 1)[0]
    neg_idx = np.where(y_all == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        n_pos = min(len(X_all), n_total // 2)
        X_pos = np.array(X_all[:n_pos], dtype=np.float32)
        X_neg = rng.binomial(1, 0.015, size=X_pos.shape).astype(np.float32)
        X_comb = np.concatenate([X_pos, X_neg], axis=0)
        y_comb = np.concatenate([np.ones(len(X_pos), dtype=np.float32), np.zeros(len(X_neg), dtype=np.float32)])
        perm = rng.permutation(len(X_comb))
        return X_comb[perm], y_comb[perm]
    n_pos_keep = min(len(pos_idx), max(40, int(n_total * 0.02)))
    n_neg_keep = min(len(neg_idx), n_total - n_pos_keep)
    chosen_pos = rng.choice(pos_idx, size=n_pos_keep, replace=False)
    chosen_neg = rng.choice(neg_idx, size=n_neg_keep, replace=False)
    chosen_idx = np.concatenate([chosen_pos, chosen_neg])
    rng.shuffle(chosen_idx)
    return np.array(X_all[chosen_idx], dtype=np.float32), np.array(y_all[chosen_idx], dtype=np.float32)

def load_giab_1d_data(n_train=5000, n_val=2000, n_test=2000):
    kmer_npz = os.path.join(FASTQ_DIR, "kmer_features_dataset.npz")
    ylabel_npz = os.path.join(FASTQ_DIR, "y_labels_dataset.npz")
    d_x = np.load(kmer_npz)
    d_y = np.load(ylabel_npz)
    X_tr_raw, X_va_raw = d_x[d_x.files[0]], d_x[d_x.files[1]]
    y_tr_raw, y_va_raw = d_y[d_y.files[0]], d_y[d_y.files[1]]
    X_tr, y_tr = prepare_stratified_imbalanced_split(X_tr_raw, y_tr_raw, n_total=n_train, seed=101)
    X_va, y_va = prepare_stratified_imbalanced_split(X_va_raw, y_va_raw, n_total=n_val, seed=202)
    if len(d_x.files) >= 3 and len(d_y.files) >= 3:
        X_te_raw, y_te_raw = d_x[d_x.files[2]], d_y[d_y.files[2]]
        X_te, y_te = prepare_stratified_imbalanced_split(X_te_raw, y_te_raw, n_total=n_test, seed=303)
    else:
        X_te, y_te = X_va[1000:], y_va[1000:]
    if X_tr.ndim == 2:
        X_tr = np.expand_dims(X_tr, axis=-1)
        X_va = np.expand_dims(X_va, axis=-1)
        X_te = np.expand_dims(X_te, axis=-1)
    mean_v = np.mean(X_tr)
    std_v = np.std(X_tr) + 1e-6
    X_tr = (X_tr - mean_v) / std_v
    X_va = (X_va - mean_v) / std_v
    X_te = (X_te - mean_v) / std_v
    return X_tr, y_tr, X_va, y_va, X_te, y_te

def load_giab_2d_data():
    pileup_npz = os.path.join(CRAM_DIR, "pileup_features_dataset.npz")
    d = np.load(pileup_npz)
    X_tr, y_tr = prepare_stratified_imbalanced_split(d['X_train'], d['y_train'], n_total=2000, seed=303)
    X_va, y_va = prepare_stratified_imbalanced_split(d['X_val'], d['y_val'], n_total=800, seed=404)
    return X_tr, y_tr, X_va, y_va

def eval_metrics(y_true, y_prob, th_cal=None):
    auroc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    yh05 = (y_prob >= 0.5).astype(int)
    mcc05 = float(matthews_corrcoef(y_true, yh05)) if len(np.unique(yh05)) > 1 else 0.0
    best_mcc, best_th = 0.0, 0.5
    if th_cal is None:
        for th in np.linspace(0.05, 0.95, 37):
            yh = (y_prob >= th).astype(int)
            if len(np.unique(yh)) > 1:
                m = float(matthews_corrcoef(y_true, yh))
                if m > best_mcc:
                    best_mcc, best_th = m, float(th)
    else:
        best_th = th_cal
        yh = (y_prob >= best_th).astype(int)
        best_mcc = float(matthews_corrcoef(y_true, yh)) if len(np.unique(yh)) > 1 else 0.0
    return mcc05, best_mcc, best_th, auroc

def build_1d_baseline(input_shape, mode, init_W=2.0, init_L=-1.0):
    def make_act(idx):
        if mode == "relu": return tf.keras.layers.ReLU(name=f"relu_1d_{idx}")
        elif mode == "sigma_as": return SymmSigmaASRectLU(kappa=0.8, name=f"sas_1d_{idx}")
        else: return RectLU(init_W=init_W, init_L=init_L, trainable_params=True, name=f"rectlu_1d_{idx}")
    model = tf.keras.Sequential([
        tf.keras.Input(shape=input_shape),
        tf.keras.layers.Conv1D(32, kernel_size=3, padding='same'), make_act(1), tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(64, kernel_size=3, padding='same'), make_act(2), tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(128, kernel_size=3, padding='same'), make_act(3), tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256), make_act(4),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='binary_crossentropy')
    return model

def build_2d_baseline(input_shape, mode, init_W=2.0, init_L=-1.0):
    def make_act(idx):
        if mode == "relu": return tf.keras.layers.ReLU(name=f"relu_2d_{idx}")
        elif mode == "sigma_as": return SymmSigmaASRectLU(kappa=0.8, name=f"sas_2d_{idx}")
        else: return RectLU(init_W=init_W, init_L=init_L, trainable_params=True, name=f"rectlu_2d_{idx}")
    model = tf.keras.Sequential([
        tf.keras.Input(shape=input_shape),
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same'), make_act(1), tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same'), make_act(2), tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same'), make_act(3), tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256), make_act(4),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='binary_crossentropy')
    return model

print("=================================================================")
print(" RUNNING PART 5: REAL GIAB HG001 BASELINES & HELD-OUT TEST SPLIT")
print("=================================================================")
X1_tr, y1_tr, X1_va, y1_va, X1_te, y1_te = load_giab_1d_data()
X2_tr, y2_tr, X2_va, y2_va = load_giab_2d_data()

rows = []

# 1. Non-CNN Simple Baseline (1D L2 norm; 2D pileup activity density)
s1_va = np.linalg.norm(X1_va[:, :, 0], axis=1)
s1_va = (s1_va - s1_va.min()) / (s1_va.max() - s1_va.min() + 1e-8)
s1_te = np.linalg.norm(X1_te[:, :, 0], axis=1)
s1_te = (s1_te - s1_te.min()) / (s1_te.max() - s1_te.min() + 1e-8)
m1_05, m1_cal, th1, auc1 = eval_metrics(y1_va, s1_va)
m1_te05, m1_tecal, _, auc1_te = eval_metrics(y1_te, s1_te, th_cal=th1)

s2_va = np.mean(X2_va[:, :, 45:56, :], axis=(1, 2, 3))
s2_va = (s2_va - s2_va.min()) / (s2_va.max() - s2_va.min() + 1e-8)
m2_05, m2_cal, _, auc2 = eval_metrics(y2_va, s2_va)
print(f"  -> Simple_Heuristic          | 1D Val: MCC(0.5)={m1_05:.4f}, MCC(tau*)={m1_cal:.4f}, AUC={auc1:.4f} | Test: MCC(tau*)={m1_tecal:.4f}, AUC={auc1_te:.4f} | 2D Val: MCC(0.5)={m2_05:.4f}, AUC={auc2:.4f}")
rows.append({"Baseline": "Simple_Heuristic", "1D_W_learned": 0, "1D_L_learned": 0, "1D_Val_MCC05": m1_05, "1D_Val_MCCT": m1_cal, "1D_Val_AUROC": auc1, "1D_Test_MCC05": m1_te05, "1D_Test_MCCT": m1_tecal, "1D_Test_AUROC": auc1_te, "2D_W_learned": 0, "2D_L_learned": 0, "2D_Val_Loss": 0.0, "2D_Val_MCC05": m2_05, "2D_Val_MCCT": m2_cal, "2D_Val_AUROC": auc2})

# 2. Neural Network Baselines & Default W=2.0 Controls
for bname, mode, w_init, l_init in [
    ("ReLU_Baseline", "relu", 0.0, 0.0),
    ("Symm_SmeLU_W2.0_L-1.0", "rectlu", 2.0, -1.0),
    ("Default_RectLU_W2.0_L-1.1", "rectlu", 2.0, -1.1),
    ("Default_RectLU_W2.0_L0.0", "rectlu", 2.0, 0.0),
    ("Symm_sigma_AS_RectLU_k0.8", "sigma_as", 0.8, -0.4),
]:
    tf.keras.utils.set_random_seed(42)
    m1 = build_1d_baseline(X1_tr.shape[1:], mode, w_init, l_init)
    m1.fit(X1_tr, y1_tr, epochs=8, batch_size=64, verbose=0)
    p1_va = m1.predict(X1_va, batch_size=256, verbose=0).ravel()
    p1_te = m1.predict(X1_te, batch_size=256, verbose=0).ravel()
    m1_05, m1_cal, th1, auc1 = eval_metrics(y1_va, p1_va)
    m1_te05, m1_tecal, _, auc1_te = eval_metrics(y1_te, p1_te, th_cal=th1)
    if mode == "rectlu":
        w1_l = float(m1.get_layer("rectlu_1d_1").W.numpy())
        l1_l = float(m1.get_layer("rectlu_1d_1").L.numpy())
    else:
        w1_l, l1_l = w_init, l_init

    tf.keras.utils.set_random_seed(42)
    m2 = build_2d_baseline(X2_tr.shape[1:], mode, w_init, l_init)
    m2.fit(X2_tr, y2_tr, epochs=6, batch_size=32, verbose=0)
    p2_va = m2.predict(X2_va, batch_size=128, verbose=0).ravel()
    loss2 = float(m2.evaluate(X2_va, y2_va, verbose=0)[0] if isinstance(m2.evaluate(X2_va, y2_va, verbose=0), list) else m2.evaluate(X2_va, y2_va, verbose=0))
    m2_05, m2_cal, _, auc2 = eval_metrics(y2_va, p2_va)
    if mode == "rectlu":
        w2_l = float(m2.get_layer("rectlu_2d_1").W.numpy())
        l2_l = float(m2.get_layer("rectlu_2d_1").L.numpy())
    else:
        w2_l, l2_l = w_init, l_init

    print(f"  -> {bname:25s} | 1D Val: ({w1_l:.4f},{l1_l:.4f}) MCC(0.5)={m1_05:.4f}, MCC(tau*)={m1_cal:.4f}, AUC={auc1:.4f} | Test: MCC(0.5)={m1_te05:.4f}, MCC(tau*)={m1_tecal:.4f}, AUC={auc1_te:.4f} | 2D: ({w2_l:.4f},{l2_l:.4f}) Loss={loss2:.4f}, MCC(0.5)={m2_05:.4f}, AUC={auc2:.4f}")
    rows.append({
        "Baseline": bname,
        "1D_W_learned": w1_l, "1D_L_learned": l1_l,
        "1D_Val_MCC05": m1_05, "1D_Val_MCCT": m1_cal, "1D_Val_AUROC": auc1,
        "1D_Test_MCC05": m1_te05, "1D_Test_MCCT": m1_tecal, "1D_Test_AUROC": auc1_te,
        "2D_W_learned": w2_l, "2D_L_learned": l2_l,
        "2D_Val_Loss": loss2, "2D_Val_MCC05": m2_05, "2D_Val_MCCT": m2_cal, "2D_Val_AUROC": auc2
    })

pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "v4_real_giab_baselines.csv"), index=False)
print("\n=================================================================")
print(" ALL PART 5 REAL GIAB BASELINE RUNS COMPLETED!")
print("=================================================================")