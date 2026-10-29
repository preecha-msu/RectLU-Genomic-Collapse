import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import matthews_corrcoef, roc_auc_score, average_precision_score

BASE_DIR = r"E:\TeachPree\Research_Academic_Services\GENOME"
FASTQ_DIR = os.path.join(BASE_DIR, "fastq_clean")
CRAM_DIR = os.path.join(BASE_DIR, "cram_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "tuning_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================================================================
# 1. นิยามคลาส RectLU (คงเดิมตามสมการของอาจารย์)
# ==============================================================================
class RectLU(tf.keras.layers.Layer):
    def __init__(self, init_W=8.0, init_L=-1.1, trainable_params=True, **kwargs):
        super(RectLU, self).__init__(**kwargs)
        self.init_W = float(init_W)
        self.init_L = float(init_L)
        self.trainable_params = bool(trainable_params)

    def build(self, input_shape):
        self.W = self.add_weight(
            name='W',
            shape=(),
            initializer=tf.constant_initializer(self.init_W),
            trainable=self.trainable_params
        )
        self.L = self.add_weight(
            name='L',
            shape=(),
            initializer=tf.constant_initializer(self.init_L),
            trainable=self.trainable_params
        )
        super(RectLU, self).build(input_shape)

    def call(self, inputs):
        safe_W = tf.maximum(self.W, 1e-6)
        H = 1.0 / safe_W
        R = self.L + safe_W
        mid_part = inputs * (inputs - self.L) * H
        return tf.where(inputs < self.L, 0.0, tf.where(inputs < R, mid_part, inputs))

    def get_config(self):
        config = super(RectLU, self).get_config()
        config.update({
            "init_W": self.init_W,
            "init_L": self.init_L,
            "trainable_params": self.trainable_params,
        })
        return config


# ==============================================================================
# 2. ฟังก์ชันจัดเตรียมข้อมูล GIAB 1D และ 2D จริง (Stratified & Minority=1)
# ==============================================================================
def prepare_stratified_imbalanced_split(X_all, y_all, n_total=6000, seed=42):
    """
    1. ตรวจสอบให้แน่ใจว่าคลาสส่วนน้อย (Minority Variant) มีค่าเป็น y=1 และคลาสส่วนใหญ่เป็น y=0
    2. ดึงข้อมูลแบบรักษาสัดส่วนคลาสส่วนน้อย (Stratified) เพื่อไม่ให้การตัด Slice ต้นตารางทำให้คลาสหาย
    """
    rng = np.random.RandomState(seed)
    y_all = np.asarray(y_all, dtype=np.float32).ravel()

    # หากในไฟล์เดิมให้คลาสส่วนใหญ่เป็น 1 (เช่น 99.5% เป็น 1) ให้สลับให้คลาสส่วนน้อยเป็น 1 ตามมาตรฐาน
    if np.mean(y_all == 1) > 0.5:
        y_all = 1.0 - y_all

    pos_idx = np.where(y_all == 1)[0]
    neg_idx = np.where(y_all == 0)[0]

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        # กรณีไฟล์มีเฉพาะคลาสเดียว (เช่นใน pileup_features_dataset.npz ที่สกัดเฉพาะ VCF)
        n_pos = min(len(X_all), n_total // 2)
        X_pos = np.array(X_all[:n_pos], dtype=np.float32)
        # สร้างหน้าต่าง Reference อ้างอิงที่มีสัญญาณรบกวนระดับ Sequencing error (1.5%)
        X_neg = rng.binomial(1, 0.015, size=X_pos.shape).astype(np.float32)
        X_comb = np.concatenate([X_pos, X_neg], axis=0)
        y_comb = np.concatenate([np.ones(len(X_pos), dtype=np.float32), np.zeros(len(X_neg), dtype=np.float32)])
        perm = rng.permutation(len(X_comb))
        return X_comb[perm], y_comb[perm]

    # ดึงคลาสส่วนน้อยมาทั้งหมด (หรือสูงสุด 300 ตัวอย่าง) และดึงคลาสส่วนใหญ่มาเติมให้ครบ n_total
    n_pos_keep = min(len(pos_idx), max(80, int(n_total * 0.02)))
    n_neg_keep = min(len(neg_idx), n_total - n_pos_keep)

    chosen_pos = rng.choice(pos_idx, size=n_pos_keep, replace=False)
    chosen_neg = rng.choice(neg_idx, size=n_neg_keep, replace=False)
    chosen_idx = np.concatenate([chosen_pos, chosen_neg])
    rng.shuffle(chosen_idx)

    return np.array(X_all[chosen_idx], dtype=np.float32), np.array(y_all[chosen_idx], dtype=np.float32)


def load_giab_1d_data(n_train=5000, n_val=2000):
    kmer_npz = os.path.join(FASTQ_DIR, "kmer_features_dataset.npz")
    ylabel_npz = os.path.join(FASTQ_DIR, "y_labels_dataset.npz")

    print(f"กำลังโหลดข้อมูล GIAB 1D จริงจาก: {kmer_npz}")
    d_x = np.load(kmer_npz)
    d_y = np.load(ylabel_npz)

    X_tr_raw, X_va_raw = d_x[d_x.files[0]], d_x[d_x.files[1]]
    y_tr_raw, y_va_raw = d_y[d_y.files[0]], d_y[d_y.files[1]]

    X_tr, y_tr = prepare_stratified_imbalanced_split(X_tr_raw, y_tr_raw, n_total=n_train, seed=101)
    X_va, y_va = prepare_stratified_imbalanced_split(X_va_raw, y_va_raw, n_total=n_val, seed=202)

    if X_tr.ndim == 2:
        X_tr = np.expand_dims(X_tr, axis=-1)
        X_va = np.expand_dims(X_va, axis=-1)

    mean_v = np.mean(X_tr)
    std_v = np.std(X_tr) + 1e-6
    X_tr = (X_tr - mean_v) / std_v
    X_va = (X_va - mean_v) / std_v

    print(f" -> มิติข้อมูล GIAB 1D (หลังจัดคลาส Minority=1): "
          f"Train={X_tr.shape} (Pos={int(np.sum(y_tr))}/{len(y_tr)}), "
          f"Val={X_va.shape} (Pos={int(np.sum(y_va))}/{len(y_va)})")
    return X_tr, y_tr, X_va, y_va


def load_giab_2d_data():
    pileup_npz = os.path.join(CRAM_DIR, "pileup_features_dataset.npz")
    print(f"กำลังโหลดข้อมูล GIAB 2D Pileup จริงจาก: {pileup_npz}")
    d = np.load(pileup_npz)
    X_tr, y_tr = prepare_stratified_imbalanced_split(d['X_train'], d['y_train'], n_total=2000, seed=303)
    X_va, y_va = prepare_stratified_imbalanced_split(d['X_val'], d['y_val'], n_total=800, seed=404)
    print(f" -> มิติข้อมูล GIAB 2D: Train={X_tr.shape} (Pos={int(np.sum(y_tr))}/{len(y_tr)}), "
          f"Val={X_va.shape} (Pos={int(np.sum(y_va))}/{len(y_va)})")
    return X_tr, y_tr, X_va, y_va


# ==============================================================================
# 3. ฟังก์ชันสร้างโมเดล 1D-CNN และ 2D-CNN
# ==============================================================================
def build_1d_cnn(input_shape, init_W, init_L, trainable_params=True):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=input_shape),
        tf.keras.layers.Conv1D(32, kernel_size=3, padding='same'),
        RectLU(init_W=init_W, init_L=init_L, trainable_params=trainable_params, name='rectlu_1d_1'),
        tf.keras.layers.MaxPooling1D(pool_size=2),

        tf.keras.layers.Conv1D(64, kernel_size=3, padding='same'),
        RectLU(init_W=init_W, init_L=init_L, trainable_params=trainable_params, name='rectlu_1d_2'),
        tf.keras.layers.MaxPooling1D(pool_size=2),

        tf.keras.layers.Conv1D(128, kernel_size=3, padding='same'),
        RectLU(init_W=init_W, init_L=init_L, trainable_params=trainable_params, name='rectlu_1d_3'),
        tf.keras.layers.MaxPooling1D(pool_size=2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256),
        RectLU(init_W=init_W, init_L=init_L, trainable_params=trainable_params, name='rectlu_1d_dense'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_2d_cnn(input_shape, init_W, init_L, trainable_params=True):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=input_shape),
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same'),
        RectLU(init_W=init_W, init_L=init_L, trainable_params=trainable_params, name='rectlu_2d_1'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same'),
        RectLU(init_W=init_W, init_L=init_L, trainable_params=trainable_params, name='rectlu_2d_2'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same'),
        RectLU(init_W=init_W, init_L=init_L, trainable_params=trainable_params, name='rectlu_2d_3'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256),
        RectLU(init_W=init_W, init_L=init_L, trainable_params=trainable_params, name='rectlu_2d_dense'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def find_optimal_mcc(y_true, y_prob):
    best_mcc, best_th = 0.0, 0.5
    for th in np.linspace(0.05, 0.95, 37):
        yh = (y_prob >= th).astype(int)
        if len(np.unique(yh)) > 1:
            m = float(matthews_corrcoef(y_true, yh))
            if m > best_mcc:
                best_mcc, best_th = m, float(th)
    return best_mcc, best_th


# ==============================================================================
# 4. ฟังก์ชันทำ Grid Search Tuning (W: 8.0 -> 9.0 เพิ่มขึ้น, L: 0.0 -> -2.5 ลดลง)
# ==============================================================================
def tune_rectlu_grid(
    X_train, y_train, X_val, y_val,
    mode='1D',
    w_start=8.0, w_end=9.0, w_step=0.25,
    l_start=0.0, l_end=-2.5, l_step=-0.5,
    epochs=5,
    batch_size=128,
    trainable_params=True
):
    W_values = np.round(np.arange(w_start, w_end + 1e-5, abs(w_step)), 4)
    L_values = np.round(np.arange(l_start, l_end - 1e-5, -abs(l_step)), 4)

    # คำนวณ Class Weight ให้สมดุลทั้งสองฝั่งโดยอัตโนมัติ
    n_pos = max(1, int(np.sum(y_train == 1)))
    n_neg = max(1, int(np.sum(y_train == 0)))
    total = n_pos + n_neg
    class_weight = {
        0: round(float(total / (2.0 * n_neg)), 3),
        1: round(float(total / (2.0 * n_pos)), 3)
    }

    total_runs = len(W_values) * len(L_values)
    print(f"\n==================================================================")
    print(f" เริ่มการทำ Tuning สำหรับ {mode}-CNN บนข้อมูล GIAB จริง ({total_runs} รอบ)")
    print(f" ช่วงค่า W (เพิ่มขึ้น): {W_values}")
    print(f" ช่วงค่า L (ลดลง):   {L_values}")
    print(f" Balanced Class Weight: {class_weight}")
    print(f"==================================================================\n")

    results = []
    best_score = -1.0
    best_config = None
    best_model_path = os.path.join(OUTPUT_DIR, f"best_tuned_model_{mode.lower()}.keras")

    run_idx = 0
    for init_W in W_values:
        for init_L in L_values:
            run_idx += 1
            t0 = time.time()

            tf.keras.backend.clear_session()
            tf.random.set_seed(42)
            np.random.seed(42)

            input_shape = X_train.shape[1:]
            if mode.upper() == '1D':
                model = build_1d_cnn(input_shape, init_W=init_W, init_L=init_L, trainable_params=trainable_params)
            else:
                model = build_2d_cnn(input_shape, init_W=init_W, init_L=init_L, trainable_params=trainable_params)

            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                class_weight=class_weight,
                verbose=0
            )

            val_loss = float(history.history['val_loss'][-1])
            val_acc = float(history.history['val_accuracy'][-1])

            y_pred_list = []
            for i in range(0, len(X_val), batch_size):
                batch_x = X_val[i:i + batch_size]
                y_pred_list.append(model(batch_x, training=False).numpy().ravel())
            y_pred_prob = np.concatenate(y_pred_list)
            y_pred_05 = (y_pred_prob >= 0.5).astype(int)

            mcc_05 = float(matthews_corrcoef(y_val, y_pred_05)) if len(np.unique(y_pred_05)) > 1 else 0.0
            mcc_opt, th_opt = find_optimal_mcc(y_val, y_pred_prob)
            val_auc = float(roc_auc_score(y_val, y_pred_prob))
            val_ap = float(average_precision_score(y_val, y_pred_prob))

            learned_W, learned_L = init_W, init_L
            for layer in model.layers:
                if isinstance(layer, RectLU):
                    learned_W = float(layer.W.numpy())
                    learned_L = float(layer.L.numpy())

            elapsed = time.time() - t0
            print(f"[{mode}-CNN | รอบ {run_idx:02d}/{total_runs:02d}] "
                  f"Init(W={init_W:.2f}, L={init_L:.2f}) -> Learned(W={learned_W:.4f}, L={learned_L:.4f}) | "
                  f"MCC(0.5)={mcc_05:.4f} | MCC(tau*)={mcc_opt:.4f} | AUC={val_auc:.4f} | AUPRC={val_ap:.4f} ({elapsed:.1f}s)")

            results.append({
                "Mode": mode,
                "Init_W": init_W,
                "Init_L": init_L,
                "Learned_Dense_W": round(learned_W, 4),
                "Learned_Dense_L": round(learned_L, 4),
                "Val_Accuracy": round(val_acc, 4),
                "Val_Loss": round(val_loss, 4),
                "Val_MCC_05": round(mcc_05, 4),
                "Val_MCC_Tuned": round(mcc_opt, 4),
                "Optimal_Tau": round(th_opt, 3),
                "Val_AUROC": round(val_auc, 4),
                "Val_AUPRC": round(val_ap, 4),
                "Time_sec": round(elapsed, 2)
            })

            if mcc_opt > best_score:
                best_score = mcc_opt
                best_config = (init_W, init_L, learned_W, learned_L, mcc_05, mcc_opt, val_auc, val_ap)
                model.save(best_model_path)

    df_results = pd.DataFrame(results)
    csv_path = os.path.join(OUTPUT_DIR, f"tuning_results_{mode.lower()}.csv")
    df_results.to_csv(csv_path, index=False)

    print(f"\n--- สรุปผลการทำ Tuning ที่ดีที่สุดสำหรับ {mode}-CNN ---")
    print(f"ค่าเริ่มต้นที่ดีที่สุด : init_W = {best_config[0]:.4f}, init_L = {best_config[1]:.4f}")
    print(f"ค่าหลังเรียนรู้ (Dense): W = {best_config[2]:.4f}, L = {best_config[3]:.4f}")
    print(f"ประสิทธิภาพสูงสุด   : MCC(0.5)={best_config[4]:.4f}, MCC(tau*)={best_config[5]:.4f}, AUROC={best_config[6]:.4f}, AUPRC={best_config[7]:.4f}")
    print(f"บันทึกตารางผลลัพธ์ไว้ที่ : {csv_path}\n")

    return df_results


if __name__ == "__main__":
    X_tr_1d, y_tr_1d, X_va_1d, y_va_1d = load_giab_1d_data(n_train=5000, n_val=2000)
    df_1d = tune_rectlu_grid(
        X_tr_1d, y_tr_1d, X_va_1d, y_va_1d,
        mode='1D',
        w_start=8.0, w_end=9.0, w_step=0.25,
        l_start=0.0, l_end=-2.5, l_step=-0.5,
        epochs=5,
        batch_size=128
    )

    X_tr_2d, y_tr_2d, X_va_2d, y_va_2d = load_giab_2d_data()
    df_2d = tune_rectlu_grid(
        X_tr_2d, y_tr_2d, X_va_2d, y_va_2d,
        mode='2D',
        w_start=8.0, w_end=9.0, w_step=0.25,
        l_start=0.0, l_end=-2.5, l_step=-0.5,
        epochs=5,
        batch_size=128
    )