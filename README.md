# A Mechanistic Study of Scale-Dependent Optimization Stagnation in C1 Piecewise-Quadratic Rectifiers Under Class Imbalance

**Authors:** Preecha Noiumkar and Thawatchai Chomsiri  
**Affiliation:** Research Center of Information Technology for the Future, Department of Information Technology, Faculty of Informatics, Mahasarakham University, Maha Sarakham 44150, Thailand  
**Release Tag:** `v5.0.0` (Intelligent Systems with Applications Submission)

---

## Overview
This repository archives the extracted datasets (`.npz`), raw per-seed confusion and diagnostic logs (`.csv`), PyTorch model implementations (`src/`), verification scripts (`scripts/`), and LaTeX manuscript sources (`paper/`) reproducing Tables 1–11 and Figures 1–5 of our study.

### Key Empirical & Mechanistic Verification Logs (`results/`)
1. **All-Layer Modular RMSNorm + SmeLU Ablation & Wall-Clock Latency (`results/v9_reviewer_3_1_real_giab2d_10seeds.csv`):**
   - Controlled 6-way ablation across $N = 10$ paired seeds on $0.25\times$ 2D NIST GIAB HG001 read-pileup tensors ($50 \times 101 \times 4$).
   - Demonstrates exact accuracy parity between explicit 3-step `RMSNorm + SmeLU + Inverse Rescaling` (`MCC(0.50) = 0.9691 ± 0.0373`, `AUROC = 0.9999 ± 0.0001`) and fused `Symmetric sigma-AS-RectLU` (`MCC(0.50) = 0.9700 ± 0.0372`, `AUROC = 0.9999 ± 0.0001`), while reducing measured training time by **11.9%** (`959.78 ms/epoch` vs. `1089.53 ms/epoch`) and running **33.6% faster** than all-layer `LayerNorm (No-Affine) + SmeLU` (`1445.10 ms/epoch`).
2. **Extended $N = 30$ Paired-Seed & $K = 6$ Disjoint Genomic Sub-Block Validation (`results/v9_reviewer_3_3_30seeds_and_subblocks.csv`):**
   - Direct empirical validation across $N = 30$ independent random seeds and $K = 6$ disjoint contiguous genomic test sub-blocks ($180$ block evaluations, $n_+ = 10, n_- = 100$ per block) without bootstrap resampling approximations.
   - Confirms $30/30$ paired seed wins (`Delta MCC(0.50) = +0.9506 ± 0.0623`, range `[+0.7564, +1.0123]`).
3. **Causal Epoch-0 $q_4$ Variance-Clamping Intervention (`results/v9_reviewer_3_4_causal_q4_intervention_10seeds.csv`):**
   - Clamping fourth-layer pre-activation variance $q_4 = \mathrm{Var}(x_4)$ at $c \in \{0.55, 3.70\}$ from Epoch 0 under Smooth RectLU ($L=0, W=2$) immediately restores Epoch-0 first-layer weight gradients by $10^7$–$10^8\times$ (`1.00e-10 -> 5.15e-3` and `3.49e-2`) and rescues convergence from `0/10` to `10/10` seeds.
4. **Cross-Optimizer Learning-Rate, Gradient-Clipping, and Prior-Bias Diagnostics (`results/v9_reviewer_3_6_cross_optimizer_ablation_10seeds.csv`):**
   - Systematic evaluation across $N = 10$ paired seeds for momentum SGD ($\eta \in \{10^{-3}, 10^{-2}, 5\times 10^{-2}\}$, gradient clipping $\|\mathbf{g}\|_2 \le 1.0$, and prior-logit bias $b_{\mathrm{out}} = \ln(p_+/p_-)$) and RMSprop.

---

## Repository Structure
- `data/`: Extracted NIST GIAB HG001 2D read-pileup tensors (`pileup_features_dataset.npz`).
- `results/`: All per-seed empirical logs (`.csv`) reproducing Tables 1–11.
- `src/`: Core PyTorch activation modules (`activations.py`) and 1D/2D CNN architectures (`models.py`).
- `scripts/`: Reproducibility scripts (`v1_edit_3p1.py`, `v1_edit_3p3.py`, `v1_edit_3p4.py`, `v1_edit_3p6.py`, `extract_pileup.py`, `generate_labels.py`).
- `paper/`: LaTeX manuscript (`main.tex`), compiled PDF, `Highlights.txt`, and submission files.
