# Bias-Offset Sensitivity, Scale-Width Mismatch, and Majority-Class Collapse in C1 Piecewise-Quadratic Rectifiers: Mechanistic Dissection, Scale-Adaptive Design, and Empirical Validation on NIST GIAB Sequencing Data

[![Release](https://img.shields.io/badge/Release-v3.0.0-blue.svg)](https://github.com/preecha-msu/RectLU-Genomic-Collapse/releases/tag/v3.0.0)
[![Journal](https://img.shields.io/badge/Target-Neurocomputing-orange.svg)](https://www.sciencedirect.com/journal/neurocomputing)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**Authors:** **Preecha Noiumkar** (1,*) and **Thawatchai Chomsiri** (1)  
(1) Research Center of Information Technology for the Future, Department of Information Technology, Faculty of Informatics, Mahasarakham University, Mahasarakham 44150, Thailand  
(*) Corresponding author: preecha.n@msu.ac.th | Co-author: thawatchai.c@msu.ac.th

---

## Abstract

**Background:** Piecewise-quadratic C1-continuous rectifiers such as Smooth ReLU (SmeLU, centered symmetrically on [-beta, +beta]) replace ReLU's non-differentiable kink with a parabolic transition. Recently proposed Smooth RectLU and its trainable variant AS-RectLU anchor the quadratic transition asymmetrically on [L, L+W] with default L = 0, W = 2. Algebraically, Smooth RectLU satisfies the exact identity f_S(x; L, W) = SmeLU_{beta=W/2}(x - L - W/2), meaning that L = 0, W = 2 is functionally equivalent to shifting symmetric SmeLU (beta = 1) by an implicit negative bias offset of Delta_b = -1 at initialization while fixing the transition scale at W = 2. Under severe class imbalance (100:1 to 200:1) on low-amplitude sparse inputs (sigma_x << W), this combination of negative initialization offset and scale-width mismatch (sigma_l << W) triggers severe signal contraction and majority-class collapse.

**Methods:** We derive Gaussian second-moment propagation equations and propose **Symmetric Per-Sample sigma-AS-RectLU**, a parameter-free, batch-independent C1 rectifier that dynamically locks transition width W_l(x) = kappa * sigma_RMS(x) and symmetric threshold L_l(x) = -W_l(x)/2 to each sample's layer root-mean-square scale, restoring degree-1 positive homogeneity (f(alpha * x) = alpha * f(x)). Across six pre-specified experiments (N = 10 paired seeds, 20 epochs) and a 30-cell hyperparameter grid search (W in [8.0, 9.0] x L in [0.0, -2.5]) on real NIST GIAB HG001 (NA12878, GRCh38 v4.2.1) FASTQ k-mer spectra and BWA-MEM read-pileup tensors, we evaluate 13 activation and normalization configurations alongside step-imbalanced MNIST digit 3 vs. 5/8 (100:1, clean and noisy sigma_noise = 1.1) and imbalanced tabular classification (200:1, n_+ = 150 test positives). We report fixed-threshold MCC(tau=0.50) (Mean +/- SD and Median [IQR]), validation-calibrated MCC(tau*), AUROC, and AUPRC.

**Results:** Exact Monte-Carlo integration (4 x 10^6 samples) confirms that shifting L from -1.0 (SmeLU, forward gain 0.5239) to L = 0.0 (W=2) contracts forward gain to 0.0874. On low-amplitude 2D spatial motifs (sigma_x = 0.22, 100:1), Smooth RectLU (L=0, W=2) and AS-RectLU suffer severe representation collapse (Median MCC(0.50) = 0.0000, calibrated MCC(tau*) <= 0.0865, AUROC = 0.6349-0.6695), whereas symmetric SmeLU (L=-1, W=2) achieves MCC(0.50) = 0.7406 +/- 0.0687 (MCC(tau*) = 0.8259, AUROC = 0.9980) and Symmetric sigma-AS-RectLU achieves MCC(0.50) = 0.7281 +/- 0.0326 (MCC(tau*) = 0.8061, AUROC = 0.9975), outperforming ReLU (0.6380 +/- 0.1132) and BatchNorm (0.6005 +/- 0.0920). Directly rescaling 2D inputs to sigma_x = 1.00 restores Smooth RectLU (MCC(0.50) = 0.5721, MCC(tau*) = 0.7688, AUROC = 0.9970). Crucially, empirical grid tuning on real NIST GIAB HG001 data confirms that while L = 0.0 causes complete 2D pileup gradient freeze (Loss = 0.6931, MCC = 0.0000, AUROC = 0.5000), shifting L in [-0.5, -2.5] immediately unlocks parameter adaptation and achieves full convergence (MCC(0.50) = 0.9851-1.0000, AUROC = 1.0000 in 2D; MCC(0.50) = 0.4628, calibrated MCC(tau*) = 0.5115, AUROC = 0.9597 in 1D).

**Conclusions:** Fixed-width asymmetric quadratic rectifiers fail when sigma_l << W and L = 0 under severe imbalance; enforcing negative threshold offsets (L < 0) or per-sample RMS scale invariance in sigma-AS-RectLU resolves both vulnerabilities.

---

## Key Empirical Highlights (Tables 1-8)

| Benchmark / Experiment | Smooth RectLU (L=0, W=2) | Symmetric SmeLU (L=-1, W=2) | Proposed Symm. sigma-AS-RectLU | Negative-L Tuned (L <= -0.5) |
| :--- | :---: | :---: | :---: | :---: |
| **1D Motif (sigma_x=0.25, 100:1)** | MCC=0.4441 (AUC=0.9181) | MCC=0.5304 (AUC=0.9735) | MCC=0.4045 (AUC=0.9696) | MCC(tau*)=0.5566 (+b_dense) |
| **2D Motif (sigma_x=0.22, 100:1)** | **Collapse** (Med. MCC=0.0000) | MCC=0.7406 (AUC=0.9980) | MCC=0.7281 (AUC=0.9975) | MCC(tau*)=0.7688 (sigma_x=1.00) |
| **Real GIAB HG001 1D k-mer** | MCC=0.0000-0.1486 (L=0.0) | - | - | MCC(0.5)=0.4628, MCC(tau*)=0.5115 (AUC=0.9597) |
| **Real GIAB HG001 2D Pileup** | **Freeze** (MCC=0.0000, Loss=0.6931) | - | - | MCC(0.5)=1.0000, AUC=1.0000 (Loss=0.0000) |
| **MNIST 3 vs. 5/8 Clean (100:1)** | MCC=0.9092 (AUC=0.9991) | MCC=0.9270 (AUC=0.9989) | **MCC=0.9314 (AUC=0.9993)** | MCC=0.9221 (b_dense=+1) |
| **Imbalanced Tabular (200:1)** | MCC=0.4128 (AUC=0.8443) | MCC=0.4294 (AUC=0.8679) | MCC=0.4172 (AUC=0.8455) | - |

---

## Repository Structure

~~~text
RectLU-Genomic-Collapse/
|-- data/
|   +-- pileup_features_dataset.npz            # Real NIST GIAB HG001 (GRCh38 v4.2.1) extracted 2D BAM/VCF pileup tensors (50x101x4)
|-- scripts/
|   |-- run_v3_complete_reviewer_fixes.py      # Reproduces Tables 1-7 (20-epoch 1D/2D benchmarks, sigma_x=1.00 controls, layer variances)
|   |-- run_all_reviewer_experiments.py        # Reproduces cross-domain MNIST (Clean/Noisy 100:1) & Tabular (200:1) benchmarks
|   |-- extract_pileup.py                      # Extracts 2D pileup tensors from GIAB HG001 BAM and truth VCF files via pysam
|   |-- tune_rectlu_w_l_giab.py                # Reproduces Table 8: 30-point (W x L) grid search on real GIAB HG001 1D & 2D datasets
|   +-- plot_giab_heatmap.py                   # Generates 300 DPI publication heatmaps (PDF/PNG) from GIAB tuning logs
|-- results/
|   |-- v3_per_seed_logs.csv                   # Complete 20-epoch per-seed logs (MCC(0.5), Median[IQR], MCC(tau*), AUROC, AUPRC)
|   |-- v3_summary_table.csv                   # Aggregated summary statistics for Tables 4, 5, and 6
|   |-- v3_layer_variances.csv                 # Directly measured layer pre-activation variances sigma_l^2 (Figure 2)
|   |-- all_controlled_per_seed_logs.csv       # Cross-domain per-seed logs for Table 7 (MNIST Clean/Noisy & Tabular 200:1)
|   |-- tuning_results_1d.csv                  # Real GIAB HG001 1D k-mer (W x L) grid search logs (Table 8)
|   |-- tuning_results_2d.csv                  # Real GIAB HG001 2D BAM pileup (W x L) grid search logs (Table 8)
|   |-- Fig_GIAB_WL_Tuning_Heatmap_300dpi.pdf  # High-resolution 300 DPI vector heatmap of GIAB 1D & 2D grid tuning
|   |-- Fig_GIAB_WL_Tuning_Heatmap_300dpi.png  # High-resolution 300 DPI raster heatmap of GIAB 1D & 2D grid tuning
|   |-- best_tuned_model_1d.keras              # Best trained 1D-CNN model checkpoint on GIAB HG001 k-mer spectra
|   +-- best_tuned_model_2d.keras              # Best trained 2D-CNN model checkpoint on GIAB HG001 read pileups
|-- README.md
+-- CITATION.cff
~~~

---

## Reproducing the Experiments

### 1. Controlled 20-Epoch Benchmarks, Scale Controls (sigma_x = 1.00), and Cross-Domain Suites (Tables 1-7)
~~~bash
python scripts/run_v3_complete_reviewer_fixes.py
python scripts/run_all_reviewer_experiments.py
~~~

### 2. Real NIST GIAB HG001 (GRCh38 v4.2.1) W x L Grid Search and 300 DPI Heatmaps (Table 8 & Fig. 5)
~~~bash
python scripts/tune_rectlu_w_l_giab.py
python scripts/plot_giab_heatmap.py
~~~

---

## Acknowledgments
This research project was financially supported by the **Faculty of Informatics, Mahasarakham University**.

## Citation
~~~bibtex
@article{Noiumkar2026RectLUGenomicCollapse,
  author  = {Preecha Noiumkar and Thawatchai Chomsiri},
  title   = {Bias-Offset Sensitivity, Scale-Width Mismatch, and Majority-Class Collapse in C1 Piecewise-Quadratic Rectifiers: Mechanistic Dissection, Scale-Adaptive Design, and Empirical Validation on NIST GIAB Sequencing Data},
  journal = {Neurocomputing},
  year    = {2026},
  url     = {https://github.com/preecha-msu/RectLU-Genomic-Collapse}
}
~~~
