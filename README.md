# Majority-Class Collapse in Asymmetric Low-Gain C1 Piecewise-Quadratic Activations

Official reproducibility repository for the manuscript:
**Majority-Class Collapse in Asymmetric Low-Gain C1 Piecewise-Quadratic Activations: Disentangling Signal Propagation from Class Imbalance in CNN Classification**
*Preecha Noiumkar and Thawatchai Chomsiri*
Faculty of Informatics, Mahasarakham University, Thailand

## Repository Structure
- `src/activations.py`: Implementations of RectLU, Smooth RectLU, SmeLU, AS-RectLU, and sigma-AS-RectLU.
- `src/models.py`: Pathway A (1D-CNN, 4-mer) and Pathway B (2D-CNN, BAM pileup) architectures (Table 3).
- `scripts/verify_monte_carlo_gains.py`: Exact Monte-Carlo second-moment integration (4,000,000 samples, Table 1 & Table 7).
- `scripts/statistical_tests_and_metrics.py`: Exact Wilcoxon signed-rank test, Holm-Bonferroni correction, Cliff's delta, and Pooled vs. Per-Seed MCC.
- `scripts/run_cross_domain_benchmarks.py`: End-to-end training on Step-Imbalanced MNIST (100:1) and Imbalanced Tabular (200:1) benchmarks (Table 9).
- `scripts/generate_figures.py`: Generates high-resolution 300 DPI PDF/PNG figures (Figure 1 and Figure 2).
- `results/`: Complete per-seed confusion matrices and summary CSV files for Tables 1, 4, 5, 6, 7, 8, and 9.
- `figures/`: 300 DPI PDF and PNG figures.

## Quick Reproduction
1. Install dependencies: `pip install -r requirements.txt`
2. Verify Monte-Carlo gains: `python scripts/verify_monte_carlo_gains.py`
3. Run exact Wilcoxon & Holm-Bonferroni tests: `python scripts/statistical_tests_and_metrics.py`
4. Run cross-domain benchmarks: `python scripts/run_cross_domain_benchmarks.py`
5. Generate figures: `python scripts/generate_figures.py`

## Acknowledgments
This research project was financially supported by the Faculty of Informatics, Mahasarakham University.
