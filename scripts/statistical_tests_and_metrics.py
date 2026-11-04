import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

def cliffs_delta(x, y):
    x, y = np.asarray(x), np.asarray(y)
    diffs = x[:, None] - y[None, :]
    return float(np.sum(diffs > 0) - np.sum(diffs < 0)) / float(len(x) * len(y))

def holm_bonferroni(p_values):
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.zeros(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = p_values[idx] * (m - rank)
        running_max = max(running_max, adj)
        adjusted[idx] = min(1.0, running_max)
    return adjusted

def verify_pathway(csv_path, ref_act='Smooth RectLU (W=2)'):
    df = pd.read_csv(csv_path)
    ref_mccs = df[df['Activation'] == ref_act].sort_values('Seed')['Per_Seed_MCC'].values
    others = [a for a in df['Activation'].unique() if a != ref_act]
    rows, raw_ps = [], []
    for act in others:
        sub = df[df['Activation'] == act].sort_values('Seed')
        mccs = sub['Per_Seed_MCC'].values
        if np.allclose(ref_mccs, mccs):
            p_val = 1.0
        else:
            res = wilcoxon(ref_mccs, mccs, zero_method='wilcox', alternative='two-sided', mode='exact')
            p_val = float(res.pvalue)
        raw_ps.append(p_val)
        rows.append({
            'Comparison': f'{ref_act} vs. {act}',
            'Ref_Mean_MCC': round(float(np.mean(ref_mccs)), 4),
            'Baseline_Mean_MCC': round(float(np.mean(mccs)), 4),
            'Baseline_Pooled_MCC': round(float(sub['Pooled_MCC_Ref'].iloc[0]), 4),
            'Exact_Wilcoxon_p': round(p_val, 6),
            'Cliffs_Delta': round(cliffs_delta(ref_mccs, mccs), 4)
        })
    adj_ps = holm_bonferroni(raw_ps)
    for i, r in enumerate(rows):
        r['Holm_Bonferroni_p_adj'] = round(float(adj_ps[i]), 4)
    return pd.DataFrame(rows)

if __name__ == '__main__':
    da = verify_pathway('results/e1_pathway_a_1d_per_seed.csv')
    db = verify_pathway('results/e1_pathway_b_2d_per_seed.csv')
    out = pd.concat([da.assign(Pathway='Pathway_A_1D'), db.assign(Pathway='Pathway_B_2D')], ignore_index=True)
    out.to_csv('results/e1_statistical_significance.csv', index=False)
    print(out.to_string(index=False))
