"""How much of the exploit-step vectors is model style, and does removing it help.

    cd sae_eda && PYTHONPATH=. uv run python explore/style_check.py

1. Predict the source model from each representation (logistic regression,
   5 folds grouped by task so a task is never in train and test).
2. K-Means (k = 25) on sae_svd refit after removing each model's mean:
   model and task concentration, seed stability (as in quality.py).
3. Features enriched on exploit steps within every model separately:
   ratio >= 5x and >= 2% of exploit steps in each of the 3 models.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import GroupKFold, cross_val_score

import representations as reps
from cluster import kmeans
from quality import K, concentration, load


def main():
    steps, strengths = load()
    ex = steps.kept.to_numpy()
    sub = steps[ex].reset_index(drop=True)
    tasks, models = sub.group_id.to_numpy(), sub.source_model.to_numpy()
    table = reps.sae_reps(strengths[ex])
    tfidf = reps.tfidf_reps(sub.body.tolist())
    folds = GroupKFold(5)
    print("1. predict model from the vectors (chance ~0.37)")
    for name, x in [("sae_idf", table["sae_idf"].matrix),
                    ("sae_svd", table["sae_svd"].matrix),
                    ("tfidf", tfidf["tfidf"].matrix),
                    ("tfidf_svd", tfidf["tfidf_svd"].matrix)]:
        acc = cross_val_score(LogisticRegression(max_iter=2000), x, models,
                              groups=tasks, cv=folds).mean()
        print(f"   {name:10s} accuracy {acc:.3f}")

    print("2. K-Means on model-centered sae_svd")
    codes = np.unique(models, return_inverse=True)[1]
    for label, x0, x1 in [
            ("sae_svd", table["sae_svd"].matrix, None),
            ("sae_svd - model mean",
             reps.centered_svd(table["sae_idf"].matrix, codes, 0), None)]:
        a, b = kmeans(x0, K, 0), kmeans(x0, K, 1)
        groups = [np.flatnonzero(a == c) for c in range(K)]
        print(f"   {label:22s} model_conc {concentration(groups, models)} "
              f"task_conc {concentration(groups, tasks)} "
              f"stability {adjusted_rand_score(a, b):.3f}")

    print("3. features enriched on exploit steps in every model")
    fired = (strengths > 0).tocsc()
    allm = steps.source_model.to_numpy()
    ok = np.ones(fired.shape[1], bool)
    ratios = []
    for m in np.unique(allm):
        e, o = ex & (allm == m), ~ex & (allm == m)
        re = np.asarray(fired[e].sum(0)).ravel() / e.sum()
        ro = np.asarray(fired[o].sum(0)).ravel() / o.sum()
        r = (re + 1e-3) / (ro + 1e-3)
        ok &= (r >= 5) & (re >= 0.02)
        ratios.append(r)
    print(f"   {ok.sum()} features; model-specific ones fail this by construction")
    for f in np.flatnonzero(ok)[np.argsort(-np.min(np.array(ratios)[:, ok], 0))][:15]:
        print("  ", f, [round(r[f], 1) for r in ratios])


if __name__ == "__main__":
    main()
