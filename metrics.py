"""Quality metrics for a grouping of documents.

A grouping is a 0/1 membership matrix, documents x groups: hard clusters
have one 1 per row, top-k groups up to k. Category labels are multi-label
(a Terminal Wrench trace has 1 to 6 categories), so:

    primary label   the rarest of a document's categories, the most
                    specific one; used only for NMI / ARI
    category_f1     for each category, the F1 of the single best group
                    when that group is read as "has this category";
                    averaged over categories (macro)
    category_auc    the same with the group's score as a ranking (soft
                    groupings only)
    task_nmi, model_nmi
                    NMI of hard clusters against task and source model; high
                    means the clusters found tasks or model styles
    task_purity     size-weighted share of each group's largest task, over
                    groups of at least 5 documents
Every label metric is also computed with the labels shuffled across
documents, which gives the chance level.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import scipy.sparse as sp
from scipy.stats import rankdata
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


class Labels:
    """The labels of a docs table, as arrays aligned to its rows."""

    def __init__(self, docs, seed: int = 0):
        self.categories = sorted({c for row in docs.column("labels").to_pylist()
                                  for c in row})
        lists = docs.column("labels").to_pylist()
        counts = Counter(c for row in lists for c in row)
        self.y = np.array([[c in row for c in self.categories] for row in lists])
        self.primary = np.array([min(row, key=counts.__getitem__) if row else ""
                                 for row in lists])
        self.task = np.array(docs.column("group_id").to_pylist())
        self.model = np.array(docs.column("source_model").to_pylist())
        hack = docs.column("is_hack").to_pylist()
        self.is_hack = None if hack[0] is None else np.array(hack, dtype=bool)
        self.shuffle = np.random.default_rng(seed).permutation(len(lists))


def one_hot(labels: np.ndarray) -> sp.csr_matrix:
    """Membership of hard labels; noise (-1) belongs to no group."""
    keep = labels >= 0
    rows = np.flatnonzero(keep)
    _, cols = np.unique(labels[keep], return_inverse=True)
    return sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                         shape=(len(labels), cols.max() + 1 if len(cols) else 1))


def best_f1(y: np.ndarray, membership: sp.csr_matrix) -> np.ndarray:
    """Per category, the best F1 of any group as a predictor of it."""
    tp = np.asarray((membership.T @ sp.csr_matrix(y.astype(np.float32)))
                    .todense()).T
    sizes = np.asarray(membership.sum(axis=0)).ravel()
    f1 = 2 * tp / (y.sum(axis=0)[:, None] + sizes[None, :] + 1e-9)
    return f1.max(axis=1)


def auc_matrix(y: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """AUC of every score column for every category: categories x columns."""
    ranks = rankdata(scores, axis=0)
    positives = y.sum(axis=0).astype(float)
    negatives = len(y) - positives
    summed = y.T.astype(float) @ ranks
    return ((summed - positives[:, None] * (positives[:, None] + 1) / 2)
            / (positives[:, None] * negatives[:, None]))


def purity(membership: sp.csr_matrix, groups: np.ndarray,
           min_size: int = 5) -> float:
    """Size-weighted share of each group's most common value of groups.

    Groups smaller than min_size are skipped: a group of one document is
    trivially pure.
    """
    total, largest = 0, 0
    for g in range(membership.shape[1]):
        rows = membership[:, g].nonzero()[0]
        if len(rows) >= min_size:
            total += len(rows)
            largest += Counter(groups[rows]).most_common(1)[0][1]
    return largest / max(total, 1)


def gini(sizes: np.ndarray) -> float:
    sizes = np.sort(sizes[sizes > 0]).astype(float)
    if len(sizes) < 2:
        return 1.0
    n = len(sizes)
    return float((2 * np.arange(1, n + 1) - n - 1) @ sizes / (n * sizes.sum()))


def hard_metrics(labels: np.ndarray, truth: Labels) -> dict:
    """Metrics of a hard clustering."""
    member = one_hot(labels)
    out = grouping_metrics(member, None, truth)
    clustered = labels >= 0
    shuffled = truth.primary[truth.shuffle]
    out.update({
        "nmi_category": normalized_mutual_info_score(
            truth.primary[clustered], labels[clustered]),
        "nmi_category_shuffled": normalized_mutual_info_score(
            shuffled[clustered], labels[clustered]),
        "ari_category": adjusted_rand_score(truth.primary[clustered],
                                            labels[clustered]),
        "task_nmi": normalized_mutual_info_score(truth.task[clustered],
                                                 labels[clustered]),
        "model_nmi": normalized_mutual_info_score(truth.model[clustered],
                                                  labels[clustered]),
        "noise_frac": float(1 - clustered.mean()),
    })
    return out


def grouping_metrics(membership: sp.csr_matrix, scores: np.ndarray | None,
                     truth: Labels) -> dict:
    """Metrics of any grouping, hard or overlapping.

    scores: documents x groups, used for category_auc when given.
    """
    sizes = np.asarray(membership.sum(axis=0)).ravel()
    f1 = best_f1(truth.y, membership)
    out = {
        "groups": int((sizes > 0).sum()),
        "covered": float((np.asarray(membership.sum(axis=1)).ravel() > 0)
                         .mean()),
        "size_gini": gini(sizes),
        "category_f1": float(f1.mean()),
        "category_f1_shuffled": float(best_f1(truth.y[truth.shuffle],
                                              membership).mean()),
        "task_purity": purity(membership, truth.task),
        "model_purity": purity(membership, truth.model),
    }
    out.update({f"f1:{c}": float(v) for c, v in zip(truth.categories, f1)})
    if scores is not None:
        aucs = auc_matrix(truth.y, scores)
        out["category_auc"] = float(aucs.max(axis=1).mean())
        out["category_auc_shuffled"] = float(
            auc_matrix(truth.y[truth.shuffle], scores).max(axis=1).mean())
    return out


def outlier_metrics(score: np.ndarray, truth: Labels, top: int = 50) -> dict:
    """AUROC and precision@top of an outlier score against is_hack."""
    if truth.is_hack is None:
        return {}
    auc = auc_matrix(truth.is_hack[:, None], score[:, None])[0, 0]
    return {"hack_auroc": float(auc),
            "hack_precision_at_50": float(
                truth.is_hack[np.argsort(-score)[:top]].mean())}


def feature_auc(weights: sp.csr_matrix, truth: Labels, min_docs: int = 10,
                top: int = 3) -> dict[str, list[tuple[int, float]]]:
    """HypotheSAEs-style: the SAE features that best rank each category."""
    df = np.bincount(weights.indices, minlength=weights.shape[1])
    columns = np.flatnonzero(df >= min_docs)
    aucs = auc_matrix(truth.y, weights[:, columns].toarray())
    return {c: [(int(columns[j]), float(aucs[i, j]))
                for j in np.argsort(-aucs[i])[:top]]
            for i, c in enumerate(truth.categories)}
