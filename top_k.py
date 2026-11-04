"""Top-k most common: groups without clustering.

Each document keeps its k highest-weighted columns (SAE features or
TF-IDF words); a kept column is a group, so a document is in up to k
groups and groups overlap. Groups are ranked by how many documents keep
them. This generalizes demos/support_intents.analyze from upstream Quail.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def top_k_membership(matrix: sp.csr_matrix, k: int = 3) -> sp.csr_matrix:
    """Return a 0/1 matrix with each row's k largest entries set."""
    rows, cols = [], []
    for r in range(matrix.shape[0]):
        start, end = matrix.indptr[r], matrix.indptr[r + 1]
        data = matrix.data[start:end]
        keep = np.argsort(-data)[:k]
        keep = keep[data[keep] > 0]
        rows.extend([r] * len(keep))
        cols.extend(matrix.indices[start:end][keep].tolist())
    return sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                         shape=matrix.shape)


def threshold_membership(matrix: sp.csr_matrix,
                         quantile: float = 0.97) -> sp.csr_matrix:
    """Return 0/1 membership where a value exceeds the nonzero quantile."""
    cut = np.quantile(matrix.data, quantile)
    member = matrix.copy()
    member.data = (member.data >= cut).astype(np.float32)
    member.eliminate_zeros()
    return member


def ranked_groups(membership: sp.csr_matrix, top: int = 50) -> list[int]:
    """Columns ordered by how many documents are in them, the top ones."""
    counts = np.asarray(membership.sum(axis=0)).ravel()
    order = np.argsort(-counts)
    return [int(c) for c in order[:top] if counts[c] > 0]


def examples(membership: sp.csr_matrix, weights: sp.csr_matrix, column: int,
             groups: list[str], count: int = 3) -> list[int]:
    """Rows in a group with the largest weight, at most one per task."""
    rows = membership[:, column].nonzero()[0]
    values = np.asarray(weights[rows, column].todense()).ravel()
    seen, found = set(), []
    for r in rows[np.argsort(-values)]:
        if groups[r] not in seen:
            seen.add(groups[r])
            found.append(int(r))
        if len(found) == count:
            break
    return found
