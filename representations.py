"""Build the document vectors the grid compares.

Every builder returns a Rep: a matrix with one row per document, in
docs.parquet order, and a name for each column.

SAE representations (from features.parquet, one row per document and
feature):
    sae      raw strengths, sparse n x 32,768
    sae_bin  1 where a feature fired, else 0
    sae_idf  strength * log(n / documents with the feature), rows L2-normalized
    sae_svd  TruncatedSVD(100) of sae_idf, rows L2-normalized
Baselines:
    tfidf      word TF-IDF of the body (sublinear tf, min_df=5)
    tfidf_svd  TruncatedSVD(100) of tfidf
    emb        Qwen3-Embedding-0.6B vectors from embed_modal.py

centered_reps rebuilds all of these with each task's (or model's) mean
removed, so a grouping only sees how a document differs from the other
documents of its task.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, svds
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

NUM_FEATURES = 32_768
SVD_DIMS = 100
SVD_SOURCE = {"sae_svd": "sae_idf", "tfidf_svd": "tfidf"}


@dataclass(frozen=True)
class Rep:
    """One representation: a document matrix and its column names."""
    name: str
    matrix: sp.csr_matrix | np.ndarray
    columns: list[str]

    @property
    def sparse(self) -> bool:
        return sp.issparse(self.matrix)


def sae_matrix(features: pa.Table, ids: pa.Array) -> sp.csr_matrix:
    """Return the strengths as a documents x features matrix in ids order.

    A step-level features table can be pooled to traces by passing step
    ids here and then max_pool.
    """
    rows = pc.index_in(features.column("id"), value_set=ids).to_numpy()
    if np.any(rows < 0):
        raise ValueError("features.parquet has ids that docs.parquet lacks")
    return sp.csr_matrix(
        (features.column("strength").to_numpy().astype(np.float32),
         (rows, features.column("feature").to_numpy())),
        shape=(len(ids), NUM_FEATURES))


def max_pool(matrix: sp.csr_matrix, parents: list[str],
             order: list[str]) -> sp.csr_matrix:
    """Return one row per parent id in `order`: the max over its rows."""
    index = {parent: i for i, parent in enumerate(order)}
    coo = matrix.tocoo()
    target = np.array([index[parents[r]] for r in coo.row])
    pooled = {}
    for r, c, v in zip(target, coo.col, coo.data):
        if v > pooled.get((r, c), 0.0):
            pooled[(r, c)] = v
    keys = np.array(list(pooled), dtype=np.int64).reshape(-1, 2)
    return sp.csr_matrix((np.fromiter(pooled.values(), np.float32),
                          (keys[:, 0], keys[:, 1])),
                         shape=(len(order), matrix.shape[1]))


def idf_weight(matrix: sp.csr_matrix) -> sp.csr_matrix:
    """Weight each column by log(n / documents with that column)."""
    df = np.bincount(matrix.indices, minlength=matrix.shape[1])
    idf = np.log(matrix.shape[0] / np.maximum(df, 1)).astype(np.float32)
    return normalize(matrix @ sp.diags(idf), norm="l2", axis=1).tocsr()


def svd(matrix, seed: int = 0) -> np.ndarray:
    """Reduce to SVD_DIMS dimensions, rows L2-normalized."""
    reduced = TruncatedSVD(SVD_DIMS, random_state=seed).fit_transform(matrix)
    return normalize(reduced).astype(np.float32)


def sae_reps(strengths: sp.csr_matrix, seed: int = 0) -> dict[str, Rep]:
    """The four SAE representations of one extraction run."""
    names = [f"f{i}" for i in range(NUM_FEATURES)]
    binary = strengths.copy()
    binary.data[:] = 1.0
    weighted = idf_weight(strengths)
    dims = [f"svd{i}" for i in range(SVD_DIMS)]
    return {"sae": Rep("sae", strengths, names),
            "sae_bin": Rep("sae_bin", binary, names),
            "sae_idf": Rep("sae_idf", weighted, names),
            "sae_svd": Rep("sae_svd", svd(weighted, seed), dims)}


def tfidf_reps(bodies: list[str], seed: int = 0) -> dict[str, Rep]:
    """Word TF-IDF of the bodies and its SVD reduction."""
    vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=5, max_df=0.95,
                                 max_features=50_000,
                                 token_pattern=r"(?u)\b[a-zA-Z_][a-zA-Z_]+\b")
    matrix = vectorizer.fit_transform(bodies).tocsr().astype(np.float32)
    words = vectorizer.get_feature_names_out().tolist()
    dims = [f"svd{i}" for i in range(SVD_DIMS)]
    return {"tfidf": Rep("tfidf", matrix, words),
            "tfidf_svd": Rep("tfidf_svd", svd(matrix, seed), dims)}


def emb_rep(directory: Path, ids: list[str]) -> Rep:
    """The embed_modal.py vectors, reordered to ids."""
    vectors = np.load(directory / "embeddings.npy")
    order = {i: n for n, i in enumerate(json.loads(
        (directory / "ids.json").read_text()))}
    vectors = vectors[[order[i] for i in ids]]
    return Rep("emb", vectors, [f"e{i}" for i in range(vectors.shape[1])])


def group_means(matrix, codes: np.ndarray) -> np.ndarray:
    """Dense groups x columns matrix of each group's mean row."""
    indicator = sp.csr_matrix((np.ones(len(codes), np.float32),
                               (codes, np.arange(len(codes)))))
    sums = indicator @ matrix
    sums = sums.toarray() if sp.issparse(sums) else sums
    return sums / np.bincount(codes)[:, None]


def positive_residual(matrix: sp.csr_matrix, codes: np.ndarray) -> sp.csr_matrix:
    """max(x - group mean, 0) on each document's nonzero entries.

    Where a document has 0 the residual is <= 0 and is dropped, so the
    result is as sparse as the input and stays non-negative (top-3 and
    the topic models need that). A group of one document becomes empty.
    """
    means = group_means(matrix, codes)
    coo = matrix.tocoo()
    data = np.maximum(coo.data - means[codes[coo.row], coo.col], 0)
    out = sp.csr_matrix((data.astype(np.float32), (coo.row, coo.col)),
                        shape=matrix.shape)
    out.eliminate_zeros()
    return out


def centered_svd(matrix: sp.csr_matrix, codes: np.ndarray,
                 seed: int = 0) -> np.ndarray:
    """SVD_DIMS-dimension SVD of (matrix - group means), rows L2-normalized.

    The centered matrix is dense, so it is never built: svds only needs
    products with it, and P = I - (group averaging) is symmetric, so
    (P X) v = P (X v) and (P X)^T u = X^T (P u).
    """
    counts = np.bincount(codes).astype(np.float32)

    def center(y):
        y = y.reshape(len(codes), -1)
        sums = np.zeros((len(counts), y.shape[1]), np.float32)
        np.add.at(sums, codes, y)
        return y - (sums / counts[:, None])[codes]

    operator = LinearOperator(
        matrix.shape, dtype=np.float32,
        matvec=lambda v: center(matrix @ v).ravel(),
        matmat=lambda v: center(matrix @ v),
        rmatvec=lambda u: matrix.T @ center(u).ravel(),
        rmatmat=lambda u: matrix.T @ center(u))
    v0 = np.random.default_rng(seed).standard_normal(min(matrix.shape))
    u, s, _ = svds(operator, k=SVD_DIMS, v0=v0.astype(np.float32))
    return normalize(u * s).astype(np.float32)


def centered_reps(table: dict[str, Rep], groups: np.ndarray,
                  seed: int = 0) -> dict[str, Rep]:
    """Every representation with each group's mean removed.

    groups holds one value per document, e.g. its task id. Sparse
    representations keep the positive residual (positive_residual);
    sae_idf and tfidf are L2-normalized again. The SVD representations
    are refit on the centered sae_idf / tfidf rather than centered after
    the fact, because the uncentered SVD spends its dimensions on the
    differences between groups. emb is centered and normalized.
    """
    codes = np.unique(groups, return_inverse=True)[1]
    out = {}
    for name, rep in table.items():
        if rep.sparse:
            matrix = positive_residual(rep.matrix, codes)
            if name in ("sae_idf", "tfidf"):
                matrix = normalize(matrix).tocsr()
        elif name in ("sae_svd", "tfidf_svd"):
            matrix = centered_svd(table[SVD_SOURCE[name]].matrix, codes, seed)
        else:
            matrix = normalize(rep.matrix - group_means(rep.matrix, codes)[codes])
        out[name] = Rep(name, matrix, rep.columns)
    return out


def load_features(run_dir: Path) -> pa.Table:
    return pq.read_table(run_dir / "features.parquet")
