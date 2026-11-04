"""Hard clustering: one cluster label per document, -1 for noise.

Every function takes a document matrix (sparse rows are L2-normalized by
the caller) and returns an int array of labels.
"""

from __future__ import annotations

import hdbscan
import numpy as np
import scipy.sparse as sp
import umap
from sklearn.cluster import (DBSCAN, AgglomerativeClustering, KMeans,
                             SpectralClustering)
from sklearn.metrics.pairwise import cosine_distances
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors, kneighbors_graph


def kmeans(x, k: int, seed: int = 0) -> np.ndarray:
    return KMeans(k, n_init=3, random_state=seed).fit_predict(x)


def kmedoids(x, k: int, seed: int = 0, iterations: int = 50) -> np.ndarray:
    """Alternating k-medoids on cosine distance with k-means++ seeding."""
    distance = cosine_distances(x).astype(np.float32)
    rng = np.random.default_rng(seed)
    medoids = [int(rng.integers(len(distance)))]
    for _ in range(k - 1):
        nearest = distance[:, medoids].min(axis=1) ** 2
        medoids.append(int(rng.choice(len(distance), p=nearest / nearest.sum())))
    medoids = np.array(medoids)
    for _ in range(iterations):
        labels = distance[:, medoids].argmin(axis=1)
        updated = medoids.copy()
        for c in range(k):
            members = np.flatnonzero(labels == c)
            if len(members):
                within = distance[np.ix_(members, members)].sum(axis=1)
                updated[c] = members[within.argmin()]
        if np.array_equal(updated, medoids):
            break
        medoids = updated
    return distance[:, medoids].argmin(axis=1)


def agglomerative(x, k: int, seed: int = 0) -> np.ndarray:
    """Ward on dense input; average linkage on cosine distance for sparse."""
    if sp.issparse(x):
        return AgglomerativeClustering(k, metric="precomputed",
                                       linkage="average").fit_predict(
            cosine_distances(x))
    return AgglomerativeClustering(k, linkage="ward").fit_predict(x)


def spectral(x, k: int, seed: int = 0) -> np.ndarray:
    graph = kneighbors_graph(x, 15, metric="cosine", include_self=False)
    graph = 0.5 * (graph + graph.T)
    return SpectralClustering(k, affinity="precomputed", random_state=seed,
                              assign_labels="cluster_qr").fit_predict(graph)


def gmm(x, k: int, seed: int = 0) -> np.ndarray:
    return GaussianMixture(k, covariance_type="full", reg_covar=1e-4,
                           random_state=seed).fit_predict(x)


def dbscan(x, k: int | None = None, seed: int = 0) -> np.ndarray:
    """eps at the elbow of the sorted 10th-neighbor cosine distances."""
    nearest = NearestNeighbors(n_neighbors=10, metric="cosine").fit(x)
    curve = np.sort(nearest.kneighbors(x)[0][:, -1])
    line = np.linspace(curve[0], curve[-1], len(curve))
    eps = float(curve[np.argmax(line - curve)])
    return DBSCAN(eps=max(eps, 1e-4), min_samples=10,
                  metric="cosine").fit_predict(x)


def umap_hdbscan(x, k: int | None = None, seed: int = 0,
                 min_cluster_size: int = 10) -> np.ndarray:
    """The BERTopic recipe: UMAP to 10 dimensions, then HDBSCAN."""
    reduced = umap.UMAP(n_components=10, n_neighbors=15, min_dist=0.0,
                        metric="cosine", random_state=seed).fit_transform(x)
    return hdbscan.HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(
        reduced)


# name: (function, needs dense input, takes k)
CLUSTERERS = {
    "kmeans": (kmeans, False, True),
    "kmedoids": (kmedoids, False, True),
    "agglomerative": (agglomerative, False, True),
    "spectral": (spectral, False, True),
    "gmm": (gmm, True, True),
    "dbscan": (dbscan, True, False),
    "hdbscan": (umap_hdbscan, True, False),
}
