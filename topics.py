"""Topic models over SAE features: each document is a mix of topics.

Every function returns (weights, topics): weights is documents x topics
(how much each document has of each topic) and topics is topics x
columns (how much each SAE feature belongs to each topic). The grid
takes argmax over a document's weights as its hard label.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import (NMF, LatentDirichletAllocation,
                                   TruncatedSVD)


def nmf(x, k: int, seed: int = 0):
    model = NMF(k, init="nndsvda", max_iter=400, random_state=seed)
    return model.fit_transform(x), model.components_


def plsa(x, k: int, seed: int = 0):
    """NMF with the KL loss, which is equivalent to pLSA."""
    model = NMF(k, init="nndsvda", beta_loss="kullback-leibler", solver="mu",
                max_iter=400, random_state=seed)
    return model.fit_transform(x), model.components_


def lda(x, k: int, seed: int = 0):
    model = LatentDirichletAllocation(k, learning_method="batch",
                                      random_state=seed)
    return model.fit_transform(x), model.components_


def lsa(x, k: int, seed: int = 0):
    """SVD components as topics; weights can be negative, so take |w|."""
    model = TruncatedSVD(k, random_state=seed)
    return np.abs(model.fit_transform(x)), model.components_


def bertopic(bodies: list[str], embeddings: np.ndarray, seed: int = 0):
    """BERTopic with precomputed embeddings; returns labels and topic words."""
    import hdbscan
    import umap
    from bertopic import BERTopic

    model = BERTopic(
        umap_model=umap.UMAP(n_components=10, n_neighbors=15, min_dist=0.0,
                             metric="cosine", random_state=seed),
        hdbscan_model=hdbscan.HDBSCAN(min_cluster_size=10,
                                      prediction_data=True),
        calculate_probabilities=False)
    labels, _ = model.fit_transform(bodies, embeddings=embeddings)
    words = {topic: [w for w, _ in model.get_topic(topic)][:8]
             for topic in set(labels) if topic != -1}
    return np.asarray(labels), words


# name: (function, input representation)
TOPIC_MODELS = {
    "nmf": (nmf, "sae_idf"),
    "plsa": (plsa, "sae"),
    "lda": (lda, "sae_bin"),
    "lsa": (lsa, "sae_idf"),
}
