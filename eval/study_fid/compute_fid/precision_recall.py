"""Computes k-NN Precision and Recall for generative models (Kynkäänniemi et al., NeurIPS 2019 / Sajjadi et al., 2018).
Used to produce the PRECISION and RECALL columns in Table 2 of the ARD-VAE paper.
"""
import numpy as np
from scipy.spatial.distance import cdist


def compute_pairwise_distances(X, Y=None):
    if Y is None:
        return cdist(X, X, metric='euclidean')
    return cdist(X, Y, metric='euclidean')


def compute_knn_radii(X, k=3):
    """Computes the distance from each point in X to its k-th nearest neighbor."""
    distances = compute_pairwise_distances(X)
    np.fill_diagonal(distances, np.inf)
    # Sort distances along each row
    sorted_distances = np.sort(distances, axis=1)
    # k-th nearest neighbor distance (0-indexed so index k-1)
    radii = sorted_distances[:, k - 1]
    return radii


def compute_precision_recall(real_features, gen_features, k=3, batch_size=1000):
    """Computes precision and recall between real and generated feature representations.

    Parameters:
    -- real_features: Array of shape (N_real, D)
    -- gen_features:  Array of shape (N_gen, D)
    -- k:             Number of nearest neighbors to define the manifold boundary (default 3)

    Returns:
    -- precision: Float in [0, 1]
    -- recall:    Float in [0, 1]
    """
    n_real = len(real_features)
    n_gen = len(gen_features)

    # Compute k-NN radii for both manifolds
    real_radii = compute_knn_radii(real_features, k=k)
    gen_radii = compute_knn_radii(gen_features, k=k)

    # Compute Precision: fraction of gen samples within the real manifold
    inside_real = 0
    for i in range(0, n_gen, batch_size):
        batch_gen = gen_features[i:i + batch_size]
        dist_matrix = compute_pairwise_distances(batch_gen, real_features)
        # Check if gen sample is within radius of at least one real sample
        is_inside = np.any(dist_matrix <= real_radii[None, :], axis=1)
        inside_real += np.sum(is_inside)

    precision = float(inside_real) / float(n_gen)

    # Compute Recall: fraction of real samples within the gen manifold
    inside_gen = 0
    for i in range(0, n_real, batch_size):
        batch_real = real_features[i:i + batch_size]
        dist_matrix = compute_pairwise_distances(batch_real, gen_features)
        # Check if real sample is within radius of at least one gen sample
        is_inside = np.any(dist_matrix <= gen_radii[None, :], axis=1)
        inside_gen += np.sum(is_inside)

    recall = float(inside_gen) / float(n_real)

    return precision, recall
