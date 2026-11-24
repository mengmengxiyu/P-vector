"""P-matrix, P-vector, JS divergence and NND, as defined in Sections 2 and 3.4 of the paper.

P-matrix: N rows and D+1 columns. Column j (j=1..D) is the fraction of the other N-1 nodes at
shortest-path distance exactly j from node i; the last column is the fraction of nodes unreachable
from i. Each row sums to 1. Logarithms are base 2, so the JS divergence lies in [0,1].
"""
import numpy as np
import networkx as nx
from scipy.sparse.csgraph import shortest_path


def distance_matrix(G):
    """Shortest-path lengths between all node pairs; -1 for unreachable pairs."""
    A = nx.to_scipy_sparse_array(G, nodelist=range(G.number_of_nodes()), format="csr")
    D = shortest_path(A, method="D", unweighted=True, directed=False)
    D[np.isinf(D)] = -1
    return D.astype(np.int32)


def _counts(D):
    """Rows of a distance matrix -> per row, the number of nodes at distance 1..dmax and the number unreachable."""
    dmax = int(D.max())
    C = np.zeros((D.shape[0], max(dmax, 1) + 1))
    for j in range(1, dmax + 1):
        C[:, j - 1] = (D == j).sum(1)
    C[:, -1] = (D == -1).sum(1)
    return C


def p_matrix(G, D=None, width=None, chunk=1000):
    """Return the P-matrix. If width is given, the distance part is zero-padded to that length so that
    different networks can be aligned. Without D, breadth-first searches are run in batches of chunk
    source nodes, so memory is about chunk×N and the full N×N distance matrix is never stored."""
    if D is not None:
        blocks = [_counts(D)]
        n = D.shape[0]
    else:
        n = G.number_of_nodes()
        A = nx.to_scipy_sparse_array(G, nodelist=range(n), format="csr")
        blocks = []
        for s in range(0, n, chunk):
            Ds = shortest_path(A, method="D", unweighted=True, directed=False, indices=range(s, min(s + chunk, n)))
            Ds[np.isinf(Ds)] = -1
            blocks.append(_counts(Ds.astype(np.int32)))
    dmax = max(b.shape[1] - 1 for b in blocks)
    w = dmax if width is None else width
    assert w >= dmax
    P = np.zeros((n, w + 1))
    r = 0
    for b in blocks:
        P[r:r + len(b), :b.shape[1] - 1] = b[:, :-1]
        P[r:r + len(b), w] = b[:, -1]
        r += len(b)
    return P / (n - 1)


def p_vector(P):
    """Global P-vector of the network: column means of the P-matrix."""
    return P.mean(0)


def entropy(P, axis=-1):
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(P > 0, P * np.log2(P), 0.0)
    return -t.sum(axis)


def js_pairwise(P, chunk=256):
    """Matrix of JS divergences between the P-vectors of all node pairs (base 2, values in [0,1])."""
    n = P.shape[0]
    H = entropy(P)
    J = np.empty((n, n))
    for s in range(0, n, chunk):
        M = (P[s:s + chunk, None, :] + P[None, :, :]) / 2
        J[s:s + chunk] = entropy(M) - (H[s:s + chunk, None] + H[None, :]) / 2
    np.clip(J, 0, 1, out=J)
    return J


def nnd(P):
    """Network node dispersion (NND): generalized JS divergence of the N node P-vectors divided by log(d+1), in [0,1]."""
    d = int(np.max(np.nonzero(P[:, :-1].any(0))[0]) + 1) if P[:, :-1].any() else 1
    J = entropy(P.mean(0)) - entropy(P).mean()
    return max(J, 0.0) / np.log2(d + 1)
