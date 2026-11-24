"""Node similarity with additional baselines (Table 4). Usage: python exp_similarity2.py [network ...]
Besides the four methods of exp_similarity.py:
  SimRank   - matrix iteration S = max(C·WᵀSW, I), C = 0.8, 10 iterations; W is the column-normalized adjacency matrix;
  DeepWalk  - closed-form matrix factorization of NetMF (Qiu et al., WSDM 2018), window T=10, negative samples b=1,
              128 dimensions, cosine similarity;
  GraphWave - heat-kernel wavelet characteristic-function embedding (Donnat et al., KDD 2018), two scales,
              50 points of t in [0,100]; similarity is the negative Euclidean distance; designed for structural roles;
  Random    - a randomly chosen other node, as a reference.
Measures: mean difference in degree, betweenness, closeness and eigenvector centrality between each node and its most
similar node; mean difference in SI spreading ability over steps 1..10; and the fraction of most similar pairs in the
same community (greedy modularity). Ties are broken at random. Output: results/similarity2_<network>.csv."""
import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse.linalg import eigsh
from networkx.algorithms.community import greedy_modularity_communities
from prep import load
from exp_similarity import similarities, most_similar, si_curves, NETS

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def simrank(A, C=0.8, it=10):
    k = A.sum(0)
    W = A / np.where(k > 0, k, 1)
    S = np.eye(len(A))
    for _ in range(it):
        S = C * W.T @ S @ W
        np.fill_diagonal(S, 1)
    return S


def netmf(A, T=10, b=1, dim=128):
    k = A.sum(1)
    Dinv = 1 / np.where(k > 0, k, 1)
    Pm = Dinv[:, None] * A
    S, Pr = np.zeros_like(A), np.eye(len(A))
    for _ in range(T):
        Pr = Pr @ Pm
        S += Pr
    M = k.sum() / (b * T) * S * Dinv[None, :]
    M = np.log(np.maximum(M, 1))
    U, s, _ = np.linalg.svd(M, hermitian=True)
    E = U[:, :dim] * np.sqrt(s[:dim])
    E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    return E @ E.T


def graphwave(A, ts=np.linspace(0, 100, 50)):
    L = np.diag(A.sum(1)) - A
    lam, U = np.linalg.eigh(L)
    nz = lam[lam > 1e-8]
    l1, ln = nz.min(), lam.max()
    feats = []
    for eta in [0.85, 0.95]:
        s = -np.log(eta) / np.sqrt(l1 * ln)
        Psi = (U * np.exp(-s * lam)) @ U.T
        phi = np.stack([np.exp(1j * t * Psi).mean(0) for t in ts], axis=1)  # one row per node; loop over t to save memory
        feats += [phi.real, phi.imag]
    E = np.concatenate(feats, axis=1)
    sq = (E ** 2).sum(1)
    return -np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * E @ E.T, 0))


def run(name):
    G = load(name)
    n = G.number_of_nodes()
    A = nx.to_numpy_array(G, nodelist=range(n))
    S = similarities(G)
    S["SimRank"] = simrank(A)
    S["DeepWalk"] = netmf(A)
    S["GraphWave"] = graphwave(A)
    rng = np.random.default_rng(2)
    S["Random"] = rng.random((n, n))
    for s in S.values():
        np.fill_diagonal(s, -np.inf)
    _, vec = eigsh(nx.to_scipy_sparse_array(G, nodelist=range(n), dtype=float), k=1, which="LA")
    bc, cc = nx.betweenness_centrality(G), nx.closeness_centrality(G)
    bench = {"degree": np.array([G.degree(v) for v in range(n)]) / (n - 1),
             "betweenness": np.array([bc[v] for v in range(n)]),
             "closeness": np.array([cc[v] for v in range(n)]),
             "eigenvector": np.abs(vec[:, 0])}
    curves = si_curves(G, np.random.default_rng(0))
    comm = np.empty(n, int)
    for c, nodes in enumerate(greedy_modularity_communities(G)):
        comm[list(nodes)] = c
    rows = []
    for meth, s in S.items():
        j = most_similar(s, "random", np.random.default_rng(1))
        r = dict(network=name, method=meth)
        for b, x in bench.items():
            r[b] = np.abs(x - x[j]).mean()
        r["SI"] = np.abs(curves - curves[j]).mean()
        r["same_community"] = (comm == comm[j]).mean()
        rows.append(r)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    for name in sys.argv[1:] or NETS:
        df = run(name)
        df.to_csv(os.path.join(OUT, f"similarity2_{name.replace(' ', '_')}.csv"), index=False)
        print(df.round(4).to_string(index=False), flush=True)
