"""Node similarity, Section 3.1: Table 2 (MDE), Table 3 (VDE), Fig. 1 (MDP, VDP).
Usage: python exp_similarity.py [network ...]; all 8 networks if none is given. Output: results/similarity_*.csv.

Four similarities: PVJS = 1 - JS(P_i, P_j); CN = A^2; LP = A^2 + 0.1 A^3; Katz3 = bA + b^2A^2 + b^3A^3, b = 1/N.
Diagonals are removed. The most similar node is the row maximum; ties are broken in two ways:
  first  - the node with the smallest index (np.argmax);
  random - a random node among the tied ones, with a fixed seed.
SI model: synchronous updates; at each step every infected node infects each susceptible neighbour
independently with probability gamma; single seed, averaged over R runs; infected fraction at steps 1..T.
"""
import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from prep import load
from pvec import p_matrix, js_pairwise

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
NETS = ["Karate", "Train bombing", "David", "Physicians", "C elegans", "Asoiaf", "Euroroad", "U.S. PowerGrid"]
GAMMA, R, T = 0.2, 100, 10


def similarities(G):
    n = G.number_of_nodes()
    A = nx.to_numpy_array(G, nodelist=range(n))
    A2 = A @ A
    A3 = A2 @ A
    b = 1 / n
    S = {"PVJS": 1 - js_pairwise(p_matrix(G)),
         "CN": A2.copy(),
         "LP": A2 + 0.1 * A3,
         "Katz3": b * A + b ** 2 * A2 + b ** 3 * A3}
    for s in S.values():
        np.fill_diagonal(s, -np.inf)
    return S


def most_similar(S, how, rng):
    if how == "first":
        return S.argmax(1)
    out = np.empty(len(S), int)
    for i, row in enumerate(S):
        cand = np.flatnonzero(np.isclose(row, row.max(), rtol=1e-12, atol=1e-15))
        out[i] = rng.choice(cand)
    return out


def si_curves(G, rng):
    """Each node as the single seed; returns an N×T matrix of the mean infected fraction at step t."""
    n = G.number_of_nodes()
    A = sp.csr_array(nx.to_scipy_sparse_array(G, nodelist=range(n), dtype=float))
    out = np.zeros((n, T))
    for s in range(n):
        I = np.zeros((n, R), bool)
        I[s] = True
        for t in range(T):
            k = A @ I.astype(float)  # number of infected neighbours of each node in each run
            I = I | (rng.random((n, R)) < 1 - (1 - GAMMA) ** k)
            out[s, t] = I.mean()
    return out


def run(name):
    rng = np.random.default_rng(0)
    G = load(name)
    S = similarities(G)
    # leading eigenvector of the adjacency matrix, unit length; same as eigenvector_centrality_numpy in
    # older networkx, computed directly because newer versions fail on disconnected networks
    _, vec = eigsh(nx.to_scipy_sparse_array(G, nodelist=range(G.number_of_nodes()), dtype=float), k=1, which="LA")
    x = np.abs(vec[:, 0])
    curves = si_curves(G, rng)
    rows = []
    for how in ["first", "random"]:
        for meth, s in S.items():
            j = most_similar(s, how, np.random.default_rng(1))
            dx = np.abs(x - x[j])
            dp = np.abs(curves - curves[j])
            rows.append(dict(network=name, tie=how, method=meth, MDE=dx.mean(), VDE=dx.var(),
                             **{f"MDP{t + 1}": dp[:, t].mean() for t in range(T)},
                             **{f"VDP{t + 1}": dp[:, t].var() for t in range(T)}))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    for name in sys.argv[1:] or NETS:
        df = run(name)
        df.to_csv(os.path.join(OUT, f"similarity_{name.replace(' ', '_')}.csv"), index=False)
        print(df[["network", "tie", "method", "MDE", "VDE", "MDP1", "MDP5", "MDP10"]].to_string(index=False), flush=True)
