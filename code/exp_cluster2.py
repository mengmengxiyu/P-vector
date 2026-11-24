"""Quantitative evaluation of node clustering (Table 5, Table A3). Usage: python exp_cluster2.py
(a) Structural-role benchmarks with known labels (following GraphWave, Donnat et al., KDD 2018):
    houses - a ring of 40 nodes with 8 equally spaced "house" motifs (5 nodes);
    varied - a ring of 40 nodes with 8 equally spaced motifs alternating house, star (centre + 4 leaves)
             and fan (centre joined to a 4-node path).
    Two label sets: local     - the usual local roles (plain ring node, ring node with a motif attached,
                                positions within each motif), as in the GraphWave paper;
                    global-WL - global equivalence classes from 1-WL colour refinement on the noise-free
                                graph, which is stricter. The number of classes is used as K.
    Noise: random edges added, 0%, 5%, 10% of the original edge count, 10 realizations each; labels are
    those of the noise-free graph.
    Methods: K-means on P-vectors; hierarchical clustering with the JS distance (ward, average, complete);
             ward clustering of GraphWave (Euclidean) and DeepWalk/NetMF (cosine) embeddings;
             greedy modularity communities. Scores: ARI, NMI.
(b) Choice of K: silhouette scores (JS distance) of K-means and ward clustering for K = 2..8 on four
    real-world networks.
Output: results/cluster2_roles.csv, cluster2_silhouette.csv."""
import os
import numpy as np
import pandas as pd
import networkx as nx
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from networkx.algorithms.community import greedy_modularity_communities
from prep import load
from pvec import p_matrix, js_pairwise
from exp_similarity2 import netmf, graphwave

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def shape(kind):
    if kind == "house":
        return nx.Graph([(0, 1), (1, 2), (2, 3), (3, 0), (2, 4), (3, 4)]), 0
    if kind == "star":
        return nx.star_graph(4), 0
    G = nx.path_graph(range(1, 5)); G.add_edges_from((0, i) for i in range(1, 5))
    return G, 0  # fan


def benchmark(kinds, ring=40):
    """Return the graph and local role labels: ring nodes are either attachment points or plain; motif nodes
    are labelled by (motif type, position class), where the position class comes from 1-WL on the motif
    with one pendant node standing for the ring."""
    G = nx.cycle_graph(ring)
    step = ring // len(kinds)
    local = ["ring"] * ring
    for i, kind in enumerate(kinds):
        S, anchor = shape(kind)
        T = S.copy(); T.add_edge(anchor, "r")
        pos = wl_roles(nx.relabel_nodes(T, {v: j for j, v in enumerate(list(S) + ["r"])}))
        off = G.number_of_nodes()
        G.add_edges_from((a + off, b + off) for a, b in S.edges())
        G.add_edge(i * step, anchor + off)
        local[i * step] = "ring-anchor"
        local += [f"{kind}-{pos[j]}" for j in range(len(S))]
    ids = {r: k for k, r in enumerate(sorted(set(local)))}
    return G, np.array([ids[r] for r in local])


def wl_roles(G):
    col = {v: 0 for v in G}
    while True:
        sig = {v: (col[v], tuple(sorted(col[u] for u in G[v]))) for v in G}
        ids = {s: i for i, s in enumerate(sorted(set(sig.values())))}
        new = {v: ids[sig[v]] for v in G}
        if len(set(new.values())) == len(set(col.values())):
            return np.array([new[v] for v in range(G.number_of_nodes())])
        col = new


def add_noise(G, frac, rng):
    G = G.copy()
    n, k = G.number_of_nodes(), int(round(frac * G.number_of_edges()))
    while k > 0:
        a, b = rng.integers(n, size=2)
        if a != b and not G.has_edge(a, b):
            G.add_edge(a, b); k -= 1
    return G


def hier(J, K, method):
    return sch.fcluster(sch.linkage(squareform(J, checks=False), method=method), K, criterion="maxclust")


def methods(G, K):
    P = p_matrix(G)
    J = js_pairwise(P); np.fill_diagonal(J, 0)
    A = nx.to_numpy_array(G, nodelist=range(G.number_of_nodes()))
    km = lambda X: KMeans(K, n_init=10, random_state=0).fit(X).labels_
    gw = -graphwave(A)  # graphwave returns negative Euclidean distances; negate to get distances
    lab = {"P-vector K-means": km(P)}
    for m in ["ward", "average", "complete"]:
        lab[f"P-vector hier-{m}"] = hier(J, K, m)
    lab["GraphWave hier-ward"] = hier(gw / gw.max(), K, "ward")
    dw = 1 - netmf(A, dim=min(128, len(A) - 1))
    np.fill_diagonal(dw, 0)
    lab["DeepWalk hier-ward"] = hier(np.clip(dw, 0, None), K, "ward")
    comm = np.empty(len(A), int)
    for c, nodes in enumerate(greedy_modularity_communities(G)):
        comm[list(nodes)] = c
    lab["greedy modularity"] = comm
    return lab


def part_a():
    rows = []
    for bname, kinds in [("houses", ["house"] * 8), ("varied", ["house", "star", "fan", "house", "star", "fan", "house", "star"])]:
        G0, y_local = benchmark(kinds)
        for label_name, y in [("local", y_local), ("global-WL", wl_roles(G0))]:
            K = len(set(y))
            for frac in [0.0, 0.05, 0.10]:
                for r in range(10 if frac > 0 else 1):
                    G = add_noise(G0, frac, np.random.default_rng(r))
                    for m, lab in methods(G, K).items():
                        rows.append(dict(benchmark=bname, labels=label_name, nodes=G0.number_of_nodes(), roles=K,
                                         noise=frac, rep=r, method=m,
                                         ARI=adjusted_rand_score(y, lab), NMI=normalized_mutual_info_score(y, lab)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "cluster2_roles.csv"), index=False)
    print(df.groupby(["benchmark", "labels", "roles", "noise", "method"])[["ARI", "NMI"]].mean().round(3).to_string())


def part_b():
    rows = []
    for name in ["911", "Train bombing", "Power-494", "U.S. Airports"]:
        G = load(name)
        P = p_matrix(G)
        J = js_pairwise(P); np.fill_diagonal(J, 0)
        for K in range(2, 9):
            for m, lab in [("K-means", KMeans(K, n_init=10, random_state=0).fit(P).labels_), ("hier-ward", hier(J, K, "ward"))]:
                rows.append(dict(network=name, K=K, method=m, silhouette=silhouette_score(J, lab, metric="precomputed")))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "cluster2_silhouette.csv"), index=False)
    print(df.pivot_table(index=["network", "method"], columns="K", values="silhouette").round(3).to_string())


if __name__ == "__main__":
    part_a()
    part_b()
