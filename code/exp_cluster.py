"""Node clustering, Section 3.2, Figs. 3-4 and Table 5. Usage: python exp_cluster.py
Three partitions:
  greedy - greedy modularity communities, for comparison;
  kmeans - K-means on node P-vectors as Euclidean coordinates, 4 clusters;
  hier   - hierarchical clustering (ward) with the JS divergence as distance, cut into 4 clusters.
Output: results/cluster_<network>.csv, one row per node.
Also computes the share of the variance of closeness centrality explained by each partition,
eta^2 = between-group variance / total variance, which measures core-to-periphery layering;
written to results/cluster_eta2.csv."""
import os
import numpy as np
import networkx as nx
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from networkx.algorithms.community import greedy_modularity_communities
from prep import load
from pvec import p_matrix, js_pairwise

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
NETS = ["911", "Train bombing", "Power-494", "U.S. Airports"]
K = 4


def cluster(G):
    n = G.number_of_nodes()
    P = p_matrix(G)
    greedy = np.empty(n, int)
    for c, nodes in enumerate(greedy_modularity_communities(G)):
        greedy[list(nodes)] = c
    km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(P).labels_
    J = js_pairwise(P)
    np.fill_diagonal(J, 0)
    hier = sch.cut_tree(sch.linkage(squareform(J, checks=False), method="ward"), n_clusters=K).ravel()
    clo = nx.closeness_centrality(G)
    return pd.DataFrame({"node": range(n), "degree": [G.degree(v) for v in range(n)],
                         "closeness": [clo[v] for v in range(n)],
                         "greedy": greedy, "kmeans": km, "hier": hier})


def eta2(df, col):
    g = df.groupby(col).closeness
    return (g.size() * (g.mean() - df.closeness.mean()) ** 2).sum() / (df.closeness.var(ddof=0) * len(df))


if __name__ == "__main__":
    eta = []
    for name in NETS:
        df = cluster(load(name))
        eta.append({"network": name, **{c: round(eta2(df, c), 3) for c in ["greedy", "kmeans", "hier"]}})
        df.to_csv(os.path.join(OUT, f"cluster_{name.replace(' ', '_')}.csv"), index=False)
        print(name, "communities", df.greedy.nunique(), "kmeans sizes", np.bincount(df.kmeans).tolist(),
              "hier sizes", np.bincount(df.hier).tolist())
    pd.DataFrame(eta).to_csv(os.path.join(OUT, "cluster_eta2.csv"), index=False)
    print(pd.DataFrame(eta).to_string(index=False))
