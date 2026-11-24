"""Table 1: basic properties of the real-world networks and estimates of the clustering coefficient.
Usage: python exp_table1.py
C_ave, C_med: the estimate evaluated at the network mean (median) of triangle count and degree;
C_node: the estimate evaluated for each node and then averaged;
C_glob: numerator and denominator summed over all nodes before dividing."""
import os
import numpy as np
import pandas as pd
import networkx as nx
from prep import SRC, load

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
rows = []
for name in SRC:
    G = load(name)
    n, m = G.number_of_nodes(), G.number_of_edges()
    k = np.array([d for _, d in G.degree()], float)
    t = np.array([nx.triangles(G)[v] for v in G], float)  # M11: number of edges among the first neighbours
    f = lambda T, K: 2 * T / (K * (K - 1)) if K > 1 else np.nan
    per = np.array([f(a, b) if b > 1 else 0.0 for a, b in zip(t, k)])
    rows.append(dict(Network=name, N=n, M=m, k_mean=round(k.mean(), 2), k_max=int(k.max()),
                     C_real=round(nx.average_clustering(G), 3), C_ER=round(2 * m / (n * (n - 1)), 3),
                     C_ave=round(f(t.mean(), k.mean()), 3), C_med=round(f(np.median(t), np.median(k)), 3),
                     C_node=round(per.mean(), 3), C_glob=round(2 * t.sum() / np.sum(k * (k - 1)), 3),
                     transitivity=round(nx.transitivity(G), 3)))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "table1.csv"), index=False)
print(df.to_string(index=False))
