"""Network evolution, Section 3.3 and Fig. 5. Usage: python exp_evolution.py
(a) BA networks with m=2 and 100..1000 nodes in steps of 50; (b) a regular ring with N=1000, k=6
rewired with probability 0.1..1.0 (WS model). 100 networks per setting; P-vectors are averaged.
Output: results/evolution_BA.csv, evolution_WS.csv; columns are distances 1..D and inf."""
import os
import numpy as np
import pandas as pd
import networkx as nx
from pvec import p_matrix, p_vector

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
REP = 100


def mean_pvec(make, seeds):
    vs = [p_vector(p_matrix(make(s))) for s in seeds]
    w = max(len(v) for v in vs)
    pad = lambda v: np.r_[v[:-1], np.zeros(w - len(v)), v[-1]]
    return np.mean([pad(v) for v in vs], 0)


def table(rows, index):
    w = max(len(r) for r in rows)
    rows = [np.r_[r[:-1], np.zeros(w - len(r)), r[-1]] for r in rows]
    return pd.DataFrame(rows, index=index, columns=[*range(1, w), "inf"])


if __name__ == "__main__":
    Ns = list(range(100, 1001, 50))
    ba = table([mean_pvec(lambda s: nx.barabasi_albert_graph(n, 2, seed=s), range(REP)) for n in Ns], Ns)
    ba.to_csv(os.path.join(OUT, "evolution_BA.csv"))
    ps = [round(0.1 * i, 1) for i in range(1, 11)]
    ws = table([mean_pvec(lambda s: nx.watts_strogatz_graph(1000, 6, p, seed=s), range(REP)) for p in ps], ps)
    ws.to_csv(os.path.join(OUT, "evolution_WS.csv"))
    print(ba.round(3).to_string()); print(ws.round(3).to_string())
