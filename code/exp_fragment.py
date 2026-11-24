"""Effect of disconnection and fragmentation on the P-vector (Appendix A, Table A1). Usage: python exp_fragment.py
(a) Decomposition over connected components: for components of sizes s_c, w_c = s_c(s_c-1) / (N(N-1)) and
      p_inf = 1 - sum_c w_c,   distance part p_j = sum_c w_c * p_j^(c),
    where p^(c) is the P-vector of component c on its own. Checked numerically on the disconnected
    networks Physicians and Euroroad.
(b) Random edge removal: 0%..90% of the edges of U.S. Airports, C elegans and Power-494, 20 realizations each.
    Records the number of components, the size of the largest component, p_inf, the JS divergence from the
    intact network, and the JS divergence of the distance distribution of reachable pairs (distance part
    divided by 1-p_inf). Output: results/fragment_decomp.csv, fragment_removal.csv."""
import os
import numpy as np
import pandas as pd
import networkx as nx
from prep import load
from pvec import p_matrix, p_vector
from exp_scale import js

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def part_a():
    rows = []
    for name in ["Physicians", "Euroroad"]:
        G = load(name)
        n = G.number_of_nodes()
        pv = p_vector(p_matrix(G))
        comps = [G.subgraph(c) for c in nx.connected_components(G)]
        w = np.array([len(c) * (len(c) - 1) / (n * (n - 1)) for c in comps])
        dist = np.zeros(len(pv) - 1)
        for wc, C in zip(w, comps):
            if len(C) > 1:
                pc = p_vector(p_matrix(nx.convert_node_labels_to_integers(C)))[:-1]
                dist[:len(pc)] += wc * pc
        rows.append(dict(network=name, components=len(comps), sizes=sorted(map(len, comps), reverse=True)[:6],
                         p_inf=pv[-1], p_inf_formula=1 - w.sum(),
                         max_abs_err_dist=np.abs(pv[:-1] - dist).max()))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "fragment_decomp.csv"), index=False)
    print(df.to_string(index=False))


def cond(p):
    return p[:-1] / (1 - p[-1])


def align(p, q):
    w = max(len(p), len(q))
    f = lambda v: np.r_[v[:-1], np.zeros(w - len(v)), v[-1]]
    return f(p), f(q)


def part_b():
    rows = []
    for name in ["U.S. Airports", "C elegans", "Power-494"]:
        G0 = load(name)
        p0 = p_vector(p_matrix(G0))
        E = list(G0.edges())
        for f in np.round(np.arange(0, 0.91, 0.1), 1):
            for r in range(20):
                rng = np.random.default_rng(r)
                G = G0.copy()
                drop = rng.choice(len(E), int(f * len(E)), replace=False)
                G.remove_edges_from(E[i] for i in drop)
                p = p_vector(p_matrix(G))
                a, b = align(p0, p)
                rows.append(dict(network=name, removed=f, rep=r, components=nx.number_connected_components(G),
                                 lcc=max(map(len, nx.connected_components(G))) / G.number_of_nodes(),
                                 p_inf=p[-1], js_full=js(a, b),
                                 js_reachable=js(np.r_[cond(a), 0], np.r_[cond(b), 0])))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "fragment_removal.csv"), index=False)
    print(df.groupby(["network", "removed"])[["components", "lcc", "p_inf", "js_full", "js_reachable"]].mean().round(3).to_string())


if __name__ == "__main__":
    part_a()
    part_b()
