"""Dependence on network size (Section 2.3). Usage: python exp_scale.py
(a) Four models with mean degree 8: ER, BA(m=4), WS(K=8, p=0.05), RPL(a=2.5); 250..8000 nodes, 20 networks each.
    JS divergences of the global P-vectors between sizes of the same model and between models of the same size,
    and the mean distance as a function of size.
    Output: results/scale_pvec.csv (one row per network), scale_js.csv (mean JS for each pair of model × size).
(b) Cross-size classification with the 4,000 networks cached by exp_classify.py: train on networks with
    fewer than 1000 nodes and test on the rest, then the reverse. Output: results/scale_cross.csv."""
import os
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from multiprocessing import Pool
from sklearn.tree import DecisionTreeClassifier
from pvec import p_matrix, p_vector, distance_matrix

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
NS = [250, 500, 1000, 2000, 4000, 8000]
REP = 20
MODELS = ["ER", "BA", "WS", "RPL"]


def make(model, n, seed):
    if model == "ER":
        return nx.fast_gnp_random_graph(n, 8 / (n - 1), seed=seed)
    if model == "BA":
        return nx.barabasi_albert_graph(n, 4, seed=seed)
    if model == "WS":
        return nx.watts_strogatz_graph(n, 8, 0.05, seed=seed)
    w = np.arange(1, n + 1) ** (-1 / 1.5)
    w = np.minimum(w * 8 * n / w.sum(), n - 1)
    return nx.expected_degree_graph(w.tolist(), seed=seed, selfloops=False)


def job(a):
    model, n, seed = a
    pv = p_vector(p_matrix(make(model, n, seed)))
    return model, n, seed, pv


def pad(vs):
    w = max(len(v) for v in vs)
    return np.array([np.r_[v[:-1], np.zeros(w - len(v)), v[-1]] for v in vs])


def js(p, q):
    m = (p + q) / 2
    h = lambda x: -np.sum(np.where(x > 0, x * np.log2(np.where(x > 0, x, 1)), 0))
    return max(h(m) - (h(p) + h(q)) / 2, 0.0)


def part_a():
    tasks = [(m, n, s) for m in MODELS for n in NS for s in range(REP)]
    with Pool(24) as pool:
        recs = pool.map(job, tasks, chunksize=1)
    P = pad([r[3] for r in recs])
    meta = pd.DataFrame([r[:3] for r in recs], columns=["model", "n", "seed"])
    dist = np.arange(1, P.shape[1])
    meta["mean_dist"] = (P[:, :-1] * dist).sum(1) / (1 - P[:, -1])
    meta["p_inf"] = P[:, -1]
    pd.concat([meta, pd.DataFrame(P, columns=[*map(str, dist), "inf"])], axis=1).to_csv(
        os.path.join(OUT, "scale_pvec.csv"), index=False)
    # mean P-vector for each (model, size), then pairwise JS
    key = meta.model + "@" + meta.n.astype(str)
    avg = {k: P[(key == k).values].mean(0) for k in key.unique()}
    rows = [dict(a=a, b=b, js=js(avg[a], avg[b])) for a in avg for b in avg if a < b]
    J = pd.DataFrame(rows)
    J[["ma", "na"]] = J.a.str.split("@", expand=True)
    J[["mb", "nb"]] = J.b.str.split("@", expand=True)
    J.to_csv(os.path.join(OUT, "scale_js.csv"), index=False)
    same_model = J[J.ma == J.mb]
    same_size = J[(J.ma != J.mb) & (J.na == J.nb)]
    print("Mean distance vs size:\n", meta.groupby(["model", "n"]).mean_dist.mean().unstack().round(2).to_string())
    print("JS between sizes of the same model:\n", same_model.groupby("ma").js.agg(["mean", "max"]).round(3).to_string())
    print("JS between models of the same size:\n", same_size.groupby("na").js.agg(["mean", "min"]).round(3).to_string())
    # whether the nearest other (model, size) group has the same model
    nearest = []
    for k in avg:
        others = [(js(avg[k], avg[o]), o) for o in avg if o != k]
        nearest.append(dict(group=k, nearest=min(others)[1], same_model=min(others)[1].split("@")[0] == k.split("@")[0]))
    nearest = pd.DataFrame(nearest)
    print("Fraction whose nearest group has the same model:", nearest.same_model.mean())


def part_b():
    recs = pickle.load(open(os.path.join(OUT, "classify_features.pkl"), "rb"))
    y = np.array([r["model"] for r in recs])
    n = np.array([r["n"] for r in recs])
    pv = pad([r["pvec"] for r in recs])
    sets = {"FV": np.array([r["fv"] for r in recs]),
            "PNN": np.c_[pv, [r["nnd"] for r in recs]], "Pvec_only": pv}
    rows = []
    for train, test, label in [(n < 1000, n >= 1000, "train N<1000, test N>=1000"),
                               (n >= 1000, n < 1000, "train N>=1000, test N<1000")]:
        for name, X in sets.items():
            clf = DecisionTreeClassifier(random_state=0).fit(X[train], y[train])
            rows.append(dict(split=label, features=name, n_train=train.sum(), n_test=test.sum(),
                             accuracy=(clf.predict(X[test]) == y[test]).mean()))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "scale_cross.csv"), index=False)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    part_b()
    part_a()
