"""Network classification, Section 3.4, Tables 6-8. Usage: python exp_classify.py [networks per model, default 1000]
Networks are generated from four models, with the number of nodes uniform in 100..2000 and the
parameter ranges of Table 6:
  ER  connection probability p ∈ [0.02, 0.5]
  WS  K = 8, rewiring probability p ∈ [0.02, 0.1]
  BA  m ∈ {2,3,4,5}
  RPL random network with power-law expected degrees (Chung-Lu), exponent a ∈ [2, 3],
      mean degree uniform in [4, 10].
Two feature sets:
  PNN - global P-vector (distance part zero-padded to the largest diameter, last entry the unreachable
        fraction) plus the NND;
  FV  - average clustering, transitivity, degree assortativity, average shortest-path length, diameter,
        effective diameter (90th percentile of distances). For disconnected networks the distance-based
        features use reachable pairs only.
Classifier: decision tree; ablations with the global P-vector only and the NND only; stratified 10-fold
cross-validation; confusion matrices and accuracies; the computing time of each feature set is recorded.
Features are cached in results/classify_features.pkl; delete it to regenerate the networks."""
import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp
from multiprocessing import Pool
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from pvec import distance_matrix, p_matrix, p_vector, nnd

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
MODELS = ["ER", "BA", "WS", "RPL"]


def generate(model, seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(100, 2001))
    if model == "ER":
        par = rng.uniform(0.02, 0.5)
        G = nx.fast_gnp_random_graph(n, par, seed=seed)
    elif model == "WS":
        par = rng.uniform(0.02, 0.1)
        G = nx.watts_strogatz_graph(n, 8, par, seed=seed)
    elif model == "BA":
        par = int(rng.integers(2, 6))
        G = nx.barabasi_albert_graph(n, par, seed=seed)
    else:
        par = rng.uniform(2, 3)
        kbar = rng.uniform(4, 10)
        w = np.arange(1, n + 1) ** (-1 / (par - 1))
        w = np.minimum(w * kbar * n / w.sum(), n - 1)
        G = nx.expected_degree_graph(w.tolist(), seed=seed, selfloops=False)
    return G, n, par


def pnn_raw(G):
    P = p_matrix(G, distance_matrix(G))
    return p_vector(P), nnd(P)


def fv(G):
    D = distance_matrix(G)
    A = sp.csr_array(nx.to_scipy_sparse_array(G, nodelist=range(G.number_of_nodes()), dtype=float))
    k = np.asarray(A.sum(1)).ravel()
    tri = np.asarray((A @ A).multiply(A).sum(1)).ravel() / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.where(k > 1, 2 * tri / (k * (k - 1)), 0.0)
    trans = 2 * tri.sum() / np.sum(k * (k - 1))
    assort = nx.degree_assortativity_coefficient(G)
    d = D[D > 0]
    return np.array([c.mean(), trans, assort, d.mean(), d.max(), np.percentile(d, 90)])


def job(args):
    model, seed = args
    G, n, par = generate(model, seed)
    t0 = time.perf_counter(); pv, nd = pnn_raw(G); t1 = time.perf_counter()
    f = fv(G); t2 = time.perf_counter()
    return dict(model=model, seed=seed, n=n, par=par, m=G.number_of_edges(),
                pvec=pv, nnd=nd, fv=f, t_pnn=t1 - t0, t_fv=t2 - t1)


def features(per_model):
    path = os.path.join(OUT, "classify_features.pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    tasks = [(m, 100000 * i + s) for i, m in enumerate(MODELS) for s in range(per_model)]
    tasks.sort(key=lambda t: t[0] != "ER")  # dense ER networks are the slowest, so submit them first
    with Pool(24) as pool:
        recs = pool.map(job, tasks, chunksize=1)
    pickle.dump(recs, open(path, "wb"))
    return recs


def evaluate(X, y):
    cv = StratifiedKFold(10, shuffle=True, random_state=0)
    pred = cross_val_predict(DecisionTreeClassifier(random_state=0), X, y, cv=cv)
    cm = pd.crosstab(pd.Series(y, name="true"), pd.Series(pred, name="pred")).reindex(index=MODELS, columns=MODELS, fill_value=0)
    return cm, (pred == y).mean()


if __name__ == "__main__":
    recs = features(int(sys.argv[1]) if len(sys.argv) > 1 else 1000)
    y = np.array([r["model"] for r in recs])
    w = max(len(r["pvec"]) for r in recs)
    pv = np.array([np.r_[r["pvec"][:-1], np.zeros(w - len(r["pvec"])), r["pvec"][-1]] for r in recs])
    nd = np.array([[r["nnd"]] for r in recs])
    sets = {"FV": np.array([r["fv"] for r in recs]), "PNN": np.c_[pv, nd],
            "Pvec_only": pv, "NND_only": nd, "FV+PNN": np.c_[np.array([r["fv"] for r in recs]), pv, nd]}  # the last three are ablations
    rows = []
    for name, X in sets.items():
        cm, acc = evaluate(X, y)
        print(f"== {name}  accuracy {acc:.4f}\n{cm.to_string()}\n")
        cm.to_csv(os.path.join(OUT, f"classify_cm_{name}.csv"))
        rows.append(dict(features=name, accuracy=acc, dims=X.shape[1]))
    t = pd.DataFrame(recs)
    rows[0]["time_s"], rows[1]["time_s"] = t.t_fv.sum(), t.t_pnn.sum()
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "classify_summary.csv"), index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print("FV/PNN time per model (s)\n", t.groupby("model")[["t_fv", "t_pnn"]].sum().round(1).to_string())
