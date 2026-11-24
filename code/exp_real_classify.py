"""Classification of real-world graphs (Table 8). Usage: python exp_real_classify.py
Data: TUDataset graph-classification datasets MUTAG, PROTEINS, IMDB-BINARY, IMDB-MULTI, REDDIT-BINARY;
graph structure only. Expected in data/raw/tu/<dataset>/ (see data/raw/SOURCES.txt).
Features: FV, PNN, global P-vector only, NND only, FV and PNN combined; defined as in exp_classify.py.
Classifiers: decision tree (as in Section 3.4) and random forest; stratified 10-fold cross-validation;
mean and standard deviation of the fold accuracies. Features are computed serially in one process and the
total time of each feature set is recorded. Output: results/real_classify.csv."""
import os
import time
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from exp_classify import pnn_raw, fv

HERE = os.path.dirname(os.path.abspath(__file__))
TU = os.path.join(HERE, "..", "data", "raw", "tu")
OUT = os.path.join(HERE, "..", "results")
SETS = ["MUTAG", "PROTEINS", "IMDB-BINARY", "IMDB-MULTI", "REDDIT-BINARY"]


def read_tu(ds):
    f = lambda s: os.path.join(TU, ds, f"{ds}_{s}.txt")
    gi = np.loadtxt(f("graph_indicator"), dtype=int)
    y = np.loadtxt(f("graph_labels"), dtype=int)
    E = np.loadtxt(f("A"), dtype=int, delimiter=",")
    graphs = [nx.Graph() for _ in y]
    for v, g in enumerate(gi, 1):
        graphs[g - 1].add_node(v)
    for a, b in E:
        if a != b:
            graphs[gi[a - 1] - 1].add_edge(a, b)
    return [nx.convert_node_labels_to_integers(G) for G in graphs], y


def run(ds):
    graphs, y = read_tu(ds)
    t0 = time.perf_counter(); raw = [pnn_raw(G) for G in graphs]; t_pnn = time.perf_counter() - t0
    t0 = time.perf_counter(); F = np.array([fv(G) for G in graphs]); t_fv = time.perf_counter() - t0
    F = np.nan_to_num(F)  # assortativity is undefined for regular graphs; set to 0
    w = max(len(p) for p, _ in raw)
    pv = np.array([np.r_[p[:-1], np.zeros(w - len(p)), p[-1]] for p, _ in raw])
    nd = np.array([[d] for _, d in raw])
    sets = {"FV": F, "PNN": np.c_[pv, nd], "Pvec_only": pv, "NND_only": nd, "FV+PNN": np.c_[F, pv, nd]}
    cv = StratifiedKFold(10, shuffle=True, random_state=0)
    rows = []
    for name, X in sets.items():
        for clf_name, clf in [("DT", DecisionTreeClassifier(random_state=0)),
                              ("RF", RandomForestClassifier(300, random_state=0, n_jobs=8))]:
            acc = cross_val_score(clf, X, y, cv=cv)
            rows.append(dict(dataset=ds, graphs=len(y), classes=len(set(y)),
                             avg_nodes=round(np.mean([G.number_of_nodes() for G in graphs]), 1),
                             features=name, classifier=clf_name, acc_mean=acc.mean(), acc_std=acc.std(),
                             time_s={"FV": t_fv, "PNN": t_pnn}.get(name)))
    return rows


if __name__ == "__main__":
    rows = []
    for ds in SETS:
        rows += run(ds)
        print(pd.DataFrame(rows[-10:])[["dataset", "features", "classifier", "acc_mean", "acc_std", "time_s"]].round(3).to_string(index=False), flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "real_classify.csv"), index=False)
