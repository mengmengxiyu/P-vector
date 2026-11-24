"""Computational cost (Table A2). Usage: python exp_complexity.py
Theory: the P-matrix needs one breadth-first search per node, O(N(N+M)) time; computed in batches, memory is
O(chunk·N + N·d).
Measured: BA(m=4) and ER (mean degree 8) networks, N = 1000..32000; single-thread computation of the P-matrix,
global P-vector and NND, each size in a separate subprocess; running time and peak memory (ru_maxrss, including
about 100 MB used by Python itself). For comparison, the time of the all-pairs shortest paths needed by the
distance-based FV features on the same networks (N ≤ 16000; larger N×N matrices do not fit in memory).
Output: results/complexity.csv."""
import os
import sys
import json
import subprocess
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results")
NS = [1000, 2000, 4000, 8000, 16000, 32000]

CHILD = r'''
import sys, time, json, resource, numpy as np, networkx as nx
sys.path.insert(0, HERE)
from pvec import p_matrix, p_vector, nnd, distance_matrix
model, n, what = sys.argv[1], int(sys.argv[2]), sys.argv[3]
G = nx.barabasi_albert_graph(n, 4, seed=0) if model == "BA" else nx.fast_gnp_random_graph(n, 8 / (n - 1), seed=0)
base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
t = time.perf_counter()
if what == "PNN":
    P = p_matrix(G, chunk=500); p_vector(P); nnd(P)
else:
    D = distance_matrix(G); d = D[D > 0]; d.mean(); d.max(); np.percentile(d, 90)
t = time.perf_counter() - t
print(json.dumps(dict(model=model, n=n, m=G.number_of_edges(), what=what, time_s=t,
                      peak_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, graph_mb=base / 1024)))
'''.replace("HERE", repr(HERE))

if __name__ == "__main__":
    rows = []
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    for model in ["BA", "ER"]:
        for n in NS:
            for what in ["PNN", "APSP"]:
                if what == "APSP" and n > 16000:
                    continue
                r = subprocess.run([sys.executable, "-c", CHILD, model, str(n), what], capture_output=True, text=True, env=env)
                rows.append(json.loads(r.stdout))
                print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "complexity.csv"), index=False)
