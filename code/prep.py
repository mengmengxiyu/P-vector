"""Build data/clean/<name>.txt from the downloads in data/raw: undirected, unweighted, without
self-loops or multi-edges, nodes relabelled from 0. Download links: data/raw/SOURCES.txt. Usage: python prep.py"""
import os
import networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "data", "raw")
CLEAN = os.path.join(HERE, "..", "data", "clean")

# name -> (raw file, format)
SRC = {
    "Karate": (None, "builtin"),
    "911": ("terrorists_911/edges.csv", "csv"),
    "Train bombing": ("moreno_train/out.moreno_train_train", "konect"),
    "David": ("adjnoun_adjacency/out.adjnoun_adjacency_adjacency", "konect"),
    "Physicians": ("moreno_innovation/out.moreno_innovation_innovation", "konect"),
    "U.S. Airports": ("inf-USAir97/inf-USAir97.mtx", "mtx"),
    "C elegans": ("bio-celegans/bio-celegans.mtx", "mtx"),
    "Power-494": ("power-494-bus/power-494-bus.mtx", "mtx"),
    "Asoiaf": ("asoiaf/out.asoiaf", "konect"),
    "Euroroad": ("subelj_euroroad/out.subelj_euroroad_euroroad", "konect"),
    "U.S. PowerGrid": ("opsahl-powergrid/out.opsahl-powergrid", "konect"),
}


def read_raw(name):
    path, fmt = SRC[name]
    if fmt == "builtin":
        return list(nx.karate_club_graph().edges())
    lines = open(os.path.join(RAW, path)).read().splitlines()
    edges = []
    for l in lines:
        if not l.strip() or l.startswith("%") or l.startswith("#"):
            continue
        t = l.replace(",", " ").split()
        edges.append((t[0], t[1]))
    if fmt == "mtx":
        edges = edges[1:]  # the first non-comment line gives the matrix size
    return edges


def build(name):
    G = nx.Graph()
    G.add_edges_from((a, b) for a, b in read_raw(name) if a != b)
    return nx.convert_node_labels_to_integers(G, ordering="sorted")


def load(name):
    G = nx.read_edgelist(os.path.join(CLEAN, name.replace(" ", "_") + ".txt"), nodetype=int)
    return nx.convert_node_labels_to_integers(G, ordering="sorted")


if __name__ == "__main__":
    os.makedirs(CLEAN, exist_ok=True)
    for name in SRC:
        G = build(name)
        nx.write_edgelist(G, os.path.join(CLEAN, name.replace(" ", "_") + ".txt"), data=False)
        print(name, G.number_of_nodes(), G.number_of_edges(), nx.number_connected_components(G))
