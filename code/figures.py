"""Figures 1-5 from the files in results/; output in results/figures/. Usage: python figures.py"""
import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from prep import load
from pvec import p_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9})
METH = {"PVJS": ("tab:red", "o"), "CN": ("tab:blue", "d"), "LP": ("tab:green", "^"), "Katz3": ("tab:orange", "*")}
SIM_NETS = ["Karate", "Train bombing", "David", "Physicians", "C elegans", "Asoiaf", "Euroroad", "U.S. PowerGrid"]
fn = lambda s: s.replace(" ", "_")


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig1(tie="random"):
    fig = plt.figure(figsize=(10, 8.5))
    gs = fig.add_gridspec(3, 6)  # each panel spans two cells; panels g and h are centred in the last row
    axes = [fig.add_subplot(gs[r, 2 * c:2 * c + 2]) for r in range(2) for c in range(3)]
    axes += [fig.add_subplot(gs[2, 1:3]), fig.add_subplot(gs[2, 3:5])]
    for ax, name, lab in zip(axes, SIM_NETS, "abcdefgh"):
        d = pd.read_csv(os.path.join(RES, f"similarity_{fn(name)}.csv"))
        d = d[d.tie == tie].set_index("method")
        for m, (c, mk) in METH.items():
            ax.errorbar(range(1, 11), d.loc[m, [f"MDP{t}" for t in range(1, 11)]],
                        yerr=d.loc[m, [f"VDP{t}" for t in range(1, 11)]], color=c, marker=mk, ms=4,
                        ls="--", lw=1, capsize=2, label=m)
        ax.set_title(f"({lab}) {name}")
        ax.set_xlabel("Step")
        ax.set_ylabel("MDP")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    save(fig, "fig1_similarity_SI")


def heat(ax, P, title):
    im = ax.imshow(P, aspect="auto", cmap="Reds", interpolation="nearest")
    w = P.shape[1]
    ax.set_xticks(range(w), [*range(1, w), "∞"], fontsize=7)
    ax.set_xlabel("distance")
    ax.set_ylabel("node")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.05)


def fig2():
    nets = [("(a) ER  N=1000, p=0.006", nx.erdos_renyi_graph(1000, 0.006, seed=0)),
            ("(b) BA  N=1000, m=2", nx.barabasi_albert_graph(1000, 2, seed=0)),
            ("(c) 911", load("911")), ("(d) Train bombing", load("Train bombing"))]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (t, G) in zip(axes.flat, nets):
        heat(ax, p_matrix(G), t)
    fig.tight_layout()
    save(fig, "fig2_pmatrix")


def order_by(lab, key):
    """Relabel groups by decreasing mean of key: P-vector clusters by closeness (core = 0, so colours run from core to periphery), communities by size."""
    lab, key = np.asarray(lab), np.asarray(key, float)
    rank = sorted(set(lab), key=lambda c: -key[lab == c].mean())
    return np.array([rank.index(c) for c in lab])


def draw_groups(ax, G, pos, lab, title, small):
    nx.draw_networkx_edges(G, pos, ax=ax, width=0.3 if small else 0.6, alpha=0.4)
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=range(len(lab)), node_color=lab, cmap="viridis", vmin=0, vmax=max(lab.max(), 1),
                           node_size=15 if small else 110, linewidths=0)
    if not small:
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=4.5, font_color="white")
    ax.set_title(title)
    ax.axis("off")


def fig34():
    for figname, nets, with_heat in [("fig3_cluster_small", ["911", "Train bombing"], False),
                                     ("fig4_cluster_large", ["Power-494", "U.S. Airports"], True)]:
        cols = 4 if with_heat else 3
        fig, axes = plt.subplots(2, cols, figsize=(3.6 * cols, 7))
        letters = iter("abcdefgh")
        for row, name in zip(axes, nets):
            G = load(name)
            d = pd.read_csv(os.path.join(RES, f"cluster_{fn(name)}.csv"))
            pos = nx.kamada_kawai_layout(G)
            small = G.number_of_nodes() > 200
            panels = row
            if with_heat:
                heat(row[0], p_matrix(G), f"({next(letters)}) {name}: P matrix")
                panels = row[1:]
            for ax, col, t in zip(panels, ["greedy", "kmeans", "hier"],
                                  ["greedy modularity", "K-means (P-vector)", "hierarchical (JS, ward)"]):
                key = d.groupby(col)["node"].transform("size") if col == "greedy" else d["closeness"]
                draw_groups(ax, G, pos, order_by(d[col], key), f"({next(letters)}) {name}: {t}", small)
        fig.tight_layout()
        save(fig, figname)


def fig5():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, f, t, yl in [(axes[0], "evolution_BA.csv", "(a) BA growth, m=2", "number of nodes"),
                         (axes[1], "evolution_WS.csv", "(b) regular → WS → ER, N=1000, k=6", "rewiring probability")]:
        d = pd.read_csv(os.path.join(RES, f), index_col=0)
        heat(ax, d.values, t)
        ax.set_yticks(range(len(d)), d.index)
        ax.set_ylabel(yl)
    fig.tight_layout()
    save(fig, "fig5_evolution")


if __name__ == "__main__":
    fig1(); fig2(); fig34(); fig5()
    print(sorted(os.listdir(FIG)))
