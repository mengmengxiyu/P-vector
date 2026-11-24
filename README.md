# Characterization of complex networks based on P-vector

Code for the paper

> M. Li, Y. Jiang, Z. Di, *Characterization of complex networks based on P-vector*, Chaos, Solitons & Fractals (under review).

The scripts compute the node P-vector, the global P-vector and the network node dispersion (NND), and reproduce all tables and figures of the paper.

## Requirements

Python 3.10 or later with

```
numpy scipy pandas networkx scikit-learn matplotlib
```

The results in `results/` were obtained with Python 3.14, numpy 2.5, scipy 1.18, pandas 3.0, networkx 3.6, scikit-learn 1.9 and matplotlib 3.11.

## Layout

```
code/               scripts (see below)
data/raw/SOURCES.txt download links of all data and extraction commands
results/            output tables (csv), logs and figures reported in the paper
```

## Usage

The data are not included. First download the files listed in `data/raw/SOURCES.txt` into `data/raw/` and extract them as described there. Then run the scripts from the `code/` directory; `prep.py` builds the edge lists in `data/clean/`, and each experiment writes its output to `results/`.

| Script | Content | Paper |
|---|---|---|
| `pvec.py` | P-matrix, P-vector, JS divergence, NND (imported by the other scripts) | Section 2 |
| `prep.py` | builds `data/clean/` from `data/raw/` | |
| `exp_table1.py` | basic properties of the real-world networks, clustering-coefficient estimate | Table 1 |
| `exp_similarity.py` | node similarity: P-vector, CN, LP, Katz | Tables 2–3, Fig. 1 |
| `exp_similarity2.py` | node similarity: additional baselines (SimRank, DeepWalk, GraphWave, random) | Table 4 |
| `exp_cluster.py` | node clustering on four real-world networks | Figs. 3–4, Table 5 |
| `exp_cluster2.py` | silhouette scores; structural-role benchmarks | Table 5, Table A3 |
| `exp_evolution.py` | P-vector during BA growth and WS rewiring | Fig. 5 |
| `exp_classify.py` | classification of 4,000 synthetic networks (ER, BA, WS, RPL) | Tables 6–8 |
| `exp_scale.py` | dependence on network size; cross-size classification | Section 2.3 |
| `exp_real_classify.py` | classification of TUDataset graphs | Table 8 |
| `exp_fragment.py` | decomposition over connected components; random edge removal | Appendix A, Table A1 |
| `exp_complexity.py` | running time and memory of the P-vector | Table A2 |
| `figures.py` | Figures 1–5 from the result files | Figs. 1–5 |

Example:

```
cd code
python prep.py
python exp_table1.py
python exp_classify.py      # writes results/classify_features.pkl
python exp_scale.py         # reads results/classify_features.pkl
python figures.py
```

`exp_classify.py` generates the synthetic networks and caches their features in `results/classify_features.pkl`; `exp_scale.py` reuses this cache. Delete the file to regenerate the networks. Generating and processing the 4,000 networks takes several hours on a single core. The other scripts finish within minutes to about an hour.

## Data

The real-world networks are taken from KONECT, Network Repository, Netzschleuder and the networkx built-in Karate club; the graph-classification datasets are from the TUDataset collection (Morris et al., 2020). The files are not redistributed here; exact download links are listed in `data/raw/SOURCES.txt`. Please cite the original sources when using these data.

## Contact

Yuanxiang Jiang (corresponding author), jiangyx@mail.bnu.edu.cn
