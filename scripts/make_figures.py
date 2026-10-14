"""Generate the README figures from the code and results/benchmarks.json."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pagerank import ECOMMERCE_SITE, build_transition_matrix, pagerank_sequential  # noqa: E402

OUT = ROOT / "images"
CREAM, INK, BURGUNDY, GOLD, ROSE, MUTED = "#FAF7F0", "#3D0F1C", "#5C1A2B", "#C9A55A", "#C58591", "#A88C7A"
CMAP = LinearSegmentedColormap.from_list("vintage", ["#FAF0E6", "#E8D5A8", GOLD, ROSE, "#8B3A4E", BURGUNDY])

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.titleweight": "bold", "axes.titlecolor": BURGUNDY, "axes.titlesize": 15,
})


def style(ax):
    ax.set_facecolor(CREAM)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, color=MUTED, linestyle="--", linewidth=0.7)


def web_graph(ranks, nodes):
    G = nx.DiGraph(ECOMMERCE_SITE)
    pos = nx.kamada_kawai_layout(G)
    values = [ranks[nodes.index(n)] for n in G.nodes()]
    fig, ax = plt.subplots(figsize=(11, 7.5), facecolor=CREAM)
    ax.set_facecolor(CREAM)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.4, arrowsize=14, min_target_margin=20, edge_color=MUTED, width=1.4,
                           connectionstyle="arc3,rad=0.12", node_size=2300)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=values, cmap=CMAP, node_size=2300,
                           edgecolors=INK, linewidths=1.8)
    for node, (x, y) in pos.items():
        dark = ranks[nodes.index(node)] > 0.075
        ax.text(x, y, node, ha="center", va="center", fontsize=7.5, fontweight="bold",
                color="#F5EDE0" if dark else INK)
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(min(values), max(values)))
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.01)
    cbar.set_label("PageRank score", color=INK)
    cbar.outline.set_edgecolor(MUTED)
    ax.set_title("15-page e-commerce site, coloured by PageRank")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "web_graph.png", dpi=150)


def speedup(results):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=CREAM,
                             gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    style(ax)
    palette = [GOLD, ROSE, BURGUNDY]
    max_w = 1
    for row, color in zip(results["data_parallel"], palette):
        ws = sorted(int(w) for w in row["workers"])
        sp = [row["sequential_s"] / row["workers"][str(w)] for w in ws]
        max_w = max(max_w, ws[-1])
        ax.plot(ws, sp, "o-", color=color, lw=2.4, ms=7, label=f"{row['n']:,} pages")
    ax.axhline(1, color=INK, lw=1, ls=":")
    ax.text(max_w, 1.03, "sequential", ha="right", va="bottom", color=INK, fontsize=9)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
    ax.set_xlabel("Worker threads")
    ax.set_ylabel("Speedup vs. vectorised sequential")
    ax.set_title("Data parallelism: speedup grows with graph size")
    ax.legend(frameon=False)

    ax = axes[1]
    style(ax)
    ax.grid(axis="x", visible=False)
    tp = results["task_parallel"]
    vals = [tp["sequential_s"], tp["parallel_s"]]
    bars = ax.bar(["Sequential", "3 tasks\nin parallel"], vals, color=[MUTED, BURGUNDY],
                  edgecolor=INK, width=0.6)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f} s", ha="center", va="bottom",
                fontweight="bold", color=INK)
    ax.set_ylim(0, max(vals) * 1.2)
    ax.set_ylabel("Wall-clock time")
    ax.set_title(f"Task parallelism: {vals[0] / vals[1]:.2f}× faster")
    cpus = results["machine"]["cpu_count"]
    fig.text(0.5, 0.005, f"{results['iterations']} iterations per run · median of 7 runs · "
             f"{cpus} CPUs · NumPy BLAS pinned to 1 thread", ha="center", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "speedup.png", dpi=150)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    M, nodes = build_transition_matrix(ECOMMERCE_SITE)
    ranks, _ = pagerank_sequential(M)
    web_graph(ranks, nodes)
    speedup(json.loads((ROOT / "results" / "benchmarks.json").read_text()))
    print("Figures written to", OUT)
