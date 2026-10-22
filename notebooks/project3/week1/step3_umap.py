import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap

from src.data import binary_label, study_table, top_k_classes, usable_findings
from src.embeddings import EMBEDDINGS_DIR, embedding_path, load_embeddings

FIG_DIR = Path("figures/week1")
GREY = "#d0d0d0"
POS, NEG = "#d62728", "#1f77b4"            # positive red, negative blue
CATEGORICAL = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
SEED = 0


def umap_coords(X: np.ndarray, n_neighbors: int, tag: str) -> np.ndarray:
    """Compute (or load cached) 2-D UMAP coordinates."""
    cache = EMBEDDINGS_DIR / f"umap_{tag}_nn{n_neighbors}.npy"
    if cache.exists():
        return np.load(cache)
    print(f"  running UMAP n_neighbors={n_neighbors} on {len(X)} studies ...", flush=True)
    Z = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, random_state=SEED).fit_transform(X)
    np.save(cache, Z)
    return Z


def plot_binary(ax, Z, y, title):
    """Clear cases in colour (positive / negative), everything else grey underneath."""
    rest = y.isna().values
    ax.scatter(*Z[rest].T, s=1, c=GREY, rasterized=True)
    for val, col, name in [(0.0, NEG, "negative"), (1.0, POS, "positive")]:
        m = (y == val).values
        ax.scatter(*Z[m].T, s=3, c=col, label=f"{name} ({m.sum()})", rasterized=True)
    ax.legend(markerscale=4, fontsize=7, loc="best")
    ax.set_title(title)


def plot_categorical(ax, Z, labels: pd.Series, title):
    """Each class its own colour (fixed order by frequency), missing / other in grey."""
    rest = labels.isna().values
    ax.scatter(*Z[rest].T, s=1, c=GREY, rasterized=True)
    for i, cls in enumerate(labels.value_counts().index[: len(CATEGORICAL)]):
        m = (labels == cls).values
        ax.scatter(*Z[m].T, s=1.5, c=CATEGORICAL[i], label=f"{cls} ({m.sum()})", rasterized=True)
    ax.legend(markerscale=6, fontsize=6, loc="best")
    ax.set_title(title)


def plot_numeric(ax, Z, values: pd.Series, title, clip_q=0.99):
    """Continuous value on a colour scale; extreme values clipped so they don't wash out the scale."""
    v = values.astype(float).values
    ok = ~np.isnan(v)
    ax.scatter(*Z[~ok].T, s=1, c=GREY, rasterized=True)
    vmax = np.nanquantile(v, clip_q)
    sc = ax.scatter(*Z[ok].T, s=1.5, c=np.minimum(v[ok], vmax), cmap="viridis", rasterized=True)
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    ax.set_title(title)


def make_figure(Z: np.ndarray, s: pd.DataFrame, n_neighbors: int) -> Path:
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    plot_binary(axes[0, 0], Z, binary_label(s, "edema"), "Edema (CheXbert)")
    plot_binary(axes[0, 1], Z, binary_label(s, "pleural_effusion"), "Pleural effusion (CheXbert)")
    plot_categorical(axes[0, 2], Z, s["health_system"], "Health system")
    plot_categorical(axes[0, 3], Z, top_k_classes(s["manufacturer"], 6), "Manufacturer (top 6)")
    plot_numeric(axes[1, 0], Z, s["age_years"], "Age (years)")
    plot_numeric(axes[1, 1], Z, s["study_year"], "Study year", clip_q=1.0)
    plot_numeric(axes[1, 2], Z, s["findings_words"], "Findings length (words, clipped at 99th pct)")
    plot_categorical(axes[1, 3], Z, s["sex"].where(s["sex"].isin(["F", "M"])), "Sex")
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"UMAP of BioClinicalBERT Findings embeddings (mean pooling), n_neighbors={n_neighbors}, "
                 f"{len(s)} studies", fontsize=14)
    fig.tight_layout()
    out = FIG_DIR / f"umap_nn{n_neighbors}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nn", type=int, nargs="+", default=[5, 15, 200])
    args = ap.parse_args()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    studies = study_table()
    X = load_embeddings(embedding_path("findings"), studies)
    keep = usable_findings(studies).values            # drop empty / very short Findings
    s, X = studies[keep].reset_index(drop=True), X[keep]
    print(f"{keep.sum()} of {len(keep)} studies have usable Findings")

    for k in args.nn:
        Z = umap_coords(X, k, tag="findings_mean_layer-1")
        print(f"  saved {make_figure(Z, s, k)}")


if __name__ == "__main__":
    main()