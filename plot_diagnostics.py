"""
Shared plotting helpers: confusion-matrix heatmap and a 2D UMAP/t-SNE view
of the feature space, colored by class. Palette follows the project's
dataviz conventions: a single-hue sequential blue ramp for the confusion
matrix (magnitude), and the first three categorical slots (blue/orange/aqua
-- the only ones validated for all-pairs separation, i.e. a 3-class scatter)
for the class colors.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"

CATEGORICAL = {  # first 3 slots -- validated for all-pairs (scatter) separation
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
}
SEQ_BLUE = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _class_colors(classes):
    hues = list(CATEGORICAL.values())
    return {c: hues[i % len(hues)] for i, c in enumerate(classes)}


def plot_confusion_matrix(cm, classes, accuracy, balanced_accuracy, out_path):
    n = len(classes)
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(1.6 * n + 2.2, 1.6 * n + 1.6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1, aspect="equal")

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(classes, fontsize=11, color=INK_PRIMARY)
    ax.set_yticklabels(classes, fontsize=11, color=INK_PRIMARY)
    ax.set_xlabel("Predicted", fontsize=11, color=INK_SECONDARY)
    ax.set_ylabel("True", fontsize=11, color=INK_SECONDARY)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i in range(n):
        for j in range(n):
            frac = cm_norm[i, j]
            count = cm[i, j]
            txt_color = "#ffffff" if frac > 0.55 else INK_PRIMARY
            weight = "bold" if i == j else "normal"
            ax.text(j, i, f"{count}\n{frac:.0%}", ha="center", va="center",
                     fontsize=12, color=txt_color, fontweight=weight)
        # thin surface gaps between cells
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=3)
    ax.tick_params(which="minor", length=0)

    fig.suptitle(f"Accuracy: {accuracy:.1%}   |   Balanced accuracy: {balanced_accuracy:.1%}",
                 fontsize=15, fontweight="bold", color=INK_PRIMARY, y=0.99)
    ax.set_title(f"Confusion matrix -- held-out test set (n={cm.sum()})",
                 fontsize=11, color=INK_SECONDARY, pad=10)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _add_cluster_clouds(ax, emb, y, classes, colors, grid_n=180, pad_frac=0.25, max_alpha=0.34):
    """Soft KDE-density 'cloud' behind each class's points, so the clusters
    read as filled regions rather than just a scatter of dots. Drawn before
    the points (lower zorder) and clipped to the data's own extent + padding
    so independent classes don't bleed into a shared haze."""
    from scipy.stats import gaussian_kde

    xmin, xmax = emb[:, 0].min(), emb[:, 0].max()
    ymin, ymax = emb[:, 1].min(), emb[:, 1].max()
    xpad = (xmax - xmin) * pad_frac or 1.0
    ypad = (ymax - ymin) * pad_frac or 1.0
    xx, yy = np.mgrid[xmin - xpad:xmax + xpad:complex(grid_n),
                       ymin - ypad:ymax + ypad:complex(grid_n)]
    grid_pts = np.vstack([xx.ravel(), yy.ravel()])

    levels = np.linspace(0, 1, 9)
    for c in classes:
        pts = emb[y == c]
        if len(pts) < 5:
            continue
        kde = gaussian_kde(pts.T)
        kde.set_bandwidth(kde.factor * 1.5)  # smoother, cloud-like rather than lumpy
        density = kde(grid_pts).reshape(xx.shape)
        density = density / density.max()

        r, g, b = to_rgb(colors[c])
        cloud_cmap = LinearSegmentedColormap.from_list(
            f"cloud_{c}", [(r, g, b, 0.0), (r, g, b, max_alpha)]
        )
        ax.contourf(xx, yy, density, levels=levels, cmap=cloud_cmap,
                    vmin=0, vmax=1, antialiased=True, zorder=1)
    return xmin - xpad, xmax + xpad, ymin - ypad, ymax + ypad


def _scatter(ax, emb, y, classes, colors, title):
    ax.set_facecolor(SURFACE)
    xlim = _add_cluster_clouds(ax, emb, y, classes, colors)
    for c in classes:
        m = y == c
        ax.scatter(emb[m, 0], emb[m, 1], s=22, color=colors[c], alpha=0.9,
                   edgecolors="white", linewidths=0.5, label=c, zorder=3)
        cx, cy = emb[m, 0].mean(), emb[m, 1].mean()
        ax.annotate(c, (cx, cy), fontsize=10, fontweight="bold", color=INK_PRIMARY,
                    ha="center", va="center", zorder=4,
                    bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec=colors[c], lw=1.3, alpha=0.95))
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(xlim[2], xlim[3])
    ax.set_title(title, fontsize=12, color=INK_PRIMARY)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(AXIS)
    ax.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK_SECONDARY)


def _silhouette(emb, y):
    from sklearn.metrics import silhouette_score
    try:
        return silhouette_score(emb, y)
    except Exception:
        return float("nan")


def plot_feature_space_2d(X, y, classes, out_path, random_state=0):
    """Standardizes X, computes both a UMAP and a t-SNE 2D embedding, scores
    cluster separation with silhouette score, and saves BOTH panels (the
    better-separated one is called out in the title) so the choice is visible
    rather than asserted."""
    from sklearn.preprocessing import StandardScaler
    Xs = StandardScaler().fit_transform(X)
    colors = _class_colors(classes)

    embeddings = {}
    try:
        import umap
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.2, random_state=random_state)
        embeddings["UMAP"] = reducer.fit_transform(Xs)
    except Exception as e:
        print("UMAP unavailable:", e)

    from sklearn.manifold import TSNE
    perplexity = min(30, max(5, len(Xs) // 20))
    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=random_state)
    embeddings["t-SNE"] = tsne.fit_transform(Xs)

    scores = {name: _silhouette(emb, y) for name, emb in embeddings.items()}
    best = max(scores, key=scores.get)

    n_panels = len(embeddings)
    fig, axes = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, 5.8))
    if n_panels == 1:
        axes = [axes]
    fig.patch.set_facecolor(SURFACE)
    for ax, (name, emb) in zip(axes, embeddings.items()):
        tag = "  (best cluster separation)" if name == best else ""
        _scatter(ax, emb, y, classes, colors, f"{name} -- silhouette={scores[name]:.3f}{tag}")

    fig.suptitle("2D view of the 97-dimensional feature space, colored by true class",
                 fontsize=14, fontweight="bold", color=INK_PRIMARY)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return scores, best
