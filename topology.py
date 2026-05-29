import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from ripser import Rips
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

try:
    import ripserplusplus as rpp
    _HAS_RIPSER_PP = True
except ImportError:
    _HAS_RIPSER_PP = False


def maxmin_subsample(data: NDArray[np.floating], n_landmarks: int, return_indices: bool = False):
    n = data.shape[0]
    n_lm = min(n_landmarks, n)
    idx = [np.random.randint(n)]
    min_dists = np.full(n, np.inf)
    for _ in range(n_lm - 1):
        d = np.sum((data - data[idx[-1]]) ** 2, axis=1)
        np.minimum(min_dists, d, out=min_dists)
        idx.append(int(np.argmax(min_dists)))
    if return_indices:
        return data[idx], np.array(idx)
    return data[idx]


def preprocess(
    data: NDArray[np.floating],
    variance_ratio: float = 0.95,
    n_landmarks: int = 500,
) -> NDArray[np.floating]:
    landmarks = maxmin_subsample(data, n_landmarks)
    pca = PCA(n_components=variance_ratio, svd_solver='full')
    return pca.fit_transform(landmarks)


def compute_topology(
    pcs: NDArray[np.floating],
    thresh: float = np.inf,
    coeff: int = 47,
    umap_n_landmarks: int = 2000,
    raw_data: NDArray[np.floating] = None,
    image_paths: list = None,
) -> dict:
    """Run ripser + UMAP. Returns raw data dict for caching."""
    if _HAS_RIPSER_PP:
        dist = pairwise_distances(pcs, metric='cosine').astype(np.float32)
        thresh_flag = f'--threshold {thresh}' if np.isfinite(thresh) else ''
        dgms = rpp.run(f'--dim 2 --coeff {coeff} --format distance {thresh_flag}', dist)
    else:
        rips = Rips(maxdim=2, coeff=coeff, thresh=thresh, verbose=False)
        dgms = rips.fit_transform(pcs, metric='cosine')

    try:
        from cuml.manifold.umap import UMAP
    except ImportError:
        from umap import UMAP
    if raw_data is not None:
        umap_data, umap_idx = maxmin_subsample(raw_data, umap_n_landmarks, return_indices=True)
    else:
        umap_data = pcs
        umap_idx = np.arange(len(pcs))
    umap_embedding = UMAP(n_components=3).fit_transform(umap_data)
    umap_image_paths = [image_paths[i] for i in umap_idx] if image_paths is not None else None

    def _sorted_lifetimes(dgm):
        lt = dgm[:, 1] - dgm[:, 0]
        return np.sort(lt[np.isfinite(lt)])[::-1]

    h0_lt = _sorted_lifetimes(dgms[0])
    h1_lt = _sorted_lifetimes(dgms[1])
    h2_lt = _sorted_lifetimes(dgms[2]) if len(dgms) > 2 else np.array([])

    metrics = {
        'h0_bars':     float(len(dgms[0])),
        'h1_bars':     float(len(dgms[1])),
        'h2_bars':     float(len(dgms[2])) if len(dgms) > 2 else 0.0,
        'h1_bar1':     float(h1_lt[0]) if len(h1_lt) > 0 else 0.0,
        'h1_bar2':     float(h1_lt[1]) if len(h1_lt) > 1 else 0.0,
        'h1_bar3':     float(h1_lt[2]) if len(h1_lt) > 2 else 0.0,
        'h2_bar1':     float(h2_lt[0]) if len(h2_lt) > 0 else 0.0,
        'n_pcs':            float(pcs.shape[1]),
        'n_landmarks':      float(pcs.shape[0]),
        'n_umap_landmarks': float(umap_data.shape[0]),
    }

    return {'pcs': pcs, 'dgms': dgms, 'umap_embedding': umap_embedding, 'umap_image_paths': umap_image_paths, 'metrics': metrics}


def figures_from_cache(entry: dict) -> tuple:
    """Build (barcode, diagram, umap) figures from a cached topology entry."""
    dgms = entry['dgms']
    umap_embedding = entry['umap_embedding']
    n_lm = int(entry['metrics']['n_landmarks'])
    n_pc = int(entry['metrics']['n_pcs'])
    image_paths = entry.get('umap_image_paths')
    return (
        _barcode_figure(dgms, n_lm, n_pc),
        _diagram_figure(dgms, n_lm, n_pc),
        *_umap_figure(umap_embedding, n_lm, n_pc, int(entry['metrics'].get('n_umap_landmarks', n_lm)), image_paths=image_paths),
    )


def persistent_homology(
    pcs: NDArray[np.floating],
    thresh: float = np.inf,
    coeff: int = 47,
) -> tuple:
    entry = compute_topology(pcs, thresh=thresh, coeff=coeff)
    figs = figures_from_cache(entry)
    return (*figs, entry['metrics'], entry)


def _barcode_figure(dgms, n_landmarks, n_pc) -> go.Figure:
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    dim_names = ['H0', 'H1', 'H2']

    traces = []
    y = 0
    tick_vals, tick_text = [], []

    for dim, dgm in enumerate(dgms):
        finite = dgm[np.isfinite(dgm[:, 1])]
        infinite = dgm[~np.isfinite(dgm[:, 1])]
        all_bars = np.vstack([finite, infinite]) if len(infinite) else finite
        cap = finite[:, 1].max() * 1.1 if len(finite) else 1.0

        xs, ys = [], []
        for bar in all_bars:
            birth = bar[0]
            death = bar[1] if np.isfinite(bar[1]) else cap
            xs += [birth, death, None]
            ys += [y, y, None]
            tick_vals.append(y)
            tick_text.append(dim_names[dim])
            y += 1

        traces.append(go.Scatter(
            x=xs, y=ys,
            mode='lines',
            line=dict(color=colors[dim], width=2),
            name=dim_names[dim],
        ))

    fig = go.Figure(traces)
    fig.update_layout(
        title=f'Persistence barcode — {n_pc} PCs, {n_landmarks} landmarks',
        xaxis_title='Filtration radius',
        yaxis=dict(tickvals=tick_vals, ticktext=tick_text, showgrid=False),
        showlegend=True,
    )
    return fig


def _diagram_figure(dgms, n_landmarks, n_pc) -> go.Figure:
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    dim_names = ['H0', 'H1', 'H2']

    all_finite = np.concatenate([dgm[np.isfinite(dgm[:, 1]), 1] for dgm in dgms if len(dgm)])
    cap = all_finite.max() * 1.1 if len(all_finite) else 1.0

    traces = []
    for dim, dgm in enumerate(dgms):
        births = dgm[:, 0]
        deaths = np.where(np.isfinite(dgm[:, 1]), dgm[:, 1], cap)
        traces.append(go.Scatter(
            x=births, y=deaths,
            mode='markers',
            marker=dict(color=colors[dim], size=5),
            name=dim_names[dim],
        ))

    traces.append(go.Scatter(
        x=[0, cap], y=[0, cap],
        mode='lines',
        line=dict(color='black', dash='dash', width=1),
        showlegend=True, name='∞',
    ))
    traces.append(go.Scatter(
        x=[0, cap], y=[cap, cap],
        mode='lines',
        line=dict(color='black', dash='dot', width=1),
        showlegend=False,
    ))

    fig = go.Figure(traces)
    fig.update_layout(
        title=f'Persistence diagram — {n_pc} PCs, {n_landmarks} landmarks',
        xaxis_title='Birth', yaxis_title='Death',
        xaxis=dict(range=[0, cap]),
        yaxis=dict(range=[0, cap * 1.05]),
    )
    return fig


def _umap_figure(umap_embedding, n_landmarks, n_pc, n_umap_landmarks=None, image_paths=None) -> tuple:
    n_umap_landmarks = n_umap_landmarks or n_landmarks

    fig_3d = go.Figure(go.Scatter3d(
        x=umap_embedding[:, 0],
        y=umap_embedding[:, 1],
        z=umap_embedding[:, 2],
        mode='markers',
        marker=dict(size=2, color='#1f77b4', opacity=0.6),
        customdata=image_paths,
    ))
    fig_3d.update_layout(
        title=f'UMAP 3D — {n_pc} PCs, {n_umap_landmarks} points',
        scene=dict(xaxis_title='UMAP 1', yaxis_title='UMAP 2', zaxis_title='UMAP 3'),
    )

    fig_2d = go.Figure(go.Scatter(
        x=umap_embedding[:, 0],
        y=umap_embedding[:, 1],
        mode='markers',
        marker=dict(size=3, color='#1f77b4', opacity=0.6),
        customdata=image_paths,
    ))
    fig_2d.update_layout(
        title=f'UMAP 2D projection — {n_pc} PCs, {n_umap_landmarks} points',
        xaxis_title='UMAP 1', yaxis_title='UMAP 2',
        xaxis=dict(showgrid=False), yaxis=dict(showgrid=False),
    )

    return fig_2d, fig_3d


_HOVER_IMAGE_JS = """
(function() {
    var gd = document.querySelector('.plotly-graph-div');
    if (!gd) return;
    var tip = document.createElement('div');
    Object.assign(tip.style, {
        position: 'fixed', background: '#fff', border: '1px solid #aaa',
        padding: '3px', borderRadius: '4px', display: 'none',
        pointerEvents: 'none', zIndex: '9999',
        boxShadow: '2px 2px 8px rgba(0,0,0,0.25)'
    });
    document.body.appendChild(tip);
    gd.on('plotly_hover', function(data) {
        var cd = data.points[0].customdata;
        if (!cd) return;
        tip.innerHTML = '<img src="file://' + cd + '" style="max-width:180px;max-height:180px;display:block;">';
        tip.style.display = 'block';
    });
    gd.on('plotly_unhover', function() { tip.style.display = 'none'; });
    document.addEventListener('mousemove', function(e) {
        if (tip.style.display !== 'none') {
            tip.style.left = (e.clientX + 16) + 'px';
            tip.style.top = Math.max(0, e.clientY - 16) + 'px';
        }
    });
})();
"""


def write_umap_html(fig: go.Figure, path: str) -> None:
    has_images = any(getattr(t, 'customdata', None) is not None for t in fig.data)
    fig.write_html(path, post_script=_HOVER_IMAGE_JS if has_images else None)


def save_pngs(entry: dict, base_path: str) -> None:
    """Save barcode, diagram, umap2d, umap3d as PNGs using matplotlib."""
    dgms = entry['dgms']
    emb = entry['umap_embedding']
    n_lm = int(entry['metrics']['n_landmarks'])
    n_pc = int(entry['metrics']['n_pcs'])
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    dim_names = ['H0', 'H1', 'H2']

    # barcode
    fig, ax = plt.subplots(figsize=(10, 6))
    y = 0
    for dim, dgm in enumerate(dgms):
        finite = dgm[np.isfinite(dgm[:, 1])]
        cap = finite[:, 1].max() * 1.1 if len(finite) else 1.0
        infinite = dgm[~np.isfinite(dgm[:, 1])]
        all_bars = np.vstack([finite, infinite]) if len(infinite) else finite
        for bar in all_bars:
            birth = bar[0]
            death = bar[1] if np.isfinite(bar[1]) else cap
            ax.plot([birth, death], [y, y], color=colors[dim], linewidth=1.5)
            y += 1
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], color=c, label=n) for c, n in zip(colors, dim_names)])
    ax.set_xlabel('Filtration radius')
    ax.set_title(f'Persistence barcode — {n_pc} PCs, {n_lm} landmarks')
    ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(base_path + '__barcode.png', dpi=150)
    plt.close(fig)

    # diagram
    fig, ax = plt.subplots(figsize=(6, 6))
    all_finite = np.concatenate([dgm[np.isfinite(dgm[:, 1]), 1] for dgm in dgms if len(dgm)])
    cap = all_finite.max() * 1.1 if len(all_finite) else 1.0
    for dim, dgm in enumerate(dgms):
        births = dgm[:, 0]
        deaths = np.where(np.isfinite(dgm[:, 1]), dgm[:, 1], cap)
        ax.scatter(births, deaths, s=8, color=colors[dim], alpha=0.7, label=dim_names[dim])
    ax.plot([0, cap], [0, cap], 'k--', linewidth=1)
    ax.axhline(cap, color='k', linestyle=':', linewidth=1)
    ax.set_xlim(0, cap)
    ax.set_ylim(0, cap * 1.05)
    ax.set_xlabel('Birth')
    ax.set_ylabel('Death')
    ax.set_title(f'Persistence diagram — {n_pc} PCs, {n_lm} landmarks')
    ax.legend()
    plt.tight_layout()
    fig.savefig(base_path + '__diagram.png', dpi=150)
    plt.close(fig)

    # umap 2d
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(emb[:, 0], emb[:, 1], s=2, color='#1f77b4', alpha=0.5)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    n_umap = int(entry['metrics'].get('n_umap_landmarks', n_lm))
    ax.set_title(f'UMAP 2D — {n_pc} PCs, {n_umap} points')
    ax.axis('off')
    plt.tight_layout()
    fig.savefig(base_path + '__umap.png', dpi=150)
    plt.close(fig)

    # umap 3d
    fig = plt.figure(figsize=(8, 7))
    ax3d = fig.add_subplot(111, projection='3d')
    ax3d.scatter(emb[:, 0], emb[:, 1], emb[:, 2], s=1, color='#1f77b4', alpha=0.4)
    ax3d.set_xlabel('UMAP 1')
    ax3d.set_ylabel('UMAP 2')
    ax3d.set_zlabel('UMAP 3')
    ax3d.set_title(f'UMAP 3D — {n_pc} PCs, {n_umap} points')
    plt.tight_layout()
    fig.savefig(base_path + '__umap_3d.png', dpi=150)
    plt.close(fig)
