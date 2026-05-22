import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from ripser import Rips
import plotly.graph_objects as go


def preprocess(
    data: NDArray[np.floating],
    variance_ratio: float = 0.95,
) -> NDArray[np.floating]:
    """PCA projection retaining at least `variance_ratio` of explained variance.

    Args:
        data: Input array of shape (N, D).
        variance_ratio: Fraction of variance to retain (0, 1].

    Returns:
        PCA-projected array of shape (N, n_components).
    """
    pca = PCA(n_components=variance_ratio, svd_solver='full')
    return pca.fit_transform(data)


def _barcode_figure(
    dgms: list,
    n_landmarks: int,
    n_pc: int,
) -> go.Figure:

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    dim_names = ['H0', 'H1', 'H2']

    bars = []
    y = 0
    tick_vals, tick_text = [], []

    for dim, dgm in enumerate(dgms):
        finite = dgm[np.isfinite(dgm[:, 1])]
        infinite = dgm[~np.isfinite(dgm[:, 1])]
        all_bars = np.vstack([finite, infinite]) if len(infinite) else finite

        for bar in all_bars:
            birth = bar[0]
            death = bar[1] if np.isfinite(bar[1]) else dgm[np.isfinite(dgm[:, 1]), 1].max() * 1.1
            bars.append(go.Scatter(
                x=[birth, death],
                y=[y, y],
                mode='lines',
                line=dict(color=colors[dim], width=2),
                showlegend=(y == 0 or (dim > 0 and all(b['line']['color'] != colors[dim] for b in bars if isinstance(b, go.Scatter)))),
                name=dim_names[dim],
            ))
            tick_vals.append(y)
            tick_text.append(f'{dim_names[dim]}')
            y += 1

    fig = go.Figure(bars)
    fig.update_layout(
        title=f'Persistence barcode — {n_pc} PCs, {n_landmarks} landmarks',
        xaxis_title='Filtration radius',
        yaxis=dict(tickvals=tick_vals, ticktext=tick_text, showgrid=False),
        showlegend=False,
    )
    return fig


def persistent_homology(
    pcs: NDArray[np.floating],
    n_landmarks: int = 500,
    thresh: float = np.inf,
    coeff: int = 47,
) -> tuple:
    """Persistent cohomology on PCA-projected data via landmark-based Ripser.

    Uses maxmin landmark subsampling and Z_p coefficients (default p=47).
    Torus signature: β=(1, 2, 1).

    Args:
        pcs: PCA-projected activations, shape (N, n_components).
        n_landmarks: Max landmark points for maxmin subsampling.
        thresh: Max filtration radius; caps simplex explosion at maxdim=2.
        coeff: Coefficient field prime (47 follows Gardner et al.).

    Returns:
        Tuple of (barcode_figure, metrics_dict).
    """
    n_pc = pcs.shape[1]
    n_lm = min(n_landmarks, pcs.shape[0])

    rips = Rips(maxdim=2, coeff=coeff, n_perm=n_lm, thresh=thresh, verbose=False)
    dgms = rips.fit_transform(pcs, metric='cosine')

    def _sorted_lifetimes(dgm: NDArray) -> NDArray:
        lt = dgm[:, 1] - dgm[:, 0]
        return np.sort(lt[np.isfinite(lt)])[::-1]

    h0_lt = _sorted_lifetimes(dgms[0])
    h1_lt = _sorted_lifetimes(dgms[1])
    h2_lt = _sorted_lifetimes(dgms[2]) if len(dgms) > 2 else np.array([])

    metrics: dict[str, float] = {
        'h0_bars':  float(len(dgms[0])),
        'h1_bars':  float(len(dgms[1])),
        'h2_bars':  float(len(dgms[2])) if len(dgms) > 2 else 0.0,
        'h1_bar1':  float(h1_lt[0]) if len(h1_lt) > 0 else 0.0,
        'h1_bar2':  float(h1_lt[1]) if len(h1_lt) > 1 else 0.0,
        'h1_bar3':  float(h1_lt[2]) if len(h1_lt) > 2 else 0.0,
        'h2_bar1':  float(h2_lt[0]) if len(h2_lt) > 0 else 0.0,
        'n_pcs':    float(n_pc),
        'n_landmarks': float(n_lm),
    }

    fig = _barcode_figure(dgms, n_lm, n_pc)
    return fig, metrics
