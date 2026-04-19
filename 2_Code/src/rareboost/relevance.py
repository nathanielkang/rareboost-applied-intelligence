"""Relevance function with adaptive bandwidth KDE and Gaussian smoothing.

The core idea: rare target values (low density under KDE) get high relevance,
common values get low relevance.  φ(y) = 1 − normalized_density(y).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter1d
from scipy.stats import skew as scipy_skew


class RelevanceFunction:
    """Map target values to relevance scores via kernel density estimation.

    Parameters
    ----------
    method : str
        Density estimation method. Currently only ``'kde'`` is supported.
    bandwidth : str or float
        ``'adaptive'`` uses Silverman's rule adjusted for skewness.
        A float sets the bandwidth directly.
    threshold : float
        Default relevance threshold for :meth:`get_rare_mask`.
    smoothing : bool
        If *True*, apply Gaussian kernel smoothing to the relevance curve
        evaluated on a fine grid, then interpolate.
    smoothing_sigma : float or None
        Standard deviation for the smoothing kernel (in grid‐index units).
        *None* → automatically set to ``max(2, n_grid // 50)``.
    """

    def __init__(
        self,
        method: str = "kde",
        bandwidth: str | float = "adaptive",
        threshold: float = 0.5,
        smoothing: bool = True,
        smoothing_sigma: float | None = None,
    ) -> None:
        self.method = method
        self.bandwidth = bandwidth
        self.threshold = threshold
        self.smoothing = smoothing
        self.smoothing_sigma = smoothing_sigma

        self._kde = None
        self._grid_y: np.ndarray | None = None
        self._grid_phi: np.ndarray | None = None
        self._y_min: float = 0.0
        self._y_max: float = 1.0
        self._is_constant: bool = False

    def fit(self, y: np.ndarray) -> "RelevanceFunction":
        """Fit KDE on target values and build the relevance lookup."""
        y = np.asarray(y, dtype=np.float64).ravel()

        if y.shape[0] < 2 or np.ptp(y) < 1e-12:
            self._is_constant = True
            self._y_min = float(y.min()) if len(y) else 0.0
            self._y_max = float(y.max()) if len(y) else 1.0
            return self

        self._is_constant = False
        self._y_min = float(y.min())
        self._y_max = float(y.max())

        bw = self._resolve_bandwidth(y)
        self._kde = gaussian_kde(y, bw_method=bw)

        n_grid = max(512, len(y))
        pad = 0.05 * (self._y_max - self._y_min)
        self._grid_y = np.linspace(self._y_min - pad, self._y_max + pad, n_grid)
        raw_density = self._kde.evaluate(self._grid_y)

        d_max = raw_density.max()
        if d_max < 1e-30:
            norm_density = np.zeros_like(raw_density)
        else:
            norm_density = raw_density / d_max

        phi = 1.0 - norm_density

        if self.smoothing:
            sigma = self.smoothing_sigma
            if sigma is None:
                sigma = max(2.0, n_grid / 50.0)
            phi = gaussian_filter1d(phi, sigma=sigma)

        phi = np.clip(phi, 0.0, 1.0)
        self._grid_phi = phi
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        """Compute relevance φ(y) for arbitrary target values."""
        y = np.asarray(y, dtype=np.float64).ravel()
        if self._is_constant:
            return np.ones_like(y) * 0.5

        return np.interp(y, self._grid_y, self._grid_phi)

    def get_rare_mask(
        self, y: np.ndarray, threshold: float | None = None
    ) -> np.ndarray:
        """Boolean mask where φ(y) exceeds *threshold*."""
        if threshold is None:
            threshold = self.threshold
        return self.transform(y) > threshold

    def _resolve_bandwidth(self, y: np.ndarray) -> float:
        """Silverman's rule, optionally widened for skewed distributions."""
        n = len(y)
        std = np.std(y, ddof=1)
        iqr = np.subtract(*np.percentile(y, [75, 25]))
        h_silverman = 0.9 * min(std, iqr / 1.34) * n ** (-0.2)

        if isinstance(self.bandwidth, (int, float)):
            return float(self.bandwidth)

        sk = abs(float(scipy_skew(y)))
        skew_factor = 1.0 + 0.3 * min(sk, 5.0)
        return max(h_silverman * skew_factor, 1e-8) / std if std > 1e-12 else 1e-4


class AdaptiveBinning:
    """Quantile‐aware binning that places finer bins in rare regions.

    Parameters
    ----------
    n_bins : int
        Base number of bins.
    method : str
        ``'quantile'`` — equal‐frequency quantile bins.
        ``'adaptive'`` — finer bins where density is low (rare regions).
    """

    def __init__(self, n_bins: int = 20, method: str = "quantile") -> None:
        self.n_bins = n_bins
        self.method = method
        self._edges: np.ndarray | None = None
        self._densities: np.ndarray | None = None

    def fit(self, y: np.ndarray) -> "AdaptiveBinning":
        y = np.asarray(y, dtype=np.float64).ravel()

        if self.method == "adaptive":
            self._edges = self._adaptive_edges(y)
        else:
            quantiles = np.linspace(0, 100, self.n_bins + 1)
            self._edges = np.unique(np.percentile(y, quantiles))

        if len(self._edges) < 2:
            self._edges = np.array([y.min() - 1e-8, y.max() + 1e-8])

        counts, _ = np.histogram(y, bins=self._edges)
        bin_widths = np.diff(self._edges)
        bin_widths = np.maximum(bin_widths, 1e-12)
        self._densities = counts / (len(y) * bin_widths)
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        """Return bin indices (0‐based) for each value in *y*."""
        y = np.asarray(y, dtype=np.float64).ravel()
        indices = np.digitize(y, self._edges) - 1
        return np.clip(indices, 0, len(self._edges) - 2)

    def get_bin_densities(self) -> np.ndarray:
        if self._densities is None:
            raise RuntimeError("Call fit() first.")
        return self._densities.copy()

    def _adaptive_edges(self, y: np.ndarray) -> np.ndarray:
        """Build edges with more bins where KDE density is lowest."""
        rf = RelevanceFunction(smoothing=False)
        rf.fit(y)
        phi = rf.transform(y)

        w = phi + 0.1
        w /= w.sum()
        sorted_idx = np.argsort(y)
        y_sorted = y[sorted_idx]
        w_sorted = w[sorted_idx]

        cumw = np.cumsum(w_sorted)
        cumw /= cumw[-1]

        target_fracs = np.linspace(0, 1, self.n_bins + 1)
        edge_indices = np.searchsorted(cumw, target_fracs, side="right")
        edge_indices = np.clip(edge_indices, 0, len(y_sorted) - 1)
        edges = np.unique(y_sorted[edge_indices])

        if edges[0] > y.min():
            edges = np.concatenate([[y.min()], edges])
        if edges[-1] < y.max():
            edges = np.concatenate([edges, [y.max()]])
        return edges
