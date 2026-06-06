#!/usr/bin/env python3
"""Assemble univariate SMARTS priors from histogram data.

- Peak detection for initialization
- Higher kappa bounds for sharp distributions
- Adaptive initialization based on peak sharpness
- Minimum fit quality threshold to force more components

Usage:
    python scripts/assemble_univariate_priors.py [--output priors.json] [--max-components 5]
    python scripts/assemble_univariate_priors.py --plot-dir plots/  # Generate diagnostic plots
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import i0  # Modified Bessel function of the first kind
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def _round_floats(obj):
    if isinstance(obj, float):
        return round(obj, 4)
    elif isinstance(obj, dict):
        return {k: _round_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_round_floats(item) for item in obj]
    return obj


# Import generic priors from bouquet
try:
    from bouquet.priors import BUILTIN_UNIVARIATE_PRIORS
    GENERIC_PRIORS = {
        k: v for k, v in BUILTIN_UNIVARIATE_PRIORS.items()
        if k in ("sp3_sp3", "sp3_sp2")
    }
except ImportError:
    # Fallback if bouquet is not installed
    GENERIC_PRIORS = {
        "sp3_sp3": {
            "description": "Generic sp3-sp3 single bond (gauche/anti)",
            "means_deg": [60.0, 180.0, 300.0],
            "concentrations": [5.0, 6.0, 5.0],
            "weights": [0.28, 0.44, 0.28],
        },
        "sp3_sp2": {
            "description": "sp3-sp2 bond (eclipsed/anti)",
            "means_deg": [0.0, 180.0],
            "concentrations": [4.0, 4.0],
            "weights": [0.5, 0.5],
        },
    }


def compute_bhattacharyya_coefficient(pdf1: np.ndarray, pdf2: np.ndarray) -> float:
    """Compute Bhattacharyya coefficient between two probability distributions.

    Args:
        pdf1: First probability distribution (will be normalized)
        pdf2: Second probability distribution (will be normalized)

    Returns:
        Bhattacharyya coefficient (0-1, higher = more similar)
    """
    # Normalize to ensure they sum to 1
    pdf1 = pdf1 / (pdf1.sum() + 1e-10)
    pdf2 = pdf2 / (pdf2.sum() + 1e-10)

    # Bhattacharyya coefficient: BC = sum(sqrt(p * q))
    bc = np.sum(np.sqrt(pdf1 * pdf2))

    return float(bc)


def compute_fit_histogram_bc(
    fit: Dict,
    counts: np.ndarray,
    angles: np.ndarray,
) -> float:
    """Compute Bhattacharyya coefficient between fitted distribution and histogram.

    Args:
        fit: Fitted prior with means_deg, concentrations, weights
        counts: Histogram counts (360 bins)
        angles: Bin centers in radians

    Returns:
        Bhattacharyya coefficient (0-1, higher = more similar)
    """
    # Fitted distribution PDF
    fit_means = np.deg2rad(np.array(fit["means_deg"]) - 180)
    fit_kappas = np.array(fit["concentrations"])
    fit_weights = np.array(fit["weights"])
    pdf_fit = mixture_von_mises_pdf(angles, fit_means, fit_kappas, fit_weights)

    # Histogram as empirical PDF
    pdf_hist = counts.copy()

    return compute_bhattacharyya_coefficient(pdf_fit, pdf_hist)


def compute_distribution_similarity(
    fit: Dict,
    generic: Dict,
    n_points: int = 360,
) -> float:
    """Compute similarity between a fitted distribution and a generic prior.

    Uses the Bhattacharyya coefficient, which ranges from 0 (no overlap)
    to 1 (identical distributions).

    Args:
        fit: Fitted prior with means_deg, concentrations, weights
        generic: Generic prior definition
        n_points: Number of points to evaluate PDFs

    Returns:
        Bhattacharyya coefficient (0-1, higher = more similar)
    """
    # Evaluate both distributions on a common grid
    angles = np.linspace(-np.pi, np.pi, n_points, endpoint=False)

    # Fitted distribution PDF
    fit_means = np.deg2rad(np.array(fit["means_deg"]) - 180)
    fit_kappas = np.array(fit["concentrations"])
    fit_weights = np.array(fit["weights"])
    pdf_fit = mixture_von_mises_pdf(angles, fit_means, fit_kappas, fit_weights)

    # Generic distribution PDF
    gen_means = np.deg2rad(np.array(generic["means_deg"]) - 180)
    gen_kappas = np.array(generic["concentrations"])
    gen_weights = np.array(generic["weights"])
    pdf_gen = mixture_von_mises_pdf(angles, gen_means, gen_kappas, gen_weights)

    return compute_bhattacharyya_coefficient(pdf_fit, pdf_gen)


def classify_fit_pattern(fit: Dict) -> str:
    """Classify a fitted distribution as 'staggered' or 'planar' based on peak positions.

    Args:
        fit: Fitted prior with means_deg, weights

    Returns:
        'staggered' if peaks are at gauche positions (60°, 300°)
        'planar' if peaks are only at 0°/180°
    """
    means = np.array(fit["means_deg"])
    weights = np.array(fit["weights"])

    gauche_weight = 0.0
    planar_weight = 0.0

    for mean, weight in zip(means, weights):
        mean = mean % 360

        # Gauche positions: 60° ± 30° or 300° ± 30°
        if 30 <= mean <= 90 or 270 <= mean <= 330:
            gauche_weight += weight
        # Planar positions: 0° ± 30° or 180° ± 30°
        elif mean <= 30 or mean >= 330 or 150 <= mean <= 210:
            planar_weight += weight

    # If less than 10% weight at gauche positions, it's planar
    if gauche_weight < 0.1:
        return "planar"
    return "staggered"


def compare_to_generic_priors(
    fit: Dict,
    similarity_threshold: float = 0.90,
) -> Tuple[Optional[str], float]:
    """Compare a fitted distribution to generic priors.

    First classifies the fit as staggered or planar, then compares to the
    appropriate generic prior (sp3_sp3 or sp3_sp2).

    Args:
        fit: Fitted prior with means_deg, concentrations, weights
        similarity_threshold: Minimum Bhattacharyya coefficient to use generic

    Returns:
        (generic_type, similarity) if similar enough, (None, best_similarity) otherwise
    """
    pattern_class = classify_fit_pattern(fit)

    if pattern_class == "staggered":
        generic_type = "sp3_sp3"
    else:
        generic_type = "sp3_sp2"

    generic = GENERIC_PRIORS[generic_type]
    similarity = compute_distribution_similarity(fit, generic)

    if similarity >= similarity_threshold:
        return generic_type, similarity
    return None, similarity


def load_torlib(torlib_path: Path) -> List[Tuple[int, str]]:
    """Load SMARTS patterns from torlib.txt."""
    patterns = []
    with open(torlib_path) as f:
        idx = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                idx = int(parts[0])
                smarts = parts[1]
                patterns.append((idx, smarts))
            else:
                # in case it's just a list of SMARTS
                idx += 1  # Increment index
                smarts = line
                patterns.append((idx, smarts))
    return patterns


def load_histogram(data_dirs: List[Path], pattern_idx: int) -> Optional[np.ndarray]:
    """Load and sum histogram counts across all data sources.

    Also mirrors counts around 180° to enforce symmetry (e.g., count at 179°
    is added to 181°, count at 270° is added to 90°).
    """
    filename = f"tl{pattern_idx}.txt"
    total_counts = np.zeros(360, dtype=np.float64)
    found_any = False

    for data_dir in data_dirs:
        filepath = data_dir / filename
        if filepath.exists():
            counts = np.loadtxt(filepath)
            if len(counts) == 360:
                total_counts += counts
                found_any = True
            else:
                print(
                    f"Warning: {filepath} has {len(counts)} lines, expected 360",
                    file=sys.stderr,
                )

    if not found_any:
        return None

    # Mirror counts around 180° to enforce symmetry
    # Angle i maps to (360 - i) % 360:
    #   179° -> 181°, 270° -> 90°, 0° -> 0°, 180° -> 180°
    mirrored_counts = np.zeros(360, dtype=np.float64)
    for i in range(360):
        mirror_idx = (360 - i) % 360
        mirrored_counts[i] = total_counts[i] + total_counts[mirror_idx]
        # 0° and 180° map to themselves, so the line above already doubles them.

    return mirrored_counts


def histogram_to_angles(counts: np.ndarray) -> np.ndarray:
    """Convert histogram counts to weighted angle samples."""
    bin_centers_deg = np.arange(360) + 0.5
    bin_centers_rad = np.deg2rad(bin_centers_deg - 180)
    return bin_centers_rad, counts


def von_mises_pdf(theta: np.ndarray, mu: float, kappa: float) -> np.ndarray:
    """Von Mises probability density function."""
    # Higher kappa limit for sharp peaks
    kappa = np.clip(kappa, 0, 10000)
    if kappa > 500:
        # Use log-space computation for numerical stability at high kappa
        # log(I0(kappa)) ≈ kappa - 0.5*log(2*pi*kappa) for large kappa
        log_i0_approx = kappa - 0.5 * np.log(2 * np.pi * kappa)
        log_pdf = kappa * np.cos(theta - mu) - np.log(2 * np.pi) - log_i0_approx
        return np.exp(log_pdf)
    return np.exp(kappa * np.cos(theta - mu)) / (2 * np.pi * i0(kappa))


def mixture_von_mises_pdf(
    theta: np.ndarray, means: np.ndarray, kappas: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Mixture of von Mises distributions."""
    pdf = np.zeros_like(theta)
    for mu, kappa, w in zip(means, kappas, weights):
        pdf += w * von_mises_pdf(theta, mu, kappa)
    return pdf


def detect_peaks(
    counts: np.ndarray, min_prominence_ratio: float = 0.1, enforce_symmetry: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect peaks in circular histogram data.

    Args:
        counts: Histogram counts (360 bins)
        min_prominence_ratio: Minimum peak prominence as fraction of max count
        enforce_symmetry: If True, ensure symmetric pairs (θ and 360-θ) are included

    Returns:
        peak_positions: Indices of detected peaks
        peak_prominences: Prominence of each peak
    """
    # Handle circular wrapping by padding
    padded = np.concatenate([counts[-30:], counts, counts[:30]])

    # Smooth slightly to reduce noise
    smoothed = gaussian_filter1d(padded, sigma=2)

    # Find peaks
    min_prominence = counts.max() * min_prominence_ratio
    peaks, properties = find_peaks(smoothed, prominence=min_prominence, distance=10)

    # Adjust indices back to original range
    peaks = peaks - 30
    peaks = peaks[(peaks >= 0) & (peaks < 360)]

    # Get prominences for the valid peaks
    prominences = []
    for p in peaks:
        # Estimate local prominence
        left_min = counts[max(0, p - 20) : p].min() if p > 0 else counts[p]
        right_min = counts[p + 1 : min(360, p + 21)].min() if p < 359 else counts[p]
        base = max(left_min, right_min)
        prominences.append(counts[p] - base)

    peaks = list(peaks)
    prominences = list(prominences)

    # Enforce symmetry: for each peak at θ, ensure there's a peak at 360-θ
    if enforce_symmetry and len(peaks) > 0:
        symmetric_peaks = []
        symmetric_prominences = []
        for p, prom in zip(peaks, prominences):
            symmetric_peaks.append(p)
            symmetric_prominences.append(prom)
            # Calculate symmetric position (360 - θ) mod 360
            mirror_p = (360 - p) % 360
            # Check if mirror peak is already in list (within tolerance)
            already_exists = any(
                abs(mirror_p - existing) < 10 or abs(mirror_p - existing) > 350
                for existing in symmetric_peaks
            )
            if not already_exists and mirror_p != p:  # Skip if at 0° or 180°
                symmetric_peaks.append(mirror_p)
                # Use prominence from the mirrored location in counts
                mirror_prom = counts[mirror_p] - np.percentile(counts, 10)
                symmetric_prominences.append(max(mirror_prom, prom * 0.5))

        peaks = symmetric_peaks
        prominences = symmetric_prominences

    return np.array(peaks), np.array(prominences)


def estimate_kappa_from_peak(counts: np.ndarray, peak_idx: int) -> float:
    """Estimate kappa (concentration) from peak sharpness.

    Uses the relationship between FWHM and kappa for von Mises distribution.
    More aggressive estimation for very sharp peaks.
    """
    peak_height = counts[peak_idx]
    baseline = np.percentile(counts, 10)  # Use 10th percentile as baseline
    half_max = (peak_height + baseline) / 2

    # Find FWHM by searching left and right (with circular wrapping)
    # Use interpolation for sub-bin precision
    left_width = 0.0
    for i in range(1, 180):
        idx = (peak_idx - i) % 360
        idx_next = (peak_idx - i + 1) % 360
        if counts[idx] < half_max:
            # Linear interpolation for sub-bin precision
            if counts[idx_next] > counts[idx]:
                frac = (half_max - counts[idx]) / (counts[idx_next] - counts[idx])
                left_width = i - frac
            else:
                left_width = i
            break
    else:
        left_width = 180

    right_width = 0.0
    for i in range(1, 180):
        idx = (peak_idx + i) % 360
        idx_prev = (peak_idx + i - 1) % 360
        if counts[idx] < half_max:
            # Linear interpolation for sub-bin precision
            if counts[idx_prev] > counts[idx]:
                frac = (half_max - counts[idx]) / (counts[idx_prev] - counts[idx])
                right_width = i - frac
            else:
                right_width = i
            break
    else:
        right_width = 180

    # Calculate FWHM in degrees (can be fractional now)
    fwhm_deg = left_width + right_width
    fwhm_deg = max(fwhm_deg, 0.5)  # Minimum 0.5 degree for very sharp peaks
    fwhm_rad = np.deg2rad(fwhm_deg)

    # Approximate kappa from FWHM
    # For von Mises, FWHM ≈ 2 * arccos(1 - ln(2)/kappa) for large kappa
    # Rearranged: kappa ≈ 2 * ln(2) / (1 - cos(FWHM/2))
    cos_half_fwhm = np.cos(fwhm_rad / 2)
    if cos_half_fwhm < 0.99999:
        kappa = 2 * np.log(2) / (1 - cos_half_fwhm)
    else:
        kappa = 10000  # Extremely sharp peak

    return np.clip(kappa, 1, 10000)


def smart_initialization(
    angles: np.ndarray, counts: np.ndarray, n_components: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate smart initial parameters based on peak detection.

    Returns:
        init_means: Initial mean angles in radians
        init_kappas: Initial concentration parameters
        init_weights: Initial mixture weights
    """
    # Detect peaks
    peak_indices, prominences = detect_peaks(counts)

    if len(peak_indices) == 0:
        # Fallback: use histogram maximum
        peak_indices = np.array([np.argmax(counts)])
        prominences = np.array([counts.max()])

    # Select top n_components peaks by prominence
    if len(peak_indices) > n_components:
        top_idx = np.argsort(-prominences)[:n_components]
        peak_indices = peak_indices[top_idx]
        prominences = prominences[top_idx]

    # Convert peak positions to radians
    init_means = np.deg2rad(peak_indices + 0.5 - 180)

    # Estimate kappa for each peak
    init_kappas = np.array([estimate_kappa_from_peak(counts, p) for p in peak_indices])

    # Initialize weights proportional to prominence
    if prominences.sum() > 0:
        init_weights = prominences / prominences.sum()
    else:
        init_weights = np.ones(len(peak_indices)) / len(peak_indices)

    # If we need more components than detected peaks, add random ones
    if len(init_means) < n_components:
        n_extra = n_components - len(init_means)
        # Add components at random positions weighted by histogram
        probs = counts / counts.sum()
        extra_indices = np.random.choice(360, size=n_extra, p=probs, replace=False)
        extra_means = np.deg2rad(extra_indices + 0.5 - 180)
        extra_kappas = np.array(
            [estimate_kappa_from_peak(counts, p) for p in extra_indices]
        )
        extra_weights = np.ones(n_extra) * 0.1 / n_extra

        init_means = np.concatenate([init_means, extra_means])
        init_kappas = np.concatenate([init_kappas, extra_kappas])
        init_weights = np.concatenate([init_weights * 0.9, extra_weights])

    return init_means, init_kappas, init_weights


def fit_symmetric_von_mises_mixture(
    angles: np.ndarray,
    counts: np.ndarray,
    n_pairs: int,
    n_restarts: int = 10,
    use_smart_init: bool = True,
) -> Dict:
    """Fit a mixture of symmetric von Mises pairs to histogram data.

    Each pair consists of two von Mises components at angles μ and -μ (360°-μ),
    sharing the same kappa and weight. This enforces dihedral angle symmetry.

    Args:
        angles: Bin centers in radians
        counts: Histogram counts
        n_pairs: Number of symmetric pairs to fit
        n_restarts: Number of optimization restarts
        use_smart_init: Whether to use peak-based initialization

    Returns:
        Dictionary with fitted parameters and statistics
    """
    if counts.sum() == 0:
        return None

    total = counts.sum()
    bin_width = angles[1] - angles[0] if len(angles) > 1 else np.deg2rad(1)

    LOG_KAPPA_MIN = -2
    LOG_KAPPA_MAX = 9.2

    def neg_log_likelihood(params):
        """Negative log likelihood for symmetric pairs."""
        # params: [n_pairs means, n_pairs log_kappas, n_pairs raw_weights]
        # Means are in [0, π] and will be mirrored to [-π, 0]
        pair_means = params[:n_pairs]
        log_kappas = np.clip(
            params[n_pairs : 2 * n_pairs], LOG_KAPPA_MIN, LOG_KAPPA_MAX
        )
        pair_kappas = np.exp(log_kappas)

        raw_weights = params[2 * n_pairs :]
        raw_weights = raw_weights - raw_weights.max()
        pair_weights = np.exp(raw_weights) / np.exp(raw_weights).sum()

        # Build full component arrays with symmetric pairs
        all_means = []
        all_kappas = []
        all_weights = []

        for mu, kappa, w in zip(pair_means, pair_kappas, pair_weights):
            # Check if this is a "self-symmetric" angle (near 0 or π)
            mu_deg = np.rad2deg(mu) + 180  # Convert to [0, 360]
            is_self_symmetric = mu_deg < 5 or mu_deg > 355 or abs(mu_deg - 180) < 5

            if is_self_symmetric:
                # Single component at symmetric axis
                all_means.append(mu)
                all_kappas.append(kappa)
                all_weights.append(w)
            else:
                # Two components: μ and -μ, each with half the weight
                all_means.append(mu)
                all_means.append(-mu)
                all_kappas.append(kappa)
                all_kappas.append(kappa)
                all_weights.append(w / 2)
                all_weights.append(w / 2)

        all_means = np.array(all_means)
        all_kappas = np.array(all_kappas)
        all_weights = np.array(all_weights)

        pdf = mixture_von_mises_pdf(angles, all_means, all_kappas, all_weights)
        pdf = np.maximum(pdf, 1e-10)

        nll = -np.sum(counts * np.log(pdf))
        return nll

    best_result = None
    best_nll = np.inf

    for restart in range(n_restarts):
        if restart == 0 and use_smart_init:
            # Use detected peaks, but only keep one from each symmetric pair
            peak_indices, prominences = detect_peaks(counts, enforce_symmetry=False)
            unique_pairs = []
            used_mirrors = set()

            for p, prom in sorted(zip(peak_indices, prominences), key=lambda x: -x[1]):
                mirror_p = (360 - p) % 360
                if p not in used_mirrors:
                    # Keep the peak in [0, 180] range
                    if p <= 180:
                        unique_pairs.append((p, prom))
                    else:
                        unique_pairs.append((mirror_p, prom))
                    used_mirrors.add(p)
                    used_mirrors.add(mirror_p)

            # Select top n_pairs
            unique_pairs = unique_pairs[:n_pairs]

            if len(unique_pairs) > 0:
                init_means_deg = np.array([p[0] for p in unique_pairs])
                init_prominences = np.array([p[1] for p in unique_pairs])
            else:
                init_means_deg = np.array([np.argmax(counts[:180])])
                init_prominences = np.array([counts.max()])

            # Pad if needed
            while len(init_means_deg) < n_pairs:
                # Add random positions in [0, 180]
                new_pos = np.random.randint(0, 180)
                init_means_deg = np.append(init_means_deg, new_pos)
                init_prominences = np.append(init_prominences, counts[new_pos])

            init_means = np.deg2rad(
                init_means_deg + 0.5 - 180
            )  # Convert to radians [-π, π]
            init_kappas = np.array(
                [estimate_kappa_from_peak(counts, int(p)) for p in init_means_deg]
            )
            init_log_kappas = np.log(np.clip(init_kappas, 0.2, 10000))
            prominence_sum = init_prominences.sum()
            if prominence_sum > 0:
                init_weights = init_prominences / prominence_sum
            else:
                # Uniform weights if all prominences are zero
                init_weights = np.ones(len(init_prominences)) / len(init_prominences)
            init_raw_weights = np.log(init_weights + 1e-10)
        else:
            # Random initialization - means in [0, π] (will be mirrored)
            init_means = np.random.uniform(-np.pi, 0, n_pairs)
            init_log_kappas = np.random.uniform(2, 7, n_pairs)
            init_raw_weights = np.zeros(n_pairs)

        init_params = np.concatenate([init_means, init_log_kappas, init_raw_weights])

        try:
            bounds = [(-np.pi, np.pi)] * n_pairs  # means
            bounds += [(LOG_KAPPA_MIN, LOG_KAPPA_MAX)] * n_pairs  # log_kappas
            bounds += [(None, None)] * n_pairs  # raw_weights

            result = minimize(
                neg_log_likelihood,
                init_params,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 2000, "ftol": 1e-10},
            )
            if result.fun < best_nll:
                best_nll = result.fun
                best_result = result
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as e:
            # Expected numerical/optimization failure for this restart: log and
            # try the next one. Unexpected exceptions propagate so real bugs surface.
            print(
                f"  Skipping fit restart: {type(e).__name__}: {e} "
                f"(init_params={init_params}, bounds={bounds})",
                file=sys.stderr,
            )
            continue

    if best_result is None:
        return None

    # Extract parameters and expand symmetric pairs
    params = best_result.x
    pair_means = params[:n_pairs]
    log_kappas = np.clip(params[n_pairs : 2 * n_pairs], LOG_KAPPA_MIN, LOG_KAPPA_MAX)
    pair_kappas = np.exp(log_kappas)
    raw_weights = params[2 * n_pairs :]
    raw_weights = raw_weights - raw_weights.max()
    pair_weights = np.exp(raw_weights) / np.exp(raw_weights).sum()

    # Expand to full component arrays
    all_means_deg = []
    all_kappas = []
    all_weights = []

    for mu, kappa, w in zip(pair_means, pair_kappas, pair_weights):
        mu_deg = np.rad2deg(mu) + 180
        is_self_symmetric = mu_deg < 5 or mu_deg > 355 or abs(mu_deg - 180) < 5

        if is_self_symmetric:
            all_means_deg.append(mu_deg % 360)
            all_kappas.append(kappa)
            all_weights.append(w)
        else:
            all_means_deg.append(mu_deg % 360)
            all_means_deg.append((360 - mu_deg) % 360)
            all_kappas.append(kappa)
            all_kappas.append(kappa)
            all_weights.append(w / 2)
            all_weights.append(w / 2)

    means_deg = np.array(all_means_deg)
    kappas = np.array(all_kappas)
    weights = np.array(all_weights)

    # Prune degenerate components
    MIN_WEIGHT = 0.01
    MIN_KAPPA = 1.0
    valid_mask = (weights >= MIN_WEIGHT) & (kappas >= MIN_KAPPA)

    if valid_mask.sum() > 0:
        means_deg = means_deg[valid_mask]
        kappas = kappas[valid_mask]
        weights = weights[valid_mask]
        weights = weights / weights.sum()
    elif len(weights) > 0:
        best_idx = np.argmax(weights)
        means_deg = np.array([means_deg[best_idx]])
        kappas = np.array([max(kappas[best_idx], MIN_KAPPA)])
        weights = np.array([1.0])

    # Sort by weight
    sort_idx = np.argsort(-weights)
    means_deg = means_deg[sort_idx]
    kappas = kappas[sort_idx]
    weights = weights[sort_idx]

    # Calculate goodness of fit
    fitted_pdf = mixture_von_mises_pdf(
        angles, np.deg2rad(means_deg - 180), kappas, weights
    )
    fitted_counts = fitted_pdf * total * bin_width

    ss_res = np.sum((counts - fitted_counts) ** 2)
    ss_tot = np.sum((counts - counts.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    rmse = np.sqrt(np.mean((counts - fitted_counts) ** 2))
    mean_count = counts.mean()
    nrmse = (rmse / mean_count * 100) if mean_count > 0 else 0.0

    nonzero = fitted_counts > 1e-10
    chi2 = np.sum(
        (counts[nonzero] - fitted_counts[nonzero]) ** 2 / fitted_counts[nonzero]
    )

    # Bhattacharyya coefficient between fit and histogram
    bc = compute_bhattacharyya_coefficient(fitted_pdf, counts)

    return {
        "means_deg": means_deg.tolist(),
        "concentrations": kappas.tolist(),
        "weights": weights.tolist(),
        "total_counts": int(total),
        "chi2": float(chi2),
        "nll": float(best_nll),
        "r2": float(r2),
        "rmse": float(rmse),
        "nrmse": float(nrmse),
        "bc": float(bc),
        "symmetric": True,
    }


def fit_von_mises_mixture(
    angles: np.ndarray,
    counts: np.ndarray,
    n_components: int,
    n_restarts: int = 10,
    use_smart_init: bool = True,
    symmetric: bool = True,
) -> Dict:
    """Fit a mixture of von Mises distributions to histogram data.

    Improved version with:
    - Smart initialization from peak detection
    - Higher kappa bounds for very sharp peaks
    - Component pruning to remove degenerate solutions
    - More restarts for robustness
    - Optional symmetric fitting (peaks at θ and 360-θ)

    Args:
        angles: Bin centers in radians
        counts: Histogram counts
        n_components: Number of components (or pairs if symmetric)
        n_restarts: Number of optimization restarts
        use_smart_init: Whether to use peak-based initialization
        symmetric: If True, fit symmetric pairs instead of independent components
    """
    # Use symmetric fitting by default
    if symmetric:
        return fit_symmetric_von_mises_mixture(
            angles, counts, n_components, n_restarts, use_smart_init
        )
    if counts.sum() == 0:
        return None

    total = counts.sum()
    bin_width = angles[1] - angles[0] if len(angles) > 1 else np.deg2rad(1)
    empirical_pdf = counts / (total * bin_width)

    # Kappa bounds - allow very high values for sharp peaks
    LOG_KAPPA_MIN = -2  # kappa_min ≈ 0.14
    LOG_KAPPA_MAX = 9.2  # kappa_max ≈ 10000

    def neg_log_likelihood(params):
        """Negative log likelihood for optimization."""
        means = params[:n_components]
        log_kappas = np.clip(
            params[n_components : 2 * n_components], LOG_KAPPA_MIN, LOG_KAPPA_MAX
        )
        kappas = np.exp(log_kappas)

        raw_weights = params[2 * n_components :]
        raw_weights = raw_weights - raw_weights.max()
        weights = np.exp(raw_weights) / np.exp(raw_weights).sum()

        pdf = mixture_von_mises_pdf(angles, means, kappas, weights)
        pdf = np.maximum(pdf, 1e-10)

        nll = -np.sum(counts * np.log(pdf))
        return nll

    best_result = None
    best_nll = np.inf

    for restart in range(n_restarts):
        if restart == 0 and use_smart_init:
            # First restart: use smart initialization
            init_means, init_kappas, init_weights = smart_initialization(
                angles, counts, n_components
            )
            init_log_kappas = np.log(np.clip(init_kappas, 0.2, 10000))
            # Convert weights to raw (pre-softmax) form
            init_raw_weights = np.log(init_weights + 1e-10)
        else:
            # Random restarts with better kappa range
            if restart < n_restarts // 2:
                # Half restarts: initialize near detected peaks with jitter
                init_means, init_kappas, _ = smart_initialization(
                    angles, counts, n_components
                )
                init_means += np.random.uniform(-0.2, 0.2, n_components)
                init_log_kappas = np.log(init_kappas) + np.random.uniform(
                    -1, 1, n_components
                )
            else:
                # Other half: random initialization
                init_means = np.random.uniform(-np.pi, np.pi, n_components)
                # Broader kappa initialization range - bias toward higher values
                init_log_kappas = np.random.uniform(2, 7, n_components)

            init_raw_weights = np.zeros(n_components)

        init_params = np.concatenate([init_means, init_log_kappas, init_raw_weights])

        try:
            # Use bounds to help optimization
            bounds = [(None, None)] * n_components  # means unbounded
            bounds += [(LOG_KAPPA_MIN, LOG_KAPPA_MAX)] * n_components  # log_kappas
            bounds += [(None, None)] * n_components  # raw_weights

            result = minimize(
                neg_log_likelihood,
                init_params,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 2000, "ftol": 1e-10},
            )
            if result.fun < best_nll:
                best_nll = result.fun
                best_result = result
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as e:
            # Expected numerical/optimization failure for this restart: log and
            # try the next one. Unexpected exceptions propagate so real bugs surface.
            print(
                f"  Skipping fit restart: {type(e).__name__}: {e} "
                f"(init_params={init_params}, bounds={bounds})",
                file=sys.stderr,
            )
            continue

    if best_result is None:
        return None

    # Extract parameters
    params = best_result.x
    means = params[:n_components]
    log_kappas = np.clip(
        params[n_components : 2 * n_components], LOG_KAPPA_MIN, LOG_KAPPA_MAX
    )
    kappas = np.exp(log_kappas)
    raw_weights = params[2 * n_components :]
    raw_weights = raw_weights - raw_weights.max()
    weights = np.exp(raw_weights) / np.exp(raw_weights).sum()

    # Convert means to degrees [0, 360)
    means_deg = np.rad2deg(means) + 180
    means_deg = means_deg % 360

    # ============ PRUNE DEGENERATE COMPONENTS ============
    # Remove components with very low weight or very low kappa (essentially uniform)
    MIN_WEIGHT = 0.01
    MIN_KAPPA = 1.0

    valid_mask = (weights >= MIN_WEIGHT) & (kappas >= MIN_KAPPA)

    if valid_mask.sum() > 0:
        means_deg = means_deg[valid_mask]
        kappas = kappas[valid_mask]
        weights = weights[valid_mask]
        # Renormalize weights after pruning
        weights = weights / weights.sum()
    # If all components are invalid, keep the best one
    elif len(weights) > 0:
        best_idx = np.argmax(weights)
        means_deg = np.array([means_deg[best_idx]])
        kappas = np.array([max(kappas[best_idx], MIN_KAPPA)])
        weights = np.array([1.0])

    # Sort by weight (descending)
    sort_idx = np.argsort(-weights)
    means_deg = means_deg[sort_idx]
    kappas = kappas[sort_idx]
    weights = weights[sort_idx]

    # Calculate goodness of fit statistics
    fitted_pdf = mixture_von_mises_pdf(
        angles, np.deg2rad(means_deg - 180), kappas, weights
    )
    fitted_counts = fitted_pdf * total * bin_width

    # R² (coefficient of determination)
    ss_res = np.sum((counts - fitted_counts) ** 2)
    ss_tot = np.sum((counts - counts.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # RMSE
    rmse = np.sqrt(np.mean((counts - fitted_counts) ** 2))

    # Normalized RMSE
    mean_count = counts.mean()
    nrmse = (rmse / mean_count * 100) if mean_count > 0 else 0.0

    # Chi-squared (use a small threshold to avoid overflow from near-zero values)
    nonzero = fitted_counts > 1e-10
    chi2 = np.sum(
        (counts[nonzero] - fitted_counts[nonzero]) ** 2 / fitted_counts[nonzero]
    )

    # Bhattacharyya coefficient between fit and histogram
    bc = compute_bhattacharyya_coefficient(fitted_pdf, counts)

    return {
        "means_deg": means_deg.tolist(),
        "concentrations": kappas.tolist(),
        "weights": weights.tolist(),
        "total_counts": int(total),
        "chi2": float(chi2),
        "nll": float(best_nll),
        "r2": float(r2),
        "rmse": float(rmse),
        "nrmse": float(nrmse),
        "bc": float(bc),
    }


def select_n_components(
    angles: np.ndarray,
    counts: np.ndarray,
    max_components: int = 6,
    min_r2: float = 0.85,
) -> Tuple[int, Dict]:
    """Select optimal number of components using BIC with minimum quality threshold.

    Improved: If BIC-selected model has R² < min_r2, increase components until
    either R² threshold is met or max_components is reached.
    """
    total = counts.sum()
    if total == 0:
        return 0, None

    # First pass: find BIC-optimal number of components
    best_bic = np.inf
    best_n = 1
    best_fit = None
    all_fits = {}

    for n in range(1, max_components + 1):
        fit = fit_von_mises_mixture(angles, counts, n)
        if fit is None:
            continue

        # Actual components after pruning
        actual_n = len(fit["means_deg"])

        # BIC with slightly reduced penalty for complex models
        # Use actual number of components after pruning
        k = 3 * actual_n - 1
        bic = 2 * fit["nll"] + 0.8 * k * np.log(total)

        fit["bic"] = bic
        fit["n_components"] = actual_n  # Use actual count after pruning
        all_fits[n] = fit

        if bic < best_bic:
            best_bic = bic
            best_n = actual_n
            best_fit = fit

    # Second pass: if fit quality is poor, try more components
    if best_fit is not None and best_fit["r2"] < min_r2:
        for n in range(best_fit["n_components"] + 1, max_components + 1):
            if n in all_fits:
                fit = all_fits[n]
            else:
                fit = fit_von_mises_mixture(angles, counts, n)
                if fit is None:
                    continue
                actual_n = len(fit["means_deg"])
                k = 3 * actual_n - 1
                fit["bic"] = 2 * fit["nll"] + 0.8 * k * np.log(total)
                fit["n_components"] = actual_n

            if fit["r2"] > best_fit["r2"]:
                best_fit = fit
                best_n = fit["n_components"]

            if fit["r2"] >= min_r2:
                break

    return best_n, best_fit


def plot_fit(
    angles: np.ndarray,
    counts: np.ndarray,
    fit: Dict,
    pattern_idx: int,
    smarts: str,
    output_path: Path,
) -> None:
    """Generate diagnostic plot comparing histogram and fitted distribution."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[3, 1])

    angles_deg = np.rad2deg(angles) + 180

    total = counts.sum()
    bin_width = angles[1] - angles[0] if len(angles) > 1 else 1.0
    fitted_pdf = mixture_von_mises_pdf(
        angles,
        np.deg2rad(np.array(fit["means_deg"]) - 180),
        np.array(fit["concentrations"]),
        np.array(fit["weights"]),
    )
    fitted_counts = fitted_pdf * total * bin_width

    # Main plot
    ax1.bar(
        angles_deg,
        counts,
        width=1.0,
        alpha=0.6,
        color="steelblue",
        label="Observed",
        edgecolor="none",
    )
    ax1.plot(
        angles_deg,
        fitted_counts,
        "r-",
        linewidth=2,
        label=f'Fit ({fit["n_components"]} components)',
    )

    # Mark component means with kappa info
    for i, (mu, kappa, w) in enumerate(
        zip(fit["means_deg"], fit["concentrations"], fit["weights"])
    ):
        label = f"μ={mu:.1f}° (κ={kappa:.0f}, w={w:.2f})" if i < 4 else None
        ax1.axvline(mu, color="orange", linestyle="--", alpha=0.7, label=label)

    ax1.set_xlim(0, 360)
    ax1.xaxis.set_major_locator(MultipleLocator(60))
    ax1.set_xlabel("Dihedral Angle (°)")
    ax1.set_ylabel("Count")

    smarts_display = smarts if len(smarts) <= 50 else smarts[:47] + "..."
    ax1.set_title(
        f"Pattern {pattern_idx}: {smarts_display}\n"
        f'BC={fit["bc"]:.4f}, RMSE={fit["rmse"]:.1f}, '
        f"n={int(total):,}"
    )
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Residual plot
    residuals = counts - fitted_counts
    ax2.bar(angles_deg, residuals, width=1.0, alpha=0.6, color="gray", edgecolor="none")
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_xlim(0, 360)
    ax2.xaxis.set_major_locator(MultipleLocator(60))
    ax2.set_xlabel("Dihedral Angle (°)")
    ax2.set_ylabel("Residual")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_diagnostics_csv(diagnostics: List[Dict], output_path: Path) -> None:
    """Write fit diagnostics to CSV file."""
    import csv

    fieldnames = [
        "pattern_idx",
        "smarts",
        "n_components",
        "total_counts",
        "bc",
        "r2",
        "rmse",
        "nrmse",
        "chi2",
        "bic",
        "means_deg",
        "concentrations",
        "weights",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(diagnostics)


def main():
    parser = argparse.ArgumentParser(
        description="Assemble univariate SMARTS priors (improved)"
    )
    parser.add_argument(
        "--torlib",
        type=Path,
        default=Path("torlib.txt"),
        help="Path to torlib.txt with SMARTS patterns",
    )
    parser.add_argument(
        "--data-dirs",
        type=Path,
        nargs="+",
        default=[Path("data/cod"), Path("data/zinc"), Path("data/pubchemqc")],
        help="Directories containing histogram files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("univariate_priors.json"),
        help="Output JSON file",
    )
    parser.add_argument(
        "--max-components",
        type=int,
        default=5,
        help="Maximum number of von Mises components",
    )
    parser.add_argument(
        "--min-r2",
        type=float,
        default=0.85,
        help="Minimum R² threshold (will try more components if below)",
    )
    parser.add_argument(
        "--min-counts",
        type=int,
        default=100,
        help="Minimum total counts to fit a prior",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Directory for diagnostic plots (optional)",
    )
    parser.add_argument(
        "--diagnostics-csv",
        type=Path,
        default=None,
        help="Path for diagnostics CSV file (optional)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Print progress")
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.90,
        help="Bhattacharyya coefficient threshold for using generic priors (0-1)",
    )
    args = parser.parse_args()

    patterns = load_torlib(args.torlib)
    print(f"Loaded {len(patterns)} SMARTS patterns from {args.torlib}")

    for d in args.data_dirs:
        if not d.exists():
            print(f"Warning: Data directory {d} does not exist", file=sys.stderr)

    if args.plot_dir:
        args.plot_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving diagnostic plots to {args.plot_dir}")

    univariate_priors = {}
    univariate_smarts = []
    diagnostics = []
    skipped = 0
    bc_values = []
    rmse_values = []
    generic_matches = {"sp3_sp3": 0, "sp3_sp2": 0}
    custom_fits = 0

    for idx, smarts in patterns:
        print(f"Processing pattern {idx}: {smarts[:50]}...")

        raw_counts = load_histogram(args.data_dirs, idx)
        if raw_counts is None:
            if args.verbose:
                print(f"  No data found, skipping")
            skipped += 1
            continue

        total = raw_counts.sum()
        if total < args.min_counts:
            if args.verbose:
                print(f"  Only {total} counts, skipping (min: {args.min_counts})")
            skipped += 1
            continue

        angles, counts = histogram_to_angles(raw_counts)
        n_comp, fit = select_n_components(
            angles, counts, args.max_components, args.min_r2
        )

        if fit is None:
            if args.verbose:
                print(f"  Fitting failed, skipping")
            skipped += 1
            continue

        # Compare to generic priors
        generic_type, similarity = compare_to_generic_priors(
            fit, args.similarity_threshold
        )

        if args.verbose:
            if generic_type:
                print(
                    f"  Fit {n_comp} components, R²={fit['r2']:.4f}, "
                    f"RMSE={fit['rmse']:.1f}, counts={int(total)} "
                    f"-> using generic '{generic_type}' (similarity={similarity:.3f})"
                )
            else:
                print(
                    f"  Fit {n_comp} components, R²={fit['r2']:.4f}, "
                    f"RMSE={fit['rmse']:.1f}, counts={int(total)} "
                    f"-> custom fit (similarity={similarity:.3f})"
                )

        bc_values.append(fit["bc"])
        rmse_values.append(fit["rmse"])

        prior_type_id = idx
        if generic_type:
            # Use generic prior - just record the SMARTS mapping
            generic_matches[generic_type] += 1
            univariate_priors[prior_type_id] = {
                "smarts": smarts,
                "description": f"Matched {generic_type} (similarity={similarity:.3f}, n={int(total)})",
                "generic_type": generic_type,
            }
        else:
            # Use custom fit
            custom_fits += 1
            univariate_priors[prior_type_id] = {
                "smarts": smarts,
                "description": f"Fit from {int(total)} obs.",
                "means_deg": fit["means_deg"],
                "concentrations": fit["concentrations"],
                "weights": fit["weights"],
            }

        diagnostics.append(
            {
                "pattern_idx": idx,
                "smarts": smarts,
                "n_components": fit["n_components"],
                "total_counts": fit["total_counts"],
                "bc": fit["bc"],
                "r2": fit["r2"],
                "rmse": fit["rmse"],
                "nrmse": fit["nrmse"],
                "chi2": fit["chi2"],
                "bic": fit["bic"],
                "means_deg": str(fit["means_deg"]),
                "concentrations": str(fit["concentrations"]),
                "weights": str(fit["weights"]),
            }
        )

        if args.plot_dir:
            plot_path = args.plot_dir / f"pattern_{idx:03d}.pdf"
            plot_fit(angles, counts, fit, idx, smarts, plot_path)

    with open(args.output, "w") as f:
        rounded_priors = _round_floats(univariate_priors)
        json.dump(rounded_priors, f, indent=2)

    if args.diagnostics_csv:
        write_diagnostics_csv(diagnostics, args.diagnostics_csv)
        print(f"Wrote diagnostics to {args.diagnostics_csv}")

    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"Patterns processed: {len(patterns)}")
    print(f"Priors fitted:      {len(univariate_priors)}")
    print(f"Skipped:            {skipped}")
    print(f"\nGeneric prior matching (threshold={args.similarity_threshold}):")
    print(f"  sp3_sp3 matches: {generic_matches['sp3_sp3']}")
    print(f"  sp3_sp2 matches: {generic_matches['sp3_sp2']}")
    print(f"  Custom fits:     {custom_fits}")
    print(f"\nFit quality statistics:")
    if bc_values:
        bc_arr = np.array(bc_values)
        rmse_arr = np.array(rmse_values)
        print(
            f"  BC:   min={bc_arr.min():.4f}, median={np.median(bc_arr):.4f}, "
            f"max={bc_arr.max():.4f}"
        )
        print(
            f"  RMSE: min={rmse_arr.min():.1f}, median={np.median(rmse_arr):.1f}, "
            f"max={rmse_arr.max():.1f}"
        )
        poor_fits = np.sum(bc_arr < 0.8)
        if poor_fits > 0:
            print(f"\n  Warning: {poor_fits} patterns have BC < 0.8")
    print(f"\nOutput written to {args.output}")


if __name__ == "__main__":
    main()
