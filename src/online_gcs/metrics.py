"""Trajectory quality metrics used by online GCS experiments."""

from typing import Protocol

import numpy as np


class Trajectory(Protocol):
    """Minimal trajectory interface needed by :func:`compute_trajectory_metrics`."""

    def start_time(self) -> float: ...

    def end_time(self) -> float: ...

    def vector_values(self, times: np.ndarray) -> np.ndarray: ...


def compute_trajectory_metrics(
    traj: Trajectory,
    num_samples: int = 500,
) -> dict[str, float]:
    """Compute arc-length-parameterized geometric trajectory metrics."""
    t0 = traj.start_time()
    tf = traj.end_time()
    duration = tf - t0
    zeros = {
        "curv_max": 0.0,
        "curv_integral": 0.0,
        "torsion_max": 0.0,
        "torsion_integral": 0.0,
        "path_length": 0.0,
    }
    if duration < 1e-10:
        return zeros

    times = np.linspace(t0, tf, num_samples)
    positions = traj.vector_values(times)

    seg_lens = np.linalg.norm(np.diff(positions, axis=1), axis=0)
    cum_len = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total_len = cum_len[-1]
    if total_len < 1e-10:
        return zeros

    target_s = np.linspace(0.0, total_len, num_samples)
    nq = positions.shape[0]
    resampled = np.zeros((nq, num_samples))
    for i in range(num_samples):
        idx = int(np.searchsorted(cum_len, target_s[i], side="right")) - 1
        idx = min(max(idx, 0), len(cum_len) - 2)
        seg = cum_len[idx + 1] - cum_len[idx]
        alpha = (target_s[i] - cum_len[idx]) / seg if seg > 1e-12 else 0.0
        resampled[:, i] = (1.0 - alpha) * positions[:, idx] + alpha * positions[:, idx + 1]

    ds = target_s[1] - target_s[0] if num_samples > 1 else 1.0
    d1 = np.diff(resampled, axis=1) / ds
    d2 = np.diff(d1, axis=1) / ds
    d3 = np.diff(d2, axis=1) / ds

    curv = np.linalg.norm(d2, axis=0)
    tors = np.linalg.norm(d3, axis=0)

    # Finite differences amplify round-off even for exactly affine samples.
    # Values below the corresponding floating-point resolution are numerical noise.
    scale = max(1.0, float(np.max(np.abs(resampled))))
    eps = np.finfo(resampled.dtype).eps
    curv = np.where(curv <= 32.0 * eps * scale / ds**2, 0.0, curv)
    tors = np.where(tors <= 32.0 * eps * scale / ds**3, 0.0, tors)

    return {
        "curv_max": float(np.max(curv)) if len(curv) else 0.0,
        "curv_integral": float(np.trapezoid(curv, dx=ds)),
        "torsion_max": float(np.max(tors)) if len(tors) else 0.0,
        "torsion_integral": float(np.trapezoid(tors, dx=ds)),
        "path_length": float(total_len),
    }
