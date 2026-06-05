"""
Adaptive source selection from precomputed pairwise subject distances.

Implements two methods (per target subject):
  1) gmm2: 2-component 1D Gaussian Mixture Model on distances; select "near" component.
  2) softmax_tau: softmax(-d/tau_t) with self-calibrated tau_t; select weights > 1/N.


"""

from __future__ import annotations

import argparse
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

# Mitigate a known Windows+MKL warning path in scikit-learn GMM init (KMeans).
# This is a small script with tiny inputs; forcing 1 thread is acceptable and keeps runs stable.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

try:
    from sklearn.mixture import GaussianMixture
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "scikit-learn is required for GMM selection. Install with: pip install scikit-learn"
    ) from e

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt

warnings.filterwarnings(
    "ignore",
    message=r"KMeans is known to have a memory leak on Windows with MKL.*",
    category=UserWarning,
)


@dataclass(frozen=True)
class Config:
    seed: int = 0
    eps: float = 1e-12
    k_min: int = 2
    k_max: int = 20
    gmm_prob_threshold: float = 0.5
    results_dir: str = "results"


# Requested Cho2017 subset (used for both target IDs and candidate source IDs)
CHO2017_SUBJECT_SUBSET: Tuple[int, ...] = (2, 4, 14, 17, 28, 30, 40, 42, 44, 45)

NPZ_REQUIRED_KEYS = ("distance_matrix", "subject_ids", "metric", "embedding_model", "dataset")
CANONICAL_DISTANCE_FILES = (
    "bci2a_pairwise_distances.npz",
    "cho2017_pairwise_distances.npz",
)


def set_seeds(seed: int) -> None:
    np.random.seed(seed)


def _npz_str(value: np.ndarray) -> str:
    return str(value.item()) if value.ndim == 0 else str(value.flat[0])


def matrix_to_long(mat: np.ndarray, row_ids: List[int], col_ids: List[int]) -> pd.DataFrame:
    if mat.shape != (len(row_ids), len(col_ids)):
        raise ValueError("Matrix shape does not match provided row/col ids.")
    records: List[Dict[str, object]] = []
    for i, sid in enumerate(row_ids):
        for j, tid in enumerate(col_ids):
            d = mat[i, j]
            if sid == tid:
                continue
            if not np.isfinite(d):
                continue
            records.append({"source_id": sid, "target_id": tid, "distance": float(d)})
    return pd.DataFrame.from_records(records)


def load_distances_npz(path: Path) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Load pairwise distances from a structured .npz artifact."""
    data = np.load(path, allow_pickle=False)
    missing = [key for key in NPZ_REQUIRED_KEYS if key not in data.files]
    if missing:
        raise ValueError(f"{path} is missing required keys: {missing}")

    mat = np.asarray(data["distance_matrix"], dtype=np.float64)
    subject_ids = np.asarray(data["subject_ids"], dtype=np.int64).tolist()
    metadata = {
        "metric": _npz_str(data["metric"]),
        "embedding_model": _npz_str(data["embedding_model"]),
        "dataset": _npz_str(data["dataset"]),
    }

    if mat.shape != (len(subject_ids), len(subject_ids)):
        raise ValueError(
            f"{path}: distance_matrix shape {mat.shape} does not match "
            f"subject_ids length {len(subject_ids)}"
        )

    pairs = matrix_to_long(mat=mat, row_ids=subject_ids, col_ids=subject_ids)
    return pairs, metadata


def softmax_stable(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.max(x)
    ex = np.exp(x)
    s = np.sum(ex)
    if s <= 0 or not np.isfinite(s):
        # fallback: uniform
        return np.ones_like(x) / max(len(x), 1)
    return ex / s


def discover_default_inputs(distances_dir: Path) -> Dict[str, Path]:
    """Discover bundled distance artifacts under distances/."""
    out: Dict[str, Path] = {}
    for filename in CANONICAL_DISTANCE_FILES:
        path = distances_dir / filename
        if not path.exists():
            continue
        _, metadata = load_distances_npz(path)
        out[str(metadata["dataset"])] = path
    return out


def compute_tau(d: np.ndarray, eps: float) -> float:
    d = np.asarray(d, dtype=float)
    med = float(np.median(d))
    if np.isfinite(med) and med > eps:
        return med
    d_pos = d[d > eps]
    if d_pos.size > 0:
        med_pos = float(np.median(d_pos))
        if np.isfinite(med_pos) and med_pos > eps:
            return med_pos
    sd = float(np.std(d))
    if np.isfinite(sd) and sd > eps:
        return sd
    return 1.0


def select_softmax_tau(
    pairs_df: pd.DataFrame, dataset: str, cfg: Config
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    method = "softmax_tau"
    rows: List[pd.DataFrame] = []
    summaries: List[Dict[str, object]] = []

    for target_id, g in pairs_df.groupby("target_id", sort=True):
        g2 = g.copy()
        d = g2["distance"].to_numpy(dtype=float)
        n = len(d)

        # Rank candidates by increasing distance (1 = closest)
        order = np.argsort(d)
        dist_rank = np.empty(n, dtype=int)
        dist_rank[order] = np.arange(1, n + 1)
        g2["distance_rank"] = dist_rank

        tau_t = compute_tau(d, eps=cfg.eps)
        logits = -d / max(tau_t, cfg.eps)
        w = softmax_stable(logits)
        thr = 1.0 / max(n, 1)

        selected = w > thr
        guardrail_used = 0

        # Guardrails
        k = int(selected.sum())
        if k < cfg.k_min and n > 0:
            guardrail_used = 1
            top_idx = np.argsort(-w)[: min(cfg.k_min, n)]
            selected = np.zeros(n, dtype=bool)
            selected[top_idx] = True
        if selected.sum() > cfg.k_max:
            guardrail_used = 1
            top_idx = np.argsort(-w)[: cfg.k_max]
            selected = np.zeros(n, dtype=bool)
            selected[top_idx] = True

        g2["dataset"] = dataset
        g2["method"] = method
        g2["tau_t"] = float(tau_t)
        g2["weight"] = w.astype(float)
        g2["threshold_uniform"] = float(thr)
        g2["selected"] = selected.astype(int)
        g2["guardrail_used"] = int(guardrail_used)

        rows.append(g2)

        sel = g2[g2["selected"] == 1].sort_values("distance", ascending=True)
        selected_sources_ordered = ",".join(str(int(x)) for x in sel["source_id"].tolist())
        summaries.append(
            {
                "dataset": dataset,
                "method": method,
                "target_id": target_id,
                "num_candidates": int(n),
                "num_selected": int(sel.shape[0]),
                "mean_distance_selected": float(sel["distance"].mean()) if sel.shape[0] else np.nan,
                "median_distance_selected": float(sel["distance"].median())
                if sel.shape[0]
                else np.nan,
                "max_weight_selected": float(sel["weight"].max()) if sel.shape[0] else np.nan,
                "tau_t": float(tau_t),
                "guardrail_used": int(guardrail_used),
                "selected_sources_ordered": selected_sources_ordered,
            }
        )

    pairs_out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    summary_out = pd.DataFrame.from_records(summaries)
    return pairs_out, summary_out


def _fallback_by_distance_rank(d: np.ndarray, cfg: Config) -> np.ndarray:
    n = len(d)
    if n == 0:
        return np.zeros(0, dtype=bool)
    # pick at least k_min, at most k_max, using smallest distances
    k = min(max(cfg.k_min, 1), n)
    k = min(k, cfg.k_max)
    idx = np.argsort(d)[:k]
    sel = np.zeros(n, dtype=bool)
    sel[idx] = True
    return sel


def select_gmm2(
    pairs_df: pd.DataFrame, dataset: str, cfg: Config
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    method = "gmm2"
    rows: List[pd.DataFrame] = []
    summaries: List[Dict[str, object]] = []

    for target_id, g in pairs_df.groupby("target_id", sort=True):
        g2 = g.copy()
        d = g2["distance"].to_numpy(dtype=float)
        n = len(d)

        # Rank candidates by increasing distance (1 = closest)
        order = np.argsort(d)
        dist_rank = np.empty(n, dtype=int)
        dist_rank[order] = np.arange(1, n + 1)
        g2["distance_rank"] = dist_rank

        fallback_used = 0
        p_near = np.full(n, np.nan, dtype=float)
        near_comp_mean = np.nan

        selected: np.ndarray
        try:
            gmm = GaussianMixture(
                n_components=2,
                random_state=cfg.seed,
                reg_covar=1e-6,
                max_iter=500,
            )
            gmm.fit(d.reshape(-1, 1))

            means = gmm.means_.reshape(-1)
            near_comp = int(np.argmin(means))
            near_comp_mean = float(means[near_comp])
            probs = gmm.predict_proba(d.reshape(-1, 1))
            p_near = probs[:, near_comp].astype(float)
            selected = p_near > cfg.gmm_prob_threshold

            if (not getattr(gmm, "converged_", True)) or selected.sum() in (0, n):
                fallback_used = 1
                # median threshold first, then rank-based fallback if still degenerate
                med = float(np.median(d)) if n else np.nan
                selected = d <= med if n else np.zeros(0, dtype=bool)
                if selected.sum() in (0, n):
                    selected = _fallback_by_distance_rank(d, cfg=cfg)
        except Exception:
            fallback_used = 1
            selected = _fallback_by_distance_rank(d, cfg=cfg)

        g2["dataset"] = dataset
        g2["method"] = method
        g2["p_near"] = p_near
        g2["near_comp_mean"] = near_comp_mean
        g2["selected"] = selected.astype(int)
        g2["fallback_used"] = int(fallback_used)

        rows.append(g2)

        sel = g2[g2["selected"] == 1].sort_values("distance", ascending=True)
        selected_sources_ordered = ",".join(str(int(x)) for x in sel["source_id"].tolist())
        summaries.append(
            {
                "dataset": dataset,
                "method": method,
                "target_id": target_id,
                "num_candidates": int(n),
                "num_selected": int(sel.shape[0]),
                "mean_distance_selected": float(sel["distance"].mean()) if sel.shape[0] else np.nan,
                "median_distance_selected": float(sel["distance"].median())
                if sel.shape[0]
                else np.nan,
                "near_comp_mean": float(near_comp_mean) if np.isfinite(near_comp_mean) else np.nan,
                "fallback_used": int(fallback_used),
                "selected_sources_ordered": selected_sources_ordered,
            }
        )

    pairs_out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    summary_out = pd.DataFrame.from_records(summaries)
    return pairs_out, summary_out


def save_plots(summary_df: pd.DataFrame, dataset: str, method: str, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Histogram of num_selected
    if "num_selected" in summary_df.columns and not summary_df.empty:
        plt.figure(figsize=(8, 4.5))
        plt.hist(summary_df["num_selected"].dropna().to_numpy(), bins=15)
        plt.title(f"{dataset} / {method} — num_selected across targets")
        plt.xlabel("num_selected")
        plt.ylabel("count(targets)")
        plt.tight_layout()
        plt.savefig(plots_dir / f"{dataset}__{method}__num_selected_hist.png", dpi=150)
        plt.close()

    # Scatter: mean_distance_selected vs num_selected
    if (
        {"mean_distance_selected", "num_selected"}.issubset(summary_df.columns)
        and not summary_df.empty
    ):
        plt.figure(figsize=(6.5, 4.5))
        x = summary_df["num_selected"].to_numpy(dtype=float)
        y = summary_df["mean_distance_selected"].to_numpy(dtype=float)
        plt.scatter(x, y, s=30, alpha=0.8)
        plt.title(f"{dataset} / {method} — mean(dist_selected) vs num_selected")
        plt.xlabel("num_selected")
        plt.ylabel("mean_distance_selected")
        plt.tight_layout()
        plt.savefig(plots_dir / f"{dataset}__{method}__mean_dist_vs_k.png", dpi=150)
        plt.close()


def ensure_results_dirs(cfg: Config) -> Tuple[Path, Path]:
    results_dir = Path(cfg.results_dir)
    plots_dir = results_dir / "plots"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    return results_dir, plots_dir


def run_for_dataset(distance_npz: Path, cfg: Config) -> None:
    pairs, metadata = load_distances_npz(distance_npz)
    dataset = str(metadata["dataset"])
    print(
        f"[{dataset}] loading {distance_npz.name} | "
        f"embedding_model={metadata['embedding_model']} | metric={metadata['metric']}"
    )

    pairs["source_id"] = pairs["source_id"].astype(int)
    pairs["target_id"] = pairs["target_id"].astype(int)
    pairs["distance"] = pairs["distance"].astype(float)
    pairs = pairs[pairs["source_id"] != pairs["target_id"]].copy()

    # For Cho2017, restrict to requested 10-subject subset
    if dataset == "cho2017":
        subset = set(CHO2017_SUBJECT_SUBSET)
        pairs = pairs[pairs["source_id"].isin(subset) & pairs["target_id"].isin(subset)].copy()

    results_dir, plots_dir = ensure_results_dirs(cfg)

    # Method 1: GMM2
    pairs_gmm, summary_gmm = select_gmm2(pairs, dataset=dataset, cfg=cfg)
    pairs_gmm.to_csv(results_dir / f"{dataset}__gmm2__pairs.csv", index=False)
    summary_gmm.to_csv(results_dir / f"{dataset}__gmm2__summary.csv", index=False)
    save_plots(summary_gmm, dataset=dataset, method="gmm2", plots_dir=plots_dir)

    # Method 2: Softmax tau
    pairs_sm, summary_sm = select_softmax_tau(pairs, dataset=dataset, cfg=cfg)
    pairs_sm.to_csv(results_dir / f"{dataset}__softmax_tau__pairs.csv", index=False)
    summary_sm.to_csv(results_dir / f"{dataset}__softmax_tau__summary.csv", index=False)
    save_plots(summary_sm, dataset=dataset, method="softmax_tau", plots_dir=plots_dir)

    # Minimal console report (auditable)
    gmm_fallback_rate = (
        float(summary_gmm["fallback_used"].mean()) if "fallback_used" in summary_gmm else np.nan
    )
    sm_guardrail_rate = (
        float(summary_sm["guardrail_used"].mean())
        if "guardrail_used" in summary_sm
        else np.nan
    )
    print(
        f"[{dataset}] wrote outputs to {results_dir}/ | "
        f"gmm2 fallback_rate={gmm_fallback_rate:.3f} | "
        f"softmax_tau guardrail_rate={sm_guardrail_rate:.3f}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=("bciciv2a", "cho2017", "all"),
        help="Dataset to run (default: all bundled datasets).",
    )
    p.add_argument(
        "--distance-file",
        type=str,
        default=None,
        help="Optional path to a single .npz distance artifact (overrides --dataset).",
    )
    p.add_argument("--results-dir", type=str, default="results", help="Output directory.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k-min", type=int, default=2)
    p.add_argument("--k-max", type=int, default=20)
    p.add_argument("--eps", type=float, default=1e-12)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(
        seed=int(args.seed),
        eps=float(args.eps),
        k_min=int(args.k_min),
        k_max=int(args.k_max),
        results_dir=str(args.results_dir),
    )
    set_seeds(cfg.seed)

    script_dir = Path(__file__).resolve().parent
    distances_dir = script_dir / "distances"

    if args.distance_file:
        inputs = [Path(args.distance_file)]
    else:
        discovered = discover_default_inputs(distances_dir)
        if args.dataset == "all":
            inputs = [
                discovered[key]
                for key in ("bciciv2a", "cho2017")
                if key in discovered
            ]
        else:
            if args.dataset not in discovered:
                raise FileNotFoundError(
                    f"No distance artifact found for dataset '{args.dataset}' under {distances_dir}."
                )
            inputs = [discovered[args.dataset]]

    if not inputs:
        raise FileNotFoundError(
            f"No distance .npz files found under {distances_dir}. "
            "Expected bci2a_pairwise_distances.npz and/or cho2017_pairwise_distances.npz."
        )

    for path in inputs:
        run_for_dataset(path, cfg=cfg)


if __name__ == "__main__":
    main()


