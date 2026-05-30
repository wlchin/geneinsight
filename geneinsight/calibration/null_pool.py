"""Null pool loader for empirical p-value recalibration.

A NullPool is a directory containing one or two CSV files plus a manifest:

    manifest.json          # n_themes, KS, bins, pipeline_version, built_at
    null_pool_pvalues.csv  # columns: p, bin, run               (for statistic='raw')
    null_pool_overlap.csv  # columns: overlap_ratio, bin, run   (for statistic='overlap')

The default pool ships inside the wheel under `geneinsight/calibration/data/`.
User-built pools (from `geneinsight build-null`) follow the same layout.

This module mirrors the lazy-load pattern of `stringdb_local.py` but without
the network leg: the default pool is bundled package data, not downloaded.
"""
from __future__ import annotations

import json
import logging
from importlib import resources
from pathlib import Path
from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

Statistic = Literal["raw", "overlap"]

_STATISTIC_CONFIG = {
    "raw": ("null_pool_pvalues.csv", "p"),
    "overlap": ("null_pool_overlap.csv", "overlap_ratio"),
}


class NullPool:
    """In-memory, bin-partitioned, pre-sorted null distribution."""

    def __init__(
        self,
        sorted_by_bin: Dict[str, np.ndarray],
        statistic: Statistic,
        manifest: Optional[dict] = None,
        source: Optional[str] = None,
    ):
        self._sorted = sorted_by_bin
        self.statistic = statistic
        self.manifest = manifest or {}
        self.source = source

    # ----- constructors -----

    @classmethod
    def from_path(cls, path: str | Path, statistic: Statistic) -> "NullPool":
        """Load a NullPool from a directory containing the expected CSVs."""
        path = Path(path)
        if not path.is_dir():
            raise FileNotFoundError(f"Null pool directory not found: {path}")
        csv_name, value_col = _STATISTIC_CONFIG[statistic]
        csv_path = path / csv_name
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Null pool CSV not found for statistic={statistic!r}: {csv_path}"
            )
        df = pd.read_csv(csv_path)
        sorted_by_bin = _build_sorted_index(df, value_col)
        manifest = _read_manifest(path / "manifest.json")
        return cls(sorted_by_bin, statistic, manifest=manifest, source=str(path))

    @classmethod
    def from_bundled(cls, statistic: Statistic = "raw") -> "NullPool":
        """Load the null pool bundled with the package."""
        csv_name, value_col = _STATISTIC_CONFIG[statistic]
        data_pkg = resources.files("geneinsight.calibration.data")
        csv_resource = data_pkg / csv_name
        if not csv_resource.is_file():
            raise FileNotFoundError(
                f"Bundled null pool CSV missing: {csv_name}. "
                "Reinstall the package or supply --null-cache."
            )
        with resources.as_file(csv_resource) as csv_path:
            df = pd.read_csv(csv_path)
        sorted_by_bin = _build_sorted_index(df, value_col)

        manifest = {}
        manifest_resource = data_pkg / "manifest.json"
        if manifest_resource.is_file():
            with resources.as_file(manifest_resource) as mpath:
                try:
                    manifest = json.loads(mpath.read_text())
                except json.JSONDecodeError:
                    logger.warning("Bundled null pool manifest is not valid JSON.")
        return cls(sorted_by_bin, statistic, manifest=manifest, source="<bundled>")

    # ----- queries -----

    def sorted_bin(self, bin_label: str) -> np.ndarray:
        """Return the pre-sorted null array for a bin. Raises KeyError if absent."""
        return self._sorted[bin_label]

    def bin_counts(self) -> Dict[str, int]:
        return {b: int(arr.size) for b, arr in self._sorted.items()}

    def total_themes(self) -> int:
        return int(sum(arr.size for arr in self._sorted.values()))

    def __repr__(self) -> str:
        return (
            f"NullPool(statistic={self.statistic!r}, "
            f"bins={len(self._sorted)}, n_themes={self.total_themes()}, "
            f"source={self.source!r})"
        )


def _build_sorted_index(df: pd.DataFrame, value_col: str) -> Dict[str, np.ndarray]:
    if value_col not in df.columns or "bin" not in df.columns:
        raise ValueError(
            f"Null pool CSV missing required columns: need 'bin' and '{value_col}', "
            f"got {list(df.columns)}"
        )
    out: Dict[str, np.ndarray] = {}
    for bin_label, group in df.groupby("bin", sort=False):
        out[str(bin_label)] = np.sort(group[value_col].to_numpy(dtype=float))
    return out


def _read_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        logger.warning("Manifest at %s is not valid JSON.", path)
        return {}
