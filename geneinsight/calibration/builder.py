"""Build a custom null pool by running random gene sets through the pipeline.

This orchestrates the size-matched single-bin mode: K random gene sets of the
same size are run serially, their Step 6 unfiltered p-values pooled, and
written out as a NullPool-compatible directory.

Concurrency safeguards (see CLAUDE.md memory `project_pipeline_concurrency_gotchas`):
    - api_parallel_jobs is hard-coded to 1 (joblib pickling bug).
    - Each run gets its own output_dir under {output_cache}/runs/{run_id}/.
    - Per-run query gene file paths are read from metadata.json, never inferred
      from filenames (lesson 9).

`run_manifest.json` is written incrementally so an interrupted build can resume
on the same --output-cache.
"""
from __future__ import annotations

import json
import logging
import os
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from .empirical_p import assign_bin, parse_overlap
from .sampling import load_background, sample_gene_set

logger = logging.getLogger(__name__)


@dataclass
class RunRecord:
    replicate: int
    seed: int
    gene_file: str
    run_dir: str
    status: str  # "completed" | "failed"
    duration_seconds: float
    error: Optional[str] = None


def _load_manifest(path: Path) -> dict:
    if not path.exists():
        return {"runs": []}
    try:
        data = json.loads(path.read_text())
        if "runs" not in data:
            data["runs"] = []
        return data
    except json.JSONDecodeError:
        logger.warning("Existing manifest %s is corrupt; starting fresh.", path)
        return {"runs": []}


def _write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    tmp.replace(path)


def _pool_run_pvalues(run_dir: Path, replicate: int, bin_label: str) -> tuple[
    Optional[pd.DataFrame], Optional[pd.DataFrame]
]:
    """Load `enriched_unfiltered.csv` from a finished run, return rows for both
    null pool CSVs (raw p and overlap ratio)."""
    candidates = [
        run_dir / "enriched_unfiltered.csv",
        # Older runs in the repo use a v2 suffix — accept both for forward-compat.
        run_dir / "enriched_unfiltered_v2.csv",
    ]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        return None, None

    df = pd.read_csv(src)
    if df.empty:
        return None, None

    run_id = run_dir.name
    raw_rows = pd.DataFrame({
        "p": df["P-value"].astype(float),
        "bin": bin_label,
        "run": run_id,
        "replicate": replicate,
    })
    overlap_rows = pd.DataFrame({
        "overlap_ratio": df["Overlap"].apply(parse_overlap).astype(float),
        "bin": bin_label,
        "run": run_id,
        "replicate": replicate,
    })
    overlap_rows = overlap_rows.dropna(subset=["overlap_ratio"])
    raw_rows = raw_rows.dropna(subset=["p"])
    return raw_rows, overlap_rows


def build_null(
    query_size: int,
    replicates: int,
    background: str | Path,
    output_cache: str | Path,
    seed: int = 42,
    species: int = 9606,
    api_model: str = "gpt-4o-mini",
    use_local_stringdb: bool = True,
    pvalue_threshold: float = 0.05,
    overlap_ratio_threshold: float = 0.0,  # don't pre-filter null observations
) -> Path:
    """Build a single-bin null pool by running K random gene sets through the pipeline.

    Returns the output_cache path (a NullPool-compatible directory).
    """
    from ..pipeline import Pipeline  # local import to avoid pulling pipeline at module load

    bin_label = assign_bin(query_size)
    if bin_label is None:
        raise ValueError(
            f"query_size={query_size} is outside the supported 1-500 range."
        )

    out_root = Path(output_cache).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    runs_root = out_root / "runs"
    gene_sets_root = out_root / "gene_sets"
    runs_root.mkdir(exist_ok=True)
    gene_sets_root.mkdir(exist_ok=True)

    manifest_path = out_root / "run_manifest.json"
    manifest = _load_manifest(manifest_path)
    manifest.setdefault("config", {})
    manifest["config"].update({
        "query_size": query_size,
        "replicates_requested": replicates,
        "background": str(Path(background).resolve()),
        "base_seed": seed,
        "species": species,
        "api_model": api_model,
        "use_local_stringdb": use_local_stringdb,
        "pvalue_threshold": pvalue_threshold,
        "overlap_ratio_threshold": overlap_ratio_threshold,
    })

    completed_reps = {r["replicate"] for r in manifest["runs"] if r.get("status") == "completed"}

    bg_info = load_background(background)
    if query_size > bg_info.size:
        raise ValueError(
            f"query_size {query_size} exceeds background size {bg_info.size}"
        )

    for rep in range(1, replicates + 1):
        if rep in completed_reps:
            logger.info("[build-null] replicate %d already complete; skipping.", rep)
            continue

        sampled = sample_gene_set(
            background=bg_info,
            size=query_size,
            replicate=rep,
            base_seed=seed,
            output_dir=gene_sets_root,
        )

        run_id = f"null_size{query_size}_rep{rep}"
        run_dir = runs_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "[build-null] starting replicate %d/%d (size=%d, seed=%d) -> %s",
            rep, replicates, query_size, sampled.seed, run_dir,
        )
        t0 = time.time()
        record_status = "completed"
        error_msg: Optional[str] = None
        try:
            pipeline = Pipeline(
                output_dir=str(run_dir),
                pvalue_threshold=pvalue_threshold,
                overlap_ratio_threshold=overlap_ratio_threshold,
                api_parallel_jobs=1,  # forced; concurrency safeguard
                api_model=api_model,
                species=species,
                use_local_stringdb=use_local_stringdb,
                empirical_p=False,  # don't recurse
                quiet_metrics=True,
            )
            pipeline.run(
                query_gene_set=str(sampled.file_path),
                background_gene_list=str(Path(background).resolve()),
                generate_report=False,
            )
        except Exception as exc:  # noqa: BLE001 — we want the build to keep going
            record_status = "failed"
            error_msg = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            logger.error("[build-null] replicate %d failed: %s", rep, exc)

        duration = time.time() - t0
        manifest["runs"].append(asdict(RunRecord(
            replicate=rep,
            seed=sampled.seed,
            gene_file=str(sampled.file_path),
            run_dir=str(run_dir),
            status=record_status,
            duration_seconds=round(duration, 2),
            error=error_msg,
        )))
        _write_manifest(manifest_path, manifest)

    _pool_and_write(out_root, runs_root, bin_label)

    # Also write a top-level manifest.json so NullPool.from_path can find it.
    _write_pool_manifest(out_root, query_size, bin_label, manifest)
    return out_root


def _pool_and_write(out_root: Path, runs_root: Path, bin_label: str) -> None:
    raw_frames: list[pd.DataFrame] = []
    overlap_frames: list[pd.DataFrame] = []
    for rd in sorted(runs_root.iterdir()):
        if not rd.is_dir():
            continue
        # replicate index is encoded in run_id "null_sizeS_repN"
        try:
            rep = int(rd.name.rsplit("_rep", 1)[1])
        except (IndexError, ValueError):
            rep = -1
        raw, overlap = _pool_run_pvalues(rd, replicate=rep, bin_label=bin_label)
        if raw is not None and not raw.empty:
            raw_frames.append(raw)
        if overlap is not None and not overlap.empty:
            overlap_frames.append(overlap)

    if raw_frames:
        pd.concat(raw_frames, ignore_index=True).to_csv(
            out_root / "null_pool_pvalues.csv", index=False
        )
    if overlap_frames:
        pd.concat(overlap_frames, ignore_index=True).to_csv(
            out_root / "null_pool_overlap.csv", index=False
        )
    logger.info(
        "[build-null] pooled %d raw-p frames, %d overlap frames -> %s",
        len(raw_frames), len(overlap_frames), out_root,
    )


def _write_pool_manifest(
    out_root: Path, query_size: int, bin_label: str, run_manifest: dict
) -> None:
    n_completed = sum(1 for r in run_manifest["runs"] if r.get("status") == "completed")
    raw_path = out_root / "null_pool_pvalues.csv"
    overlap_path = out_root / "null_pool_overlap.csv"

    n_themes = 0
    if raw_path.exists():
        try:
            n_themes = sum(1 for _ in raw_path.open()) - 1
        except OSError:
            n_themes = 0

    manifest = {
        "kind": "custom",
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "query_size": query_size,
        "bin": bin_label,
        "n_runs_completed": n_completed,
        "n_themes": n_themes,
        "statistics_available": [
            s for s, p in [("raw", raw_path), ("overlap", overlap_path)] if p.exists()
        ],
        "build_config": run_manifest.get("config", {}),
        "notes": (
            "Single-bin custom null built by `geneinsight build-null`. "
            "Only the bin matching --query-size has coverage; out-of-bin queries will "
            "fall back to NaN empirical-p."
        ),
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
