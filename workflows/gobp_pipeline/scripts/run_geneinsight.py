#!/usr/bin/env python3
"""
Run the GeneInsight pipeline for a single gene list and persist metadata.
"""

import json
import os
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from geneinsight.pipeline import Pipeline  # noqa: E402


def load_genes(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def prepare_output_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def run_pipeline_for_sample(sample_id: str, gene_list: Path, background: Path, pipeline_cfg: dict, sample_dir: Path) -> dict:
    genes = load_genes(gene_list)
    gene_set_name = gene_list.stem

    # Resolve tmp_output_root. Default = system tempfile (outside sample_dir).
    # NEVER default to a path inside sample_dir, because prepare_output_dir()
    # below calls rmtree(sample_dir) and would wipe the pipeline's outputs
    # before the copy loop can read them. (Empirically observed bug 2026-05-17:
    # produces empty sample_dirs after a full ~50-min pipeline run with all
    # API spend wasted.)
    if pipeline_cfg.get("output_dir"):
        tmp_output_root = Path(pipeline_cfg["output_dir"])
    else:
        tmp_output_root = Path(tempfile.mkdtemp(prefix=f"geneinsight_{gene_set_name}_"))

    # Defensive assertion: explicit output_dir paths must not sit inside sample_dir.
    sample_dir_abs = Path(sample_dir).resolve()
    tmp_output_abs = tmp_output_root.resolve()
    try:
        inside = tmp_output_abs.is_relative_to(sample_dir_abs)
    except AttributeError:
        # Python <3.9 fallback
        inside = str(tmp_output_abs).startswith(str(sample_dir_abs) + os.sep) or tmp_output_abs == sample_dir_abs
    if inside:
        raise ValueError(
            f"output_dir ({tmp_output_abs}) is inside sample_dir ({sample_dir_abs}). "
            f"prepare_output_dir() will rmtree sample_dir and wipe the pipeline "
            f"outputs before the copy loop runs. Use an output_dir outside sample_dir."
        )

    pipeline_cfg = pipeline_cfg.copy()
    pipeline_cfg["output_dir"] = str(tmp_output_root)

    # Record start time
    start_time = time.perf_counter()

    pipeline = Pipeline(**pipeline_cfg)
    pipeline.run(
        query_gene_set=str(gene_list),
        background_gene_list=str(background),
        generate_report=False,
        report_title=None,
    )

    # Record end time and compute duration
    end_time = time.perf_counter()
    duration_seconds = end_time - start_time

    run_id = f"{gene_set_name}_{pipeline.timestamp}"
    run_dir = Path(pipeline.dirs["final"]) / run_id
    if not run_dir.exists():
        zip_path = run_dir.with_suffix(".zip")
        if zip_path.exists():
            with zipfile.ZipFile(zip_path, "r") as archive:
                archive.extractall(zip_path.parent)
        else:
            raise FileNotFoundError(f"Pipeline run directory not found: {run_dir}")
    # Snapshot the items we'll copy BEFORE wiping sample_dir, so even if a
    # future regression puts run_dir back inside sample_dir, we crash early.
    items_to_copy = list(run_dir.iterdir())
    if not items_to_copy:
        raise RuntimeError(
            f"Pipeline produced no output in {run_dir}. Refusing to "
            f"prepare_output_dir(sample_dir) — that would create an empty "
            f"sample_dir and lose the API spend already made."
        )

    prepare_output_dir(sample_dir)
    # copytree preserving metadata
    for item in items_to_copy:
        dest = sample_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)

    # Verify the copy actually landed before reporting success.
    final_files = list(sample_dir.iterdir())
    if not final_files:
        raise RuntimeError(
            f"sample_dir ({sample_dir}) is empty after copy from {run_dir}. "
            f"This is the 'silent empty dir' failure mode."
        )

    metadata = {
        "run_id": run_id,
        "timestamp": pipeline.timestamp,
        "gene_list": str(gene_list),
        "background": str(background),
        "gene_count": len(genes),
        "result_dir": str(sample_dir),
        "duration_seconds": duration_seconds,
        "n_files_in_sample_dir": len(final_files),
    }
    return metadata


def main():
    params = snakemake.params  # type: ignore # noqa: F821
    inputs = snakemake.input  # type: ignore # noqa: F821
    outputs = snakemake.output  # type: ignore # noqa: F821

    gene_list = Path(inputs.gene_list).resolve()
    background = Path(inputs.background).resolve()
    sample_dir = Path(params.sample_dir).resolve()
    metadata_path = Path(outputs.metadata).resolve()
    done_path = Path(outputs.done).resolve()

    try:
        metadata = run_pipeline_for_sample(
            sample_id=params.sample_id,
            gene_list=gene_list,
            background=background,
            pipeline_cfg=params.pipeline,
            sample_dir=sample_dir,
        )
        metadata.update(
            {
                "sample_id": params.sample_id,
                "group": params.group,
            }
        )
    except ValueError as exc:
        message = str(exc)
        skip_reasons = {
            "No overlap found between query gene set and background": "skipped_no_overlap",
            "Cannot filter topics: input DataFrame is empty": "skipped_empty_topics",
            "No enrichment or documents returned from StringDB": "skipped_no_enrichment",
            "Hypergeometric enrichment resulted in an empty DataFrame": "skipped_no_hypergeometric",
        }
        for substring, status in skip_reasons.items():
            if substring in message:
                metadata = {
                    "sample_id": params.sample_id,
                    "group": params.group,
                    "gene_list": str(gene_list),
                    "background": str(background),
                    "gene_count": len(load_genes(gene_list)),
                    "status": status,
                    "message": message,
                    "duration_seconds": None,
                }
                metadata_path.parent.mkdir(parents=True, exist_ok=True)
                metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                done_path.parent.mkdir(parents=True, exist_ok=True)
                done_path.touch()
                return
        raise
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    done_path.parent.mkdir(parents=True, exist_ok=True)
    done_path.touch()


if __name__ == "__main__":
    main()
