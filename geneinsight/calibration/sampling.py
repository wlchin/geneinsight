"""Random gene-set sampling for null-pool construction.

Ported from `geneinsight/output/rebuttal_docs/modules/sampling.py` so that the
installable `geneinsight` package does not depend on the rebuttal-only directory.
Deterministic seed scheme is preserved verbatim:

    seed = base_seed + size * 1000 + replicate
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List


@dataclass
class BackgroundInfo:
    genes: List[str]
    sha256: str

    @property
    def size(self) -> int:
        return len(self.genes)


@dataclass
class SampledGeneSet:
    size: int
    replicate: int
    genes: List[str]
    file_path: Path
    sha256: str
    seed: int


def _sha256_from_lines(lines: Iterator[str]) -> str:
    sha = hashlib.sha256()
    for line in lines:
        sha.update(line.strip().encode("utf-8"))
    return sha.hexdigest()


def load_background(path: str | Path) -> BackgroundInfo:
    """Load a background gene list (one gene per line, dedup preserved order)."""
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Background gene list not found: {path}")

    seen: set[str] = set()
    genes: List[str] = []
    with path_obj.open("r", encoding="utf-8") as fh:
        for line in fh:
            gene = line.strip()
            if not gene:
                continue
            if gene not in seen:
                seen.add(gene)
                genes.append(gene)

    if not genes:
        raise ValueError("Background gene list is empty after filtering")

    return BackgroundInfo(genes=genes, sha256=_sha256_from_lines(iter(genes)))


def derive_seed(base_seed: int, size: int, replicate: int) -> int:
    return int(base_seed) + size * 1000 + int(replicate)


def sample_gene_set(
    background: BackgroundInfo,
    size: int,
    replicate: int,
    base_seed: int,
    output_dir: str | Path,
    prefix: str = "genes",
) -> SampledGeneSet:
    """Draw `size` genes (without replacement) from the background, write a file."""
    if size > background.size:
        raise ValueError(
            f"Requested gene set size {size} exceeds background size {background.size}"
        )

    rng_seed = derive_seed(base_seed, size, replicate)
    rng = random.Random(rng_seed)
    genes = rng.sample(background.genes, size)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    file_path = out_dir / f"{prefix}_size{size}_rep{replicate}.txt"
    with file_path.open("w", encoding="utf-8") as fh:
        for gene in genes:
            fh.write(gene + "\n")

    return SampledGeneSet(
        size=size,
        replicate=replicate,
        genes=genes,
        file_path=file_path,
        sha256=_sha256_from_lines(iter(genes)),
        seed=rng_seed,
    )
