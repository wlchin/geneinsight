"""Tests for Pipeline._generate_report and the Step 11 caller."""

import os
from unittest import mock

import pytest


@pytest.fixture
def pipeline_instance(tmp_path, monkeypatch):
    """Build a minimal Pipeline instance that can call _generate_report."""
    # Avoid OpenAI key check; we won't actually call APIs.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    from geneinsight.pipeline import Pipeline
    p = Pipeline(
        output_dir=str(tmp_path / "out"),
        temp_dir=str(tmp_path / "tmp"),
        n_samples=1,
        num_topics=2,
        pvalue_threshold=0.05,
        api_service="openai",
        api_model="gpt-4o-mini",
        species=9606,
        enable_metrics=False,
    )
    return p


def test_generate_report_raises_when_report_pipeline_fails(pipeline_instance, monkeypatch):
    """If reports_pipeline.run_pipeline raises, _generate_report must let the
    error propagate (no silent return None)."""
    def fake_run_pipeline(**kwargs):
        raise RuntimeError("Sphinx build failed for gene_set='foo'")

    # Patch the import inside _generate_report.
    from geneinsight.report import pipeline as report_pipeline_module
    monkeypatch.setattr(report_pipeline_module, "run_pipeline", fake_run_pipeline)

    with pytest.raises(RuntimeError, match="Sphinx build failed"):
        pipeline_instance._generate_report(
            output_path="/tmp/does-not-exist",
            query_gene_set="/tmp/fake_genes.txt",
        )


def test_sphinx_builds_dir_not_created_at_init(tmp_path, monkeypatch):
    """Pipeline init must not create an empty sphinx_builds/ directory; that
    only exists if Step 11 actually runs. Otherwise Step 12 zips an empty
    placeholder into a misleading 22-byte sphinx_builds.zip artifact."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    from geneinsight.pipeline import Pipeline
    p = Pipeline(
        output_dir=str(tmp_path / "out"),
        temp_dir=str(tmp_path / "tmp"),
        n_samples=1,
        num_topics=2,
        pvalue_threshold=0.05,
        api_service="openai",
        api_model="gpt-4o-mini",
        species=9606,
        enable_metrics=False,
    )

    sphinx_builds_dir = os.path.join(p.dirs["final"], "sphinx_builds")
    assert not os.path.exists(sphinx_builds_dir), (
        f"Expected {sphinx_builds_dir} to NOT exist at init time; was eagerly created. "
        "This leads to empty sphinx_builds.zip artifacts when reports are disabled."
    )
