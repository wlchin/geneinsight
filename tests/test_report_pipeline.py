"""Tests for geneinsight.report.pipeline.run_pipeline."""

import os
from unittest import mock

import pandas as pd
import pytest

from geneinsight.report import pipeline as report_pipeline


@pytest.fixture
def fake_input_dir(tmp_path):
    """Create a directory with the minimum 7 required input CSVs."""
    inp = tmp_path / "input"
    inp.mkdir()
    minimal = pd.DataFrame({"Term": ["term1"], "Cluster": [0]})
    pd.DataFrame({"query": ["term1"], "unique_genes": ["{'G1': 1}"], "context": ["ctx"]}).to_csv(inp / "summary.csv", index=False)
    minimal.to_csv(inp / "clustered.csv", index=False)
    minimal.rename(columns={"Term": "Topic"}).to_csv(inp / "api_results.csv", index=False)
    pd.DataFrame({"Topic": ["t1"], "Genes": ["G1"]}).to_csv(inp / "topics.csv", index=False)
    pd.DataFrame({"Term": ["term1"], "P-value": [0.01]}).to_csv(inp / "enrichment.csv", index=False)
    pd.DataFrame({"query": ["term1"], "ontology_dict": ["{'GO:1': 'G1'}"]}).to_csv(inp / "ontology_dict.csv", index=False)
    pd.DataFrame({
        "Term": ["term1"],
        "Odds Ratio": [1.5],
        "Adjusted P-value": [0.05],
        "P-value": [0.01],
        "Combined Score": [5.0],
    }).to_csv(inp / "enriched.csv", index=False)
    return str(inp)


def test_run_pipeline_raises_when_sphinx_build_fails(fake_input_dir, tmp_path, monkeypatch):
    """If build_sphinx_docs returns False, run_pipeline must raise RuntimeError
    rather than returning (False, None) silently."""
    # Mock everything before Sphinx so the test runs fast and offline.
    monkeypatch.setattr(report_pipeline, "generate_context", lambda **kw: None)
    monkeypatch.setattr(report_pipeline, "merge_context_ontology", lambda **kw: None)
    monkeypatch.setattr(report_pipeline, "generate_heatmaps", lambda **kw: None)
    monkeypatch.setattr(report_pipeline, "generate_circle_plot", lambda **kw: None)
    monkeypatch.setattr(report_pipeline, "generate_json_summary", lambda **kw: None)
    monkeypatch.setattr(report_pipeline, "generate_rst_files", lambda **kw: None)
    monkeypatch.setattr(report_pipeline, "generate_summary_page", lambda **kw: None)
    monkeypatch.setattr(report_pipeline, "generate_download_rst", lambda **kw: None)
    monkeypatch.setattr(report_pipeline, "copy_data_files", lambda **kw: None)

    # build_sphinx_docs returns False (simulating a build failure)
    monkeypatch.setattr(report_pipeline, "build_sphinx_docs", lambda **kw: False)

    with pytest.raises(RuntimeError, match="Sphinx build failed"):
        report_pipeline.run_pipeline(
            input_folder=fake_input_dir,
            output_folder=str(tmp_path / "output"),
            gene_set="test_gs",
            context_api_key="dummy",
        )
