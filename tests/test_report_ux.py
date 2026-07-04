"""Tests for report-generation user-experience improvements."""

import os
import re
import pathlib
import pytest


@pytest.fixture
def pipeline_instance(tmp_path, monkeypatch):
    """Minimal Pipeline instance for unit testing finalization helpers."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    from geneinsight.pipeline import Pipeline
    return Pipeline(
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


def test_zip_results_folders_keeps_sphinx_builds_dir(pipeline_instance, tmp_path):
    """After _zip_results_folders runs, sphinx_builds/ must still exist
    alongside its zip — so the user can browse the report without unzipping.
    Other directories should be zipped and removed as before."""
    results_dir = pipeline_instance.dirs["final"]
    os.makedirs(results_dir, exist_ok=True)

    sphinx_dir = os.path.join(results_dir, "sphinx_builds")
    os.makedirs(os.path.join(sphinx_dir, "html_build_test", "build", "html"))
    pathlib.Path(sphinx_dir, "html_build_test", "build", "html", "index.html").write_text("<html>test</html>")

    other_dir = os.path.join(results_dir, "other_dir")
    os.makedirs(other_dir)
    pathlib.Path(other_dir, "data.csv").write_text("a,b\n1,2\n")

    pipeline_instance._zip_results_folders(results_dir)

    assert os.path.exists(os.path.join(results_dir, "sphinx_builds.zip"))
    assert os.path.isdir(sphinx_dir), (
        "sphinx_builds/ directory should be preserved after zipping so the "
        "user can browse the report without unzipping"
    )
    assert os.path.exists(os.path.join(sphinx_dir, "html_build_test", "build", "html", "index.html"))

    assert os.path.exists(os.path.join(results_dir, "other_dir.zip"))
    assert not os.path.exists(other_dir), (
        "other directories should still be zipped-and-removed (only sphinx_builds is preserved)"
    )


def test_write_report_redirect_creates_meta_refresh_html(pipeline_instance, tmp_path):
    """_write_report_redirect must create results/report.html that meta-refreshes
    to the actual deep index.html. Use a RELATIVE path so the redirect file is
    self-contained (works after the results dir is moved or shared)."""
    results_dir = pipeline_instance.dirs["final"]
    os.makedirs(results_dir, exist_ok=True)

    deep_index = os.path.join(
        results_dir, "sphinx_builds", "html_build_sample", "build", "html", "index.html"
    )
    os.makedirs(os.path.dirname(deep_index), exist_ok=True)
    pathlib.Path(deep_index).write_text("<html>real report</html>")

    pipeline_instance._write_report_redirect(deep_index)

    redirect_path = os.path.join(results_dir, "report.html")
    assert os.path.exists(redirect_path)

    content = pathlib.Path(redirect_path).read_text()
    assert '<meta http-equiv="refresh"' in content
    assert 'sphinx_builds/html_build_sample/build/html/index.html' in content
    assert results_dir not in content


def test_write_report_redirect_url_encodes_special_chars(pipeline_instance, tmp_path):
    """Gene-set names with spaces or special chars must be URL-encoded in the
    redirect target so the meta-refresh resolves correctly."""
    results_dir = pipeline_instance.dirs["final"]
    os.makedirs(results_dir, exist_ok=True)

    deep_index = os.path.join(
        results_dir, "sphinx_builds", "html_build_my gene set", "build", "html", "index.html"
    )
    os.makedirs(os.path.dirname(deep_index), exist_ok=True)
    pathlib.Path(deep_index).write_text("<html>real report</html>")

    pipeline_instance._write_report_redirect(deep_index)

    redirect_path = os.path.join(results_dir, "report.html")
    content = pathlib.Path(redirect_path).read_text()
    assert "my%20gene%20set" in content
    assert "my gene set" not in content


def test_write_report_redirect_returns_redirect_path(pipeline_instance, tmp_path):
    """The method should return the absolute path of the created redirect so
    callers can include it in log/console output."""
    results_dir = pipeline_instance.dirs["final"]
    os.makedirs(results_dir, exist_ok=True)

    deep_index = os.path.join(
        results_dir, "sphinx_builds", "html_build_x", "build", "html", "index.html"
    )
    os.makedirs(os.path.dirname(deep_index), exist_ok=True)
    pathlib.Path(deep_index).write_text("<html/>")

    returned = pipeline_instance._write_report_redirect(deep_index)
    assert returned == os.path.join(results_dir, "report.html")
    assert os.path.isabs(returned)


def test_completion_summary_includes_file_url_when_report_generated(pipeline_instance, capsys):
    """The success message must print a file:// URL pointing at report.html
    so the user can cmd-click to open the report."""
    pipeline_instance._print_completion_summary(
        output_zip="/tmp/fake/results.zip",
        report_redirect_path="/tmp/fake/results/report.html",
    )
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "file:///tmp/fake/results/report.html" in combined
    assert "Pipeline completed successfully" in combined


def test_completion_summary_omits_report_url_when_no_report(pipeline_instance, capsys):
    """When generate_report=False (no redirect path), the success message must
    not print a report URL (would point to nothing)."""
    pipeline_instance._print_completion_summary(
        output_zip="/tmp/fake/results.zip",
        report_redirect_path=None,
    )
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "file://" not in combined
    assert "Pipeline completed successfully" in combined
    assert "/tmp/fake/results.zip" in combined


def test_completion_summary_uses_rich_link_markup(pipeline_instance, capsys):
    """The output should contain the URL text such that modern terminals
    render it as a clickable hyperlink. Either OSC-8 escape sequence (rich's
    [link=...] markup) or plain file:// URL text (terminals like iTerm2
    auto-linkify file:// URLs)."""
    pipeline_instance._print_completion_summary(
        output_zip="/tmp/fake/results.zip",
        report_redirect_path="/tmp/fake/results/report.html",
    )
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert combined.count("file:///tmp/fake/results/report.html") >= 1


def test_completion_summary_handles_brackets_in_path(pipeline_instance, capsys):
    """Filesystem paths can legally contain '[' (e.g. proj[2026]/). The helper
    must not crash when interpolating such a path into rich markup."""
    # Should not raise.
    pipeline_instance._print_completion_summary(
        output_zip="/tmp/proj[2026]/results.zip",
        report_redirect_path="/tmp/proj[2026]/results/report.html",
    )
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    # The literal URL must appear in output (not stripped or escaped to gibberish).
    assert "file:///tmp/proj[2026]/results/report.html" in combined


def test_compute_post_reorganize_report_path_returns_deterministic_path(pipeline_instance):
    """Pipeline.run must compute the post-reorganize index.html path
    (not trust the stale path returned by _generate_report) before calling
    _write_report_redirect. This test verifies the helper produces the
    correct deterministic path for a given gene_set_name."""
    computed = pipeline_instance._post_reorganize_report_path("sample")
    expected = os.path.join(
        pipeline_instance.dirs["sphinx_builds"],
        "html_build_sample", "build", "html", "index.html",
    )
    assert computed == expected


def test_post_reorganize_report_path_handles_dashes_and_underscores(pipeline_instance):
    """Sanity check the deterministic helper with realistic gene-set names."""
    base = pipeline_instance.dirs["sphinx_builds"]
    assert pipeline_instance._post_reorganize_report_path("my-gene-set_v2") == os.path.join(
        base, "html_build_my-gene-set_v2", "build", "html", "index.html"
    )


def test_pipeline_run_wire_up_uses_post_reorganize_path():
    """Source-level check: Pipeline.run must call _post_reorganize_report_path
    (not pass report_output directly to _write_report_redirect). This is a
    guard against future refactors silently reverting the fix."""
    import inspect
    from geneinsight.pipeline import Pipeline
    src = inspect.getsource(Pipeline.run)
    assert "_post_reorganize_report_path" in src, (
        "Pipeline.run must call _post_reorganize_report_path before "
        "_write_report_redirect; the cached report_output from "
        "_generate_report is stale after Step 12."
    )
