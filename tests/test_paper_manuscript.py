"""Tests verifying the DRIGS paper manuscript files exist, are non-empty, contain
expected key metrics, and that all LaTeX section files are present."""

import json
import pathlib
import re

REPO_ROOT = pathlib.Path(__file__).parent.parent
DOCS_PAPER = REPO_ROOT / "docs" / "paper"
DOCS = REPO_ROOT / "docs"
KAGGLE_RESULTS = REPO_ROOT / "DRIGS_benchmark_results" / "kaggle_results"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_exp_results() -> dict:
    return json.loads((KAGGLE_RESULTS / "experiment_results.json").read_text())


# ---------------------------------------------------------------------------
# 1. File existence
# ---------------------------------------------------------------------------


def test_markdown_manuscript_exists() -> None:
    assert (DOCS / "PAPER_MANUSCRIPT.md").exists()


def test_latex_main_tex_exists() -> None:
    assert (DOCS_PAPER / "main.tex").exists()


def test_latex_abstract_exists() -> None:
    assert (DOCS_PAPER / "abstract.tex").exists()


def test_latex_introduction_exists() -> None:
    assert (DOCS_PAPER / "introduction.tex").exists()


def test_latex_architecture_exists() -> None:
    assert (DOCS_PAPER / "architecture.tex").exists()


def test_latex_scheduling_exists() -> None:
    assert (DOCS_PAPER / "scheduling.tex").exists()


def test_latex_evaluation_exists() -> None:
    assert (DOCS_PAPER / "evaluation.tex").exists()


def test_latex_related_work_exists() -> None:
    assert (DOCS_PAPER / "related_work.tex").exists()


def test_latex_conclusion_exists() -> None:
    assert (DOCS_PAPER / "conclusion.tex").exists()


def test_bibtex_references_exists() -> None:
    assert (DOCS_PAPER / "references.bib").exists()


# ---------------------------------------------------------------------------
# 2. Files are non-empty
# ---------------------------------------------------------------------------


def test_markdown_manuscript_non_empty() -> None:
    content = _read(DOCS / "PAPER_MANUSCRIPT.md")
    assert len(content) > 5000, "PAPER_MANUSCRIPT.md appears too short"


def test_all_latex_sections_non_empty() -> None:
    for fname in ("abstract.tex", "introduction.tex", "architecture.tex",
                  "scheduling.tex", "evaluation.tex", "related_work.tex",
                  "conclusion.tex"):
        content = _read(DOCS_PAPER / fname)
        assert len(content) > 200, f"{fname} appears too short"


def test_bibtex_has_entries() -> None:
    content = _read(DOCS_PAPER / "references.bib")
    assert "@inproceedings" in content or "@article" in content or "@online" in content


# ---------------------------------------------------------------------------
# 3. Nomenclature: DistilBERT never called "LLM"
# ---------------------------------------------------------------------------


def test_distilbert_not_called_llm_in_markdown() -> None:
    content = _read(DOCS / "PAPER_MANUSCRIPT.md")
    # Find any paragraph containing 'DistilBERT' and ensure 'LLM' is not nearby
    paragraphs = content.split("\n\n")
    for para in paragraphs:
        if "DistilBERT" in para or "distilbert" in para.lower():
            assert "LLM" not in para, (
                "DistilBERT referenced as 'LLM' — must use 'Transformer Encoder Model'"
            )


def test_distilbert_not_called_llm_in_latex() -> None:
    for fname in ("abstract.tex", "introduction.tex", "evaluation.tex"):
        if not (DOCS_PAPER / fname).exists():
            continue
        content = _read(DOCS_PAPER / fname)
        paragraphs = content.split("\n\n")
        for para in paragraphs:
            if "DistilBERT" in para or "distilbert" in para.lower():
                assert "LLM" not in para, (
                    f"{fname}: DistilBERT referenced as 'LLM'"
                )


# ---------------------------------------------------------------------------
# 4. Metric consistency — manuscript must contain verified empirical numbers
# ---------------------------------------------------------------------------


def test_p95_priority_metric_in_markdown() -> None:
    """P95 wait time for Priority scheduler: 2.740 s (from experiment_results.json)."""
    data = _load_exp_results()
    p95 = data["experiments"]["experiment_a"]["results"]["Priority"]["p95_wait_time_sec"]
    assert p95 == 2.74
    content = _read(DOCS / "PAPER_MANUSCRIPT.md")
    assert "2.740" in content, "Expected P95 Priority wait time 2.740 s in manuscript"


def test_p95_fifo_metric_in_markdown() -> None:
    """P95 wait time for FIFO: 6.866 s."""
    data = _load_exp_results()
    p95 = data["experiments"]["experiment_a"]["results"]["FIFO"]["p95_wait_time_sec"]
    assert p95 == 6.866
    content = _read(DOCS / "PAPER_MANUSCRIPT.md")
    assert "6.866" in content, "Expected P95 FIFO wait time 6.866 s in manuscript"


def test_topology_score_improvement_in_markdown() -> None:
    """Topology score improvement: 22.19%."""
    data = _load_exp_results()
    improvement = data["experiments"]["experiment_b"]["topology_score_improvement_pct"]
    assert improvement == 22.19
    content = _read(DOCS / "PAPER_MANUSCRIPT.md")
    assert "22.19" in content, "Expected +22.19% topology improvement in manuscript"


def test_topology_score_topology_aware_in_markdown() -> None:
    """TopologyAware mean score: 74.89."""
    data = _load_exp_results()
    score = data["experiments"]["experiment_b"]["topology_score_topology_aware"]
    assert score == 74.89
    content = _read(DOCS / "PAPER_MANUSCRIPT.md")
    assert "74.89" in content


def test_topology_score_random_in_markdown() -> None:
    """Random placement mean score: 61.29."""
    data = _load_exp_results()
    score = data["experiments"]["experiment_b"]["topology_score_random"]
    assert score == 61.29
    content = _read(DOCS / "PAPER_MANUSCRIPT.md")
    assert "61.29" in content


def test_control_plane_reschedule_latency_in_manuscript() -> None:
    """Control-plane rescheduling overhead: 0.496 ms (Experiment E)."""
    data = _load_exp_results()
    latency = data["experiments"]["experiment_e"]["mean_rescheduling_latency_ms"]
    assert latency == 0.496
    content = _read(DOCS / "PAPER_MANUSCRIPT.md")
    assert "0.496" in content, "Expected 0.496 ms control-plane latency in manuscript"


def test_e2e_recovery_latency_in_manuscript() -> None:
    """Physical end-to-end wall-clock recovery: 129.13 ms."""
    e2e_results_path = REPO_ROOT / "kaggle_results" / "e2e_recovery_results.json"
    data = json.loads(e2e_results_path.read_text())
    total_ms = data["metrics_ms"]["total_e2e_recovery_latency_ms"]
    assert total_ms == 129.13
    content = _read(DOCS / "PAPER_MANUSCRIPT.md")
    assert "129.13" in content, "Expected 129.13 ms E2E recovery latency in manuscript"


def test_latency_labels_distinct() -> None:
    """0.496 ms and 129.13 ms must both appear — one is control-plane, other is wall-clock."""
    content = _read(DOCS / "PAPER_MANUSCRIPT.md")
    assert "0.496" in content
    assert "129.13" in content


def test_vram_saturation_result_in_manuscript() -> None:
    """95% VRAM saturation -> 40.0% placement efficiency."""
    content = _read(DOCS / "PAPER_MANUSCRIPT.md")
    assert "40.0" in content, "Expected 40.0% placement efficiency at 95% VRAM saturation"


def test_oom_recovery_latency_in_manuscript() -> None:
    """OOM exception recovery latency: 0.021 ms."""
    content = _read(DOCS / "PAPER_MANUSCRIPT.md")
    assert "0.021" in content, "Expected 0.021 ms CUDA OOM recovery latency in manuscript"


def test_main_tex_includes_all_sections() -> None:
    """main.tex must \\input all six section files."""
    content = _read(DOCS_PAPER / "main.tex")
    for section in ("abstract", "introduction", "architecture", "scheduling",
                    "evaluation", "related_work", "conclusion"):
        assert f"\\input{{{section}}}" in content, (
            f"main.tex missing \\input{{{section}}}"
        )


def test_bibtex_has_slurm_and_kubernetes() -> None:
    content = _read(DOCS_PAPER / "references.bib")
    assert "slurm" in content.lower()
    assert "kubernetes" in content.lower() or "Kubernetes" in content


def test_related_work_covers_slurm_ray_alpa() -> None:
    content = _read(DOCS_PAPER / "related_work.tex")
    for name in ("Slurm", "Ray", "Alpa", "Kubernetes"):
        assert name in content, f"related_work.tex missing reference to {name}"
