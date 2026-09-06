"""Publication LaTeX Data Exporter & Paper Figures Generator.

This script transforms DRIGS benchmark results (JSON/CSV) into formatted LaTeX table code
(\\input{tables.tex}), publication plots (figure_schedulers.png, figure_topology.png, figure_fault_recovery.png),
and research paper summary documentation.
"""

import argparse
import csv
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("drigs-paper-export")


def load_benchmark_data(input_dir: Path) -> Dict[str, Any]:
    """Load master JSON dataset or fallback to CSV files in input_dir."""
    json_path = input_dir / "experiment_results.json"
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            if "experiments" in data:
                return data["experiments"]
            return data
        except Exception as err:
            logger.warning("Failed to parse %s (%s). Falling back to CSV search.", json_path, err)

    # Fallback to CSV parsing
    exp_a_path = input_dir / "experiment_a_schedulers.csv"
    exp_b_path = input_dir / "experiment_b_topology.csv"
    exp_e_path = input_dir / "experiment_e_fault_recovery.csv"

    exp_a_results: Dict[str, Any] = {}
    if exp_a_path.exists():
        with open(exp_a_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sched = row["Scheduler"]
                exp_a_results[sched] = {
                    "num_trials": int(row.get("NumTrials", 5)),
                    "throughput_mean_jobs_per_sec": float(row.get("Throughput_Jobs_Per_Sec", 0.0)),
                    "mean_wait_time_sec": float(row.get("MeanWait_Sec", 0.0)),
                    "p95_wait_time_sec": float(row.get("P95Wait_Sec", 0.0)),
                    "vram_fragmentation_pct": float(row.get("VRAM_Fragmentation_Pct", 0.0)),
                    "placement_efficiency_pct": float(row.get("Placement_Efficiency_Pct", 100.0)),
                }

    exp_b_data: Dict[str, Any] = {}
    if exp_b_path.exists():
        with open(exp_b_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                exp_b_data["topology_score_topology_aware"] = float(row.get("TopologyAware_Score", 0.0))
                exp_b_data["topology_score_random"] = float(row.get("Random_Score", 0.0))
                exp_b_data["topology_score_improvement_pct"] = float(row.get("Improvement_Pct", 0.0))

    exp_e_data: Dict[str, Any] = {}
    if exp_e_path.exists():
        with open(exp_e_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                metric = row["Metric"]
                val = float(row["Value"])
                if metric == "Rescheduling_Latency_MS":
                    exp_e_data["mean_rescheduling_latency_ms"] = val
                elif metric == "Job_Completion_Rate_Pct":
                    exp_e_data["mean_job_completion_rate_pct"] = val
                elif metric == "Total_Restored_Checkpoints":
                    exp_e_data["total_restored_checkpoints"] = int(val)

    return {
        "experiment_a": {"results": exp_a_results},
        "experiment_b": exp_b_data,
        "experiment_e": exp_e_data,
    }


def generate_latex_tables(experiments: Dict[str, Any], output_path: Path) -> Path:
    """Generate LaTeX tabular markup for paper submission."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []

    lines.append("% Auto-generated DRIGS Publication Benchmark Tables")
    lines.append("% Insert directly into paper LaTeX document via \\input{tables.tex}\n")

    # Table 1: Experiment A
    lines.append("% Table 1: Scheduling Policy Comparison")
    lines.append("\\begin{table}[h!]")
    lines.append("\\centering")
    lines.append("\\caption{Comparative Performance of DRIGS Scheduling Policies under Synthetic AI Workloads}")
    lines.append("\\label{tab:schedulers}")
    lines.append("\\begin{tabular}{lcccccc}")
    lines.append("\\hline")
    lines.append(
        "\\textbf{Scheduler} & \\textbf{Throughput} & \\textbf{Mean Wait} & \\textbf{P95 Wait} & "
        "\\textbf{VRAM Frag.} & \\textbf{Placement Eff.} \\\\"
    )
    lines.append(" & (jobs/sec) & (sec) & (sec) & (\\%) & (\\%) \\\\")
    lines.append("\\hline")

    exp_a_results = experiments.get("experiment_a", {}).get("results", {})
    for sched_name, m in exp_a_results.items():
        tp = f"{m.get('throughput_mean_jobs_per_sec', 0.0):.3f}"
        mw = f"{m.get('mean_wait_time_sec', 0.0):.2f}"
        p95 = f"{m.get('p95_wait_time_sec', 0.0):.2f}"
        frag = f"{m.get('vram_fragmentation_pct', 0.0):.1f}\\%"
        eff = f"{m.get('placement_efficiency_pct', 100.0):.1f}\\%"
        lines.append(f"{sched_name} & {tp} & {mw} & {p95} & {frag} & {eff} \\\\")

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}\n")

    # Table 2: Experiment B
    exp_b = experiments.get("experiment_b", {})
    lines.append("% Table 2: Interconnect Topology Awareness Evaluation")
    lines.append("\\begin{table}[h!]")
    lines.append("\\centering")
    lines.append("\\caption{GPU Interconnect Topology Score Comparison (Multi-GPU NVLink/PCIe Workloads)}")
    lines.append("\\label{tab:topology}")
    lines.append("\\begin{tabular}{lcc}")
    lines.append("\\hline")
    lines.append("\\textbf{Placement Strategy} & \\textbf{Mean Topology Score} & \\textbf{Relative Gain} \\\\")
    lines.append("\\hline")

    topo_score = exp_b.get("topology_score_topology_aware", 0.0)
    rand_score = exp_b.get("topology_score_random", 0.0)
    imp_pct = exp_b.get("topology_score_improvement_pct", 0.0)

    lines.append(f"TopologyAware & {topo_score:.2f} / 100.0 & +{imp_pct:.1f}\\% \\\\")
    lines.append(f"Random Placement (Baseline) & {rand_score:.2f} / 100.0 & 0.0\\% \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}\n")

    # Table 3: Experiment E
    exp_e = experiments.get("experiment_e", {})
    lines.append("% Table 3: Fault Recovery Overhead")
    lines.append("\\begin{table}[h!]")
    lines.append("\\centering")
    lines.append("\\caption{DRIGS Fault Detector & Checkpoint Rescheduler Latency Metrics}")
    lines.append("\\label{tab:fault_recovery}")
    lines.append("\\begin{tabular}{lc}")
    lines.append("\\hline")
    lines.append("\\textbf{Metric} & \\textbf{Value} \\\\")
    lines.append("\\hline")

    latency_ms = exp_e.get("mean_rescheduling_latency_ms", 0.0)
    completion_pct = exp_e.get("mean_job_completion_rate_pct", 0.0)
    total_restored = exp_e.get("total_restored_checkpoints", 0)

    lines.append(f"Mean Rescheduling Latency & {latency_ms:.2f} ms \\\\")
    lines.append(f"Job Completion Rate under Node Failure & {completion_pct:.1f}\\% \\\\")
    lines.append(f"Total Restored Checkpoints & {total_restored} \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved publication LaTeX tables to %s", output_path)
    return output_path


def generate_paper_figures(experiments: Dict[str, Any], output_dir: Path) -> List[Path]:
    """Generate high-resolution PNG plots for paper figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_figures: List[Path] = []

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("Matplotlib is not installed. Skipping figure generation.")
        return generated_figures

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Figure 1: Scheduler Comparison Bar Chart
    exp_a_results = experiments.get("experiment_a", {}).get("results", {})
    if exp_a_results:
        fig, ax1 = plt.subplots(figsize=(8, 4.5))

        sched_names = list(exp_a_results.keys())
        throughputs = [m.get("throughput_mean_jobs_per_sec", 0.0) for m in exp_a_results.values()]
        mean_waits = [m.get("mean_wait_time_sec", 0.0) for m in exp_a_results.values()]

        x = range(len(sched_names))
        width = 0.35

        rects1 = ax1.bar([i - width / 2 for i in x], throughputs, width, label="Throughput (jobs/s)", color="#2b5c8f")
        ax1.set_ylabel("Throughput (jobs/sec)", color="#2b5c8f", fontsize=11, fontweight="bold")
        ax1.tick_params(axis="y", labelcolor="#2b5c8f")
        ax1.set_xticks(x)
        ax1.set_xticklabels(sched_names, rotation=15, fontsize=10)

        ax2 = ax1.twinx()
        rects2 = ax2.bar([i + width / 2 for i in x], mean_waits, width, label="Mean Wait (s)", color="#d95f02", alpha=0.85)
        ax2.set_ylabel("Mean Queue Wait Time (sec)", color="#d95f02", fontsize=11, fontweight="bold")
        ax2.tick_params(axis="y", labelcolor="#d95f02")

        plt.title("DRIGS Scheduling Policy Evaluation (Throughput vs Latency)", fontsize=12, fontweight="bold", pad=12)
        fig.tight_layout()

        fig1_path = output_dir / "figure_schedulers.png"
        plt.savefig(fig1_path, dpi=300)
        plt.close(fig)
        generated_figures.append(fig1_path)
        logger.info("Generated Figure 1: %s", fig1_path)

    # 2. Figure 2: Interconnect Topology Score Comparison
    exp_b = experiments.get("experiment_b", {})
    if exp_b:
        fig, ax = plt.subplots(figsize=(6, 4))
        strategies = ["TopologyAware", "Random Placement"]
        scores = [
            exp_b.get("topology_score_topology_aware", 0.0),
            exp_b.get("topology_score_random", 0.0),
        ]
        colors = ["#1b9e77", "#7570b3"]

        bars = ax.bar(strategies, scores, color=colors, width=0.5)
        ax.set_ylabel("Mean Topology Interconnect Score (0-100)", fontsize=11, fontweight="bold")
        ax.set_ylim(0, 105)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        plt.title("NVLink & PCIe Interconnect Placement Score Comparison", fontsize=11, fontweight="bold", pad=12)
        fig.tight_layout()

        fig2_path = output_dir / "figure_topology.png"
        plt.savefig(fig2_path, dpi=300)
        plt.close(fig)
        generated_figures.append(fig2_path)
        logger.info("Generated Figure 2: %s", fig2_path)

    # 3. Figure 3: Fault Recovery Overhead
    exp_e = experiments.get("experiment_e", {})
    if exp_e:
        fig, ax = plt.subplots(figsize=(5, 4))
        metrics = ["Rescheduling Latency (ms)", "Job Completion Rate (%)"]
        vals = [
            exp_e.get("mean_rescheduling_latency_ms", 0.0),
            exp_e.get("mean_job_completion_rate_pct", 0.0),
        ]

        bars = ax.bar(metrics, vals, color=["#e7298a", "#66a61e"], width=0.4)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        plt.title("DRIGS Checkpoint Rescheduler & Fault Recovery Overhead", fontsize=11, fontweight="bold", pad=12)
        fig.tight_layout()

        fig3_path = output_dir / "figure_fault_recovery.png"
        plt.savefig(fig3_path, dpi=300)
        plt.close(fig)
        generated_figures.append(fig3_path)
        logger.info("Generated Figure 3: %s", fig3_path)

    return generated_figures


def generate_paper_summary_markdown(experiments: Dict[str, Any], output_path: Path) -> Path:
    """Generate Markdown paper research report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []

    lines.append("# DRIGS Research Paper Empirical Evaluation Summary\n")
    lines.append("This document summarizes empirical benchmark results for paper submission.\n")

    lines.append("## 1. Scheduling Policy Evaluation (Experiment A)")
    lines.append("| Scheduler | Throughput (jobs/sec) | Mean Wait (s) | P95 Wait (s) | VRAM Frag. (%) | Placement Eff. (%) |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")

    exp_a_results = experiments.get("experiment_a", {}).get("results", {})
    for sched_name, m in exp_a_results.items():
        tp = m.get("throughput_mean_jobs_per_sec", 0.0)
        mw = m.get("mean_wait_time_sec", 0.0)
        p95 = m.get("p95_wait_time_sec", 0.0)
        frag = m.get("vram_fragmentation_pct", 0.0)
        eff = m.get("placement_efficiency_pct", 100.0)
        lines.append(f"| **{sched_name}** | {tp:.3f} | {mw:.2f} | {p95:.2f} | {frag:.1f}% | {eff:.1f}% |")

    exp_b = experiments.get("experiment_b", {})
    lines.append("\n## 2. Interconnect Topology Awareness (Experiment B)")
    lines.append(f"- **TopologyAware Score**: `{exp_b.get('topology_score_topology_aware', 0.0):.2f}` / 100.0")
    lines.append(f"- **Random Placement Score**: `{exp_b.get('topology_score_random', 0.0):.2f}` / 100.0")
    lines.append(f"- **Relative Interconnect Bandwidth Improvement**: `+{exp_b.get('topology_score_improvement_pct', 0.0):.2f}%`")

    exp_e = experiments.get("experiment_e", {})
    lines.append("\n## 3. Fault Recovery & Rescheduling Latency (Experiment E)")
    lines.append(f"- **Mean Rescheduling Latency**: `{exp_e.get('mean_rescheduling_latency_ms', 0.0):.2f} ms`")
    lines.append(f"- **Job Completion Rate under Failure**: `{exp_e.get('mean_job_completion_rate_pct', 0.0):.1f}%`")
    lines.append(f"- **Total Restored Checkpoints**: `{exp_e.get('total_restored_checkpoints', 0)}`")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved research paper summary markdown to %s", output_path)
    return output_path


def export_paper_artifacts(input_dir: Path, output_dir: Path, paper_doc_path: Optional[Path] = None) -> Dict[str, Any]:
    """Execute complete paper data export pipeline."""
    experiments = load_benchmark_data(input_dir)

    latex_path = output_dir / "tables.tex"
    generate_latex_tables(experiments, latex_path)

    figures = generate_paper_figures(experiments, output_dir)

    doc_target = paper_doc_path if paper_doc_path else Path("docs/PAPER_EXPERIMENTS.md")
    generate_paper_summary_markdown(experiments, doc_target)

    return {
        "latex_tables": latex_path,
        "figures": figures,
        "paper_doc": doc_target,
    }


def main():
    parser = argparse.ArgumentParser(description="DRIGS Publication LaTeX Data Exporter")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="kaggle_results",
        help="Input directory containing JSON/CSV benchmark results (default: kaggle_results)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="paper_outputs",
        help="Output directory for generated tables.tex and plot images (default: paper_outputs)",
    )
    parser.add_argument(
        "--paper-doc",
        type=str,
        default="docs/PAPER_EXPERIMENTS.md",
        help="Output markdown path for paper documentation (default: docs/PAPER_EXPERIMENTS.md)",
    )

    args = parser.parse_args()
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    doc_path = Path(args.paper_doc)

    export_paper_artifacts(input_path, output_path, doc_path)


if __name__ == "__main__":
    main()
