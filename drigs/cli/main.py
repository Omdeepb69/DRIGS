"""DRIGS Command Line Interface (Typer)."""

import json
from pathlib import Path
from typing import Optional
import httpx
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="drigs",
    help="Distributed Resource & Intelligent GPU Scheduling CLI",
    add_completion=False,
)
console = Console()

DEFAULT_SERVER_URL = "http://127.0.0.1:8000"


def _get_client(server_url: str) -> httpx.Client:
    return httpx.Client(base_url=server_url.rstrip("/"), timeout=10.0)


@app.command("submit")
def submit_cmd(
    spec_path: str = typer.Argument(..., help="Path to workload spec YAML/JSON file"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Override job name"),
    priority: int = typer.Option(0, "--priority", "-p", help="Job scheduling priority"),
    url: str = typer.Option(DEFAULT_SERVER_URL, "--url", "-u", help="DRIGS Control Plane API URL"),
):
    """Submit a workload spec to the DRIGS cluster."""
    path = Path(spec_path)
    if not path.exists():
        console.print(f"[bold red]Error:[/] Spec file '{spec_path}' does not exist.")
        raise typer.Exit(code=1)

    try:
        from drigs.core.spec import parse_workload_spec
        spec = parse_workload_spec(path.read_text(encoding="utf-8"))
    except Exception as e:
        console.print(f"[bold red]Error parsing workload spec:[/] {e}")
        raise typer.Exit(code=1)

    payload = {
        "name": name or spec.name,
        "spec": spec.model_dump(mode="json"),
        "priority": priority,
    }

    try:
        with _get_client(url) as client:
            res = client.post("/v1/jobs", json=payload)
            if res.status_code != 201:
                console.print(f"[bold red]Submission failed ({res.status_code}):[/] {res.text}")
                raise typer.Exit(code=1)

            data = res.json()
            console.print(f"[bold green]Job Submitted Successfully![/]")
            console.print(f"Job ID: [cyan]{data['job_id']}[/]")
            console.print(f"Status: [yellow]{data['status']}[/]")
    except httpx.RequestError as e:
        console.print(f"[bold red]Connection Error:[/] Could not connect to DRIGS API at '{url}': {e}")
        raise typer.Exit(code=1)


@app.command("jobs")
def list_jobs_cmd(
    status: Optional[str] = typer.Option(None, "--status", "-s", help="Filter by job status"),
    url: str = typer.Option(DEFAULT_SERVER_URL, "--url", "-u", help="DRIGS Control Plane API URL"),
):
    """List all submitted jobs and their status."""
    try:
        params = {"status": status.upper()} if status else {}
        with _get_client(url) as client:
            res = client.get("/v1/jobs", params=params)
            if res.status_code != 200:
                console.print(f"[bold red]Failed to fetch jobs ({res.status_code}):[/] {res.text}")
                raise typer.Exit(code=1)

            jobs = res.json()
            if not jobs:
                console.print("[dim]No jobs found.[/]")
                return

            table = Table(title="DRIGS Jobs", show_header=True, header_style="bold magenta")
            table.add_column("Job ID", style="cyan")
            table.add_column("Name", style="bold")
            table.add_column("Status", style="yellow")
            table.add_column("Priority", justify="right")
            table.add_column("Worker Node", style="blue")
            table.add_column("Submitted At")

            for j in jobs:
                alloc = j.get("allocation") or {}
                worker_id = alloc.get("worker_id", "-")
                table.add_row(
                    j.get("job_id", "-"),
                    j.get("name", "-"),
                    j.get("status", "-"),
                    str(j.get("priority", 0)),
                    worker_id,
                    str(j.get("submitted_at", "-"))[:19],
                )
            console.print(table)
    except httpx.RequestError as e:
        console.print(f"[bold red]Connection Error:[/] Could not connect to DRIGS API at '{url}': {e}")
        raise typer.Exit(code=1)


@app.command("workers")
def list_workers_cmd(
    url: str = typer.Option(DEFAULT_SERVER_URL, "--url", "-u", help="DRIGS Control Plane API URL"),
):
    """List all registered worker nodes in the DRIGS cluster."""
    try:
        with _get_client(url) as client:
            res = client.get("/v1/workers")
            if res.status_code != 200:
                console.print(f"[bold red]Failed to fetch workers ({res.status_code}):[/] {res.text}")
                raise typer.Exit(code=1)

            workers = res.json()
            if not workers:
                console.print("[dim]No workers registered.[/]")
                return

            table = Table(title="DRIGS Workers", show_header=True, header_style="bold magenta")
            table.add_column("Worker ID", style="cyan")
            table.add_column("Hostname", style="bold")
            table.add_column("IP Address")
            table.add_column("Status", style="green")
            table.add_column("CPUs", justify="right")
            table.add_column("GPUs", justify="right")

            for w in workers:
                devices = w.get("devices", [])
                gpus_count = sum(1 for d in devices if str(d.get("device_type", "")).upper() in ("GPU", "DEVICETYPE.GPU"))
                table.add_row(
                    w.get("worker_id", "-"),
                    w.get("hostname", "-"),
                    w.get("ip_address", "-"),
                    w.get("status", "-"),
                    str(w.get("total_cpus", "-")),
                    str(gpus_count),
                )
            console.print(table)
    except httpx.RequestError as e:
        console.print(f"[bold red]Connection Error:[/] Could not connect to DRIGS API at '{url}': {e}")
        raise typer.Exit(code=1)


@app.command("gpus")
def list_gpus_cmd(
    url: str = typer.Option(DEFAULT_SERVER_URL, "--url", "-u", help="DRIGS Control Plane API URL"),
):
    """List all compute devices (GPUs) available in the cluster."""
    try:
        with _get_client(url) as client:
            res = client.get("/v1/gpus")
            if res.status_code != 200:
                console.print(f"[bold red]Failed to fetch GPUs ({res.status_code}):[/] {res.text}")
                raise typer.Exit(code=1)

            gpus = res.json()
            if not gpus:
                console.print("[dim]No GPUs found.[/]")
                return

            table = Table(title="DRIGS GPU Devices", show_header=True, header_style="bold magenta")
            table.add_column("Device ID", style="cyan")
            table.add_column("Model Name", style="bold")
            table.add_column("VRAM (Total)", justify="right")
            table.add_column("State", style="green")

            for g in gpus:
                mem_gb = round(g.get("total_memory_bytes", 0) / (1024**3), 2)
                table.add_row(
                    g.get("device_id", "-"),
                    g.get("model_name", "-"),
                    f"{mem_gb} GB",
                    g.get("state", "-"),
                )
            console.print(table)
    except httpx.RequestError as e:
        console.print(f"[bold red]Connection Error:[/] Could not connect to DRIGS API at '{url}': {e}")
        raise typer.Exit(code=1)


@app.command("inspect")
def inspect_job_cmd(
    job_id: str = typer.Argument(..., help="ID of job to inspect"),
    url: str = typer.Option(DEFAULT_SERVER_URL, "--url", "-u", help="DRIGS Control Plane API URL"),
):
    """Inspect detailed specification and status of a specific job."""
    try:
        with _get_client(url) as client:
            res = client.get(f"/v1/jobs/{job_id}")
            if res.status_code == 404:
                console.print(f"[bold red]Error:[/] Job '{job_id}' not found.")
                raise typer.Exit(code=1)
            elif res.status_code != 200:
                console.print(f"[bold red]Failed to inspect job ({res.status_code}):[/] {res.text}")
                raise typer.Exit(code=1)

            job_data = res.json()
            pretty_json = json.dumps(job_data, indent=2)
            console.print(Panel(pretty_json, title=f"Job Details: {job_id}", border_style="cyan"))
    except httpx.RequestError as e:
        console.print(f"[bold red]Connection Error:[/] Could not connect to DRIGS API at '{url}': {e}")
        raise typer.Exit(code=1)


@app.command("cancel")
def cancel_job_cmd(
    job_id: str = typer.Argument(..., help="ID of job to cancel"),
    url: str = typer.Option(DEFAULT_SERVER_URL, "--url", "-u", help="DRIGS Control Plane API URL"),
):
    """Cancel a queued or running job in the cluster."""
    try:
        with _get_client(url) as client:
            res = client.post(f"/v1/jobs/{job_id}/cancel")
            if res.status_code != 200:
                console.print(f"[bold red]Failed to cancel job '{job_id}' ({res.status_code}):[/] {res.text}")
                raise typer.Exit(code=1)

            data = res.json()
            console.print(f"[bold green]Job '{job_id}' cancelled successfully.[/]")
    except httpx.RequestError as e:
        console.print(f"[bold red]Connection Error:[/] Could not connect to DRIGS API at '{url}': {e}")
        raise typer.Exit(code=1)


@app.command("logs")
def logs_cmd(
    job_id: str = typer.Argument(..., help="ID of job to retrieve logs for"),
    lines: Optional[int] = typer.Option(None, "--lines", "-l", help="Number of recent log lines to print"),
    url: str = typer.Option(DEFAULT_SERVER_URL, "--url", "-u", help="DRIGS Control Plane API URL"),
):
    """View log outputs for a specific job."""
    try:
        with _get_client(url) as client:
            res = client.get(f"/v1/jobs/{job_id}")
            if res.status_code != 200:
                console.print(f"[bold red]Error fetching job '{job_id}':[/] {res.text}")
                raise typer.Exit(code=1)

            job = res.json()
            metadata = job.get("metadata", {})
            log_file = metadata.get("log_file")
            if log_file and Path(log_file).exists():
                text = Path(log_file).read_text(encoding="utf-8", errors="replace")
                if lines:
                    text_lines = text.splitlines()[-lines:]
                    text = "\n".join(text_lines)
                console.print(text)
            else:
                console.print(f"[dim]No log output available for job '{job_id}'.[/]")
    except httpx.RequestError as e:
        console.print(f"[bold red]Connection Error:[/] Could not connect to DRIGS API at '{url}': {e}")
        raise typer.Exit(code=1)


@app.command("diagnose")
def diagnose_cmd(
    job_id: str = typer.Argument(..., help="ID of job to diagnose"),
    url: str = typer.Option(DEFAULT_SERVER_URL, "--url", "-u", help="DRIGS Control Plane API URL"),
):
    """Diagnose job execution bottlenecks, memory pressure, and status."""
    try:
        with _get_client(url) as client:
            res = client.get(f"/v1/jobs/{job_id}")
            if res.status_code != 200:
                console.print(f"[bold red]Error fetching job '{job_id}':[/] {res.text}")
                raise typer.Exit(code=1)

            job = res.json()
            status = job.get("status", "UNKNOWN")
            req = job.get("spec", {}).get("resources", {})
            alloc = job.get("allocation", {})

            console.print(f"[bold cyan]Diagnostic Report for Job:[/] {job_id}")
            console.print(f"Status: [yellow]{status}[/]")
            console.print(f"Requested Resources: CPUs={req.get('cpus')}, GPUs={req.get('gpus')}, VRAM={req.get('gpu_memory_bytes')} bytes")
            if alloc:
                console.print(f"Assigned Worker: [green]{alloc.get('worker_id')}[/]")
                console.print(f"Assigned Device IDs: {alloc.get('assigned_device_ids')}")
            else:
                console.print("Assigned Worker: [red]None (Pending / Unallocated)[/]")

            if status == "FAILED":
                console.print(f"[bold red]Failure Error:[/] {job.get('error_message', 'No error message recorded')}")
            elif status == "QUEUED":
                console.print("[bold yellow]Recommendation:[/] Workload is waiting for sufficient GPU/CPU resource availability.")
            else:
                console.print("[bold green]Health Check:[/] No critical bottlenecks detected.")
    except httpx.RequestError as e:
        console.print(f"[bold red]Connection Error:[/] Could not connect to DRIGS API at '{url}': {e}")
        raise typer.Exit(code=1)


def main():
    app()


if __name__ == "__main__":
    main()
