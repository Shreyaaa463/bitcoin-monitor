"""
Optional scalability benchmark: generate synthetic datasets of increasing size and run
the full pipeline on each, recording wall time, per-stage time and peak memory, to see
where this single-machine implementation becomes slow or runs out of memory.

    python scripts/benchmark_scalability.py                       # 10K, 100K
    python scripts/benchmark_scalability.py --sizes 10000 100000 1000000
    python scripts/benchmark_scalability.py --sizes 50000 --timeout 1800

Everything is written under benchmarks/ (datasets, isolated runs directory, ingest
cache, results.json, results.md). Your TRACE-X dataset, runs/ and data/ are never
touched. Each size runs in its own subprocess so peak memory is measured per size.

This measures THIS machine at THESE sizes. It does not demonstrate billion- or
trillion-record capability, and the synthetic data (uniformly random IPs/wallets) is
not representative of every real-world distribution.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def peak_rss_mb() -> float | None:
    """Peak resident memory of THIS process in MB (Linux/macOS via resource, Windows via psapi)."""
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / 1024.0 if sys.platform != "darwin" else peak / (1024.0 * 1024.0)
    except ImportError:
        pass
    try:  # Windows: psutil ships with streamlit's dependencies
        import psutil
        return psutil.Process().memory_info().peak_wset / (1024.0 * 1024.0)
    except Exception:
        pass
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        if not k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            return None
        return pmc.PeakWorkingSetSize / (1024.0 * 1024.0)
    except Exception:
        return None


def child(dataset: str, config: str):
    """Run one pipeline in this (fresh) process and print a JSON result line."""
    from src.pipeline import run_pipeline
    t0 = time.time()
    res = run_pipeline(dataset, config, force=True)
    m = res["manifest"]
    print("BENCH_RESULT " + json.dumps({
        "wall_seconds": round(time.time() - t0, 2),
        "peak_rss_mb": round(peak_rss_mb() or -1, 1),
        "stages": {k: v.get("seconds") for k, v in m["stages"].items()},
        "record_counts": m.get("record_counts"),
        "funnel": m.get("funnel"),
        "settings": {k: m["resolved_settings"].get(k) for k in
                     ("large_dataset_mode", "graph_feature_backend", "clustering_scope",
                      "singleton_min_priority")},
        "run_id": res["run_id"],
    }), flush=True)


def _write_bench_config(bench_dir: Path) -> Path:
    import yaml
    cfg = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text())
    cfg["paths"]["processed_dir"] = str(bench_dir / "processed")
    cfg["runs"]["root"] = str(bench_dir / "runs")
    p = bench_dir / "benchmark_config.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def _dir_bytes(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="+", default=[10_000, 100_000])
    ap.add_argument("--out", default=str(PROJECT_ROOT / "benchmarks"))
    ap.add_argument("--timeout", type=int, default=3 * 3600, help="seconds per size before giving up")
    ap.add_argument("--keep-runs", action="store_true", help="keep each size's run directory")
    ap.add_argument("--child", nargs=2, metavar=("DATASET", "CONFIG"), help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.child:
        child(*args.child)
        return

    bench = Path(args.out)
    (bench / "datasets").mkdir(parents=True, exist_ok=True)
    cfg = _write_bench_config(bench)
    results = []
    for n in args.sizes:
        ds = bench / "datasets" / f"synthetic_{n}.jsonl"
        if not ds.exists():
            print(f"[bench] generating {n:,} records ...", flush=True)
            subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "generate_synthetic_dataset.py"),
                            "--n", str(n), "--format", "jsonl", "--out", str(ds)], check=True)
        print(f"[bench] running pipeline on {n:,} records (timeout {args.timeout}s) ...", flush=True)
        t0 = time.time()
        entry = {"records": n, "dataset_mb": round(ds.stat().st_size / 1e6, 1)}
        try:
            proc = subprocess.run([sys.executable, __file__, "--child", str(ds), str(cfg)],
                                  capture_output=True, text=True, timeout=args.timeout, cwd=str(PROJECT_ROOT))
            line = next((l for l in proc.stdout.splitlines() if l.startswith("BENCH_RESULT ")), None)
            if proc.returncode == 0 and line:
                entry.update(json.loads(line[len("BENCH_RESULT "):]), status="completed")
                run_dir = bench / "runs" / entry["run_id"]
                entry["run_dir_mb"] = round(_dir_bytes(run_dir) / 1e6, 1)
                if not args.keep_runs:
                    import shutil
                    shutil.rmtree(run_dir, ignore_errors=True)
            else:
                tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
                entry.update(status=f"failed (exit {proc.returncode})", error=" | ".join(tail),
                             wall_seconds=round(time.time() - t0, 1))
        except subprocess.TimeoutExpired:
            entry.update(status=f"did not finish within {args.timeout}s", wall_seconds=args.timeout)
        print(f"[bench] {n:,}: {entry.get('status')} in {entry.get('wall_seconds')}s, "
              f"peak RSS {entry.get('peak_rss_mb', '?')} MB", flush=True)
        results.append(entry)

    import multiprocessing
    report = {"machine": {"platform": platform.platform(terse=True), "python": platform.python_version(),
                          "cpus": multiprocessing.cpu_count()},
              "results": results}
    (bench / "results.json").write_text(json.dumps(report, indent=2))

    stage_names = ["ingest", "clean", "correlate", "features", "anomaly_model", "anomaly_scoring",
                   "clustering", "explanations", "alerts", "cases", "forensics", "finalize"]
    lines = ["| records | status | wall s | peak RSS MB | " + " | ".join(stage_names) + " |",
             "|---" * (4 + len(stage_names)) + "|"]
    for r in results:
        st = r.get("stages") or {}
        lines.append(f"| {r['records']:,} | {r['status']} | {r.get('wall_seconds', '')} | "
                     f"{r.get('peak_rss_mb', '')} | " + " | ".join(str(st.get(s, "")) for s in stage_names) + " |")
    (bench / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"[bench] wrote {bench / 'results.json'} and {bench / 'results.md'}")


if __name__ == "__main__":
    main()
