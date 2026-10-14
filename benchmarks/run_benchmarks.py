"""Benchmark sequential vs data-parallel vs task-parallel PageRank.

Usage:  python benchmarks/run_benchmarks.py [--quick]
Writes results/benchmarks.json, which scripts/make_figures.py turns into charts.
"""

import os

# Pin NumPy's BLAS to one thread *before* importing NumPy, so that our own
# threads are the only source of parallelism (otherwise BLAS threads and
# Python threads fight over the same cores and the comparison is meaningless).
for var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from statistics import median

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pagerank import (  # noqa: E402
    ECOMMERCE_SITE,
    build_transition_matrix,
    pagerank_data_parallel,
    pagerank_naive,
    pagerank_sequential,
    synthetic_web_matrix,
)
from pagerank.tasks import run_tasks_parallel, run_tasks_sequential  # noqa: E402

ITERS = 50  # fixed amount of work per run (tol=0 disables early stopping)


def timeit(fn, *args, repeats=7, **kwargs):
    fn(*args, **kwargs)  # warm-up
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn(*args, **kwargs)
        times.append(time.perf_counter() - start)
    return median(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="smaller graphs, fewer repeats")
    args = parser.parse_args()
    sizes = [500, 1000, 2000] if args.quick else [1000, 2000, 4000]
    workers = [1, 2, 4, 8]
    repeats = 3 if args.quick else 7
    fixed = dict(max_iter=ITERS, tol=0.0)

    results = {
        "machine": {
            "cpu_count": os.cpu_count(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "iterations": ITERS,
    }

    # 1) Vectorisation: Python loop vs NumPy, both single-threaded.
    n = sizes[0]
    M = synthetic_web_matrix(n)
    t_naive = timeit(pagerank_naive, M, repeats=3, **fixed)
    t_vec = timeit(pagerank_sequential, M, repeats=repeats, **fixed)
    results["vectorisation"] = {"n": n, "naive_s": t_naive, "vectorised_s": t_vec}
    print(f"[vectorisation] n={n}: loop {t_naive:.3f}s, NumPy {t_vec:.4f}s "
          f"-> {t_naive / t_vec:.0f}x")

    # 2) Data parallelism: strong scaling over graph sizes and worker counts.
    results["data_parallel"] = []
    for n in sizes:
        M = synthetic_web_matrix(n)
        base = timeit(pagerank_sequential, M, repeats=repeats, **fixed)
        row = {"n": n, "sequential_s": base, "workers": {}}
        for w in workers:
            t = timeit(pagerank_data_parallel, M, n_workers=w, repeats=repeats, **fixed)
            row["workers"][w] = t
            print(f"[data parallel] n={n:5d} workers={w}: {t:.4f}s "
                  f"speedup {base / t:.2f}x")
        results["data_parallel"].append(row)

    # 3) Tiny graph: parallel overhead dominates.
    M_small, _ = build_transition_matrix(ECOMMERCE_SITE)
    small_seq = timeit(pagerank_sequential, M_small, repeats=repeats, **fixed)
    small_par = timeit(pagerank_data_parallel, M_small, n_workers=4, repeats=repeats, **fixed)
    results["small_graph"] = {"n": 15, "sequential_s": small_seq, "parallel_4_s": small_par}
    print(f"[15-page site] sequential {small_seq * 1e3:.2f}ms, "
          f"4 workers {small_par * 1e3:.2f}ms")

    # 4) Task parallelism: PageRank + HITS + graph statistics.
    M = synthetic_web_matrix(sizes[-1])
    seq = median(run_tasks_sequential(M)[0] for _ in range(repeats))
    par = median(run_tasks_parallel(M)[0] for _ in range(repeats))
    results["task_parallel"] = {"n": sizes[-1], "sequential_s": seq, "parallel_s": par}
    print(f"[task parallel] n={sizes[-1]}: {seq:.3f}s -> {par:.3f}s "
          f"speedup {seq / par:.2f}x")

    out = Path(__file__).resolve().parents[1] / "results" / "benchmarks.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
