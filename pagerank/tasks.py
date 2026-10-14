"""Task parallelism: independent analyses of the same graph run concurrently."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .core import pagerank_sequential


def task_pagerank(M: np.ndarray) -> np.ndarray:
    """PageRank scores (who is linked to by important pages)."""
    return pagerank_sequential(M)[0]


def task_hits(M: np.ndarray, max_iter: int = 100, tol: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    """HITS hub and authority scores (Kleinberg, 1999), computed by power iteration."""
    A = (M > 0).astype(float)  # A[i, j] = 1 if j links to i
    hubs = np.ones(A.shape[0])
    for _ in range(max_iter):
        auth = A @ hubs
        auth /= np.linalg.norm(auth)
        new_hubs = A.T @ auth
        new_hubs /= np.linalg.norm(new_hubs)
        if np.abs(new_hubs - hubs).sum() < tol:
            hubs = new_hubs
            break
        hubs = new_hubs
    return hubs, auth


def task_graph_stats(M: np.ndarray) -> dict[str, float]:
    """Structural statistics: degrees, density, reciprocity, dangling pages."""
    A = M > 0
    in_deg = A.sum(axis=1)
    out_deg = A.sum(axis=0)
    links = int(out_deg.sum())
    n = A.shape[0]
    return {
        "pages": n,
        "links": links,
        "avg_out_degree": links / n,
        "max_in_degree": int(in_deg.max()),
        "density": links / (n * (n - 1)),
        "reciprocity": float((A & A.T).sum() / links) if links else 0.0,
        "dangling_pages": int((out_deg == 0).sum()),
    }


TASKS = (task_pagerank, task_hits, task_graph_stats)


def run_tasks_sequential(M: np.ndarray) -> tuple[float, list]:
    start = time.perf_counter()
    results = [task(M) for task in TASKS]
    return time.perf_counter() - start, results


def run_tasks_parallel(M: np.ndarray) -> tuple[float, list]:
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(TASKS)) as pool:
        futures = [pool.submit(task, M) for task in TASKS]
        results = [f.result() for f in futures]
    return time.perf_counter() - start, results
