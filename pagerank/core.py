"""PageRank: sequential, vectorised and data-parallel implementations.

All functions take a column-stochastic transition matrix ``M`` where
``M[i, j] = 1 / outdegree(j)`` if page ``j`` links to page ``i``.
Columns of dangling pages (no outgoing links) are all zero; their rank is
redistributed uniformly, as in the original PageRank paper.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Mapping, Sequence

import numpy as np

DAMPING = 0.85
MAX_ITER = 100
TOL = 1e-10


def build_transition_matrix(
    graph: Mapping[str, Sequence[str]],
) -> tuple[np.ndarray, list[str]]:
    """Turn an adjacency list ``{page: [linked pages]}`` into a transition matrix.

    Pages that only appear as link targets are added as nodes too.
    Returns ``(M, nodes)`` where ``nodes[i]`` is the page for row/column ``i``.
    """
    nodes = list(dict.fromkeys([*graph, *(d for ds in graph.values() for d in ds)]))
    index = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    M = np.zeros((n, n))
    for src, dests in graph.items():
        dests = list(dict.fromkeys(dests))  # ignore duplicate links
        for dest in dests:
            M[index[dest], index[src]] = 1.0 / len(dests)
    return M, nodes


def _dangling_mask(M: np.ndarray) -> np.ndarray:
    return M.sum(axis=0) == 0


def pagerank_naive(
    M: np.ndarray, damping: float = DAMPING, max_iter: int = MAX_ITER, tol: float = TOL
) -> tuple[np.ndarray, int]:
    """Reference implementation: updates one node at a time in a Python loop.

    Kept to show how much vectorisation alone buys, separately from parallelism.
    """
    n = M.shape[0]
    dangling = _dangling_mask(M)
    ranks = np.full(n, 1.0 / n)
    for it in range(1, max_iter + 1):
        leak = damping * ranks[dangling].sum() / n
        new = np.empty(n)
        for i in range(n):
            new[i] = (1 - damping) / n + leak + damping * np.dot(M[i, :], ranks)
        converged = np.abs(new - ranks).sum() < tol
        ranks = new
        if converged:
            break
    return ranks, it


def pagerank_sequential(
    M: np.ndarray, damping: float = DAMPING, max_iter: int = MAX_ITER, tol: float = TOL
) -> tuple[np.ndarray, int]:
    """Vectorised power iteration on a single thread. This is the fair baseline."""
    n = M.shape[0]
    dangling = _dangling_mask(M)
    ranks = np.full(n, 1.0 / n)
    for it in range(1, max_iter + 1):
        leak = damping * ranks[dangling].sum() / n
        new = (1 - damping) / n + leak + damping * (M @ ranks)
        converged = np.abs(new - ranks).sum() < tol
        ranks = new
        if converged:
            break
    return ranks, it


def pagerank_data_parallel(
    M: np.ndarray,
    n_workers: int = 4,
    damping: float = DAMPING,
    max_iter: int = MAX_ITER,
    tol: float = TOL,
) -> tuple[np.ndarray, int]:
    """Data parallelism: each worker updates a contiguous block of rows.

    NumPy releases the GIL inside the matrix-vector product, so the threads run
    truly in parallel. Rows are split into contiguous slices (views, not copies)
    and the thread pool is created once and reused for every iteration.
    """
    n = M.shape[0]
    dangling = _dangling_mask(M)
    bounds = np.linspace(0, n, min(n_workers, n) + 1, dtype=int)
    blocks = [(lo, hi) for lo, hi in zip(bounds[:-1], bounds[1:]) if hi > lo]
    ranks = np.full(n, 1.0 / n)
    new = np.empty(n)

    def update(block: tuple[int, int], ranks: np.ndarray, base: float) -> None:
        lo, hi = block
        np.dot(M[lo:hi], ranks, out=new[lo:hi])
        new[lo:hi] *= damping
        new[lo:hi] += base

    with ThreadPoolExecutor(max_workers=len(blocks)) as pool:
        for it in range(1, max_iter + 1):
            base = (1 - damping) / n + damping * ranks[dangling].sum() / n
            # list() waits for every block, i.e. a barrier between iterations
            list(pool.map(update, blocks, [ranks] * len(blocks), [base] * len(blocks)))
            converged = np.abs(new - ranks).sum() < tol
            ranks, new = new, ranks
            if converged:
                break
    return ranks, it
