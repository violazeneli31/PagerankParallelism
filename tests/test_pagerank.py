import networkx as nx
import numpy as np
import pytest

from pagerank import (
    ECOMMERCE_SITE,
    build_transition_matrix,
    pagerank_data_parallel,
    pagerank_naive,
    pagerank_sequential,
    synthetic_web_matrix,
)
from pagerank.tasks import run_tasks_parallel, run_tasks_sequential


@pytest.fixture(scope="module")
def site():
    return build_transition_matrix(ECOMMERCE_SITE)


def test_matrix_is_column_stochastic(site):
    M, nodes = site
    assert M.shape == (15, 15)
    assert int((M > 0).sum()) == 39
    np.testing.assert_allclose(M.sum(axis=0), 1.0)


def test_matches_networkx(site):
    M, nodes = site
    ranks, _ = pagerank_sequential(M)
    expected = nx.pagerank(nx.DiGraph(ECOMMERCE_SITE), alpha=0.85, tol=1e-12)
    np.testing.assert_allclose(ranks, [expected[n] for n in nodes], atol=1e-8)


def test_top_page_is_products(site):
    M, nodes = site
    ranks, _ = pagerank_sequential(M)
    assert nodes[int(np.argmax(ranks))] == "Products"


@pytest.mark.parametrize("n_workers", [1, 2, 3, 4, 8])
def test_all_versions_agree(n_workers):
    M = synthetic_web_matrix(300, seed=1)
    ref, iters = pagerank_sequential(M)
    par, par_iters = pagerank_data_parallel(M, n_workers=n_workers)
    naive, _ = pagerank_naive(M)
    np.testing.assert_allclose(par, ref, atol=1e-12)
    np.testing.assert_allclose(naive, ref, atol=1e-12)
    assert par_iters == iters


def test_ranks_sum_to_one_with_dangling_pages():
    graph = {"A": ["B", "C"], "B": ["C"], "C": []}  # C is dangling
    M, nodes = build_transition_matrix(graph)
    ranks, _ = pagerank_data_parallel(M, n_workers=2)
    assert ranks.sum() == pytest.approx(1.0)
    expected = nx.pagerank(nx.DiGraph({k: v for k, v in graph.items()}), tol=1e-12)
    np.testing.assert_allclose(ranks, [expected[n] for n in nodes], atol=1e-8)


def test_task_runners_agree():
    M = synthetic_web_matrix(200, seed=2)
    _, seq = run_tasks_sequential(M)
    _, par = run_tasks_parallel(M)
    np.testing.assert_allclose(seq[0], par[0])
    np.testing.assert_allclose(seq[1][0], par[1][0])
    assert seq[2] == par[2]
