"""PageRank with structured parallelism (data parallelism and task parallelism)."""

from .core import (
    build_transition_matrix,
    pagerank_data_parallel,
    pagerank_naive,
    pagerank_sequential,
)
from .graphs import ECOMMERCE_SITE, synthetic_web_matrix

__all__ = [
    "build_transition_matrix",
    "pagerank_naive",
    "pagerank_sequential",
    "pagerank_data_parallel",
    "ECOMMERCE_SITE",
    "synthetic_web_matrix",
]
