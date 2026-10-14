"""Test graphs: the hand-made e-commerce site and a synthetic web-like graph."""

from __future__ import annotations

import numpy as np

# A small online shop: 15 pages, 39 links.
ECOMMERCE_SITE: dict[str, list[str]] = {
    "Home": ["Products", "Search", "About", "Blog"],
    "Products": ["Product1", "Product2", "Product3", "Cart"],
    "Product1": ["Cart", "Reviews", "Related"],
    "Product2": ["Cart", "Reviews", "Related"],
    "Product3": ["Cart", "Reviews", "Related"],
    "Cart": ["Checkout", "Products"],
    "Checkout": ["Payment", "Home"],
    "Payment": ["Confirmation"],
    "Confirmation": ["Home", "Products"],
    "Search": ["Products", "Product1", "Product2", "Product3"],
    "Reviews": ["Product1", "Product2", "Product3"],
    "Related": ["Product1", "Product2", "Product3"],
    "About": ["Home", "Contact"],
    "Contact": ["Home"],
    "Blog": ["Home", "Products"],
}


def synthetic_web_matrix(
    n: int,
    avg_out_degree: int = 10,
    dangling_fraction: float = 0.05,
    zipf_exponent: float = 0.9,
    seed: int = 42,
) -> np.ndarray:
    """Build the transition matrix of a random web-like graph with ``n`` pages.

    Link targets follow a Zipf-like popularity distribution, so a few pages
    collect most links (as on the real web), and a small fraction of pages
    have no outgoing links at all (dangling pages).
    """
    rng = np.random.default_rng(seed)
    popularity = 1.0 / np.arange(1, n + 1) ** zipf_exponent
    popularity = rng.permutation(popularity / popularity.sum())

    M = np.zeros((n, n))
    out_degrees = rng.poisson(avg_out_degree, size=n).clip(1, n - 1)
    dangling = rng.random(n) < dangling_fraction
    for src in np.flatnonzero(~dangling):
        p = popularity.copy()
        p[src] = 0.0  # no self-links
        targets = rng.choice(n, size=out_degrees[src], replace=False, p=p / p.sum())
        M[targets, src] = 1.0 / len(targets)
    return M
