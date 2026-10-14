# PageRank with Parallel Programming

PageRank (the algorithm Google used to rank web pages) written in Python, made faster by running work on several CPU cores at the same time.

Final project for **CEN 330 – Parallel Programming**, Epoka University (May 2026).

**Live demo:** https://violazeneli31.github.io/PagerankParallelism/

## How PageRank works

A page is important if important pages link to it. Every page starts with the same score, then keeps sharing its score with the pages it links to until the scores stop changing.

- Damping factor `0.85`: a visitor clicks a link 85% of the time and jumps to a random page 15% of the time.

## Three versions

| Version | How it works |
|---|---|
| **Sequential** | One thread does everything. This is the baseline. |
| **Data parallelism** | The pages are split into groups, one per thread. Every thread does the same calculation on its own group. |
| **Task parallelism** | Three different jobs run at the same time: PageRank, HITS (another ranking method), and graph statistics. |

## Results

Measured on a 4-core Intel MacBook Air. Each time is the median of 7 runs.

| Test | Result |
|---|---|
| Same scores as the NetworkX library | ✅ all versions |
| Data parallelism, 4,000 pages, 4 threads | **1.43× faster** |
| Task parallelism, 3 jobs | **1.51× faster** |
| Data parallelism, 15 pages, 4 threads | **80× slower** |

- **Small graphs get slower.** Managing threads costs more than the actual work. The speedup starts at about 2,000 pages.
- **Memory limits the speedup.** The cores spend their time waiting for data from memory, so the speedup stops at about 1.4×.
- **The test is fair.** The sequential version also uses fast NumPy code, so the speedup comes only from parallelism.

## Project structure

| Folder | What's inside | How to use it |
|---|---|---|
| `pagerank/` | The main code | Import it in Python (see below) |
| `notebooks/` | A step-by-step walkthrough | Open in Jupyter or VS Code and run all cells |
| `docs/` | The animated demo (`index.html`) | Open the live demo link, or double-click the file |
| `benchmarks/` | The speed test | `python benchmarks/run_benchmarks.py` |
| `results/` | The saved speed results | Created by the speed test |
| `scripts/` | The chart maker | `python scripts/make_figures.py` |
| `images/` | Charts used in the notebook | Created by the chart maker |
| `tests/` | Correctness checks | `pytest` |
| `report/` | The project report (PDF and Word) | Open the PDF |

## How to run

```
pip install -r requirements.txt
pytest
python benchmarks/run_benchmarks.py
```

Use it in your own code:

```python
from pagerank import ECOMMERCE_SITE, build_transition_matrix, pagerank_data_parallel

M, pages = build_transition_matrix(ECOMMERCE_SITE)
ranks, rounds = pagerank_data_parallel(M, n_workers=4)
```

## Built with

Python · NumPy · NetworkX · Matplotlib · Jupyter · HTML/JavaScript

## Team

Viola Zeneli · Sidrit Rrustemi · Arsildo Veliu · Megi Almadhi · Artemisa Hasalami

**Lecturer:** Shkumbin Fida
