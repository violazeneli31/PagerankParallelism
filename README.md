# PageRank with Parallel Programming

This project computes **PageRank**, the algorithm Google first used to rank web pages, and speeds it up by running parts of it at the same time on several CPU cores. It compares a normal one-thread version with two parallel versions and measures how much faster they really are.

Final project for **CEN 330 – Parallel Programming**, Epoka University (May 2026).

**Live demo:** https://violazeneli31.github.io/PagerankParallelism/

## What is PageRank?

A page is important if important pages link to it.

Every page starts with the same score. Then, round after round, each page shares its score with the pages it links to. After enough rounds the scores stop changing, and those are the final ranks.

```
PR(page) = (1 − d) / N  +  d × sum of ( PR(q) / links(q) )   for every page q that links to the page
```

- `N` is the number of pages.
- `d = 0.85`: a visitor clicks a link 85% of the time and jumps to a random page 15% of the time.

## The two parallel versions

**Data parallelism: same job, data split up.**
The pages are divided into groups, one group per thread. Every thread does the same calculation on its own group, and the threads wait for each other before starting the next round.

**Task parallelism: different jobs at the same time.**
Three separate jobs on the same graph run at once:
- PageRank
- HITS, another ranking method that scores pages as "hubs" and "authorities"
- graph statistics, such as the number of links and the most-linked page

## Results

Measured on a 4-core Intel MacBook Air. Each number is the median of 7 runs.

| Test | Result |
|---|---|
| All versions give the same scores as the NetworkX library | ✅ |
| Data parallelism, 4,000 pages, 4 threads | **1.43× faster** |
| Task parallelism, 3 jobs at once | **1.51× faster** |
| Data parallelism, small 15-page site, 4 threads | **80× slower** |

**What we learned**
- **Small problems don't benefit.** On 15 pages, starting and syncing threads takes much longer than the calculation itself. The speedup only shows up from about 2,000 pages.
- **Memory is the limit.** Each round reads the whole page-link table from memory. The cores end up waiting for memory, so the speedup stops at about 1.4×, and more threads don't help.
- **The comparison is fair.** The one-thread version is also optimised (it uses NumPy instead of slow Python loops), so the speedup comes from parallelism only.

## Project structure

| Folder / file | What's inside | How to use it |
|---|---|---|
| `pagerank/` | The main code. `core.py` has the one-thread and data-parallel versions, `tasks.py` has the task-parallel version, and `graphs.py` has the test websites. | Import it in Python (example below). |
| `notebooks/` | `pagerank_parallelism.ipynb`, a step-by-step walkthrough with explanations and results. | Open it in Jupyter or VS Code and run the cells from top to bottom. |
| `docs/` | `index.html`, an animated demo that races 1 worker against 4 workers. | Open https://violazeneli31.github.io/PagerankParallelism/ or double-click the file. Nothing to install. |
| `benchmarks/` | `run_benchmarks.py`, which measures the speed of every version. | `python benchmarks/run_benchmarks.py` (takes about 1 minute) |
| `results/` | `benchmarks.json`, the saved measurements. | Created by the benchmark script. Read by the notebook and the charts script. |
| `scripts/` | `make_figures.py`, which draws the charts. | `python scripts/make_figures.py` |
| `images/` | The charts used in the notebook. | Created by the charts script. |
| `tests/` | Checks that every version gives correct and identical results. | `pytest` |
| `report/` | The written project report (PDF and Word). | Open the PDF. It shows the numbers from the original submission. |
| `requirements.txt` | The Python libraries the project needs. | `pip install -r requirements.txt` |

## How to run it

1. Install Python 3.10 or newer.
2. Open a terminal in the project folder and install the libraries:
   ```
   pip install -r requirements.txt
   ```
3. Run whichever part you want:
   ```
   pytest                                   # check everything works
   python benchmarks/run_benchmarks.py      # measure the speed on your computer
   python scripts/make_figures.py           # redraw the charts
   jupyter notebook notebooks/pagerank_parallelism.ipynb
   ```

Using the code in your own Python file:

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
