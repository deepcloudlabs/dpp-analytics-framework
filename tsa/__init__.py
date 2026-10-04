"""Textile Sustainability Analytics (TSA) research prototype.

Reference implementation accompanying the manuscript "From passport data to
sustainability intelligence". It implements, at prototype level, the data model,
indicator engine, analytical provenance graph, data-quality model, compliance
rules, machine-learning tasks and interoperability exports described in the
paper, together with a documented synthetic scenario generator used for the
experimental evaluation. All data produced by this package are SYNTHETIC.
"""

import os as _os

# Gradient boosting on small tables is dominated by OpenMP overhead with many threads.
_os.environ.setdefault("OMP_NUM_THREADS", "4")

__version__ = "0.1.0"
