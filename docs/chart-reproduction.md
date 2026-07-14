---
title: Chart data reproduction
layout: default
permalink: /chart-reproduction
---

The published charts are derived from a full simulation CSV. Generate that CSV
with the same settings used by the charts:

```bash
uv run python scripts/generate_published_results.py --elections 15000 \
  --output artifacts/published-results
```

The command prints the generated filename. To explore that data with the
historical R analysis, install `data.table` and `scatterD3`, then pass the CSV
path explicitly:

```bash
Rscript vseCheck.R artifacts/published-results1.csv
```

The generated CSV is intentionally not committed. It records the model,
methods, seed, and iteration count in its first line.
