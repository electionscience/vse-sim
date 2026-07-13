---
title: IRV/RCV VSE simulation details
layout: default
permalink: /irv-simulation
---

This page records the simulation configuration and results used for IRV/RCV on this site.

The simulation uses 15,000 elections, the KS voter model (`dcdecay=(1,3)`, `wcdecay=(1.5,3)`, `dccut=.2`, `wcalpha=1.5`), 40 voters, 6 candidates, fuzzy media, and seed `target15000`. The reproducible command is:

```bash
uv run python scripts/recalculate_irv_pages.py
```

The runner divides the work into ten fixed, seeded chunks and executes the chunks concurrently when CPU cores are available.

| Ballot strategy            |    VSE |
| -------------------------- | -----: |
| 100% honest                | 86.58% |
| 25% strategic / 75% honest | 82.15% |
| 50% strategic / 50% honest | 72.97% |
| 75% strategic / 25% honest | 65.43% |
| 100% strategic             | 60.50% |

The embedded PNG charts on this site use these values.
