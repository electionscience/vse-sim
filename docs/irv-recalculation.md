---
title: IRV/RCV VSE recalculation
layout: default
permalink: /irv-recalculation
---

The legacy IRV/RCV values on this site were recalculated after correcting two implementation defects: the simulator passed rank vectors to a tabulator that expects candidate IDs in preference order, and it evaluated IRV's finish ordering as though it were a score vector.

The recalculation uses the historical run specification recorded in `vse.py`: 15,000 elections, the KS voter model (`dcdecay=(1,3)`, `wcdecay=(1.5,3)`, `dccut=.2`, `wcalpha=1.5`), 40 voters, 6 candidates, fuzzy media, and seed `target15000`. The reproducible command is:

```
uv run python scripts/recalculate_irv_pages.py
```

| Ballot strategy | VSE |
| --- | ---: |
| 100% honest | 87.17% |
| 25% strategic / 75% honest | 82.71% |
| 50% strategic / 50% honest | 73.08% |
| 75% strategic / 25% honest | 65.30% |
| 100% strategic | 60.73% |

The embedded PNG charts have been regenerated from this corrected data. The interactive legacy charts still contain pre-correction IRV points and should not be used for IRV comparisons until they are regenerated.
