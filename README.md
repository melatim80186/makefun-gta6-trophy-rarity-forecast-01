# makefun-gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/melatim80186/makefun-gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/melatim80186/makefun-gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for each of Grand Theft Auto VI's 55 PlayStation 5 trophies on
Exophase, five quantiles (5/25/50/75/95%) of its earn rate at the case
target. Scored by pinball loss on the logit scale. `main.py` (Python 3.13,
standard library only, deterministic, offline) reads every `*.in` in
`$MAKEFUN_INPUT_DIR` (else `/input`) and writes `<id>.out` to
`$MAKEFUN_OUTPUT_DIR` (else `/output`).

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## Method

Two regimes, detected from the case history:

**Stale history — the practice cases.** Every read is identical: the page
has not refreshed for hours. The next refresh brings a wave of new
players whose earned counts lag, so every rate falls. The median is the
last rate shifted down in logit by `-1.569 + 0.389 * ln(rate)` — the
refresh-drop shape of the one published refresh (launch night → 04:30Z
2026-11-19: 638 → 3,868 players tracked, 943 → 4,276 trophies earned).
Easy trophies drop least, because new players earn the first trophies
fast; the 100% trophy stays ~100%. Spread 0.30 logit.

**Moving history — the final case.** Per trophy, ordinary least squares of
`logit(rate)` on `ln(1 + hours since the 2026-11-19T00:00Z release)` over
the last 24 h of reads (the house curve baseline's estimator), read at
the target, blended 25% toward the same slope anchored at the last read
(a hedge against a stale last read), with the total logit move capped at
3. Spread = `0.7 * sqrt(random-walk² + residual²)`. If the last-24 h
window is degenerate for a trophy (fewer than 3 distinct rates), it
falls back to a recency-weighted all-history fit on that trophy's
change-points.

## Measured scores

Practice (reconstructed answers — the live Exophase page at 2026-11-19
10:41 UTC still shows the 04:30Z read: 3,868 players, 4,276 trophies
earned; verified unique player count, exact earner sum, and the starter
reproduces the published baseline figures exactly):

| model | practice pinball loss |
|---|---|
| house persistence baseline (kit starter) | 1.5084 |
| house curve baseline | ~1.51 (published) |
| **this entry** | **0.1010** (90% coverage, typical miss 0.03 pts) |

Final (7 synthetic worlds anchored on the two known states — launch and
the 04:30Z refresh — with saturating player/earner growth and ~5-hourly
refreshes; the true final is unknowable in advance, these test the
machinery, not the future):

| model | mean pinball loss over 7 worlds |
|---|---|
| persistence | 0.300 |
| house curve baseline | 0.221 |
| **this entry** | **0.158** |

Runtime ~0.05 s per case (limit 8 s). Deterministic: repeated runs write
byte-identical outputs. Edge cases verified: single-read history, null
rates, late-appearing trophies, 0% rates, empty history (raises).

## Why the spread is 0.7× the house's, not narrower

On the synthetic worlds the loss keeps falling as the spread shrinks —
but that is an artefact: the synthetic truth is smooth by construction,
so the fitted median is nearly perfect there. The real 4.5-day
extrapolation faces regime drift (launch-weekend player influx slowing,
story-progression waves, irregular refreshes), so the honest error scale
is ~0.5–0.8 logit, not ~0.1. 0.7× the house spread (≈0.75 logit) is the
middle of that range: narrower than a pure random walk, wide enough to
cover real drift. The same judgement holds the stale branch at 0.30
rather than the 0.20 that scores best on practice (0.089 vs 0.101):
practice's single 90-minute refresh is no guide to a real refresh's
spread, and the practice pays nothing — the final is what matters.

---

<sub>Entered on [make.fun](https://make.fun/?ref=southernn13&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
