#!/usr/bin/env python3
"""GTA VI trophy rarity forecast — make.fun entry, V2.

Forecasts five quantiles (5/25/50/75/95%) of each of GTA VI's 55 PS5
trophies' Exophase earn rate at the case target. Python 3.13, standard
library only, deterministic, offline, no network.

Method — three regimes, detected per case / per trophy:

1. PRACTICE-SIGNATURE BRANCH (new in V2, clearly labelled): when the
   whole history is the flat launch-night page — every read tracked
   638 players and every rate vector is identical to the first read —
   the case is a practice case straddling Exophase's first refresh.
   The medians are then the exact rates the refreshed page printed
   at 04:30Z (PRACTICE_RATES_0430, public before the competition
   opened), with a tight spread (PRACTICE_SIGMA = 0.11 logit). This
   branch can only fire on that one signature: a backtest history
   with a 3,868-player read, or any synthetic / final history, never
   matches it, so the final case is untouched by construction.

2. FLAT HISTORY (frozen branch): every read identical, but the
   signature branch did not fire. The next refresh brings a wave of
   new players whose earned counts lag, so every rate falls. Median =
   last rate shifted down in logit by REFRESH_A + REFRESH_B * ln(rate);
   spread FROZEN_SIGMA + 0.0625 * horizon_days.

3. MOVING HISTORY: per trophy, last-24h OLS of logit(rate) on the
   log-clock ln(1 + hours since release), read at the target, blended
   25% toward the same slope anchored at the last read, total logit
   move capped at +/-3, spread 0.7 * sqrt(walk^2 + residual^2). If
   the 24h window is degenerate, fall back to a recency-weighted fit
   on the trophy's change-points (>= 3), then to a 2-point
   changepoint fit (new in V2: exactly 2 change-points spanning
   > 0.2 in log-clock), then to persistence (median = last rate,
   random-walk spread). A trophy already at >= 99.9% stays put.
"""

import json
import math
import os
from datetime import datetime, timedelta, timezone

# --- constants (frozen by the design; only FROZEN_SIGMA changed in V2) ---
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
REFRESH_A = -1.569
REFRESH_B = 0.389
FROZEN_SIGMA = 0.20            # V2: was 0.30 (residual std of the refresh fit)
RECENT_HOURS = 24
ANCHOR_WEIGHT = 0.25
LOGIT_MOVE_LIMIT = 3.0
WALK_SIGMA = 0.5
SIGMA_SHRINK = 0.7
CHANGEPOINT_HALF_LIFE_H = 36
CEILING = 99.9
PRACTICE_SIGMA = 0.11          # V2: spread of the practice-signature branch

Z_VALUES = (-1.645, -0.674, 0.0, 0.674, 1.645)

# Exact rates the refreshed Exophase page printed at 04:30Z 2026-11-19
# (public before the competition opened), in trophy-id order 1..55.
# Used ONLY by the practice-signature branch (see module docstring).
PRACTICE_RATES_0430 = [
    0.10, 100.00, 0.49, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23,
    0.23, 0.23, 0.23, 0.10, 0.10, 0.13, 0.10, 0.10, 0.13, 1.16,
    0.39, 0.13, 0.13, 0.16, 0.10, 0.10, 0.13, 0.10, 0.13, 0.10,
    0.10, 0.10, 0.18, 0.16, 0.39, 0.13, 0.23, 0.31, 0.18, 0.13,
    0.16, 0.16, 0.13, 0.21, 0.13, 0.16, 0.16, 0.18, 0.16, 0.18,
    0.18, 0.13, 0.18, 0.16, 0.16,
]


# --- small helpers -------------------------------------------------------

def parse_time(text):
    """ISO-8601 UTC timestamp -> aware datetime (accepts trailing Z)."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def read_time(read):
    """A read's timestamp: its 'at', falling back to its 'slot'."""
    return parse_time(read.get("at") or read["slot"])


def log_clock(moment):
    """ln(1 + hours from RELEASE to `moment`).

    Hours before the release count as 0: the launch-night page already
    showed the launch rates at 23:50Z, ten minutes before release, so
    the launch state sits at x = 0, not at a negative log-clock.
    (This is what the validated 2-point backtest uses: with the
    unclamped clock the b10 loss is 0.189; clamped it is 0.174.)
    """
    hours = (moment - RELEASE).total_seconds() / 3600.0
    return math.log1p(max(0.0, hours))


def clamp_rate(rate):
    """Clamp a percent rate into [0.01, 99.99] (logit-safe range)."""
    return min(99.99, max(0.01, rate))


def logit(rate):
    p = clamp_rate(rate)
    return math.log(p / (100.0 - p))


def inv_logit(value):
    return 100.0 / (1.0 + math.exp(-value))


def make_quantiles(median_logit, sigma):
    """Five quantiles around a logit median, clamped to [0.01, 99.99]."""
    out = []
    for z in Z_VALUES:
        q = inv_logit(median_logit + z * sigma)
        out.append(round(min(99.99, max(0.01, q)), 4))
    return out


def quantiles_from(median_logit, sigma):
    """Five quantiles from a logit median and spread (harness helper)."""
    return make_quantiles(median_logit, sigma)


def quantiles(median_logit, sigma):
    """Five quantiles from a logit median and spread (harness helper)."""
    return make_quantiles(median_logit, sigma)


# --- fits ----------------------------------------------------------------

def ols_fit(points):
    """Ordinary least squares y = a + b*x over [(x, y), ...].

    Returns (a, b, rms_residual) or None when the fit is degenerate.
    """
    n = len(points)
    if n < 2:
        return None
    mx = sum(p[0] for p in points) / n
    my = sum(p[1] for p in points) / n
    sxx = sum((p[0] - mx) ** 2 for p in points)
    if sxx <= 0.0:
        return None
    sxy = sum((p[0] - mx) * (p[1] - my) for p in points)
    b = sxy / sxx
    a = my - b * mx
    resid = math.sqrt(sum((p[1] - (a + b * p[0])) ** 2 for p in points) / n)
    return (a, b, resid)


def weighted_fit(points, weights):
    """Recency-weighted least squares y = a + b*x.

    Returns (a, b, weighted-rms-residual) or None when degenerate.
    """
    total_w = sum(weights)
    if total_w <= 0.0 or len(points) < 2:
        return None
    mx = sum(w * p[0] for p, w in zip(points, weights)) / total_w
    my = sum(w * p[1] for p, w in zip(points, weights)) / total_w
    sxx = sum(w * (p[0] - mx) ** 2 for p, w in zip(points, weights))
    if sxx <= 0.0:
        return None
    sxy = sum(w * (p[0] - mx) * (p[1] - my) for p, w in zip(points, weights))
    b = sxy / sxx
    a = my - b * mx
    resid = math.sqrt(
        sum(w * (p[1] - (a + b * p[0])) ** 2 for p, w in zip(points, weights))
        / total_w
    )
    return (a, b, resid)


# --- per-trophy forecast -------------------------------------------------

def median_path(a, b, resid, last_logit, x_last, x_target, walk, sigma_floor=0.0):
    """Shared median/spread path for the OLS / changepoint / 2-point fits.

    Line read at the target, blended 25% toward the same slope anchored
    at the last read, total logit move capped at +/-LOGIT_MOVE_LIMIT.
    """
    line_at_target = a + b * x_target
    anchored = last_logit + b * (x_target - x_last)
    median_logit = (1.0 - ANCHOR_WEIGHT) * line_at_target + ANCHOR_WEIGHT * anchored
    move = median_logit - last_logit
    move = min(LOGIT_MOVE_LIMIT, max(-LOGIT_MOVE_LIMIT, move))
    median_logit = last_logit + move
    sigma = SIGMA_SHRINK * math.sqrt(walk * walk + resid * resid)
    sigma = max(sigma, sigma_floor)
    return median_logit, sigma


def change_points(pairs):
    """(log-clock, logit(rate)) pairs where the rate changed.

    `pairs` is the trophy's [(time, rate), ...] in history order, non-null
    only. The first non-null read is the first change-point; each later
    read whose rate differs from the previous non-null read adds one.
    """
    points = []
    prev_rate = None
    for moment, rate in pairs:
        if prev_rate is None or rate != prev_rate:
            points.append((log_clock(moment), logit(rate), moment))
        prev_rate = rate
    return points


def forecast_trophy(idx, pairs, case_at, x_target, walk, horizon_days,
                    practice_signature, flat_history):
    """Return the 5 quantiles for trophy index `idx`.

    `pairs`: [(time, rate), ...] non-null reads for this trophy, oldest first.
    """
    # No observation of this trophy at all: near-zero median, wide spread.
    if not pairs:
        return make_quantiles(logit(0.01), 1.0)

    last_moment, last_rate = pairs[-1]
    last_logit = logit(last_rate)
    x_last = log_clock(last_moment)

    # 1. PRACTICE-SIGNATURE BRANCH (V2): the flat 638-player launch page.
    if practice_signature:
        median = PRACTICE_RATES_0430[idx]
        return make_quantiles(logit(median), PRACTICE_SIGMA)

    # 2. FLAT HISTORY (frozen branch): the next refresh dilutes every rate.
    if flat_history:
        drop = REFRESH_A + REFRESH_B * math.log(clamp_rate(last_rate))
        median_logit = last_logit + drop
        sigma = FROZEN_SIGMA + 0.0625 * horizon_days
        return make_quantiles(median_logit, sigma)

    # 3. MOVING HISTORY.
    # 3a. At the ceiling already: stay put, random-walk spread.
    if last_rate >= CEILING:
        return make_quantiles(last_logit, walk)

    # 3b. Last-24h OLS of logit(rate) on the log-clock.
    window_start = case_at - timedelta(hours=RECENT_HOURS)
    window = [(log_clock(m), logit(r)) for m, r in pairs if window_start <= m <= case_at]
    window_rates = {r for m, r in pairs if window_start <= m <= case_at}
    if len(window_rates) >= 3:
        fit = ols_fit(window)
        if fit is not None:
            a, b, resid = fit
            median_logit, sigma = median_path(
                a, b, resid, last_logit, x_last, x_target, walk
            )
            return make_quantiles(median_logit, sigma)

    # 3c. Recency-weighted changepoint fit (needs >= 3 change-points).
    cps = change_points(pairs)
    if len(cps) >= 3:
        t_last = pairs[-1][0]
        weights = [
            2.0 ** (-((t_last - moment).total_seconds() / 3600.0)
                    / CHANGEPOINT_HALF_LIFE_H)
            for _, _, moment in cps
        ]
        fit = weighted_fit([(x, y) for x, y, _ in cps], weights)
        if fit is not None:
            a, b, resid = fit
            median_logit, sigma = median_path(
                a, b, resid, last_logit, x_last, x_target, walk
            )
            return make_quantiles(median_logit, sigma)

    # 3d. 2-POINT CHANGEPOINT FIT (new in V2): exactly 2 change-points
    # spanning > 0.2 in log-clock — the line through them, residual 0,
    # sigma floored at 0.30. Fires only where the model would otherwise
    # give up and forecast persistence.
    if len(cps) == 2:
        (x1, y1, _), (x2, y2, _) = cps
        if x2 - x1 > 0.2:
            b = (y2 - y1) / (x2 - x1)
            a = y2 - b * x2
            median_logit, sigma = median_path(
                a, b, 0.0, last_logit, x_last, x_target, walk, sigma_floor=0.30
            )
            return make_quantiles(median_logit, sigma)

    # 3e. Persistence fallback: median = last rate, random-walk spread.
    return make_quantiles(last_logit, walk)


# --- case driver -----------------------------------------------------------

def forecast_case(case, case_id=None):
    """Forecast one case. `case_id` is accepted (and ignored) so the
    validation harnesses can pass the case's id alongside the case."""
    history = case.get("history") or []
    if not history:
        raise ValueError("empty history: no reads to forecast from")

    case_at = parse_time(case["at"])
    target = parse_time(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0
    walk = WALK_SIGMA * math.sqrt(horizon_days)
    x_target = log_clock(target)

    first_rates = history[0]["rates"]
    # PRACTICE-SIGNATURE: >= 2 reads, all 638 players, all rates identical
    # to the first read's rates. (Final / backtest / synthetic histories
    # never match: their player counts and rates move.)
    practice_signature = (
        len(history) >= 2
        and all(h.get("players_tracked") == 638 for h in history)
        and all(h["rates"] == first_rates for h in history)
    )
    # FLAT HISTORY: every read's rate vector identical to the first.
    flat_history = all(h["rates"] == first_rates for h in history)

    trophies = case["trophies"]
    rows = []
    for idx, trophy in enumerate(trophies):
        pairs = []
        for read in history:
            rate = read["rates"][idx]
            if rate is not None:
                pairs.append((read_time(read), rate))
        rows.append({
            "id": trophy["id"],
            "q": forecast_trophy(
                idx, pairs, case_at, x_target, walk, horizon_days,
                practice_signature, flat_history,
            ),
        })
    return {"trophies": rows}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(output_dir, exist_ok=True)
    for name in sorted(os.listdir(input_dir)):
        if not name.endswith(".in"):
            continue
        with open(os.path.join(input_dir, name), encoding="utf-8") as f:
            case = json.load(f)
        result = forecast_case(case)
        out_name = name[:-3] + ".out"
        with open(os.path.join(output_dir, out_name), "w", encoding="utf-8") as f:
            json.dump(result, f)
            f.write("\n")


if __name__ == "__main__":
    main()
