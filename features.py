from math import sqrt

import numpy as np

# (name, start, end) - end is exclusive. From the competition's data documentation.
GROUPS = {
    "i0_bid_p": (0, 11), "i0_ask_p": (11, 22), "i0_bid_v": (22, 33), "i0_ask_v": (33, 44),
    "i0_trade_p": (44, 48), "i0_trade_v": (48, 52),
    "i1_bid_p": (52, 63), "i1_ask_p": (63, 74), "i1_bid_v": (74, 85), "i1_ask_v": (85, 96),
    "i1_trade_p": (96, 100), "i1_trade_v": (100, 104),
    "a": (104, 112),
}

# Exponentially weighted moving averages: value_new = alpha * x + (1 - alpha) * value_old
ALPHA_FAST = 0.05    # roughly the last 20 rows
ALPHA_SLOW = 0.005   # roughly the last 200 rows
A7_FLOOR = -5.0      # a7 sits at -5.199337 about two thirds of the time
EPS = 1e-6

# Group boundaries as one list, so every group statistic can be computed in a
# single NumPy call with reduceat (much faster per row than slicing each group).
_ORDER = ["i0_bid_p", "i0_ask_p", "i0_bid_v", "i0_ask_v", "i0_trade_p", "i0_trade_v",
          "i1_bid_p", "i1_ask_p", "i1_bid_v", "i1_ask_v", "i1_trade_p", "i1_trade_v", "a"]
_STARTS = np.array([GROUPS[g][0] for g in _ORDER])
_SIZES = np.array([GROUPS[g][1] - GROUPS[g][0] for g in _ORDER], dtype=np.float64)
_IX = {g: i for i, g in enumerate(_ORDER)}

STATELESS_NAMES = []
for _i in ("i0", "i1"):
    STATELESS_NAMES += [f"{_i}_best_bid", f"{_i}_best_ask", f"{_i}_spread_best",
                        f"{_i}_spread_best_sq", f"{_i}_bid_p_std", f"{_i}_ask_p_std",
                        f"{_i}_bid_p_range", f"{_i}_ask_p_range",
                        f"{_i}_bid_v_std", f"{_i}_ask_v_std", f"{_i}_bid_v_max",
                        f"{_i}_ask_v_max", f"{_i}_imbalance_ratio", f"{_i}_imbalance_sq",
                        f"{_i}_imbalance_x_spread", f"{_i}_trade_v_ratio"]
STATELESS_NAMES += ["imbalance_product", "imbalance_sign_agree",
                    "a7_at_floor", "a6_x_a7", "a2_x_a3"]

SIGNAL_NAMES = ["i0_mid", "i0_imbalance", "i1_mid", "i1_imbalance",
                "i0_spread", "i0_trade_v", "a6", "a7"]
VOL_SIGNALS = ("i0_mid", "i0_imbalance")
_VOL_IX = [SIGNAL_NAMES.index(n) for n in VOL_SIGNALS]

MEMORY_NAMES = []
for _s in SIGNAL_NAMES:
    MEMORY_NAMES += [f"{_s}_diff", f"{_s}_dev_fast", f"{_s}_dev_slow"]
MEMORY_NAMES += [f"{_s}_vol" for _s in VOL_SIGNALS]

N_RAW = 112
N_STATELESS = len(STATELESS_NAMES)
N_MEMORY = len(MEMORY_NAMES)


def _group_stats(x):
    """Sum, max, min and standard deviation of every column group, in four passes.

    reduceat applies a reduction to consecutive slices in one call, so all 13
    groups are handled at once instead of 13 separate slice operations.
    """
    sums = np.add.reduceat(x, _STARTS, axis=1)
    sumsq = np.add.reduceat(x * x, _STARTS, axis=1)
    maxs = np.maximum.reduceat(x, _STARTS, axis=1)
    mins = np.minimum.reduceat(x, _STARTS, axis=1)
    means = sums / _SIZES
    var = sumsq / _SIZES - means * means
    stds = np.sqrt(np.maximum(var, 0.0))
    return sums, maxs, mins, stds


def _core(x):
    """Stateless features and base signals, computed together from one pass."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    sums, maxs, mins, stds = _group_stats(x)
    out = np.empty((n, N_STATELESS))
    signals = np.empty((n, len(SIGNAL_NAMES)))

    col = 0
    stored = {}
    for i in ("i0", "i1"):
        bp, ap = _IX[f"{i}_bid_p"], _IX[f"{i}_ask_p"]
        bv, av = _IX[f"{i}_bid_v"], _IX[f"{i}_ask_v"]
        tv = _IX[f"{i}_trade_v"]

        best_bid, best_ask = maxs[:, bp], mins[:, ap]
        spread = best_ask - best_bid
        bid_total, ask_total = sums[:, bv], sums[:, av]
        imbalance = bid_total - ask_total
        depth = np.abs(bid_total) + np.abs(ask_total)
        stored[i] = (best_bid, best_ask, spread, imbalance)

        out[:, col] = best_bid;                         col += 1
        out[:, col] = best_ask;                         col += 1
        out[:, col] = spread;                           col += 1
        out[:, col] = spread * spread;                  col += 1
        out[:, col] = stds[:, bp];                      col += 1
        out[:, col] = stds[:, ap];                      col += 1
        out[:, col] = maxs[:, bp] - mins[:, bp];        col += 1
        out[:, col] = maxs[:, ap] - mins[:, ap];        col += 1
        out[:, col] = stds[:, bv];                      col += 1
        out[:, col] = stds[:, av];                      col += 1
        out[:, col] = maxs[:, bv];                      col += 1
        out[:, col] = maxs[:, av];                      col += 1
        out[:, col] = imbalance / (depth + EPS);        col += 1
        out[:, col] = imbalance * imbalance;            col += 1
        out[:, col] = imbalance * spread;               col += 1
        out[:, col] = sums[:, tv] / (depth + EPS);      col += 1

    imb0, imb1 = stored["i0"][3], stored["i1"][3]
    out[:, col] = imb0 * imb1;                                  col += 1
    out[:, col] = np.sign(imb0) * np.sign(imb1);                col += 1
    a = x[:, GROUPS["a"][0]:GROUPS["a"][1]]
    out[:, col] = (a[:, 7] <= A7_FLOOR).astype(np.float64);     col += 1
    out[:, col] = a[:, 6] * a[:, 7];                            col += 1
    out[:, col] = a[:, 2] * a[:, 3];                            col += 1

    # base signals that the memory features are built on
    signals[:, 0] = (stored["i0"][0] + stored["i0"][1]) / 2     # i0 mid
    signals[:, 1] = stored["i0"][3]                             # i0 imbalance
    signals[:, 2] = (stored["i1"][0] + stored["i1"][1]) / 2     # i1 mid
    signals[:, 3] = stored["i1"][3]                             # i1 imbalance
    signals[:, 4] = stored["i0"][2]                             # i0 spread
    signals[:, 5] = sums[:, _IX["i0_trade_v"]]                  # i0 trade volume
    signals[:, 6] = a[:, 6]
    signals[:, 7] = a[:, 7]
    return out, signals


def build_stateless(x):
    """Stateless features and their names."""
    out, _ = _core(x)
    return out, list(STATELESS_NAMES)


def base_signals(x):
    """The handful of summary numbers that memory features are built on."""
    _, signals = _core(x)
    return signals, list(SIGNAL_NAMES)


def build_memory_batch(signals):
    """Differences and EWMA deviations, computed causally down the sequence."""
    import pandas as pd    # training only; not available in the submission container

    df = pd.DataFrame(signals, columns=SIGNAL_NAMES)
    out = []
    for s in SIGNAL_NAMES:
        series = df[s]
        diff = series.diff().fillna(0.0)
        # adjust=False gives the recursive form, identical to the streaming update
        fast = series.ewm(alpha=ALPHA_FAST, adjust=False).mean()
        slow = series.ewm(alpha=ALPHA_SLOW, adjust=False).mean()
        out += [diff.to_numpy(), (series - fast).to_numpy(), (series - slow).to_numpy()]
    for s in VOL_SIGNALS:
        d = df[s].diff().fillna(0.0).abs()
        out.append(d.ewm(alpha=ALPHA_FAST, adjust=False).mean().to_numpy())
    return np.column_stack(out)


def build_sequence(x):
    """Full feature matrix for one complete sequence, in row order.

    Returns (features, stateless_names, memory_names).
    """
    x = np.asarray(x, dtype=np.float64)
    stateless, signals = _core(x)
    memory = build_memory_batch(signals)
    return np.hstack([x, stateless, memory]), list(STATELESS_NAMES), list(MEMORY_NAMES)


# ---------------------------------------------------------------------
#  Single-row fast path
# ---------------------------------------------------------------------
# For one row at a time, NumPy's per-call overhead dominates: each tiny
# operation costs more than the arithmetic itself. So the row path does four
# NumPy reductions, converts to plain Python floats, and finishes in Python.
# The arithmetic is identical to _core(), and test_features.py proves it.

_SIZES_LIST = [float(s) for s in _SIZES]


def _sign(v):
    return (v > 0.0) - (v < 0.0)


def _core_row(x):
    """Stateless features and base signals for a single 1-D row."""
    sums = np.add.reduceat(x, _STARTS).tolist()
    sumsq = np.add.reduceat(x * x, _STARTS).tolist()
    maxs = np.maximum.reduceat(x, _STARTS).tolist()
    mins = np.minimum.reduceat(x, _STARTS).tolist()

    stds = []
    for g in range(len(_SIZES_LIST)):
        size = _SIZES_LIST[g]
        mean = sums[g] / size
        var = sumsq[g] / size - mean * mean
        stds.append(sqrt(var) if var > 0.0 else 0.0)

    out = []
    keep = {}
    for i in ("i0", "i1"):
        bp, ap = _IX[i + "_bid_p"], _IX[i + "_ask_p"]
        bv, av = _IX[i + "_bid_v"], _IX[i + "_ask_v"]
        tv = _IX[i + "_trade_v"]

        best_bid, best_ask = maxs[bp], mins[ap]
        spread = best_ask - best_bid
        bid_total, ask_total = sums[bv], sums[av]
        imbalance = bid_total - ask_total
        depth = abs(bid_total) + abs(ask_total)
        keep[i] = (best_bid, best_ask, spread, imbalance)

        out += [best_bid, best_ask, spread, spread * spread,
                stds[bp], stds[ap], maxs[bp] - mins[bp], maxs[ap] - mins[ap],
                stds[bv], stds[av], maxs[bv], maxs[av],
                imbalance / (depth + EPS), imbalance * imbalance,
                imbalance * spread, sums[tv] / (depth + EPS)]

    imb0, imb1 = keep["i0"][3], keep["i1"][3]
    a6, a7 = x[110], x[111]
    out += [imb0 * imb1, float(_sign(imb0) * _sign(imb1)),
            1.0 if a7 <= A7_FLOOR else 0.0, a6 * a7, x[106] * x[107]]

    signals = [(keep["i0"][0] + keep["i0"][1]) / 2, imb0,
               (keep["i1"][0] + keep["i1"][1]) / 2, imb1,
               keep["i0"][2], sums[_IX["i0_trade_v"]], float(a6), float(a7)]
    return out, signals


class StreamingFeatures:
    """Row-by-row version of build_sequence(). Call reset() when seq_ix changes."""

    N_SIGNALS = len(SIGNAL_NAMES)

    def __init__(self):
        self._out = np.empty(N_RAW + N_STATELESS + N_MEMORY)
        self.reset()

    def reset(self):
        self.prev = None                              # previous row's base signals
        self.fast = None                              # fast EWMA state
        self.slow = None                              # slow EWMA state
        self.vol = [0.0] * len(VOL_SIGNALS)           # EWMA of |difference|

    def update(self, row):
        """Take one raw 112-feature row, update state, return the full feature vector."""
        x = np.asarray(row, dtype=np.float64)
        # A non-finite input would poison the EWMA state for the rest of the
        # sequence, so clean it here rather than only fixing the prediction.
        if not np.isfinite(x).all():
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        stateless, s = _core_row(x)
        n = self.N_SIGNALS

        if self.prev is None:                         # first row of the sequence
            diff = [0.0] * n
            self.fast = list(s)
            self.slow = list(s)
            self.vol = [0.0] * len(VOL_SIGNALS)
        else:
            prev, fast, slow = self.prev, self.fast, self.slow
            diff = [s[j] - prev[j] for j in range(n)]
            for j in range(n):
                fast[j] += ALPHA_FAST * (s[j] - fast[j])
                slow[j] += ALPHA_SLOW * (s[j] - slow[j])
            for k, j in enumerate(_VOL_IX):
                self.vol[k] += ALPHA_FAST * (abs(diff[j]) - self.vol[k])

        self.prev = s
        memory = []
        for j in range(n):                            # diff, dev_fast, dev_slow per signal
            memory += [diff[j], s[j] - self.fast[j], s[j] - self.slow[j]]
        memory += self.vol

        out = self._out
        out[:N_RAW] = x
        out[N_RAW:N_RAW + N_STATELESS] = stateless
        out[N_RAW + N_STATELESS:] = memory
        return out
