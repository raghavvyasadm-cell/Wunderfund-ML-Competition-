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
# Smaller alpha remembers further back; 2/alpha is roughly the window length.
SPANS = {"vfast": 0.2, "fast": 0.05, "slow": 0.005, "vslow": 0.001}   # ~5, 20, 200, 1000 rows
SPAN_ORDER = ["vfast", "fast", "slow", "vslow"]

A7_FLOOR = -5.0      # a7 sits at -5.199337 about two thirds of the time
EPS = 1e-6
# The volatility-scaled deviation divides by volatility, which goes to nearly
# zero when a signal is flat. A tiny floor would multiply ordinary rounding by
# a million and let the feature reach absurd values, so the floor is generous
# and the result is clipped.
Z_FLOOR = 1e-3
Z_CLIP = 20.0

_ORDER = ["i0_bid_p", "i0_ask_p", "i0_bid_v", "i0_ask_v", "i0_trade_p", "i0_trade_v",
          "i1_bid_p", "i1_ask_p", "i1_bid_v", "i1_ask_v", "i1_trade_p", "i1_trade_v", "a"]
_STARTS = np.array([GROUPS[g][0] for g in _ORDER])
_SIZES = np.array([GROUPS[g][1] - GROUPS[g][0] for g in _ORDER], dtype=np.float64)
_SIZES_LIST = [float(s) for s in _SIZES]
_IX = {g: i for i, g in enumerate(_ORDER)}

# ---------------------------------------------------------------------
#  Stateless features (current row only)
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
#  Memory features
# ---------------------------------------------------------------------
# Each base signal gets a profile saying what to remember about it.
#   "standard"  difference, and deviation from a 20-row and a 200-row average
#   "rich"      four averages, the fast average itself, volatility, and a
#               volatility-scaled deviation. Used for trade volume, because
#               i0_trade_v_dev_fast was by far the strongest feature so far.
SIGNAL_SPECS = [
    ("i0_mid", "standard"), ("i0_imbalance", "standard"),
    ("i1_mid", "standard"), ("i1_imbalance", "standard"),
    ("i0_spread", "standard"),
    ("i0_trade_v", "rich"), ("i1_trade_v", "rich"),
    ("a6", "standard"), ("a7", "standard"),
    ("i0_best_bid", "standard"), ("i0_best_ask", "standard"),
    ("i0_bid_total", "standard"), ("i0_ask_total", "standard"),
    ("a2", "standard"), ("a3", "standard"),
]
SIGNAL_NAMES = [n for n, _ in SIGNAL_SPECS]
PROFILES = dict(SIGNAL_SPECS)

# Signals that also get volatility and a volatility-scaled deviation
VOL_SIGNALS = ("i0_mid", "i0_imbalance", "i0_trade_v", "i1_trade_v")
_VOL_IX = [SIGNAL_NAMES.index(n) for n in VOL_SIGNALS]
_VOL_POS = {n: k for k, n in enumerate(VOL_SIGNALS)}


def _memory_plan():
    """What to emit per signal, in order: list of (signal, kind, span)."""
    plan = []
    for name, profile in SIGNAL_SPECS:
        plan.append((name, "diff", None))
        spans = SPAN_ORDER if profile == "rich" else ["fast", "slow"]
        for span in spans:
            plan.append((name, "dev", span))
        if profile == "rich":
            plan.append((name, "level", "fast"))
        if name in VOL_SIGNALS:
            plan.append((name, "vol", None))
            plan.append((name, "z", "fast"))
    return plan


MEMORY_PLAN = _memory_plan()
MEMORY_NAMES = [f"{s}_{k}" if span is None else f"{s}_{k}_{span}"
                for s, k, span in MEMORY_PLAN]

_SIGNAL_POS = {n: j for j, n in enumerate(SIGNAL_NAMES)}

# Only update an average for the signals that actually use it: "standard"
# signals need two of the four spans, so this roughly halves the per-row work.
SPAN_MEMBERS = {span: [] for span in SPAN_ORDER}
for _name, _profile in SIGNAL_SPECS:
    for _span in (SPAN_ORDER if _profile == "rich" else ["fast", "slow"]):
        SPAN_MEMBERS[_span].append(_SIGNAL_POS[_name])
SPAN_UPDATES = [(SPANS[_span], SPAN_MEMBERS[_span], _span) for _span in SPAN_ORDER]

# The emit plan with every lookup resolved: (code, signal index, span, vol slot)
_CODES = {"diff": 0, "dev": 1, "level": 2, "vol": 3, "z": 4}
PLAN_RESOLVED = [(_CODES[_k], _SIGNAL_POS[_s], _span,
                  _VOL_POS[_s] if _s in _VOL_POS else -1)
                 for _s, _k, _span in MEMORY_PLAN]

# Bumped whenever the feature set changes, so a model trained on an older set
# cannot be used by accident: the numbers would line up but mean something else.
FEATURE_VERSION = "v5-trade-family"

N_RAW = 112
N_STATELESS = len(STATELESS_NAMES)
N_MEMORY = len(MEMORY_NAMES)
N_SIGNALS = len(SIGNAL_NAMES)


# =====================================================================
#  Group statistics
# =====================================================================

def _group_stats(x):
    """Sum, max, min and standard deviation of every column group, in four passes."""
    sums = np.add.reduceat(x, _STARTS, axis=1)
    sumsq = np.add.reduceat(x * x, _STARTS, axis=1)
    maxs = np.maximum.reduceat(x, _STARTS, axis=1)
    mins = np.minimum.reduceat(x, _STARTS, axis=1)
    means = sums / _SIZES
    stds = np.sqrt(np.maximum(sumsq / _SIZES - means * means, 0.0))
    return sums, maxs, mins, stds


def _core(x):
    """Stateless features and base signals for a block of rows."""
    x = np.asarray(x, dtype=np.float64)
    sums, maxs, mins, stds = _group_stats(x)
    out = np.empty((len(x), N_STATELESS))
    signals = np.empty((len(x), N_SIGNALS))

    col = 0
    keep = {}
    for i in ("i0", "i1"):
        bp, ap = _IX[f"{i}_bid_p"], _IX[f"{i}_ask_p"]
        bv, av = _IX[f"{i}_bid_v"], _IX[f"{i}_ask_v"]
        tv = _IX[f"{i}_trade_v"]

        best_bid, best_ask = maxs[:, bp], mins[:, ap]
        spread = best_ask - best_bid
        bid_total, ask_total = sums[:, bv], sums[:, av]
        imbalance = bid_total - ask_total
        depth = np.abs(bid_total) + np.abs(ask_total)
        keep[i] = (best_bid, best_ask, spread, imbalance, bid_total, ask_total, sums[:, tv])

        out[:, col] = best_bid;                        col += 1
        out[:, col] = best_ask;                        col += 1
        out[:, col] = spread;                          col += 1
        out[:, col] = spread * spread;                 col += 1
        out[:, col] = stds[:, bp];                     col += 1
        out[:, col] = stds[:, ap];                     col += 1
        out[:, col] = maxs[:, bp] - mins[:, bp];       col += 1
        out[:, col] = maxs[:, ap] - mins[:, ap];       col += 1
        out[:, col] = stds[:, bv];                     col += 1
        out[:, col] = stds[:, av];                     col += 1
        out[:, col] = maxs[:, bv];                     col += 1
        out[:, col] = maxs[:, av];                     col += 1
        out[:, col] = imbalance / (depth + EPS);       col += 1
        out[:, col] = imbalance * imbalance;           col += 1
        out[:, col] = imbalance * spread;              col += 1
        out[:, col] = sums[:, tv] / (depth + EPS);     col += 1

    imb0, imb1 = keep["i0"][3], keep["i1"][3]
    a = x[:, GROUPS["a"][0]:GROUPS["a"][1]]
    out[:, col] = imb0 * imb1;                                 col += 1
    out[:, col] = np.sign(imb0) * np.sign(imb1);               col += 1
    out[:, col] = (a[:, 7] <= A7_FLOOR).astype(np.float64);    col += 1
    out[:, col] = a[:, 6] * a[:, 7];                           col += 1
    out[:, col] = a[:, 2] * a[:, 3];                           col += 1

    values = {
        "i0_mid": (keep["i0"][0] + keep["i0"][1]) / 2, "i0_imbalance": imb0,
        "i1_mid": (keep["i1"][0] + keep["i1"][1]) / 2, "i1_imbalance": imb1,
        "i0_spread": keep["i0"][2], "i0_trade_v": keep["i0"][6], "i1_trade_v": keep["i1"][6],
        "a6": a[:, 6], "a7": a[:, 7], "a2": a[:, 2], "a3": a[:, 3],
        "i0_best_bid": keep["i0"][0], "i0_best_ask": keep["i0"][1],
        "i0_bid_total": keep["i0"][4], "i0_ask_total": keep["i0"][5],
    }
    for j, name in enumerate(SIGNAL_NAMES):
        signals[:, j] = values[name]
    return out, signals


def build_stateless(x):
    out, _ = _core(x)
    return out, list(STATELESS_NAMES)


def base_signals(x):
    _, signals = _core(x)
    return signals, list(SIGNAL_NAMES)


# =====================================================================
#  Memory, whole sequence at once (training)
# =====================================================================

def build_memory_batch(signals):
    """Every memory feature, computed causally down the sequence."""
    import pandas as pd    # training only; not available in the submission container

    df = pd.DataFrame(signals, columns=SIGNAL_NAMES)
    diffs, ewmas, vols = {}, {}, {}
    for name in SIGNAL_NAMES:
        series = df[name]
        diffs[name] = series.diff().fillna(0.0)
        for span, alpha in SPANS.items():
            ewmas[(name, span)] = series.ewm(alpha=alpha, adjust=False).mean()
        if name in VOL_SIGNALS:
            vols[name] = diffs[name].abs().ewm(alpha=SPANS["fast"], adjust=False).mean()

    columns = []
    for name, kind, span in MEMORY_PLAN:
        series = df[name]
        if kind == "diff":
            columns.append(diffs[name].to_numpy())
        elif kind == "dev":
            columns.append((series - ewmas[(name, span)]).to_numpy())
        elif kind == "level":
            columns.append(ewmas[(name, span)].to_numpy())
        elif kind == "vol":
            columns.append(vols[name].to_numpy())
        elif kind == "z":   # deviation measured in recent typical moves
            z = (series - ewmas[(name, span)]) / (vols[name] + Z_FLOOR)
            columns.append(z.clip(-Z_CLIP, Z_CLIP).to_numpy())
    return np.column_stack(columns)


def build_sequence(x):
    """Full feature matrix for one complete sequence, in row order."""
    x = np.asarray(x, dtype=np.float64)
    stateless, signals = _core(x)
    memory = build_memory_batch(signals)
    return np.hstack([x, stateless, memory]), list(STATELESS_NAMES), list(MEMORY_NAMES)


# =====================================================================
#  Single-row fast path
# =====================================================================
# For one row at a time, NumPy's per-call overhead dominates: each tiny
# operation costs more than the arithmetic itself. So the row path does four
# NumPy reductions, converts to plain Python floats, and finishes in Python.
# The arithmetic is identical to _core(), and test_features.py proves it.

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
        keep[i] = (best_bid, best_ask, spread, imbalance, bid_total, ask_total, sums[tv])

        out += [best_bid, best_ask, spread, spread * spread,
                stds[bp], stds[ap], maxs[bp] - mins[bp], maxs[ap] - mins[ap],
                stds[bv], stds[av], maxs[bv], maxs[av],
                imbalance / (depth + EPS), imbalance * imbalance,
                imbalance * spread, sums[tv] / (depth + EPS)]

    imb0, imb1 = keep["i0"][3], keep["i1"][3]
    a2, a3, a6, a7 = x[106], x[107], x[110], x[111]
    out += [imb0 * imb1, float(_sign(imb0) * _sign(imb1)),
            1.0 if a7 <= A7_FLOOR else 0.0, a6 * a7, a2 * a3]

    values = {
        "i0_mid": (keep["i0"][0] + keep["i0"][1]) / 2, "i0_imbalance": imb0,
        "i1_mid": (keep["i1"][0] + keep["i1"][1]) / 2, "i1_imbalance": imb1,
        "i0_spread": keep["i0"][2], "i0_trade_v": keep["i0"][6], "i1_trade_v": keep["i1"][6],
        "a6": float(a6), "a7": float(a7), "a2": float(a2), "a3": float(a3),
        "i0_best_bid": keep["i0"][0], "i0_best_ask": keep["i0"][1],
        "i0_bid_total": keep["i0"][4], "i0_ask_total": keep["i0"][5],
    }
    return out, [values[n] for n in SIGNAL_NAMES]


class StreamingFeatures:
    """Row-by-row version of build_sequence(). Call reset() when seq_ix changes."""

    def __init__(self):
        self._out = np.empty(N_RAW + N_STATELESS + N_MEMORY)
        self.reset()

    def reset(self):
        self.prev = None                                   # previous row's signals
        self.ewma = {span: None for span in SPAN_ORDER}    # one average per span
        self.vol = [0.0] * len(VOL_SIGNALS)                # EWMA of |difference|

    def update(self, row):
        """Take one raw 112-feature row, update state, return the full feature vector."""
        x = np.asarray(row, dtype=np.float64)
        # A non-finite input would poison the EWMA state for the rest of the
        # sequence, so clean it here rather than only fixing the prediction.
        if not np.isfinite(x).all():
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        stateless, s = _core_row(x)

        if self.prev is None:                              # first row of the sequence
            diff = [0.0] * N_SIGNALS
            for span in SPAN_ORDER:
                self.ewma[span] = list(s)
            self.vol = [0.0] * len(VOL_SIGNALS)
        else:
            prev = self.prev
            diff = [s[j] - prev[j] for j in range(N_SIGNALS)]
            for alpha, members, span in SPAN_UPDATES:
                state = self.ewma[span]
                for j in members:
                    state[j] += alpha * (s[j] - state[j])
            alpha_fast = SPANS["fast"]
            vol = self.vol
            for k, j in enumerate(_VOL_IX):
                vol[k] += alpha_fast * (abs(diff[j]) - vol[k])

        self.prev = s
        ewma, vol = self.ewma, self.vol
        memory = []
        append = memory.append
        for code, j, span, vpos in PLAN_RESOLVED:
            if code == 0:
                append(diff[j])
            elif code == 1:
                append(s[j] - ewma[span][j])
            elif code == 2:
                append(ewma[span][j])
            elif code == 3:
                append(vol[vpos])
            else:
                z = (s[j] - ewma[span][j]) / (vol[vpos] + Z_FLOOR)
                append(Z_CLIP if z > Z_CLIP else (-Z_CLIP if z < -Z_CLIP else z))

        out = self._out
        out[:N_RAW] = x
        out[N_RAW:N_RAW + N_STATELESS] = stateless
        out[N_RAW + N_STATELESS:] = memory
        # Return a copy, not the scratch buffer. Handing back the buffer is
        # marginally faster but lets a caller keep a reference that silently
        # changes on the next row - a bug that is very hard to see.
        return out.copy()
