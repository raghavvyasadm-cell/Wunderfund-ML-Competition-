"""Group the fitted weights by feature family and check what each family carries.

Run from the project root:

    python inspect_weights.py
"""
import numpy as np

from utils import FEATURE_COLUMNS

# Position ranges from the data documentation (start, end) - end is exclusive
GROUPS = [
    ("i0 bid price",  0,  11), ("i0 ask price", 11,  22),
    ("i0 bid volume", 22, 33), ("i0 ask volume", 33, 44),
    ("i0 trade price", 44, 48), ("i0 trade volume", 48, 52),
    ("i1 bid price", 52, 63), ("i1 ask price", 63, 74),
    ("i1 bid volume", 74, 85), ("i1 ask volume", 85, 96),
    ("i1 trade price", 96, 100), ("i1 trade volume", 100, 104),
    ("a0-a7", 104, 112),
]


def main():
    model = np.load("linear_model.npz")
    w0, w1 = model["coef"][0], model["coef"][1]      # weights for t0 and for t1

    # Every weight pair splits into a "together" and a "seesaw" component,
    # exactly like the targets do.
    together = (w0 + w1) / 2
    seesaw = (w0 - w1) / 2

    print(f"{'group':<17}{'n':>3}{'|w| t0':>9}{'|w| t1':>9}"
          f"{'|together|':>12}{'|seesaw|':>10}{'carries':>10}")
    for name, start, end in GROUPS:
        n = end - start
        a, b = np.abs(w0[start:end]).mean(), np.abs(w1[start:end]).mean()
        t, s = np.abs(together[start:end]).mean(), np.abs(seesaw[start:end]).mean()
        carries = "together" if t > s else "seesaw"
        print(f"{name:<17}{n:>3}{a:>9.4f}{b:>9.4f}{t:>12.4f}{s:>10.4f}{carries:>10}")

    print("\nDoes the bid side lean towards t0 and the ask side towards t1?")
    for name, start, end in GROUPS[:4] + GROUPS[6:10]:
        lean = np.abs(w0[start:end]).mean() - np.abs(w1[start:end]).mean()
        direction = "t0" if lean > 0 else "t1"
        print(f"  {name:<17} leans towards {direction}  (difference {lean:+.4f})")

    print("\nTotal weight size per group (how much the model relies on it):")
    totals = [(np.abs(model['coef'][:, s:e]).sum(), name) for name, s, e in GROUPS]
    for total, name in sorted(totals, reverse=True):
        print(f"  {name:<17} {total:.3f}")


if __name__ == "__main__":
    main()