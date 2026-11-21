"""Symbolic regression on the shifted complex-viscosity master curves.

Rebuilds the same shifted |eta*| data that trios_analysis.ipynb fits with the
Cross model (same shift factors, same b_T, same excluded HT points), then asks
PySR to find formulas eta(x), x = a_T * omega, without being told the form.

Usage (from the "trios data" folder):
    python symbolic_regression.py --prep-only     # check the data, no PySR needed
    python symbolic_regression.py                 # all three polymers
    python symbolic_regression.py --polymer FEP --holdout 350
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

HERE = Path(__file__).resolve().parent

POLYMERS = {
    "FEP": {
        "ref": 330,
        "shift_file": "FEP/FEP 9835 2007390112 mfr 24.8 330°C.txt",
        "sweeps": {
            330: "FEP/FEP 9835 2007390112 mfr 24.8 330°C.txt",
            350: "FEP/TE9835 Lot 2007390112 at 350°C.txt",
            370: "FEP/TE9835 Lot 2007390112 at 370°C.txt",
        },
    },
    "PFA": {
        "ref": 330,
        "shift_file": "PFA/PFA 440HPAX Lot 2103470009 at 330°C.txt",
        "sweeps": {
            330: "PFA/PFA 440HPAX Lot 2103470009 at 330°C.txt",
            350: "PFA/PFA 440  Lot 2103470009 mfr 15.5 at 350°C.txt",
            370: "PFA/PFA 440  Lot 2103470009 mfr 15.5 at 370°C.txt",
        },
    },
    "HT": {
        "ref": 300,
        "shift_file": "HT/HT 2181 1212PASK 22 MFR 6,4 315°C.txt",
        "sweeps": {
            285: "HT/HT 2181 1212PASK 22 MFR 6,4 285°C.txt",
            300: "HT/HT 2181 1212PASK 22 MFR 6,4 300°C.txt",
            315: "HT/HT 2181 1212PASK 22 MFR 6,4 315°C.txt",
        },
    },
}


# ---------- TRIOS loading (same logic as trios_analysis.ipynb) ----------
def _num(x):
    try:
        return float(x)
    except ValueError:
        return np.nan


def read_steps(path):
    steps, cur = [], None
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("Step name\t"):
            cur = {"name": line.split("\t", 1)[1], "vars": None, "rows": []}
            steps.append(cur)
        elif cur is None:
            continue
        elif line.startswith("Variables\t"):
            cur["vars"] = line.split("\t")[1:]
        elif line.startswith("Data point\t"):
            cur["rows"].append([_num(x) for x in line.split("\t")[1:]])
    return steps


def column(step, name):
    idx = step["vars"].index(name)
    return np.array([r[idx] for r in step["rows"]], float)


# Sweeps run from high to low frequency, so the lowest-frequency points are measured
# last, 10-35 min into the test. In the terminal region |eta*| should level off as
# omega drops. A point that falls below the highest |eta*| already measured at higher
# frequency means the sample changed during the test (most likely degradation), so it
# is dropped. DROP_TOL keeps sub-1% point-to-point noise from triggering the rule.
DROP_TOL = 0.01


def degraded_points(w, eta, tol=DROP_TOL):
    """Boolean mask of points whose |eta*| is more than tol below the running maximum
    taken from high to low frequency."""
    order = np.argsort(w)[::-1]
    running_max = np.maximum.accumulate(eta[order])
    bad = np.zeros_like(w, bool)
    bad[order] = eta[order] < (1 - tol) * running_max
    return bad


def shifted_data(name, verbose=True):
    """Return (x = a_T*omega, eta_shifted, temperature label) for one polymer."""
    cfg = POLYMERS[name]
    shift_step = next(s for s in read_steps(HERE / cfg["shift_file"]) if "shift" in s["name"].lower())
    Tt, aTt = column(shift_step, "Temperature"), column(shift_step, "aT (x variable)")
    xs, etas, temps = [], [], []
    for T, f in cfg["sweeps"].items():
        step = next(s for s in read_steps(HERE / f) if s["name"].startswith("Frequency"))
        w, eta = column(step, "Angular frequency"), column(step, "Complex viscosity")
        aT = aTt[np.argmin(np.abs(Tt - T))]
        eta = eta * ((cfg["ref"] + 273.15) / (T + 273.15)) / aT  # b_T / a_T, as in the notebook
        bad = degraded_points(w, eta)
        if verbose and bad.any():
            print(f"  {name} {T} °C: dropped omega = {np.round(np.sort(w[bad]), 4).tolist()} rad/s")
        ok = ~bad & (w > 0) & (eta > 0) & np.isfinite(eta)
        xs.append(w[ok] * aT)
        etas.append(eta[ok])
        temps.append(np.full(ok.sum(), T))
    return np.concatenate(xs), np.concatenate(etas), np.concatenate(temps)


# ---------- Cross baseline, scored the same way as PySR ----------
def cross(x, eta0, lam, m):
    return eta0 / (1 + (lam * x) ** m)


def log_mse(pred, y):
    return float(np.mean((np.log(pred) - np.log(y)) ** 2))


def fit_cross(x, y):
    p, _ = curve_fit(lambda lx, le, ll, m: np.log(cross(np.exp(lx), np.exp(le), np.exp(ll), m)),
                     np.log(x), np.log(y), p0=[np.log(y.max()), np.log(1 / np.median(x)), 0.7])
    return np.exp(p[0]), np.exp(p[1]), p[2]


# ---------- PySR ----------
# Loss is squared error in ln(eta), matching how the Cross model was fit.
# Negative predictions get a large penalty instead of a NaN.
LOG_LOSS = "loss(p, t) = p > 0 ? (log(p) - log(t))^2 : 100 + abs(p)"


def run_pysr(name, x, y, iterations, seed):
    from pysr import PySRRegressor  # imported here so --prep-only works without PySR

    model = PySRRegressor(
        niterations=iterations,
        binary_operators=["+", "-", "*", "/", "^"],
        unary_operators=["exp", "log"],
        elementwise_loss=LOG_LOSS,
        maxsize=20,
        # keep exponents simple: no deep nesting inside a power, and no exp(exp(...))
        constraints={"^": (-1, 5)},
        nested_constraints={"^": {"^": 1, "exp": 0, "log": 0}, "exp": {"exp": 0, "log": 0},
                            "log": {"exp": 0, "log": 0}},
        model_selection="best",
        random_state=seed,
        progress=True,
    )
    model.fit(x.reshape(-1, 1), y, variable_names=["x"])
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--polymer", choices=list(POLYMERS), nargs="*", default=list(POLYMERS))
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--holdout", type=int, default=None,
                    help="leave this sweep temperature out of the fit and test on it")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prep-only", action="store_true")
    args = ap.parse_args()

    for name in args.polymer:
        x, eta, temps = shifted_data(name)
        # Scale eta so the plateau is about 1. PySR finds O(1) constants more easily.
        scale = float(np.median(eta[x < np.quantile(x, 0.1)]))
        y = eta / scale

        train = np.ones_like(x, bool) if args.holdout is None else temps != args.holdout
        test = ~train

        eta0, lam, m = fit_cross(x[train], y[train])
        print(f"\n===== {name}: {train.sum()} train points, {test.sum()} test points, scale = {scale:.4g} Pa s =====")
        print(f"x range {x.min():.3g} to {x.max():.3g} rad/s")
        print(f"Cross baseline: eta0 = {eta0 * scale:.4g} Pa s, lambda = {lam:.4g} s, m = {m:.3f}, "
              f"train log-MSE = {log_mse(cross(x[train], eta0, lam, m), y[train]):.3e}"
              + (f", test log-MSE = {log_mse(cross(x[test], eta0, lam, m), y[test]):.3e}" if test.any() else ""))
        if args.prep_only:
            continue

        model = run_pysr(name, x[train], y[train], args.iterations, args.seed)
        eqs = model.equations_
        print("\nPareto front (y = eta / scale):")
        for i, row in eqs.iterrows():
            line = f"  complexity {row['complexity']:>2}  train log-MSE {row['loss']:.3e}"
            if test.any():
                pred = model.predict(x[test].reshape(-1, 1), index=i)
                line += f"  test log-MSE {log_mse(np.clip(pred, 1e-300, None), y[test]):.3e}"
            print(line + f"   {row['sympy_format']}")
        out = HERE / f"pysr_{name}.csv"
        eqs.drop(columns=[c for c in eqs.columns if c in ("lambda_format", "torch_format", "jax_format")]
                 ).to_csv(out, index=False)
        print(f"saved {out.name}")


if __name__ == "__main__":
    main()
