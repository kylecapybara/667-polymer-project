"""Convergence of 1-D root/optimisation methods for the line-speed limit V*.

For the optimum design (FEP 0.20 mm, HT 0.0285 mm, die at 330 °C), V* is the
single root of g(ln V) = max(|tau_w|, |tau_d|)/tau_c - 1. Each method is run
on this function, and every call to g is one full die solve (the cost unit).

Metric: relative error |V_best - V*| / V*, where V_best is the evaluated point
with the smallest |g| so far, plotted against the number of die solves.
"""
from pathlib import Path
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
from scipy.optimize import bisect, brentq, brenth, ridder, toms748, newton, minimize_scalar

sys.path.insert(0, str(Path("..").resolve()))
from wirecoating import WireCoatingDie, CrossLayer, WireCoatingSolverError

K0 = 273.15
R_W, R_D, L_LAND = 0.577e-3 / 2, 1.25e-3 / 2, 0.10e-3
TAU_C = 2e5
T_FEP_MIN = 0.20e-3
T_DIE = 330 + K0
T_HT_OPT = 0.02852e-3  # optimum HT thickness at 330 °C (notebook Section 4b)

FEP = CrossLayer(eta0=3044, lam=0.0103, m=0.685, E_a=51.4e3, T_ref=330 + K0,
                 name="FEP 9835", rho_melt=1400, rho_solid=2100)
HT = CrossLayer(eta0=12780, lam=1.0345, m=0.529, E_a=92.7e3, T_ref=300 + K0,
                name="HT 2181", rho_melt=1400, rho_solid=1700)
THICKNESSES = [T_FEP_MIN, T_HT_OPT]

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)


class Counted:
    """Wraps g(lnV) so each call is counted and logged, with a warm start from the last solve."""

    def __init__(self, cold=False):
        self.calls = []          # (lnV, g) in evaluation order
        self.warm = None
        self.cold = cold

    def __call__(self, lnV):
        V = float(np.exp(lnV))
        die = WireCoatingDie(R_W, R_D, V, [FEP, HT], L=L_LAND)
        try:
            sol = die.solve(thicknesses=THICKNESSES, T=T_DIE,
                            warm_start=None if self.cold else self.warm)
        except WireCoatingSolverError:
            sol = die.solve(thicknesses=THICKNESSES, T=T_DIE)  # retry cold
        self.warm = sol
        g = max(abs(sol.tau_wire), abs(sol.tau_die)) / TAU_C - 1.0
        self.calls.append((float(lnV), g))
        return g


def reference_root():
    # Tight Brent root as the reference answer.
    f = Counted(cold=True)
    lnV = brentq(f, np.log(1e-4), np.log(20.0), xtol=1e-14, rtol=1e-14)
    return float(np.exp(lnV))


def error_history(calls, V_ref):
    """Relative error of the best point so far, after each evaluation."""
    best_abs_g, best_V, errs, Vs = np.inf, None, [], []
    for lnV, g in calls:
        if abs(g) < best_abs_g:
            best_abs_g, best_V = abs(g), float(np.exp(lnV))
        errs.append(abs(best_V - V_ref) / V_ref)
        Vs.append(float(np.exp(lnV)))
    return np.array(errs), np.array(Vs)


def run_methods(V_ref):
    lo, hi = np.log(1e-4), np.log(20.0)
    results = {}

    def bracketed(name, solver):
        f = Counted()
        t0 = time.perf_counter()
        solver(f)
        results[name] = (*error_history(f.calls, V_ref), time.perf_counter() - t0)

    # Bracketing root finders (need g(lo) < 0 < g(hi)).
    bracketed("Bisection", lambda f: bisect(f, lo, hi, xtol=1e-10, maxiter=500))
    bracketed("Ridder", lambda f: ridder(f, lo, hi, xtol=1e-10, maxiter=500))
    bracketed("Brent (brenth)", lambda f: brenth(f, lo, hi, xtol=1e-10, maxiter=500))
    bracketed("Brent (brentq)", lambda f: brentq(f, lo, hi, xtol=1e-10, maxiter=500))
    bracketed("TOMS748", lambda f: toms748(f, lo, hi, xtol=1e-10, maxiter=500))

    # Secant method from two starting guesses (no bracket required).
    bracketed("Secant", lambda f: newton(f, np.log(1e-2), x1=np.log(1e-1), tol=1e-10, maxiter=500))

    # Golden-section and bounded Brent minimisation of |g|, which is minimal (zero) at V*.
    def abs_counted(f):
        return lambda lnV: abs(f(lnV))
    f1 = Counted()
    t0 = time.perf_counter()
    minimize_scalar(abs_counted(f1), bracket=None, bounds=(lo, hi), method="bounded",
                    options={"xatol": 1e-10, "maxiter": 500})
    results["Brent (bounded min. of |g|)"] = (*error_history(f1.calls, V_ref), time.perf_counter() - t0)

    f2 = Counted()
    t0 = time.perf_counter()
    minimize_scalar(abs_counted(f2), bracket=(lo, np.log(0.05), hi), method="golden",
                    options={"xtol": 1e-10, "maxiter": 500})
    results["Golden section (min. of |g|)"] = (*error_history(f2.calls, V_ref), time.perf_counter() - t0)
    return results


def main():
    V_ref = reference_root()
    print(f"reference V* = {V_ref:.10f} m/s  ({V_ref*60:.4f} m/min)")
    results = run_methods(V_ref)

    print(f"{'method':32s} {'solves':>7s} {'final rel. err':>15s} {'time [s]':>9s}")
    for name, (errs, Vs, dt) in results.items():
        print(f"{name:32s} {len(errs):7d} {errs[-1]:15.2e} {dt:9.2f}")

    styles = {
        "Bisection": ("-", "o", "#1f77b4"),
        "Ridder": ("-", "s", "#ff7f0e"),
        "Brent (brenth)": ("-", "^", "#2ca02c"),
        "Brent (brentq)": ("--", "v", "#d62728"),
        "TOMS748": ("-", "D", "#9467bd"),
        "Secant": ("-", "P", "#8c564b"),
        "Brent (bounded min. of |g|)": ("--", "X", "#e377c2"),
        "Golden section (min. of |g|)": (":", "*", "#7f7f7f"),
    }
    plt.rcParams.update({"font.size": 10})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0), gridspec_kw={"width_ratios": [1.2, 1]})
    for name, (errs, Vs, _) in results.items():
        ls, mk, c = styles[name]
        k = np.arange(1, len(errs) + 1)
        ax1.plot(k, np.maximum(errs, 1e-16), ls=ls, marker=mk, ms=4, color=c, mec="k", mew=0.4, lw=1.3, label=name)
        ax2.plot(k, Vs, ls="none", marker=mk, ms=4, color=c, mec="k", mew=0.4, alpha=0.8)
    ax1.set_yscale("log")
    ax1.set_xlabel("Number of die solves")
    ax1.set_ylabel(r"Relative error $|V_\mathrm{best}-V^*|/V^*$")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="upper right", frameon=True, fontsize=8)
    ax2.axhline(V_ref, color="k", lw=1.0, ls="--")
    ax2.set_xlabel("Number of die solves")
    ax2.set_ylabel(r"Evaluated line speed $V$ [m/s]")
    ax2.set_yscale("log")
    ax2.grid(alpha=0.25)
    ax1.set_xlim(0, max(len(e) for e, _, _ in results.values()) + 2)
    ax2.set_xlim(0, max(len(e) for e, _, _ in results.values()) + 2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"method_convergence.{ext}", dpi=600, bbox_inches="tight")
    print("saved", FIG_DIR / "method_convergence.png")


if __name__ == "__main__":
    main()
