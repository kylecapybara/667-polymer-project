"""Spider plot of t*_HT sensitivity for the selected inputs: die temperature T, HT E_a, HT eta0, HT lambda.

Origin: die at 330 °C, FEP 0.20 mm. For each input p, scale it by (1 + x/100), re-optimise
the HT thickness (maximise the line-speed limit V* over t_HT), and plot the percent change in t*_HT.
Die temperature is scaled in kelvin (603.15 K at the origin), so x = -38 % is 373 K and x = +65 % is 1000 K,
the limits the model accepts.
"""
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq, minimize_scalar

sys.path.insert(0, str(Path("..").resolve()))
from wirecoating import WireCoatingDie, CrossLayer, WireCoatingSolverError

K0 = 273.15
R_W, R_D, L_LAND = 0.577e-3 / 2, 1.25e-3 / 2, 0.10e-3
TAU_C = 2e5
T_BASE = 330 + K0                 # origin: 330 °C
T_FEP = 0.20e-3                   # origin: FEP 0.20 mm
T_TOTAL_MAX = 0.25e-3
T_HT_LO, T_HT_HI = 0.002e-3, (T_TOTAL_MAX - T_FEP)

FEP0 = CrossLayer(eta0=3044, lam=0.0103, m=0.685, E_a=51.4e3, T_ref=330 + K0,
                  name="FEP 9835", rho_melt=1400, rho_solid=2100)
HT0 = CrossLayer(eta0=12780, lam=1.0345, m=0.529, E_a=92.7e3, T_ref=300 + K0,
                 name="HT 2181", rho_melt=1400, rho_solid=1700)

OUT = Path("figures")
OUT.mkdir(exist_ok=True)

# (label, kind, attribute). kind "T": die temperature; "ht": HT layer parameter.
PARAMS = [
    ("Die temperature $T$", "T", None),
    ("HT $E_a$", "ht", "E_a"),
    ("HT $\\eta_0$", "ht", "eta0"),
    ("HT $\\lambda$", "ht", "lam"),
]

GRID_PCT = np.r_[-99.0, np.arange(-90.0, 100.1, 10.0)]     # parameters with a hard lower bound of zero
GRID_T_PCT = np.arange(-35.0, 65.1, 5.0)                    # 392 K to 995 K, inside the model's 373–1000 K


def save_fig(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=600, bbox_inches="tight")


def speed_limit(ht, T, t_ht):
    """Line speed V* [m/s] at which max wall shear stress = TAU_C, for die temperature T [K], HT thickness t_ht [m]."""
    state = {"warm": None}
    thicknesses = [T_FEP, t_ht]

    def g(lnV):
        die = WireCoatingDie(R_W, R_D, float(np.exp(lnV)), [FEP0, ht], L=L_LAND)
        try:
            sol = die.solve(thicknesses=thicknesses, T=T, warm_start=state["warm"])
        except WireCoatingSolverError:
            sol = die.solve(thicknesses=thicknesses, T=T)
        state["warm"] = sol
        return max(abs(sol.tau_wire), abs(sol.tau_die)) / TAU_C - 1.0

    lnV = brentq(g, np.log(1e-4), np.log(20.0), xtol=1e-10)
    return float(np.exp(lnV))


def t_star(ht, T):
    """Optimal HT thickness [m] (maximises V* over t_HT) at die temperature T [K]."""
    res = minimize_scalar(lambda t: -speed_limit(ht, T, t), bounds=(T_HT_LO, T_HT_HI),
                          method="bounded", options={"xatol": 1e-9})
    return float(res.x)


def scaled(kind, attr, x):
    """Return (HT layer, die temperature) with one input scaled by (1 + x/100)."""
    if kind == "T":
        return HT0, T_BASE * (1.0 + x / 100.0)
    new = replace(HT0, **{attr: getattr(HT0, attr) * (1.0 + x / 100.0)})
    return new, T_BASE


def main():
    t0 = time.perf_counter()
    t_base = t_star(HT0, T_BASE)
    print(f"base: t*_HT = {t_base*1e3:.5f} mm ({time.perf_counter()-t0:.1f} s)", flush=True)

    rows, results = [], {}
    for label, kind, attr in PARAMS:
        grid = GRID_T_PCT if kind == "T" else GRID_PCT
        pct = []
        for x in grid:
            try:
                ht, T = scaled(kind, attr, x)
                t = t_star(ht, T)
                pct.append(100.0 * (t / t_base - 1.0))
                rows.append([label, x, T - K0, t * 1e3, pct[-1]])
            except (ValueError, RuntimeError) as err:
                pct.append(np.nan)
                rows.append([label, x, np.nan, np.nan, np.nan])
                print(f"  skipped {label} at {x:+.0f} %: {err}", flush=True)
        results[label] = (grid, np.array(pct))
        print(f"{label}: done ({time.perf_counter()-t0:.1f} s)", flush=True)

    df = pd.DataFrame(rows, columns=["parameter", "change_%", "T_die_C", "t_HT_star_mm", "change_tHT_%"])
    df.to_csv(OUT / "tHT_spider_selected.csv", index=False)
    plot(results)
    print(f"total {time.perf_counter()-t0:.1f} s")


def plot(results):
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    colors = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"]
    styles = ["-", "--", "-.", ":"]
    for (label, _, _), c, ls in zip(PARAMS, colors, styles):
        grid, y = results[label]
        ok = np.isfinite(y)
        ax.plot(grid[ok], y[ok], ls, color=c, lw=2, marker="o", ms=4, mec="k", mew=0.5, label=label)
    ax.axhline(0, color="k", ls="--", lw=1)
    ax.set_xlim(-100, 100)
    ax.set_xlabel("Percent change in parameter (%)")
    ax.set_ylabel(r"Percent change in $t^*_{HT}$ (%)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    save_fig(fig, "tHT_spider_selected")
    plt.close(fig)


if __name__ == "__main__":
    main()
