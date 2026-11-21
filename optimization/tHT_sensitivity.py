"""Spider plot of the sensitivity of the optimal HT thickness t*_HT to each input.

Base case: die at 330 °C, FEP 0.20 mm. For every parameter p, scale it by (1 + x/100),
re-optimise the HT thickness (maximise the line-speed limit V* over t_HT, as in
line_speed_optimization.ipynb), and plot the percent change in t*_HT against x.
"""
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter  # noqa: F401 (kept for consistency)
from scipy.optimize import brentq, minimize_scalar

sys.path.insert(0, str(Path("..").resolve()))
from wirecoating import WireCoatingDie, CrossLayer, WireCoatingSolverError

K0 = 273.15
R_W, R_D, L_LAND = 0.577e-3 / 2, 1.25e-3 / 2, 0.10e-3
TAU_C0 = 2e5                      # melt-fracture limit [Pa]
T_DIE = 330 + K0                  # origin: 330 °C
T_FEP = 0.20e-3                   # origin: FEP 0.20 mm
T_TOTAL_MAX = 0.25e-3
T_HT_LO, T_HT_HI = 0.002e-3, (T_TOTAL_MAX - T_FEP)

FEP0 = CrossLayer(eta0=3044, lam=0.0103, m=0.685, E_a=51.4e3, T_ref=330 + K0,
                  name="FEP 9835", rho_melt=1400, rho_solid=2100)
HT0 = CrossLayer(eta0=12780, lam=1.0345, m=0.529, E_a=92.7e3, T_ref=300 + K0,
                 name="HT 2181", rho_melt=1400, rho_solid=1700)

OUT = Path("figures")
OUT.mkdir(exist_ok=True)


def save_fig(fig, name, **kwargs):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=600, bbox_inches="tight", **kwargs)


def speed_limit(fep, ht, tau_c, t_ht):
    """Line speed V* [m/s] at which max wall shear stress = tau_c, for HT thickness t_ht [m]."""
    state = {"warm": None}
    thicknesses = [T_FEP, t_ht]

    def g(lnV):
        die = WireCoatingDie(R_W, R_D, float(np.exp(lnV)), [fep, ht], L=L_LAND)
        try:
            sol = die.solve(thicknesses=thicknesses, T=T_DIE, warm_start=state["warm"])
        except WireCoatingSolverError:
            sol = die.solve(thicknesses=thicknesses, T=T_DIE)
        state["warm"] = sol
        return max(abs(sol.tau_wire), abs(sol.tau_die)) / tau_c - 1.0

    lnV = brentq(g, np.log(1e-4), np.log(20.0), xtol=1e-10)
    return float(np.exp(lnV))


def t_star(fep, ht, tau_c):
    """Optimal HT thickness [m] (maximises V* over t_HT) and the corresponding V* [m/s]."""
    res = minimize_scalar(lambda t: -speed_limit(fep, ht, tau_c, t),
                          bounds=(T_HT_LO, T_HT_HI), method="bounded",
                          options={"xatol": 1e-9})
    return float(res.x), -float(res.fun)


# (label, group, attribute); group "fep" / "ht" / "tau"
PARAMS = [
    ("FEP $\\eta_0$", "fep", "eta0"),
    ("FEP $\\lambda$", "fep", "lam"),
    ("FEP $m$", "fep", "m"),
    ("FEP $E_a$", "fep", "E_a"),
    ("HT $\\eta_0$", "ht", "eta0"),
    ("HT $\\lambda$", "ht", "lam"),
    ("HT $m$", "ht", "m"),
    ("HT $E_a$", "ht", "E_a"),
    ("Critical stress $\\tau_c$", "tau", None),
]

# Percent changes to test. -100 % would give a zero or negative parameter, so the
# grid stops at -99 %. Values the model rejects (e.g. m > 1) are skipped.
GRID = np.r_[-99.0, np.arange(-90.0, 100.1, 10.0)]


def scaled_layers(group, attr, x):
    """FEP and HT layers with one parameter scaled by (1 + x/100); raises ValueError if invalid."""
    f = 1.0 + x / 100.0
    fep, ht, tau_c = FEP0, HT0, TAU_C0
    if group == "tau":
        tau_c = TAU_C0 * f
    else:
        layer = FEP0 if group == "fep" else HT0
        new = replace(layer, **{attr: getattr(layer, attr) * f})
        if group == "fep":
            fep = new
        else:
            ht = new
    return fep, ht, tau_c


def main():
    t0 = time.perf_counter()
    t_base, V_base = t_star(FEP0, HT0, TAU_C0)
    print(f"base: t*_HT = {t_base*1e3:.5f} mm, V* = {V_base:.5f} m/s "
          f"({time.perf_counter()-t0:.1f} s)")

    rows = []
    results = {}
    for label, group, attr in PARAMS:
        pct = []
        for x in GRID:
            try:
                fep, ht, tau_c = scaled_layers(group, attr, x)
                t, V = t_star(fep, ht, tau_c)
                pct.append(100.0 * (t / t_base - 1.0))
                rows.append([label, x, t * 1e3, pct[-1]])
            except (ValueError, RuntimeError) as err:
                pct.append(np.nan)
                rows.append([label, x, np.nan, np.nan])
                print(f"  skipped {label} at {x:+.0f} %: {err}")
        results[label] = np.array(pct)
        print(f"{label}: done ({time.perf_counter()-t0:.1f} s)", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows, columns=["parameter", "change_%", "t_HT_star_mm", "change_tHT_%"])
    df.to_csv(OUT / "tHT_sensitivity.csv", index=False)

    plot_spider(results)
    print(f"total {time.perf_counter()-t0:.1f} s")


def plot_spider(results):
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(PARAMS)))
    styles = ["-"] * 4 + ["--"] * 4 + [":"]
    for (label, _, _), c, ls in zip(PARAMS, colors, styles):
        y = results[label]
        ok = np.isfinite(y)
        # HT m above base: the line-speed solver finds different roots of the wall-stress
        # condition there, so t*_HT is not reliable. Those points stay in the CSV only.
        if label == "HT $m$":
            ok &= GRID <= 0
        ax.plot(GRID[ok], y[ok], ls, color=c, lw=2, label=label)
    ax.axhline(0, color="k", ls="--", lw=1)
    ax.set_xlim(-100, 100)
    ax.set_xlabel("Percent change in parameter (%)")
    ax.set_ylabel(r"Percent change in $t^*_{HT}$ (%)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", frameon=True, fontsize=8, ncol=2)
    save_fig(fig, "tHT_spider")
    plt.close(fig)


def replot_from_csv():
    import pandas as pd
    df = pd.read_csv(OUT / "tHT_sensitivity.csv")
    results = {}
    for label, _, _ in PARAMS:
        sub = df[df["parameter"] == label].set_index("change_%")["change_tHT_%"]
        results[label] = sub.reindex(GRID).to_numpy()
    plot_spider(results)


if __name__ == "__main__":
    if "--replot" in sys.argv:
        replot_from_csv()
    else:
        main()
