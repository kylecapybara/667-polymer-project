'''
Multilayer wire coating in the die land: flow, shrinkage and draw-down.

author: @kwode

Quick start
-----------
Power-law layers (the notebook model):
    die = WireCoatingDie(
        R_wire=0.50e-3, R_die=1.00e-3, V_wire=1.0, L=5e-3,
        layers=[PowerLawLayer(K=4000, n=0.40, name="Insulation"),
                PowerLawLayer(K=12000, n=0.30, name="Jacket")],
    )
    sol = die.solve(thicknesses=[0.10e-3, 0.20e-3])    # room-temperature thickness per layer [m]
    print(sol.pressure_drop, sol.wire_force, sol.draw_down_ratio)
    die.report(sol)
    die.plot_profiles(sol)

Fluoropolymer (inside) and HT 2181 (outside), Cross rheology with a time-temperature
shift and room-temperature shrinkage:
    FEP = CrossLayer(eta0=3044, lam=0.0103, m=0.685, E_a=51.4e3, T_ref=603.15,
                     name="FEP 9835", rho_melt=1400, rho_solid=2100)
    HT = CrossLayer(eta0=12780, lam=1.0345, m=0.529, E_a=92.7e3, T_ref=573.15,
                    name="HT 2181", rho_melt=1400, rho_solid=1700)
    die = WireCoatingDie(R_wire=0.577e-3 / 2, R_die=1.25e-3 / 2, V_wire=0.5,
                         layers=[FEP, HT], L=0.10e-3)
    sol = die.solve(thicknesses=[0.20e-3, 0.05e-3], T=603.15)     # die temperature [K]
    print(sol.layer_max_stress, sol.draw_down_ratio, sol.final_thicknesses)

For sweeps and optimisers, pass the previous solution as a warm start. It is faster,
and the answer is the same as a cold start:
    sol2 = die.solve(thicknesses=[0.11e-3, 0.21e-3], warm_start=sol)

Units are SI throughout: metres, seconds, Pa, Pa s^n, m^3/s, kelvin, J/mol, kg/m^3.

Importing (from the folder containing this file):
    from wirecoating import WireCoatingDie, PowerLawLayer, CrossLayer
or copy the file to a name such as `wire_coating.py` and use a normal import.
'''

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.special import expit

__all__ = [
    "MAX_LAYERS",
    "PowerLawLayer",
    "CrossLayer",
    "WireCoatingDie",
    "CoatingSolution",
    "WireCoatingSolverError",
]

MAX_LAYERS = 5          # the solver accepts 1-5 concentric layers; the examples use 2
R_GAS = 8.314           # gas constant [J/(mol K)]

# Tanh-sinh (double-exponential) quadrature on [0, 1]. The integrands have an algebraic
# singularity at the zero-stress radius, |r - r*|^p with p = 1/n for a power law and
# p = 1 + m for the Cross model. Splitting each layer at r* and using these nodes
# handles that to round-off, for any p, without special weight functions.
TS_STEP = 1.0 / 8.0
TS_LIMIT = 3.0
_TS_T = np.arange(-TS_LIMIT, TS_LIMIT + 0.5 * TS_STEP, TS_STEP)
_TS_S = 0.5 * np.pi * np.sinh(_TS_T)
_TS_U = 1.0 / (1.0 + np.exp(-2.0 * _TS_S))                            # nodes in (0, 1)
_TS_W = 0.25 * np.pi * TS_STEP * np.cosh(_TS_T) / np.cosh(_TS_S) ** 2  # weights on [0, 1]

# One colour per layer, wire outward (colour-blind checked)
LAYER_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


class WireCoatingSolverError(RuntimeError):
    """The nonlinear solver did not converge for the requested inputs.

    Subclasses RuntimeError, so code written against the notebook's
    `except RuntimeError` still works. In an optimisation loop, catch this
    exception and treat the design point as infeasible."""


# ============================================================================
# Physics kernels
# ============================================================================

def shear_stress(r, G, C):
    """Shear stress tau_rz [Pa] at radius r [m].

    From the integrated z-momentum balance (Eq. 2.3):
        tau_rz(r) = -G r / 2 + C / r
    G = -dp/dz [Pa/m] and C [Pa m] are the two integration constants.
    They are the same in every layer, because r*tau_rz is continuous at
    each interface. Works for scalar r or numpy arrays."""
    return -0.5 * G * r + C / r


def zero_stress_radius(G, C):
    """Radius r* [m] where tau_rz = 0, or None if there is no such radius.

    Setting -G r/2 + C/r = 0 gives r*^2 = 2C/G, which is real only if C/G > 0.
    r* may lie inside or outside the annulus; the solver handles both cases.
    When r* is inside, it is the radius of maximum (or minimum) velocity."""
    if G != 0 and C / G > 0:
        return np.sqrt(2.0 * C / G)
    return None


def _segment_nodes(p, q):
    """Tanh-sinh nodes and weights on each segment [p, q] (p <= q), shapes (m, nodes)."""
    length = (q - p)[:, None]
    return p[:, None] + length * _TS_U[None, :], length * _TS_W[None, :]


def _invert_log_stress(fluid, s):
    """Shear rate magnitude gdot >= 0 such that |tau| = s, for any fluid with a
    monotone log_stress(x), x = ln gdot.

    Newton's method on ln|tau|(x) - ln s. The function is increasing and concave in x,
    so the iteration converges from the starting guess (the line through gdot = 1 1/s)."""
    s = np.asarray(s, dtype=float)
    out = np.zeros(s.shape)
    pos = s > 0
    if not np.any(pos):
        return out
    target = np.log(s[pos])
    L0, dL0 = fluid.log_stress(0.0)
    x = (target - L0) / dL0
    for _ in range(100):
        L, dL = fluid.log_stress(x)
        step = (L - target) / dL
        x = np.clip(x - step, -300.0, 300.0)
        if np.max(np.abs(step)) < 1e-14:
            break
    out[pos] = np.exp(x)
    return out


@dataclass(frozen=True)
class _PowerLawFluid:
    """Power law at one temperature: tau = K |gdot|^(n-1) gdot."""

    K: float
    n: float

    def log_stress(self, x):
        x = np.asarray(x, dtype=float)
        return np.log(self.K) + self.n * x, np.full(x.shape, self.n)

    def shear_rate(self, s):
        return (np.asarray(s, dtype=float) / self.K) ** (1.0 / self.n)

    def viscosity(self, gdot):
        return self.K * np.asarray(gdot, dtype=float) ** (self.n - 1.0)

    def equivalent_power_law(self, gd):
        return self.K, self.n

    def describe(self):
        return f"power law: K = {self.K:.4g} Pa s^n, n = {self.n:.4g}"


@dataclass(frozen=True)
class _CrossFluid:
    """Cross model at one temperature: eta = eta0 / (1 + (lam gdot)^m)."""

    eta0: float
    lam: float
    m: float

    def log_stress(self, x):
        x = np.asarray(x, dtype=float)
        z = self.m * (np.log(self.lam) + x)                  # m ln(lam gdot)
        L = np.log(self.eta0) + x - np.logaddexp(0.0, z)     # ln|tau|
        dL = 1.0 - self.m * expit(z)                         # d ln|tau| / d ln gdot, in [1-m, 1]
        return L, dL

    def shear_rate(self, s):
        return _invert_log_stress(self, s)

    def viscosity(self, gdot):
        return self.eta0 / (1.0 + (self.lam * np.asarray(gdot, dtype=float)) ** self.m)

    def equivalent_power_law(self, gd):
        """Power law that matches this fluid's stress and local slope at shear rate gd."""
        L, dL = self.log_stress(np.log(gd))
        return float(np.exp(L - dL * np.log(gd))), float(dL)

    def describe(self):
        return (f"Cross: eta0 = {self.eta0:.4g} Pa s, lam = {self.lam:.4g} s, "
                f"m = {self.m:.3f} (n = {1 - self.m:.3f} at high shear)")


@dataclass(frozen=True)
class _BlendedFluid:
    """Log-linear blend of a reference and a target fluid with weight t in [0, 1]:
        ln tau_t(gdot) = (1 - t) ln tau_ref(gdot) + t ln tau_target(gdot).
    Used by the continuation stage of the solver (never returned to the user)."""

    ref: _PowerLawFluid
    target: object
    t: float

    def log_stress(self, x):
        Lr, dr = self.ref.log_stress(x)
        Lt, dt = self.target.log_stress(x)
        return (1.0 - self.t) * Lr + self.t * Lt, (1.0 - self.t) * dr + self.t * dt

    def shear_rate(self, s):
        return _invert_log_stress(self, s)


def _blend(ref, target, t):
    """Fluid at blend weight t. A power-law target stays a power law (closed form)."""
    if isinstance(target, _PowerLawFluid):
        K = np.exp((1.0 - t) * np.log(ref.K) + t * np.log(target.K))
        n = (1.0 - t) * ref.n + t * target.n
        return _PowerLawFluid(float(K), float(n))
    return _BlendedFluid(ref, target, t)


def layer_integrals(a, b, G, C, fluid):
    """Integrals over one layer occupying a <= r <= b.

    Returns (du, dq) with
        du = u(b) - u(a)                        [m/s]
        dq = int_a^b du/dr (b^2 - r^2)/2 dr     [m^3/s per radian]
    Eq. 2.8 then gives the layer flow rate: Q = 2 pi [ u(a) (b^2 - a^2)/2 + dq ].

    The layer is split at the zero-stress radius (where the stress changes sign), and
    each piece is integrated with tanh-sinh quadrature. a and b may be scalars or arrays
    that broadcast together; array inputs are evaluated in one vectorised pass. Scalar
    inputs return a pair of floats."""
    a, b = np.broadcast_arrays(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    shape = a.shape
    a, b = a.ravel(), b.ravel()
    r_star = zero_stress_radius(G, C)
    # split point: r* if it lies inside the layer, otherwise an end point (one empty piece)
    c = a.copy() if r_star is None else np.clip(r_star, a, b)
    du = np.zeros(a.size)
    dq = np.zeros(a.size)
    for p, q in ((a, c), (c, b)):
        r, w = _segment_nodes(p, q)
        tau = shear_stress(r, G, C)
        g = np.sign(tau) * fluid.shear_rate(np.abs(tau))              # du/dr at the nodes
        du += np.sum(w * g, axis=1)
        dq += np.sum(w * g * 0.5 * (b[:, None] ** 2 - r ** 2), axis=1)
    du, dq = du.reshape(shape), dq.reshape(shape)
    if shape == ():
        return float(du), float(dq)
    return du, dq


def march_through_layers(G, C, radii, fluids, V_wire):
    """Velocity at every interface and flow rate of every layer, for given G, C and radii.

    Starts from no slip at the wire, u(R_w) = V_wire, and integrates outward
    one layer at a time. Velocity continuity at each interface is automatic,
    because each layer starts from the velocity at the end of the previous one.

    radii : array [R_w, r_1, ..., R_d] of length N + 1
    fluids : one constitutive fluid per layer, wire outward
    Returns (u_edges, Q_layers):
        u_edges[i]  = velocity at radii[i]                  [m/s]
        Q_layers[i] = volumetric flow rate of layer i       [m^3/s]"""
    N = len(fluids)
    u_edges = np.empty(N + 1)
    u_edges[0] = V_wire
    Q_layers = np.zeros(N)
    for i in range(N):
        a, b = radii[i], radii[i + 1]
        du, dq = layer_integrals(a, b, G, C, fluids[i])
        # Eq. 2.8:  Q_i = 2 pi [ u(a) (b^2 - a^2)/2 + int_a^b (u - u(a)) r dr ]
        Q_layers[i] = 2.0 * np.pi * (0.5 * (b * b - a * a) * u_edges[i] + dq)
        u_edges[i + 1] = u_edges[i] + du
    return u_edges, Q_layers


def G_and_C_from_wall_stresses(tau_wire, tau_die, R_wire, R_die):
    """Convert the two wall shear stresses into the constants G and C (Eq. 2.11).

    Writing Eq. 2.3 at r = R_w and at r = R_d gives two linear equations:
        R_w tau_w = -G R_w^2 / 2 + C
        R_d tau_d = -G R_d^2 / 2 + C"""
    G = 2.0 * (R_wire * tau_wire - R_die * tau_die) / (R_die ** 2 - R_wire ** 2)
    C = R_wire * tau_wire + 0.5 * G * R_wire ** 2
    return G, C


def _bracket_root(f, x0, step=1.0, grow=1.6, maxit=200):
    """Find a root of an INCREASING function f: widen [a, b] around x0 until f
    changes sign, then use Brent's method (scipy.optimize.brentq)."""
    a, b = x0 - step, x0 + step
    fa, fb = f(a), f(b)
    for _ in range(maxit):
        if fa <= 0 <= fb:
            break
        if fa > 0:                 # root lies to the left: move the bracket left
            b, fb = a, fa
            step *= grow
            a -= step
            fa = f(a)
        else:                      # root lies to the right: move the bracket right
            a, fa = b, fb
            step *= grow
            b += step
            fb = f(b)
    else:
        raise RuntimeError("could not bracket root")
    if fa == 0:
        return a
    if fb == 0:
        return b
    return brentq(f, a, b, xtol=1e-13, rtol=1e-15, maxiter=500)


def _uniform_fluid_solution(Q_frac, Q_total, ref, R_wire, R_die, V_wire, V_die, tau_ref):
    """Solve exactly for a single fluid (`ref`, a power law) filling the annulus, then place
    the layer interfaces where the cumulative flow from the wire reaches each layer's share.

    Uses two facts:
      * for fixed G, the die-wall velocity u(R_d) increases monotonically with C;
      * the total flow increases monotonically with G = -dp/dz.
    So two nested 1-D root finds, each guaranteed to converge, give G and C.
    Returns (tau_wire, tau_die, radii). Used only as the starting guess."""
    G_ref = tau_ref / (R_die - R_wire)

    def die_velocity_error(C, G):
        du, _ = layer_integrals(R_wire, R_die, G, C, ref)
        return V_wire + du - V_die

    def C_for_given_G(G):
        z = _bracket_root(lambda x: die_velocity_error(R_wire * tau_ref * np.sinh(x), G) / tau_ref, 0.0)
        return R_wire * tau_ref * np.sinh(z)

    def flow_error(zG):
        G = G_ref * np.sinh(zG)
        C = C_for_given_G(G)
        _, dq = layer_integrals(R_wire, R_die, G, C, ref)
        Q = 2.0 * np.pi * (0.5 * (R_die ** 2 - R_wire ** 2) * V_wire + dq)
        return (Q - Q_total) / Q_total

    G = G_ref * np.sinh(_bracket_root(flow_error, 0.0))
    C = C_for_given_G(G)

    def cumulative_flow(r):          # flow between the wire and radius r (scalar or array)
        r = np.asarray(r, dtype=float)
        _, dq = layer_integrals(R_wire, np.maximum(r, R_wire), G, C, ref)
        Q = 2.0 * np.pi * (0.5 * (r * r - R_wire ** 2) * V_wire + dq)
        return np.where(r <= R_wire, 0.0, Q)

    r_grid = np.linspace(R_wire, R_die, 2001)
    Q_grid = cumulative_flow(r_grid)                      # one vectorised pass over the grid
    radii, j0 = [R_wire], 0
    for target in np.cumsum(Q_frac)[:-1] * Q_total:
        # first grid interval at or after j0 that brackets the target and lies beyond
        # the last interface
        brackets = (Q_grid[j0:-1] - target) * (Q_grid[j0 + 1:] - target) <= 0
        hits = np.nonzero(brackets & (r_grid[j0 + 1:] > radii[-1]))[0]
        if hits.size == 0:
            raise RuntimeError("could not place interface")
        j = j0 + int(hits[0])
        radii.append(brentq(lambda v: float(cumulative_flow(v)) - target,
                            max(r_grid[j], radii[-1]), r_grid[j + 1], xtol=1e-15 * R_die))
        j0 = j
    radii.append(R_die)
    return shear_stress(R_wire, G, C), shear_stress(R_die, G, C), np.array(radii)


def _style(ax):
    """Light grid, no top or right frame lines."""
    ax.grid(alpha=0.25, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _draw_cross_section(ax, die, solution):
    """Concentric rings for the die cross-section: wire (grey), layers (colours), die body (dark)."""
    from matplotlib.patches import Circle

    Rw, Rd, s = die.R_wire, die.R_die, 1e3
    ax.add_patch(Circle((0, 0), Rd * s * 1.12, color="0.25"))
    for i in reversed(range(die.n_layers)):
        ax.add_patch(Circle((0, 0), solution.radii[i + 1] * s, color=LAYER_COLORS[i]))
    ax.add_patch(Circle((0, 0), Rw * s, color="#b0b0b0"))
    ax.set_xlim(-Rd * s * 1.2, Rd * s * 1.2)
    ax.set_ylim(-Rd * s * 1.2, Rd * s * 1.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Cross-section in the die", pad=14)
    ax.text(0, 0, "wire", ha="center", va="center", fontsize=8, color="0.15")


# ============================================================================
# Public classes
# ============================================================================

def _check_layer(layer):
    """Validate the optional temperature shift and density fields of a layer."""
    label = layer.name or "(unnamed)"
    if (layer.E_a is None) != (layer.T_ref is None):
        raise ValueError(f"Layer {label}: give both E_a and T_ref for a temperature shift, or neither.")
    if layer.T_ref is not None and not layer.T_ref > 0:
        raise ValueError(f"Layer {label}: T_ref must be positive [K]; got {layer.T_ref}.")
    if (layer.rho_melt is None) != (layer.rho_solid is None):
        raise ValueError(f"Layer {label}: give both rho_melt and rho_solid, or neither.")
    if layer.rho_melt is not None and not (layer.rho_melt > 0 and layer.rho_solid > 0):
        raise ValueError(f"Layer {label}: densities must be positive [kg/m^3].")


def _shift_factor(E_a, T_ref, T):
    """Arrhenius time-temperature shift factor a_T = eta0(T) / eta0(T_ref)."""
    return float(np.exp(E_a / R_GAS * (1.0 / T - 1.0 / T_ref)))


def _kelvin(T, name):
    """Check that a temperature was given for a layer that needs one."""
    if T is None:
        raise ValueError(f"The layer '{name}' has a temperature shift: pass the die temperature T [K] to solve().")
    if not 373.15 <= T <= 1000.0:          # melt processing: above 100 C, below 1000 K
        raise ValueError(f"T must be the die temperature in kelvin (373 to 1000 K); got {T}.")
    return float(T)


@dataclass(frozen=True)
class PowerLawLayer:
    """One concentric power-law melt layer in the die land.

    The constitutive law is tau = K |gamma_dot|^(n-1) gamma_dot, so the apparent
    viscosity is eta = K |gamma_dot|^(n-1).

    Time-temperature shift (optional): if E_a and T_ref are given, the layer at
    temperature T has consistency K a_T^n, with a_T = exp[E_a/R (1/T - 1/T_ref)].
    The index n does not change with temperature.

    Attributes
    ----------
    K : float
        Consistency at T_ref [Pa s^n]. For a Newtonian layer, K is the viscosity [Pa s].
    n : float
        Power-law index [-]. n < 1 is shear thinning, n = 1 is Newtonian.
    name : str
        Label used in reports and plots. Empty names are filled in by WireCoatingDie.
    E_a : float, optional
        Activation energy of the shift [J/mol].
    T_ref : float, optional
        Temperature at which K and n apply [K].
    rho_melt, rho_solid : float, optional
        Melt density and room-temperature solid density [kg/m^3]. Used for the
        shrinkage of the coating on cooling. If omitted, the layer does not shrink.
    """

    K: float
    n: float
    name: str = ""
    E_a: float | None = None
    T_ref: float | None = None
    rho_melt: float | None = None
    rho_solid: float | None = None

    def __post_init__(self):
        if not (self.K > 0 and self.n > 0):
            raise ValueError(
                f"Layer {self.name or '(unnamed)'}: K and n must be positive; got K={self.K}, n={self.n}.")
        _check_layer(self)

    def fluid(self, T=None):
        """Constitutive law at temperature T [K]. T is needed only if E_a is set."""
        if self.E_a is None:
            return _PowerLawFluid(self.K, self.n)
        a = _shift_factor(self.E_a, self.T_ref, _kelvin(T, self.name))
        return _PowerLawFluid(self.K * a ** self.n, self.n)


@dataclass(frozen=True)
class CrossLayer:
    """One concentric melt layer obeying the Cross model.

        eta(gdot) = eta0 / (1 + (lam gdot)^m),    tau = eta gdot

    At high shear the model is a power law with n = 1 - m. The Cross form has a
    Newtonian plateau at low shear, which the power law lacks.

    Time-temperature shift (optional): if E_a and T_ref are given, the layer at
    temperature T has eta0 -> a_T eta0 and lam -> a_T lam, with
    a_T = exp[E_a/R (1/T - 1/T_ref)]. The exponent m does not change with temperature.

    Attributes
    ----------
    eta0 : float
        Zero-shear viscosity at T_ref [Pa s].
    lam : float
        Cross time constant at T_ref [s]. Must be positive.
    m : float
        Cross exponent, 0 < m <= 1 (n = 1 - m).
    name : str
        Label used in reports and plots.
    E_a, T_ref : float, optional
        Activation energy [J/mol] and reference temperature [K] of the shift.
    rho_melt, rho_solid : float, optional
        Melt density and room-temperature solid density [kg/m^3] (shrinkage).
    """

    eta0: float
    lam: float
    m: float
    name: str = ""
    E_a: float | None = None
    T_ref: float | None = None
    rho_melt: float | None = None
    rho_solid: float | None = None

    def __post_init__(self):
        if not (self.eta0 > 0 and self.lam > 0 and 0 < self.m <= 1):
            raise ValueError(
                f"Layer {self.name or '(unnamed)'}: need eta0 > 0, lam > 0 and 0 < m <= 1; "
                f"got eta0={self.eta0}, lam={self.lam}, m={self.m}.")
        _check_layer(self)

    def fluid(self, T=None):
        """Constitutive law at temperature T [K]. T is needed only if E_a is set."""
        if self.E_a is None:
            return _CrossFluid(self.eta0, self.lam, self.m)
        a = _shift_factor(self.E_a, self.T_ref, _kelvin(T, self.name))
        return _CrossFluid(self.eta0 * a, self.lam * a, self.m)


@dataclass(frozen=True, eq=False)
class CoatingSolution:
    """Converged solution of one die-land calculation.

    Returned by WireCoatingDie.solve(). Derived quantities (pressure drop, wire
    force, backflow, draw-down, shrunk coating thickness, ...) are properties, so they
    can be used directly as objective or constraint values.

    Attributes
    ----------
    G : float
        Pressure gradient G = -dp/dz [Pa/m]. Positive means pressure falls toward the exit.
    C : float
        Stress constant in tau(r) = -G r/2 + C/r [Pa m].
    r0 : float or None
        Zero-stress radius r* [m], where the velocity has an extremum. None if it does not exist.
    tau_wire, tau_die : float
        Shear stress at the wire and at the die wall [Pa]. Negative means the melt
        next to the wire lags the moving wire (it resists the wire).
    radii : ndarray
        Interface radii inside the land [R_wire, r_1, ..., R_die] [m].
    u_edge : ndarray
        Axial velocity at each entry of `radii` [m/s].
    Q : ndarray
        Melt flow rate of each layer, wire outward [m^3/s].
    Q_calc : ndarray
        Flow rate of each layer recomputed from the solution [m^3/s].
    residual : float
        Root-sum-square of the scaled equation errors. About 1e-15 when converged.
    die : WireCoatingDie
        The die this solution belongs to.
    warm_started : bool
        True if the solution was reached directly from a `warm_start` guess. False if the
        solver used its full cold start (or no guess was given).
    T : float or None
        Die temperature [K], or None if none was needed.
    fluids : tuple
        The constitutive law of each layer at T, wire outward.
    """

    G: float
    C: float
    r0: float | None
    tau_wire: float
    tau_die: float
    radii: np.ndarray
    u_edge: np.ndarray
    Q: np.ndarray
    Q_calc: np.ndarray
    residual: float
    die: WireCoatingDie
    warm_started: bool = False
    T: float | None = None
    fluids: tuple = ()

    @property
    def dpdz(self) -> float:
        """Axial pressure gradient dp/dz [Pa/m] (equal to -G)."""
        return -self.G

    @property
    def pressure_drop(self) -> float | None:
        """Pressure drop p_in - p_out over the land [Pa], or None if the land length L was not given."""
        return None if self.die.L is None else self.G * self.die.L

    @property
    def wire_force_per_length(self) -> float:
        """Axial force of the melt on the wire per unit land length [N/m], 2 pi R_w tau_w.

        Negative means the melt resists the wire."""
        return 2.0 * np.pi * self.die.R_wire * self.tau_wire

    @property
    def wire_force(self) -> float | None:
        """Axial force of the melt on the wire over the land [N], 2 pi R_w L tau_w.

        Negative means the melt resists the wire. None if the land length L was not given."""
        return None if self.die.L is None else self.wire_force_per_length * self.die.L

    @property
    def velocity_range(self) -> tuple[float, float]:
        """(minimum, maximum) axial velocity in the land [m/s].

        Exact, not sampled. Within each layer the velocity is monotonic except at the
        zero-stress radius, so the extremes occur only at the wire, at an interface,
        at the die wall, or at r*. Those points are evaluated directly."""
        Rw, Rd = self.die.R_wire, self.die.R_die
        candidates = list(self.u_edge)                     # wire, interfaces, die wall
        if self.r0 is not None and Rw < self.r0 < Rd:
            i = int(np.searchsorted(self.radii, self.r0, side="right")) - 1   # layer containing r*
            du_to_r0, _ = layer_integrals(self.radii[i], self.r0, self.G, self.C, self.fluids[i])
            candidates.append(self.u_edge[i] + du_to_r0)
        return float(min(candidates)), float(max(candidates))

    @property
    def has_backflow(self) -> bool:
        """True if any melt in the land flows backward (negative axial velocity)."""
        umin, _ = self.velocity_range
        return umin < -self._velocity_tolerance()

    @property
    def regime(self) -> str:
        """Flow regime, from the sign of G (Section 2.9)."""
        scale = (abs(self.tau_wire) + abs(self.tau_die)) / (self.die.R_die - self.die.R_wire)
        if abs(self.G) < 1e-9 * scale:
            return "pure drag (no pressure gradient)"
        if self.G > 0:
            return "pressure-assisted (pressure falls toward the die exit)"
        return "pressure build-up (pressure rises toward the die exit)"

    @property
    def draw_down_ratio(self) -> float | None:
        """Draw-down ratio DDR = (die-land area) / (molten coating area after the die).

        The melt fills the land, so the land area is pi (R_die^2 - R_wire^2). The molten
        area after the die is sum(Q) / V_wire, so DDR = pi (R_die^2 - R_wire^2) V_wire / sum(Q).
        It depends on the coating design, not on the line speed. None if V_wire <= 0."""
        if self.die.V_wire <= 0:
            return None
        land_area = np.pi * (self.die.R_die ** 2 - self.die.R_wire ** 2)
        return float(land_area * self.die.V_wire / self.Q.sum())

    @property
    def layer_max_stress(self) -> np.ndarray:
        """Largest shear stress magnitude in each layer [Pa], wire outward.

        For melt fracture, compare with the critical wall shear stress of the layer.
        |tau| has no interior maximum (its only stationary point is a minimum), so the
        largest value in a layer is at one of its two boundaries."""
        G, C = self.G, self.C
        out = np.empty(len(self.fluids))
        for i in range(len(self.fluids)):
            a, b = self.radii[i], self.radii[i + 1]
            out[i] = max(abs(shear_stress(a, G, C)), abs(shear_stress(b, G, C)))
        return out

    @property
    def molten_radii(self) -> np.ndarray | None:
        """Outer radius of each layer just after the die, while still molten [m].

        From the mass balance Q_k = V_wire pi (R_k^2 - R_{k-1}^2). None if V_wire <= 0."""
        if self.die.V_wire <= 0:
            return None
        return np.sqrt(self.die.R_wire ** 2 + np.cumsum(self.Q) / (np.pi * self.die.V_wire))

    @property
    def molten_thicknesses(self) -> np.ndarray | None:
        """Thickness of each layer just after the die, while still molten [m], wire outward."""
        radii = self.molten_radii
        if radii is None:
            return None
        return np.diff(np.concatenate([[self.die.R_wire], radii]))

    @property
    def final_radii(self) -> np.ndarray | None:
        """Outer radius of each layer on the wire at room temperature [m].

        The mass of each layer is conserved on cooling, so its room-temperature area is
        (rho_melt / rho_solid) times its molten area. Neglects die swell. None if V_wire <= 0."""
        if self.die.V_wire <= 0:
            return None
        ratio = self.die._melt_to_solid_ratios()
        return np.sqrt(self.die.R_wire ** 2 + np.cumsum(self.Q * ratio) / (np.pi * self.die.V_wire))

    @property
    def final_thicknesses(self) -> np.ndarray | None:
        """Thickness of each layer on the wire at room temperature [m], wire outward.

        These are the thicknesses to compare with the product requirements. None if V_wire <= 0."""
        radii = self.final_radii
        if radii is None:
            return None
        return np.diff(np.concatenate([[self.die.R_wire], radii]))

    def _velocity_tolerance(self) -> float:
        """Round-off threshold for velocities [m/s], scaled to the flow."""
        die = self.die
        scale = max(abs(die.V_wire), abs(die.V_die),
                    self.Q.sum() / (np.pi * (die.R_die ** 2 - die.R_wire ** 2)))
        return 1e-9 * scale


@dataclass(frozen=True)
class WireCoatingDie:
    """Fully developed, isothermal, laminar flow of 1 to 5 concentric layers in the die
    land of a wire-coating crosshead. Each layer is a PowerLawLayer or a CrossLayer.

    The wire (radius R_wire, axial speed V_wire) runs along the axis through an annular
    land bounded by the die wall (radius R_die, speed V_die, normally 0). Layers are
    listed from the wire outward.

    Instances are immutable. To explore a design, build a new one, e.g.
    `dataclasses.replace(die, V_wire=2.0)`, and call `solve` on it.

    Parameters
    ----------
    R_wire : float
        Wire radius [m].
    R_die : float
        Die-land radius [m]. Must be greater than R_wire.
    V_wire : float
        Line speed, the axial speed of the wire [m/s].
    layers : sequence of PowerLawLayer or CrossLayer
        The layers from the wire outward (1 to 5).
    V_die : float, optional
        Die-wall speed [m/s]. Default 0.
    L : float, optional
        Die-land length [m]. Used only for the pressure-drop and wire-force totals.
        The flow solution does not depend on it.

    Notes
    -----
    Governing equations (notebook Section 2):
        momentum:       tau(r) = -G r / 2 + C / r,      G = -dp/dz
        constitutive:   du/dr = sign(tau) gdot(|tau|), where gdot is the inverse of
                        |tau| = eta(gdot) gdot (power law or Cross)
        flow rate:      Q_i = 2 pi [ (b^2 - a^2)/2 u(a) + int_a^b du/dr (b^2 - r^2)/2 dr ]
        mass balance:   Q_k = V_wire pi (R_k^2 - R_{k-1}^2) rho_solid/rho_melt
                        (the room-temperature radii R_k give the thicknesses)

    Examples
    --------
    >>> die = WireCoatingDie(0.5e-3, 1.0e-3, 1.0,
    ...                      [PowerLawLayer(4000, 0.40, "Insulation"),
    ...                       PowerLawLayer(12000, 0.30, "Jacket")], L=5e-3)
    >>> sol = die.solve(thicknesses=[0.10e-3, 0.20e-3])
    >>> round(sol.pressure_drop / 1e6, 2)      # MPa
    1.36
    """

    R_wire: float
    R_die: float
    V_wire: float
    layers: tuple
    V_die: float = 0.0
    L: float | None = None

    def __post_init__(self):
        layers = tuple(self.layers)
        if not 1 <= len(layers) <= MAX_LAYERS:
            raise ValueError(f"Number of layers must be between 1 and {MAX_LAYERS}; got {len(layers)}.")
        if not all(isinstance(layer, (PowerLawLayer, CrossLayer)) for layer in layers):
            raise TypeError("layers must be PowerLawLayer or CrossLayer objects.")
        if not 0 < self.R_wire < self.R_die:
            raise ValueError("Need 0 < R_wire < R_die.")
        if self.L is not None and self.L <= 0:
            raise ValueError("The land length L must be positive, or None.")
        # give unnamed layers a default label, wire outward
        named = tuple(layer if layer.name else replace(layer, name=f"Layer {i + 1}")
                      for i, layer in enumerate(layers))
        object.__setattr__(self, "layers", named)

    @property
    def n_layers(self) -> int:
        """Number of concentric layers."""
        return len(self.layers)

    def _melt_to_solid_ratios(self) -> np.ndarray:
        """rho_melt / rho_solid of each layer (1 for a layer without densities)."""
        return np.array([1.0 if layer.rho_melt is None else layer.rho_melt / layer.rho_solid
                         for layer in self.layers])

    def _fluids_at(self, T):
        """Constitutive law of every layer at the die temperature T [K] (wire outward)."""
        return tuple(layer.fluid(T) for layer in self.layers)

    def flow_rates_from_coating(self, thicknesses):
        """Melt flow rate of each layer [m^3/s] for the requested ROOM-TEMPERATURE thicknesses.

        Downstream of the die the coating solidifies and moves with the wire. Its mass is
        conserved, so each layer's melt volume rate is its room-temperature volume rate
        times rho_solid / rho_melt:
            Q_k = V_wire pi (R_k^2 - R_{k-1}^2) rho_solid,k / rho_melt,k
        where R_k are the room-temperature radii. Die swell and draw-down are neglected here.

        Parameters
        ----------
        thicknesses : sequence of float
            Room-temperature thickness of each layer [m], from the wire outward.

        Returns
        -------
        ndarray
            Melt flow rate of each layer [m^3/s], wire outward."""
        t = np.asarray(thicknesses, dtype=float)
        R = self.R_wire + np.concatenate([[0.0], np.cumsum(t)])
        solid_area = np.pi * np.diff(R ** 2)
        return self.V_wire * solid_area / self._melt_to_solid_ratios()

    def solve(self, *, thicknesses=None, flow_rates=None, T=None,
              warm_start: CoatingSolution | None = None) -> CoatingSolution:
        """Solve the die-land flow for a given coating split at die temperature T.

        Give exactly one of `thicknesses` or `flow_rates`.

        Parameters
        ----------
        thicknesses : sequence of float, optional
            Room-temperature thickness of each layer on the wire [m], wire outward.
            Converted to melt flow rates with the mass balance (see flow_rates_from_coating).
        flow_rates : sequence of float, optional
            Melt volumetric flow rate of each layer [m^3/s], wire outward.
        T : float, optional
            Die temperature [K]. Required if any layer has a temperature shift (E_a set).
        warm_start : CoatingSolution, optional
            A nearby converged solution for a die with the same number of layers, for
            example the previous point in a sweep or an optimiser's last iterate. The
            solver tries it first and skips the cold start if it converges. If it does
            not, the full cold start runs, so the answer does not depend on the guess.

        Returns
        -------
        CoatingSolution
            Pressure gradient, wall stresses, interface positions, velocities, the
            flow-rate check and the shrunk coating. Derived quantities are properties.

        Raises
        ------
        ValueError
            If the inputs are the wrong length, non-positive, or both or neither
            coating argument is given, if T is missing or not in kelvin, or if
            `warm_start` has a different number of layers.
        WireCoatingSolverError
            If the nonlinear solver does not converge.
        """
        if (thicknesses is None) == (flow_rates is None):
            raise ValueError("Give exactly one of thicknesses or flow_rates.")
        if warm_start is not None and warm_start.die.n_layers != self.n_layers:
            raise ValueError(f"warm_start has {warm_start.die.n_layers} layers; this die has {self.n_layers}.")
        if thicknesses is not None:
            t = np.asarray(thicknesses, dtype=float)
            if t.shape != (self.n_layers,) or np.any(t <= 0):
                raise ValueError(f"thicknesses must be {self.n_layers} positive values, wire outward.")
            if self.V_wire <= 0:
                raise ValueError("Converting thicknesses to flow rates needs V_wire > 0.")
            Q = self.flow_rates_from_coating(t)
        else:
            Q = np.asarray(flow_rates, dtype=float)
            if Q.shape != (self.n_layers,) or np.any(Q <= 0):
                raise ValueError(f"flow_rates must be {self.n_layers} positive values, wire outward.")
        fluids = self._fluids_at(T)                  # raises if T is needed and missing
        return self._solve_flow(Q, fluids, T, warm_start)

    def _solve_flow(self, Q: np.ndarray, fluids, T, warm_start: CoatingSolution | None = None) -> CoatingSolution:
        """Solve for given layer melt flow rates Q [m^3/s] (the core of the notebook's solve_annular).

        Unknowns (N + 1):   tau_wire, tau_die and the N - 1 interface radii
        Equations (N + 1):  Q_i(calculated) = Q_i(given) for every layer, and u(R_d) = V_die

        Strategy (Section 3.4): if a warm start is given, try it first (Stage 0). Otherwise,
        or if it fails, exact uniform-fluid start, then continuation from that fluid to the
        real layer fluids, then seeded multistart as a last resort."""
        N = self.n_layers
        Ri, Ro = self.R_wire, self.R_die
        V_in, V_out = self.V_wire, self.V_die

        # ---- scales used to make the residuals dimensionless -----------------
        A = np.pi * (Ro ** 2 - Ri ** 2)                    # annulus area
        Q_tot = Q.sum()
        U = max(Q_tot / A, abs(V_in), abs(V_out))           # velocity scale
        frac0 = Q / Q_tot                                   # flow fraction of each layer
        h = Ro - Ri                                         # gap
        gd_ref = (6.0 * Q_tot / A + abs(V_in - V_out)) / h  # typical shear rate
        # Reference fluid: the flow-weighted power-law equivalent of the layers at that shear rate.
        equiv = [fluid.equivalent_power_law(gd_ref) for fluid in fluids]
        Kref = np.exp(np.sum(frac0 * np.log([K for K, _ in equiv])))
        nref = np.sum(frac0 * np.array([n for _, n in equiv]))
        ref = _PowerLawFluid(float(Kref), float(nref))
        tau_ref = Kref * gd_ref ** nref                     # typical stress

        # ---- map the unknown vector x to physical quantities -----------------
        # x[0], x[1] : wall stresses through asinh, so that both signs and many
        #              decades are covered smoothly (tau = tau_ref * sinh(x)).
        # x[2:]      : "softmax" weights for the layer thicknesses; they always give
        #              positive thicknesses that add up to the gap.
        def unpack(x):
            tau_i = tau_ref * np.sinh(x[0])
            tau_o = tau_ref * np.sinh(x[1])
            G, C = G_and_C_from_wall_stresses(tau_i, tau_o, Ri, Ro)
            w = np.concatenate([x[2:], [0.0]])
            t = np.exp(w - w.max())
            t /= t.sum()
            radii = Ri + h * np.concatenate([[0.0], np.cumsum(t)])
            radii[-1] = Ro
            return G, C, radii

        def pack(tau_i, tau_o, radii):                      # inverse of unpack
            t = np.maximum(np.diff(radii), 1e-300)
            w = np.log(t[:-1] / t[-1]) if N > 1 else np.array([])
            return np.concatenate([[np.arcsinh(tau_i / tau_ref), np.arcsinh(tau_o / tau_ref)], w])

        # ---- residuals: zero when all N + 1 equations are satisfied ----------
        def make_residuals(fls):
            def residuals(x):
                G, C, radii = unpack(x)
                with np.errstate(all="ignore"):
                    u, Qc = march_through_layers(G, C, radii, fls, V_in)
                    r = np.concatenate([(Qc - Q) / Q,               # N flow-rate errors (relative)
                                        [(u[-1] - V_out) / U]])     # die-wall velocity error
                return np.where(np.isfinite(r), r, 1e6)
            return residuals

        lb = np.concatenate([[-120.0, -120.0], -40 * np.ones(N - 1)])
        ub = -lb

        def run(x0, fls, nfev=400):
            """One nonlinear least-squares solve (trust-region reflective)."""
            x0 = np.clip(x0, lb + 1e-9, ub - 1e-9)
            return least_squares(make_residuals(fls), x0, bounds=(lb, ub), method="trf",
                                 xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=nfev * (N + 2),
                                 x_scale="jac")

        converged = 1e-20                                    # cost = 0.5 * sum(residual^2)

        # ---- Stage 0: warm start from a nearby solution, if one was given ----
        best = None
        warm_started = False
        if warm_start is not None:
            x_warm = pack(warm_start.tau_wire, warm_start.tau_die, warm_start.radii)
            if np.all(np.isfinite(x_warm)):                  # a non-finite guess goes straight to the cold start
                trial = run(x_warm, fluids)
                if trial.cost <= converged:
                    best, warm_started = trial, True

        if best is None:
            # ---- Stage 1: exact uniform-fluid start, then a direct solve ---------
            try:
                x_ref = pack(*_uniform_fluid_solution(frac0, Q_tot, ref, Ri, Ro, V_in, V_out, tau_ref))
            except RuntimeError:
                x_ref = np.concatenate([[0.0, 0.0], np.log(frac0[:-1] / frac0[-1]) if N > 1 else []])
            best = run(x_ref, fluids)

            # ---- Stage 2: continuation (morph the reference fluid into the layer fluids) --
            if best.cost > converged:
                x = run(x_ref, [ref] * N).x
                lam, step = 0.0, 0.1
                while lam < 1.0 and step > 1e-4:
                    lt = min(1.0, lam + step)
                    trial = run(x, [_blend(ref, f, lt) for f in fluids])
                    if trial.cost < 1e-16:                   # step accepted: go on, slightly bigger
                        x, lam, step = trial.x, lt, min(step * 1.5, 0.3)
                    else:                                    # step failed: halve it
                        step /= 2
                best = run(x, fluids)

        # ---- Stage 3: seeded multi-start (last resort; seeded, so reproducible) --
        if best.cost > converged:
            rng = np.random.default_rng(12345)
            for _batch in range(4):
                screened = []
                for _ in range(100):
                    if rng.random() < 0.5:
                        x0 = np.concatenate([rng.uniform(-5, 5, 2), rng.uniform(-3, 3, N - 1)])
                    else:
                        x0 = x_ref + rng.normal(0, 1.0, N + 1)
                    screen = run(x0, fluids, nfev=25)         # cheap screening
                    screened.append((screen.cost, screen.x))
                screened.sort(key=lambda item: item[0])
                for _cost, x0 in screened[:6]:                # polish the 6 best
                    trial = run(x0, fluids)
                    if trial.cost < best.cost:
                        best = trial
                    if best.cost < converged:
                        break
                if best.cost < converged:
                    break
        if best.cost > 1e-15:
            raise WireCoatingSolverError(
                f"Solver did not converge (residual {np.sqrt(2 * best.cost):.2e}) "
                f"for flow rates {Q} m^3/s.")

        # ---- package the result ---------------------------------------------
        G, C, radii = unpack(best.x)
        u, Qc = march_through_layers(G, C, radii, fluids, V_in)
        return CoatingSolution(
            G=float(G), C=float(C), r0=zero_stress_radius(G, C),
            tau_wire=float(shear_stress(Ri, G, C)), tau_die=float(shear_stress(Ro, G, C)),
            radii=radii, u_edge=u, Q=Q, Q_calc=Qc,
            residual=float(np.sqrt(2 * best.cost)), die=self, warm_started=warm_started,
            T=None if T is None else float(T), fluids=tuple(fluids))

    def profiles(self, solution: CoatingSolution, points_per_layer: int = 200) -> list[dict]:
        """Velocity, shear stress, shear rate and viscosity on a radial grid, per layer.

        Parameters
        ----------
        solution : CoatingSolution
            A converged solution from `solve`.
        points_per_layer : int
            Number of radial points in each layer.

        Returns
        -------
        list of dict
            One dict per layer, with arrays 'r' [m], 'u' [m/s], 'tau' [Pa], 'gdot' [1/s]
            and 'eta' [Pa s]. For a power law with n < 1 the viscosity is infinite at the
            zero-stress radius (a property of the power-law model, not of the melt). The
            Cross viscosity there is eta0."""
        out = []
        for i in range(self.n_layers):
            fluid = solution.fluids[i]
            a, b = solution.radii[i], solution.radii[i + 1]
            r = np.linspace(a, b, points_per_layer)
            du, _ = layer_integrals(a, r, solution.G, solution.C, fluid)   # u(r) - u(a), all points at once
            u = solution.u_edge[i] + du
            tau = shear_stress(r, solution.G, solution.C)
            gdot = np.sign(tau) * fluid.shear_rate(np.abs(tau))
            with np.errstate(divide="ignore", invalid="ignore"):
                eta = fluid.viscosity(np.abs(gdot))
            out.append(dict(r=r, u=u, tau=tau, gdot=gdot, eta=eta))
        return out

    def report(self, solution: CoatingSolution) -> None:
        """Print the solution in practical units.

        Shows the die temperature, the pressure gradient and the drop over the land, the
        flow regime, the wall stresses, the wire force, the zero-stress radius, the velocity
        range (with a backflow flag), the draw-down ratio, the largest stress in each layer,
        the solver residual, the constitutive law of each layer, and a per-layer table with
        flow rates, interface radii and the coating thicknesses after the die and at room
        temperature."""
        N = self.n_layers
        Rw, Rd, Vw = self.R_wire, self.R_die, self.V_wire
        r = solution.radii
        umin, umax = solution.velocity_range
        if abs(umin) < solution._velocity_tolerance():       # round-off at the stationary die wall
            umin = 0.0

        head = (f"Wire radius {Rw * 1e3:.4g} mm, die radius {Rd * 1e3:.4g} mm, wire speed {Vw:.4g} m/s"
                + (f", die wall speed {self.V_die:.4g} m/s" if self.V_die else ""))
        if solution.T is not None:
            head += f", die temperature {solution.T:.2f} K ({solution.T - 273.15:.1f} degC)"
        print(head)
        print(f"Pressure gradient  -dp/dz = {solution.G:+.4e} Pa/m = {solution.G * 1e-6:+.4f} MPa/m   "
              f"-> {solution.regime}")
        if self.L:
            print(f"Pressure change over land length {self.L * 1e3:.3g} mm: "
                  f"p_in - p_out = {solution.pressure_drop * 1e-6:+.3f} MPa")
        print(f"Wall shear stress: wire {solution.tau_wire:+.4e} Pa, die {solution.tau_die:+.4e} Pa")
        Fp = solution.wire_force_per_length
        print(f"Force of the melt on the wire: {Fp:+.4e} N per m of land"
              + (f"  ({solution.wire_force:+.4g} N over L)" if self.L else "")
              + ("  (resists the wire)" if Fp < 0 else ""))
        if solution.r0 is not None and Rw < solution.r0 < Rd:
            print(f"Zero-stress radius (velocity extremum) r* = {solution.r0 * 1e3:.4f} mm")
        else:
            print("Zero-stress radius: outside the annulus (velocity monotonic from wire to die)")
        print(f"Velocity range {umin:.4g} to {umax:.4g} m/s"
              + ("   ** backflow present **" if solution.has_backflow else ""))
        ddr = solution.draw_down_ratio
        if ddr is not None:
            print(f"Draw-down ratio DDR = {ddr:.3f}  (die-land area / molten coating area)")
        maxs = solution.layer_max_stress
        print("Largest shear stress per layer: "
              + ", ".join(f"{layer.name} {s:.4e} Pa" for layer, s in zip(self.layers, maxs)))
        print(f"Solver residual = {solution.residual:.1e}\n")

        for layer, fluid in zip(self.layers, solution.fluids):
            print(f"  {layer.name:<12} {fluid.describe()}")
        print()

        final = solution.final_thicknesses is not None
        hdr = (f"{'':<12}{'Q [m3/s]':>11}{'Q frac':>8}{'r_land [mm]':>13}{'t_land [mm]':>13}"
               f"{'u_out [m/s]':>13}{'max|tau| [Pa]':>16}" + (f"{'t_melt [mm]':>13}{'t_RT [mm]':>12}" if final else ""))
        print(hdr)
        print("-" * len(hdr))
        melt_t = solution.molten_thicknesses
        rt_t = solution.final_thicknesses
        for i, layer in enumerate(self.layers):
            u_out = self.V_die if i == N - 1 else solution.u_edge[i + 1]
            row = (f"{layer.name:<12}{solution.Q[i]:>11.3e}{solution.Q[i] / solution.Q.sum():>8.4f}"
                   f"{r[i + 1] * 1e3:>13.4f}{(r[i + 1] - r[i]) * 1e3:>13.4f}{u_out:>13.4g}{maxs[i]:>16.4e}")
            if final:
                row += f"{melt_t[i] * 1e3:>13.4f}{rt_t[i] * 1e3:>12.4f}"
            print(row)
        if final:
            print(f"\nRoom-temperature coating radius {solution.final_radii[-1] * 1e3:.4f} mm "
                  f"(molten {solution.molten_radii[-1] * 1e3:.4f} mm, die land {Rd * 1e3:.4g} mm); "
                  f"die swell neglected")

    def plot_profiles(self, solution: CoatingSolution, title: str | None = None, show: bool = True):
        """Plot the cross-section and the velocity, shear-rate and viscosity profiles.

        The wire is at the bottom of each profile plot and the die at the top. Dashed
        lines mark the interfaces, and a dotted line marks r* when it lies inside the gap.

        Parameters
        ----------
        solution : CoatingSolution
            A converged solution from `solve`.
        title : str, optional
            Figure title. A default is built from the pressure gradient if omitted.
        show : bool
            Call plt.show() before returning. Set False in scripts and optimisation loops.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker

        N = self.n_layers
        names = [layer.name for layer in self.layers]
        pr = self.profiles(solution, 300)
        Rw, Rd = self.R_wire, self.R_die
        has_r_star = solution.r0 is not None and Rw < solution.r0 < Rd

        fig = plt.figure(figsize=(15, 5.2))
        gs = fig.add_gridspec(1, 4, width_ratios=[0.9, 1.5, 1, 1])
        _draw_cross_section(fig.add_subplot(gs[0]), self, solution)
        axes = [fig.add_subplot(gs[1])]
        axes += [fig.add_subplot(gs[k], sharey=axes[0]) for k in (2, 3)]
        for ax in axes:                                      # shading, interfaces, r*, grid
            for i in range(N):
                ax.axhspan(solution.radii[i] * 1e3, solution.radii[i + 1] * 1e3,
                           color=LAYER_COLORS[i], alpha=0.08, lw=0)
            for ri in solution.radii[1:-1]:
                ax.axhline(ri * 1e3, color="0.45", lw=0.8, ls="--")
            if has_r_star:
                ax.axhline(solution.r0 * 1e3, color="0.2", lw=0.8, ls=":")
            _style(ax)
            ax.set_ylim(Rw * 1e3, Rd * 1e3)
        for ax in axes[1:]:
            plt.setp(ax.get_yticklabels(), visible=False)

        ax = axes[0]                                         # velocity
        for i, p in enumerate(pr):
            ax.plot(p["u"], p["r"] * 1e3, color=LAYER_COLORS[i], lw=2.2, label=names[i])
        umin = min(0.0, min(p["u"].min() for p in pr))
        umax = max(p["u"].max() for p in pr)
        pad = 0.05 * (umax - umin)
        ax.set_xlim(umin - (pad if umin < 0 else 0), umax + pad)
        ax.axvline(0, color="0.3", lw=0.8)
        ax.plot(ax.get_xlim(), [Rw * 1e3] * 2, color="#808080", lw=5, solid_capstyle="butt", clip_on=False)
        ax.plot(ax.get_xlim(), [Rd * 1e3] * 2, color="0.15", lw=4, solid_capstyle="butt", clip_on=False)
        ax.set_xlabel("Axial velocity u [m/s]")
        ax.set_ylabel("Radius r [mm]  (wire at bottom, die at top)")
        ax.set_title(f"Velocity  (wire {self.V_wire:.3g} m/s)", pad=14)
        ax.legend(loc="best", fontsize=9, frameon=False)

        ax = axes[1]                                         # shear rate
        for i, p in enumerate(pr):
            ax.plot(p["gdot"], p["r"] * 1e3, color=LAYER_COLORS[i], lw=2)
        ax.axvline(0, color="0.3", lw=0.8)
        ax.set_xlabel("Shear rate du/dr [1/s]")
        ax.set_title("Shear rate", pad=14)

        ax = axes[2]                                         # viscosity (log scale)
        for i, p in enumerate(pr):
            ok = np.isfinite(p["eta"])
            ax.plot(p["eta"][ok], p["r"][ok] * 1e3, color=LAYER_COLORS[i], lw=2)
        if has_r_star:                                       # viscosity is infinite at r*: cap the axis
            far = np.concatenate([p["eta"][(np.abs(p["r"] - solution.r0) > 0.05 * (Rd - Rw))
                                           & np.isfinite(p["eta"])] for p in pr])
            if far.size:
                ax.set_xlim(far.min() / 1.5, far.max() * 3)
        ax.set_xscale("log")
        lo, hi = ax.get_xlim()
        if np.log10(hi / lo) < 1.5:
            ax.xaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1, 2, 5)))
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.set_xlabel(r"Viscosity $\eta$ [Pa s]")
        ax.set_title("Local viscosity", pad=14)

        fig.suptitle(title or f"{N}-layer annular flow (-dp/dz = {solution.G * 1e-6:+.3g} MPa/m; "
                     "dashed = interface, dotted = zero stress)", fontsize=12)
        fig.tight_layout()
        if show:
            plt.show()
        return fig
