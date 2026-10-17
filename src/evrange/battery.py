"""Battery pack equivalent-circuit model: OCV(SOC) + R0 + one RC pair.

Terminal power  P = (V_oc - V_rc) I - R0 I^2   (I > 0 discharging)
  -> I = [(V_oc - V_rc) - sqrt((V_oc - V_rc)^2 - 4 R0 P)] / (2 R0)
RC update       V_rc[k+1] = V_rc a + R1 I (1 - a),  a = exp(-dt / tau)
SOC update      SOC[k+1]  = SOC - I dt / (3600 Q)
Current is limited so the terminal voltage stays within [V_min, V_max].
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class BatteryPack:
    cells_series: int
    cells_parallel: int
    cell_capacity_ah: float
    cell_r0_ohm: float
    cell_r1_ohm: float
    cell_tau_s: float
    cell_v_min: float
    cell_v_max: float
    soc_min: float
    soc_max: float
    ocv_soc: tuple = field(default=())
    ocv_v: tuple = field(default=())

    def __post_init__(self) -> None:
        object.__setattr__(self, "ocv_soc", tuple(float(x) for x in self.ocv_soc))
        object.__setattr__(self, "ocv_v", tuple(float(x) for x in self.ocv_v))
        if self.cells_series < 1 or self.cells_parallel < 1:
            raise ValueError("cells_series and cells_parallel must be >= 1")
        if min(self.cell_capacity_ah, self.cell_r0_ohm, self.cell_tau_s) <= 0 or self.cell_r1_ohm < 0:
            raise ValueError("capacity, R0 and tau must be > 0; R1 >= 0")
        if not 0 <= self.soc_min < self.soc_max <= 1:
            raise ValueError("need 0 <= soc_min < soc_max <= 1")
        if len(self.ocv_soc) < 2 or len(self.ocv_soc) != len(self.ocv_v):
            raise ValueError("ocv_soc and ocv_v must have equal length >= 2")
        if np.any(np.diff(self.ocv_soc) <= 0) or np.any(np.diff(self.ocv_v) < 0):
            raise ValueError("ocv_soc must increase and ocv_v must not decrease")
        if not self.cell_v_min < self.cell_v_max:
            raise ValueError("cell_v_min must be < cell_v_max")

    # ---- pack-level parameters -------------------------------------------------
    @property
    def capacity_ah(self) -> float:
        return self.cell_capacity_ah * self.cells_parallel

    @property
    def r0_ohm(self) -> float:
        return self.cell_r0_ohm * self.cells_series / self.cells_parallel

    @property
    def r1_ohm(self) -> float:
        return self.cell_r1_ohm * self.cells_series / self.cells_parallel

    @property
    def v_min(self) -> float:
        return self.cell_v_min * self.cells_series

    @property
    def v_max(self) -> float:
        return self.cell_v_max * self.cells_series

    def ocv(self, soc: np.ndarray | float) -> np.ndarray | float:
        return np.interp(soc, self.ocv_soc, self.ocv_v) * self.cells_series

    def energy_kwh(self, soc_low: float, soc_high: float, n: int = 2001) -> float:
        """Open-circuit energy between two SOCs: Q * integral of OCV dSOC."""
        s = np.linspace(soc_low, soc_high, n)
        v = self.ocv(s)
        return float(np.sum(0.5 * (v[1:] + v[:-1]) * np.diff(s)) * self.capacity_ah / 1000.0)

    @property
    def usable_energy_kwh(self) -> float:
        return self.energy_kwh(self.soc_min, self.soc_max)

    @property
    def nominal_energy_kwh(self) -> float:
        return self.energy_kwh(0.0, 1.0)

    # ---- time simulation ------------------------------------------------------
    def run(self, power_w: np.ndarray, time_s: np.ndarray, soc0: float, v_rc0: float = 0.0) -> dict:
        """Simulate the pack for a terminal power profile (+ = discharge).

        Power at sample k is held over [t_k, t_k+1]. Returns arrays of length N
        (states at each sample) and per-interval quantities of length N-1.
        """
        p_req = np.asarray(power_w, dtype=float)
        t = np.asarray(time_s, dtype=float)
        n = len(t)
        dt = np.diff(t)

        soc = np.empty(n)
        v_rc = np.empty(n)
        current = np.zeros(n - 1)
        v_term = np.zeros(n - 1)
        p_actual = np.zeros(n - 1)
        limited = np.zeros(n - 1, dtype=bool)

        soc[0], v_rc[0] = soc0, v_rc0
        q_as = self.capacity_ah * 3600.0
        r0, r1, tau = self.r0_ohm, self.r1_ohm, self.cell_tau_s

        for k in range(n - 1):
            e = float(self.ocv(soc[k])) - v_rc[k]
            p = p_req[k]
            disc = e * e - 4.0 * r0 * p
            if disc < 0:                      # more power than the pack can deliver
                disc, limited[k] = 0.0, True
            i = (e - np.sqrt(disc)) / (2.0 * r0)

            i_max = (e - self.v_min) / r0     # keep terminal voltage >= V_min
            i_min = (e - self.v_max) / r0     # keep terminal voltage <= V_max
            if i > i_max:
                i, limited[k] = i_max, True
            elif i < i_min:
                i, limited[k] = i_min, True

            a = np.exp(-dt[k] / tau)
            current[k] = i
            v_term[k] = e - r0 * i
            p_actual[k] = v_term[k] * i
            v_rc[k + 1] = v_rc[k] * a + r1 * i * (1.0 - a)
            soc[k + 1] = soc[k] - i * dt[k] / q_as

        ocv_mid = self.ocv(soc[:-1])
        return {
            "soc": soc,
            "v_rc": v_rc,
            "current_a": current,
            "v_terminal": v_term,
            "p_terminal_w": p_actual,
            "p_chemical_w": ocv_mid * current,
            "p_loss_w": ocv_mid * current - p_actual,
            "limited": limited,
        }
