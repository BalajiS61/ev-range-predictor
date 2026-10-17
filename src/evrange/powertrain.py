"""Single-speed drivetrain and PMSM motor/inverter with an analytical loss model.

Motor + inverter losses (Larminie & Lowry form):
    P_loss = k_c T^2 + k_i w + k_w w^3 + C      (only while the motor operates)
Motoring:   P_elec = P_shaft + P_loss
Generating: P_elec = P_shaft + P_loss  (P_shaft < 0, losses reduce recovered power)
Torque envelope: |T| <= min(T_peak, P_peak / w)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RPM_TO_RADS = 2.0 * np.pi / 60.0


@dataclass(frozen=True)
class Drivetrain:
    gear_ratio: float
    wheel_radius_m: float
    gear_efficiency: float

    def __post_init__(self) -> None:
        if self.gear_ratio <= 0 or self.wheel_radius_m <= 0:
            raise ValueError("gear_ratio and wheel_radius_m must be > 0")
        if not 0 < self.gear_efficiency <= 1:
            raise ValueError("gear_efficiency must be in (0, 1]")

    def motor_speed_rads(self, vehicle_speed_ms: np.ndarray) -> np.ndarray:
        return np.asarray(vehicle_speed_ms) / self.wheel_radius_m * self.gear_ratio

    def shaft_power(self, wheel_power_w: np.ndarray) -> np.ndarray:
        """Motor shaft power needed for a wheel power (losses both directions)."""
        p = np.asarray(wheel_power_w, dtype=float)
        return np.where(p >= 0, p / self.gear_efficiency, p * self.gear_efficiency)


@dataclass(frozen=True)
class Motor:
    peak_torque_nm: float
    peak_power_w: float
    max_speed_rpm: float
    loss_copper: float
    loss_iron: float
    loss_windage: float
    loss_constant_w: float

    def __post_init__(self) -> None:
        if min(self.peak_torque_nm, self.peak_power_w, self.max_speed_rpm) <= 0:
            raise ValueError("peak_torque_nm, peak_power_w and max_speed_rpm must be > 0")
        if min(self.loss_copper, self.loss_iron, self.loss_windage, self.loss_constant_w) < 0:
            raise ValueError("loss coefficients must be >= 0")

    @property
    def max_speed_rads(self) -> float:
        return self.max_speed_rpm * RPM_TO_RADS

    def max_torque(self, speed_rads: np.ndarray) -> np.ndarray:
        """Continuous torque envelope: constant torque, then constant power."""
        w = np.maximum(np.asarray(speed_rads, dtype=float), 1e-9)
        t = np.minimum(self.peak_torque_nm, self.peak_power_w / w)
        return np.where(np.asarray(speed_rads) > self.max_speed_rads, 0.0, t)

    def losses(self, torque_nm: np.ndarray, speed_rads: np.ndarray) -> np.ndarray:
        t = np.asarray(torque_nm, dtype=float)
        w = np.abs(np.asarray(speed_rads, dtype=float))
        operating = (np.abs(t) > 1e-9) & (w > 1e-9)
        p = self.loss_copper * t**2 + self.loss_iron * w + self.loss_windage * w**3 + self.loss_constant_w
        return np.where(operating, p, 0.0)

    def electrical_power(self, torque_nm: np.ndarray, speed_rads: np.ndarray) -> np.ndarray:
        shaft = np.asarray(torque_nm) * np.asarray(speed_rads)
        return shaft + self.losses(torque_nm, speed_rads)

    def efficiency(self, torque_nm: np.ndarray, speed_rads: np.ndarray) -> np.ndarray:
        """Motoring: P_shaft/P_elec. Generating: P_elec/P_shaft. NaN where idle."""
        t = np.asarray(torque_nm, dtype=float)
        w = np.asarray(speed_rads, dtype=float)
        shaft = t * w
        elec = self.electrical_power(t, w)
        with np.errstate(divide="ignore", invalid="ignore"):
            eff = np.where(shaft > 0, shaft / elec, elec / shaft)
        eff = np.where(np.abs(shaft) < 1e-6, np.nan, eff)
        return np.clip(eff, 0.0, 1.0)

    def efficiency_map(self, n_speed: int = 120, n_torque: int = 120):
        """Grid (speed_rpm, torque_nm, efficiency) over the motoring envelope."""
        speed_rpm = np.linspace(1.0, self.max_speed_rpm, n_speed)
        torque = np.linspace(1.0, self.peak_torque_nm, n_torque)
        S, T = np.meshgrid(speed_rpm, torque)
        W = S * RPM_TO_RADS
        eff = self.efficiency(T, W)
        eff = np.where(T <= self.max_torque(W) + 1e-9, eff, np.nan)
        return S, T, eff
