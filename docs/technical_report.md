# EV Energy Consumption and Range Prediction: Physics Model and ML Surrogate

**Balaji S** — Technical report, v1.0.0, October 2026
Code: <https://github.com/BalajiS61/ev-range-predictor>

## Abstract

A backward-facing energy model of a compact battery-electric vehicle was built in Python, covering road load, regenerative braking, gearbox, a PMSM + inverter loss map, a battery equivalent-circuit model and cabin climate control. Range is obtained from a full-depletion test in which a drive cycle is repeated until the usable state-of-charge (SOC) window is used up. Official EPA drive cycles reproduce their published distances within 0.7%, and the model's energy balance closes to machine precision. With generic compact-EV parameters, range at 25 °C is 397–423 km across UDDS, HWFET, NEDC and MIDC. Sensitivity studies show that winter heating with a resistive heater cuts city range by 55%, versus 32% at 120 km/h, and that hills with zero net elevation still cost up to 16% at an 8% grade. A gradient-boosting surrogate trained on 3,000 physics-labelled trips predicts range with 1.4% mean absolute percentage error; an extrapolation test shows that tree models fail on driving patterns absent from training, while a physics-feature linear model degrades more gracefully on some of them.

## 1. Introduction

Published EV range figures come from standardised laboratory cycles and often differ from what drivers achieve, mainly because of ambient temperature, speed and load. The aim of this project was a transparent model that (i) reproduces cycle-based consumption from first principles, (ii) quantifies the main real-world effects one at a time and in combination, and (iii) provides a fast surrogate suitable for interactive use.

## 2. Model

### 2.1 Road load

The tractive force at the wheels is

F = ½ ρ C<sub>d</sub> A v² + m g C<sub>rr</sub> cos θ + m g sin θ + m (1 + δ) dv/dt,

with total mass m = 1,880 kg (curb + driver), C<sub>d</sub> = 0.28, A = 2.30 m², C<sub>rr</sub> = 0.009 and rotational-inertia factor δ = 0.05. Rolling resistance is applied only while moving. Acceleration is obtained by central differences of the 1 Hz speed trace.

### 2.2 Braking and drivetrain

Negative wheel power is shared between regeneration (80%, front-axle share) and friction brakes. Regeneration fades linearly to zero between 10 and 5 km/h and is limited to 70 kW electrical. A single-speed 9:1 reduction with 97% efficiency in both directions connects the wheels (r = 0.33 m) to the motor.

### 2.3 Motor and inverter

Combined PMSM and inverter losses follow the analytical form of Larminie and Lowry,

P<sub>loss</sub> = k<sub>c</sub> T² + k<sub>i</sub> ω + k<sub>w</sub> ω³ + C,

with k<sub>c</sub> = 0.12 W/(N·m)², k<sub>i</sub> = 1.0 W/(rad/s), k<sub>w</sub> = 3×10⁻⁶ W/(rad/s)³ and C = 500 W, applied only while the motor operates. This gives a peak efficiency of 95.6%. Torque is limited to |T| ≤ min(310 N·m, 150 kW / ω) up to 16,000 rpm. In generating mode the regen torque limit is the root of the loss equation at which electrical power equals −70 kW.

### 2.4 Battery pack

The 96s3p pack of 58 Ah NMC cells (174 Ah, 62.5 kWh nominal) uses an OCV(SOC) table, a series resistance R<sub>0</sub> (32 mΩ pack) and one RC pair (19 mΩ, τ = 30 s). For a requested terminal power P the current is

I = [(V<sub>oc</sub> − V<sub>RC</sub>) − √((V<sub>oc</sub> − V<sub>RC</sub>)² − 4 R<sub>0</sub> P)] / (2 R<sub>0</sub>),

clipped so that the terminal voltage stays within 2.8–4.2 V per cell. The interval-average power is used for each 1 s step, consistent with trapezoidal integration of the other energy terms. The usable window is SOC 5–97%, giving 57.8 kWh of open-circuit energy.

### 2.5 Ambient conditions

Air density follows the ideal-gas law at sea-level pressure. Cabin HVAC electrical power comes from a steady-state heat balance with UA = 150 W/K, a 21 °C set-point with a ±3 °C deadband, a 300 W solar load when cooling, heater COP 1.0 (PTC) or 2.5 (heat pump), cooling COP 2.5 and a 6 kW cap. The battery is assumed to be at ambient temperature: R<sub>0</sub> and R<sub>1</sub> scale with an Arrhenius factor (E<sub>a</sub> = 25 kJ/mol, reference 25 °C), and available capacity falls to 0.88 at −10 °C and 0.80 at −20 °C. Battery ageing scales capacity by the state of health (SOH) and resistance by a growth factor.

### 2.6 Range test

The cycle is repeated from SOC 97% until SOC reaches 5%, with the battery state (SOC, RC voltage) carried between repetitions. The final partial cycle is interpolated to the exact SOC crossing.

## 3. Verification and validation

- **Physics checks.** At constant speed the simulated consumption equals the hand calculation to 0.000%. A lossless vehicle accelerating and braking returns to zero net energy. The energy balance — cell energy equals the sum of aero, rolling, grade, net inertia, friction-brake, gearbox, motor, battery and auxiliary terms — closes to a relative error of 1e-6 in the test suite (about 1e-14 in practice).
- **Drive cycles.** Computed distances: UDDS 11.990 km (published 11.99), HWFET 16.507 km (16.51), NEDC 10.931 km (11.007). MIDC, built as 4 × ECE-15 + low-powered EUDC, gives 10.586 km over 1,180 s.
- **Battery.** A 1C constant-current discharge for 30 minutes removes exactly 50% SOC; impossible charge and discharge requests are clipped at the voltage limits.
- **Test suite.** 57 pytest tests cover every module.

Validation against measured vehicle data has not been performed; see Section 7.

## 4. Results at reference conditions

| Cycle | Avg speed [km/h] | Idle [%] | Stops | Consumption [Wh/km] | Range [km] |
|---|---|---|---|---|---|
| UDDS | 31.5 | 18.0 | 17 | 137.3 | 418 |
| HWFET | 77.7 | 0.5 | 1 | 140.2 | 409 |
| NEDC | 33.4 | 23.7 | 13 | 144.7 | 397 |
| MIDC | 32.3 | 23.7 | 13 | 135.7 | 423 |

Compared with a constant-efficiency model (90% drivetrain), the detailed model lowers range by 8–10%. Cycle-average motoring efficiency is only 85–87% against a 95.6% peak, because most driving happens at low torque where the fixed losses dominate.

The energy breakdown shows rolling resistance at 46.1 Wh/km on every cycle (it scales with mass, not speed), aerodynamic loss rising from 24.0 Wh/km on UDDS to 56.7 Wh/km on HWFET, and regeneration returning 41.2 Wh/km on UDDS against 9.5 Wh/km on HWFET. Battery I²R loss is below 1 Wh/km because peak current stays under 1C.

## 5. Sensitivity and scenarios

| Scenario | UDDS [km] | HWFET [km] | 120 km/h [km] |
|---|---|---|---|
| Mild day, 25 °C | 403 | 408 | 278 |
| Summer, 35 °C + A/C | 359 | 392 | 278 |
| Winter, −10 °C + heater, 2 people | 180 | 244 | 188 |
| Family trip, +300 kg, 4% hills | 351 | 359 | 263 |
| Aged battery, SOH 80% | 322 | 325 | 222 |
| Worst case (winter, loaded, hills, aged) | 132 | 173 | 137 |

- **Temperature and HVAC.** Heating power is constant in time, so its energy per km is inversely proportional to average speed: about 133 Wh/km on UDDS at −10 °C against about 35 Wh/km at 120 km/h. City driving therefore suffers most (−55% vs −32%). In summer at motorway speed, the extra air-conditioning load is offset by about 3% lower air density, leaving range unchanged.
- **Speed.** Steady-speed range falls from 575 km at 40 km/h to 278 km at 120 km/h and 224 km at 140 km/h, driven by the v² aerodynamic term.
- **Hills.** Sinusoidal grades with zero net elevation reduce range by 1–2% at 2% amplitude and 13–16% at 8%, because energy passes through the drivetrain twice and steep descents exceed the regen power limit.
- **Payload and ageing.** Payload reduces range roughly linearly, by about 2–3% per 100 kg on the city cycle; SOH 80% reduces range almost proportionally (−20%).
- **Label estimate.** An EPA-style two-cycle estimate (55% city, 45% highway, ×0.70 adjustment) is implemented for comparison with window-sticker values.

## 6. ML surrogate

**Data.** 3,000 trips were generated with random ambient temperature (−20 to 40 °C), HVAC on/off, payload (0–450 kg), hills (0–8%, half of trips flat), SOH (0.7–1.0, with resistance growth linked to SOH), extra electrical load (0–400 W) and either a speed-scaled standard cycle (×0.7–1.2) or a constant-speed trip (30–140 km/h). Each trip is labelled with its cell-level consumption, averaged over passes starting at 85%, 50% and 15% SOC; range is usable cell energy divided by this consumption. On twelve random trips this label agrees with the full-depletion range test to 0.26% on average.

**Features.** Raw features describe the conditions and the cycle (average and moving speed, maximum speed, idle and cruise share, stops per km, positive kinetic energy, mean positive acceleration). Three physics-informed features are added: distance-weighted mean v² (an aerodynamic proxy), HVAC energy per km and auxiliary energy per km.

**Results.** Gradient boosting with physics features is best (MAE 2.77 Wh/km, R² 0.996, range MAPE 1.38%; 5-fold CV R² 0.994). The physics features reduce linear-regression error from 26.1 to 3.4 Wh/km. HVAC energy per km is the most important feature by permutation importance.

**Extrapolation.** Leaving one cycle family out of training shows the limits of each approach. Gradient boosting stays within 1.7–4.6% on held-out standard cycles but reaches 15.1% on constant-speed trips, a pattern with no stops or accelerations that trees cannot extrapolate to. The ridge model with physics features is within 3.0% on constant speed but 20.9% off on NEDC. No single model is best everywhere; the dashboard therefore flags inputs outside the training range.

## 7. Limitations and further work

The parameters are generic and uncalibrated, so absolute ranges are indicative and relative effects are the main result. The model is quasi-static and backward-facing, the battery has no thermal management, and cabin warm-up transients are ignored, which makes the cold-weather results an upper bound on losses. WLTC Class 3b is supported but not included because its trace was not freely downloadable from the source used. Planned work: calibration against a published vehicle using EPA test coefficients and label values, a battery thermal model with preconditioning, WLTC from UNECE GTR 15, and training the ML model on real trip logs.

## 8. Conclusion

A transparent, tested Python model reproduces standard-cycle EV consumption from first principles and quantifies the real-world factors behind range variation, with cabin heating at low speed as the dominant effect. A physics-informed ML surrogate reproduces the model within about 1.4% at interactive speed, and the extrapolation study shows why such surrogates need training data that covers the intended operating range.

## References

1. US EPA, *Dynamometer Drive Schedules*, https://www.epa.gov/vehicle-and-fuel-emissions-testing/dynamometer-drive-schedules
2. UNECE, Regulation No. 83 — emissions of M1 and N1 vehicles (ECE-15, EUDC, NEDC).
3. UNECE, Global Technical Regulation No. 15 — Worldwide harmonized Light vehicles Test Procedure (WLTP).
4. J. Larminie, J. Lowry, *Electric Vehicle Technology Explained*, 2nd ed., Wiley, 2012.
5. G. L. Plett, *Battery Management Systems, Vol. I: Battery Modeling*, Artech House, 2015.
