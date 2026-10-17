"""EV Range Predictor - interactive dashboard.

Run from the project root:   streamlit run app/app.py
"""

from __future__ import annotations

from dataclasses import astuple

import matplotlib.pyplot as plt
import streamlit as st

from evrange.app_logic import (
    CONSTANT, TripInputs, available_cycles, cycle_label, fig_breakdown, fig_efficiency_map,
    fig_range_trace, fig_temperature, fig_trip_signals, load_base, load_predictor, ml_range,
    ml_warnings, run_trip, temperature_curve,
)

st.set_page_config(page_title="EV Range Predictor", page_icon="🔋", layout="wide")


@st.cache_resource
def get_base():
    return load_base()


@st.cache_resource
def get_cycles():
    return available_cycles()


@st.cache_resource
def get_predictor():
    return load_predictor()


@st.cache_resource(max_entries=64)
def cached_trip(key: tuple):
    base, env = get_base()
    return run_trip(base, env, get_cycles(), TripInputs(*key))


@st.cache_resource(max_entries=16)
def cached_curve(key: tuple):
    base, env = get_base()
    return temperature_curve(base, env, get_cycles(), TripInputs(*key))


def show(fig) -> None:
    st.pyplot(fig)
    plt.close(fig)


base, env = get_base()
cycles = get_cycles()
predictor = get_predictor()

st.title("🔋 EV Range Predictor")
st.caption(
    "Physics-based EV model (road load, motor efficiency map, battery equivalent circuit, "
    "cabin HVAC) with a machine-learning surrogate trained on 3,000 simulated trips."
)
if not cycles:
    st.warning("No standard drive cycles found - only constant speed is available. "
               "Run `python scripts/download_cycles.py` to add UDDS, HWFET, NEDC and MIDC.")

# ---- sidebar inputs ---------------------------------------------------------------
with st.sidebar:
    st.header("Trip")
    options = list(cycles) + [CONSTANT]
    cycle_key = st.selectbox("Drive cycle", options, format_func=cycle_label)
    speed = st.slider("Constant speed [km/h]", 30, 150, 100, 5, disabled=cycle_key != CONSTANT)

    st.header("Conditions")
    ambient = st.slider("Ambient temperature [°C]", -20, 40, 20)
    hvac = st.checkbox("Cabin heating / air-conditioning on", value=True)
    heater = st.radio("Heater type", ["PTC heater (COP 1.0)", "Heat pump (COP 2.5)"], disabled=not hvac)
    payload = st.slider("Extra payload [kg]", 0, 450, 0, 25, help="Passengers and luggage beyond the driver")
    hills = st.slider("Hill grade amplitude [%]", 0.0, 8.0, 0.0, 0.5,
                      help="Rolling hills with zero net elevation over the trip")
    extra_aux = st.slider("Extra electrical load [W]", 0, 1000, 0, 50,
                          help="Seat heaters, lights, infotainment, ...")

    st.header("Battery")
    soh = st.slider("State of health [%]", 70, 100, 100)

inputs = TripInputs(
    cycle_key=cycle_key,
    constant_speed_kmh=float(speed),
    ambient_c=float(ambient),
    hvac=hvac,
    heater_cop=2.5 if heater.startswith("Heat pump") else 1.0,
    extra_payload_kg=float(payload),
    hill_amplitude_pct=float(hills),
    soh=soh / 100.0,
    extra_aux_w=float(extra_aux),
)

with st.spinner("Running the physics model (full-depletion range test)..."):
    res = cached_trip(astuple(inputs))
ml = ml_range(predictor, base, env, cycles, inputs, res["usable_kwh"])

# ---- headline metrics -------------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Range (physics)", f"{res['range_km']:.0f} km")
if ml is None:
    c2.metric("Range (ML)", "not trained")
else:
    c2.metric("Range (ML)", f"{ml:.0f} km", delta=f"{ml - res['range_km']:+.1f} km vs physics",
              delta_color="off")
c3.metric("Consumption", f"{res['consumption_wh_per_km']:.0f} Wh/km")
c4.metric("HVAC power", f"{res['hvac_w'] / 1000:.1f} kW")
c5.metric("Usable energy", f"{res['usable_kwh']:.1f} kWh")

if ml is None:
    st.info("Train the ML model with `python scripts/step5_ml.py` to see the ML prediction.")
else:
    for note in ml_warnings(inputs, env):
        st.warning(f"ML caution: {note}.")
if res["trace_misses"]:
    st.warning(f"The motor could not follow the speed trace for {res['trace_misses']} samples.")

# ---- detail tabs -----------------------------------------------------------------
tabs = st.tabs(["Trip signals", "Energy breakdown", "Range test", "Range vs temperature",
                "Motor map", "About"])

with tabs[0]:
    show(fig_trip_signals(res["run"], f"{cycle_label(cycle_key)} - one pass from 80% SOC"))

with tabs[1]:
    show(fig_breakdown(res["breakdown_wh_per_km"], res["regen_wh_per_km"]))
    st.dataframe(res["breakdown_wh_per_km"].round(1).rename("Wh/km"))

with tabs[2]:
    show(fig_range_trace(res["range_trace"]))
    st.caption("The cycle is repeated from 97% to 5% SOC, like a chassis-dynamometer range test.")

with tabs[3]:
    st.write("Range across ambient temperatures with all other inputs as selected.")
    if st.button("Compute temperature curve (13 range tests)"):
        with st.spinner("Running 13 range tests..."):
            st.session_state["curve"] = (astuple(inputs), cached_curve(astuple(inputs)))
    stored = st.session_state.get("curve")
    if stored is not None:
        key, curve = stored
        if key != astuple(inputs):
            st.caption("Curve shown for previous inputs - press the button again to update.")
        show(fig_temperature(curve, inputs.ambient_c))
        st.dataframe(curve.round(0), hide_index=True)

with tabs[4]:
    show(fig_efficiency_map(res["model"], res["run"]))
    st.caption(f"Average motoring efficiency on this trip: {res['motor_eff_pct']:.1f}%")

with tabs[5]:
    st.markdown(
        """
**Model chain:** drive cycle → road load (aero, rolling, grade, inertia) → braking blend with regen
fade-out → gearbox → PMSM + inverter loss map with torque/power limits → battery pack ECM
(OCV + R0 + RC, Arrhenius temperature dependence) with cabin HVAC and auxiliary loads.

**Range** is a full-depletion test: the trip is repeated until the usable SOC window is used up.

**ML surrogate:** gradient boosting on physics-informed features, trained on synthetic trips labelled
by the physics model. Predictions outside the training range are flagged.

Vehicle and powertrain parameters are generic values for a compact EV and are set in `configs/`.
"""
    )
