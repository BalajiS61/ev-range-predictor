from pathlib import Path

import numpy as np
import pytest

from evrange import concat_cycles, cycle_statistics, get_cycle, load_cycle_file, piecewise_cycle
from evrange.cycle_library import COMPOSITES, SOURCES, list_cycles
from evrange.cycles import MPH_TO_MS


def _write(path: Path, header: str, rows: list[tuple[float, float]]) -> Path:
    lines = ["CYCLE.TXT\tTest cycle", header] + [f"{t:g}\t{v:g}" for t, v in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_load_epa_mph_file(tmp_path):
    f = _write(tmp_path / "a.txt", "Test Time, secs Target Speed, mph", [(0, 0), (1, 10), (2, 20)])
    cyc = load_cycle_file(f)
    assert np.allclose(cyc.speed_ms, [0, 10 * MPH_TO_MS, 20 * MPH_TO_MS])
    assert cyc.name == "a"


def test_load_kph_file(tmp_path):
    f = _write(tmp_path / "b.txt", "Test Time, secs\tVehicle Speed, kph", [(0, 0), (1, 36), (2, 72)])
    cyc = load_cycle_file(f)
    assert np.allclose(cyc.speed_ms, [0, 10, 20])


def test_explicit_unit_overrides_header(tmp_path):
    f = _write(tmp_path / "c.txt", "time speed", [(0, 0), (1, 10)])
    assert np.isclose(load_cycle_file(f, unit="mph").speed_ms[1], 10 * MPH_TO_MS)
    with pytest.raises(ValueError):
        load_cycle_file(f, unit="furlongs")


def test_empty_file_rejected(tmp_path):
    f = tmp_path / "empty.txt"
    f.write_text("header only\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_cycle_file(f)


def test_concat_drops_shared_zero_sample():
    a = piecewise_cycle("a", [(0, 0), (10, 36), (20, 0)])
    b = piecewise_cycle("b", [(0, 0), (5, 18), (10, 0)])
    c = concat_cycles("ab", [a, b])
    assert c.duration_s == pytest.approx(30.0)
    assert len(c.time_s) == len(a.time_s) + len(b.time_s) - 1
    assert c.distance_km == pytest.approx(a.distance_km + b.distance_km)


def test_resample_preserves_distance():
    a = piecewise_cycle("a", [(0, 0), (10, 36), (30, 36), (40, 0)])
    fine = a.resample(0.1)
    assert fine.duration_s == pytest.approx(a.duration_s)
    assert fine.distance_km == pytest.approx(a.distance_km, rel=1e-6)


def test_statistics_on_trapezoid():
    # 10 s idle, 10 s accel to 36 km/h, 20 s cruise, 10 s decel, 10 s idle
    cyc = piecewise_cycle("trap", [(0, 0), (10, 0), (20, 36), (40, 36), (50, 0), (60, 0)])
    s = cycle_statistics(cyc)
    assert s["distance_km"] == pytest.approx(0.3, rel=1e-6)   # 5 + 20 + 5 s-equivalent at 10 m/s
    assert s["max_speed_kmh"] == pytest.approx(36)
    assert s["stops"] == 1
    assert s["idle_pct"] == pytest.approx(100 * 20 / 60)
    assert s["max_accel_ms2"] == pytest.approx(1.0)
    assert s["max_decel_ms2"] == pytest.approx(-1.0)
    assert s["pke_ms2"] == pytest.approx(100 / 300)          # v^2 gain 100 over 300 m


def test_composite_from_files(tmp_path):
    ece = [(t, 0.0) for t in range(0, 196)]
    ece[50:60] = [(t, 15.0) for t in range(50, 60)]
    eudc = [(t, 0.0) for t in range(0, 401)]
    eudc[100:200] = [(t, 70.0) for t in range(100, 200)]
    _write(tmp_path / SOURCES["ece"].filename, "Vehicle Speed, kph", ece)
    _write(tmp_path / SOURCES["eudc"].filename, "Vehicle Speed, kph", eudc)
    nedc = get_cycle("nedc", tmp_path)
    assert nedc.duration_s == pytest.approx(4 * 195 + 400)
    assert np.max(nedc.speed_kmh) == pytest.approx(70.0)


def test_missing_file_gives_instructions(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_cycles"):
        get_cycle("udds", tmp_path)
    with pytest.raises(FileNotFoundError, match="dieselnet"):
        get_cycle("wltc3b", tmp_path)
    with pytest.raises(KeyError):
        get_cycle("not_a_cycle", tmp_path)


def test_library_lists_all_cycles():
    keys = list_cycles()
    for key in ["udds", "hwfet", "ece", "eudc", "eudc_lp", "wltc3b", "nedc", "midc"]:
        assert key in keys
    assert set(COMPOSITES["midc"].parts) == {"ece", "eudc_lp"}
