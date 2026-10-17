"""Standard drive cycles: sources, download, and composite cycles (NEDC, MIDC).

Single cycles are 1 Hz text files published by the US EPA (public domain).
The WLTC Class 3b trace is downloaded manually (see WLTC_INSTRUCTIONS).
"""

from __future__ import annotations

import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from evrange.cycles import DriveCycle, concat_cycles, load_cycle_file

EPA_BASE = "https://www.epa.gov/sites/default/files/2015-10/"


@dataclass(frozen=True)
class CycleSource:
    key: str
    filename: str
    url: str | None
    description: str
    unit: str
    ref_duration_s: float | None = None
    ref_distance_km: float | None = None


SOURCES: dict[str, CycleSource] = {
    "udds": CycleSource(
        "udds", "uddscol.txt", EPA_BASE + "uddscol.txt",
        "EPA Urban Dynamometer Driving Schedule (city, LA4)", "mph", 1369, 11.99,
    ),
    "hwfet": CycleSource(
        "hwfet", "hwycol.txt", EPA_BASE + "hwycol.txt",
        "EPA Highway Fuel Economy Test", "mph", 765, 16.51,
    ),
    "ece": CycleSource(
        "ece", "ececol.txt", EPA_BASE + "ececol.txt",
        "UN/ECE R83 elementary urban cycle (ECE-15)", "kmh", 195, 0.994,
    ),
    "eudc": CycleSource(
        "eudc", "eudccol.txt", EPA_BASE + "eudccol.txt",
        "UN/ECE R83 extra-urban driving cycle (120 km/h)", "kmh", 400, 6.955,
    ),
    "eudc_lp": CycleSource(
        "eudc_lp", "eudclcol.txt", EPA_BASE + "eudclcol.txt",
        "UN/ECE R83 extra-urban cycle, low-powered vehicles (90 km/h)", "kmh", 400, None,
    ),
    "wltc3b": CycleSource(
        "wltc3b", "wltp3b.txt", None,
        "WLTC Class 3b (UNECE GTR 15)", "kmh", 1800, 23.266,
    ),
}


@dataclass(frozen=True)
class CompositeCycle:
    key: str
    parts: tuple[str, ...]
    description: str
    ref_duration_s: float | None = None
    ref_distance_km: float | None = None


COMPOSITES: dict[str, CompositeCycle] = {
    "nedc": CompositeCycle(
        "nedc", ("ece",) * 4 + ("eudc",),
        "New European Driving Cycle: 4 x ECE-15 + EUDC", 1180, 11.007,
    ),
    "midc": CompositeCycle(
        "midc", ("ece",) * 4 + ("eudc_lp",),
        "Modified Indian Driving Cycle: 4 x ECE-15 + low-powered EUDC (90 km/h)", 1180, None,
    ),
}

WLTC_INSTRUCTIONS = (
    "WLTC Class 3b is not downloaded automatically. To add it:\n"
    "  1. Open https://dieselnet.com/standards/cycles/wltp.php in your browser\n"
    "  2. Click the 'WLTC Class 3b' link at the top (a text page opens)\n"
    "  3. Press Ctrl+S and save it as data/cycles/wltp3b.txt in this project"
)


def list_cycles() -> list[str]:
    """Keys of every cycle the library can build."""
    return list(SOURCES) + list(COMPOSITES)


def describe(key: str) -> str:
    if key in SOURCES:
        return SOURCES[key].description
    if key in COMPOSITES:
        return COMPOSITES[key].description
    raise KeyError(f"Unknown cycle {key!r}. Available: {list_cycles()}")


def reference(key: str) -> tuple[float | None, float | None]:
    """Published (duration_s, distance_km) for validation, None where not stored."""
    if key in SOURCES:
        s = SOURCES[key]
        return s.ref_duration_s, s.ref_distance_km
    if key in COMPOSITES:
        c = COMPOSITES[key]
        return c.ref_duration_s, c.ref_distance_km
    raise KeyError(f"Unknown cycle {key!r}. Available: {list_cycles()}")


def download_cycles(
    data_dir: str | Path,
    keys: list[str] | None = None,
    overwrite: bool = False,
    timeout: float = 30.0,
) -> dict[str, str]:
    """Download the EPA cycle files into data_dir. Returns {key: status}."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    keys = keys or [k for k, s in SOURCES.items() if s.url]
    status: dict[str, str] = {}

    for key in keys:
        src = SOURCES[key]
        target = data_dir / src.filename
        if src.url is None:
            status[key] = "manual" if not target.exists() else "present"
            continue
        if target.exists() and not overwrite:
            status[key] = "present"
            continue
        request = urllib.request.Request(src.url, headers={"User-Agent": "Mozilla/5.0 (evrange)"})
        tmp = target.with_suffix(".part")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                tmp.write_bytes(response.read())
            load_cycle_file(tmp, name=key, unit=src.unit)  # validate before keeping it
            tmp.replace(target)
            status[key] = "downloaded"
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
            tmp.unlink(missing_ok=True)
            status[key] = f"failed: {exc}"
    return status


def get_cycle(key: str, data_dir: str | Path) -> DriveCycle:
    """Load a single or composite cycle from the files in data_dir."""
    data_dir = Path(data_dir)
    if key in COMPOSITES:
        comp = COMPOSITES[key]
        parts = [get_cycle(p, data_dir) for p in comp.parts]
        return concat_cycles(key, parts)

    if key not in SOURCES:
        raise KeyError(f"Unknown cycle {key!r}. Available: {list_cycles()}")
    src = SOURCES[key]
    path = data_dir / src.filename
    if not path.exists():
        hint = WLTC_INSTRUCTIONS if src.url is None else "Run: python scripts/download_cycles.py"
        raise FileNotFoundError(f"{path} not found.\n{hint}")
    return load_cycle_file(path, name=key, unit=src.unit)
