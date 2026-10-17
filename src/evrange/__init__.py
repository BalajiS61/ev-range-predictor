"""evrange: EV drive-cycle energy consumption and range prediction."""

from evrange.vehicle import Vehicle
from evrange.dynamics import simulate, summarize
from evrange.cycles import DriveCycle, concat_cycles, load_cycle_file, piecewise_cycle
from evrange.cycle_stats import cycle_statistics
from evrange.cycle_library import download_cycles, get_cycle, list_cycles

__all__ = [
    "Vehicle",
    "simulate",
    "summarize",
    "DriveCycle",
    "piecewise_cycle",
    "load_cycle_file",
    "concat_cycles",
    "cycle_statistics",
    "download_cycles",
    "get_cycle",
    "list_cycles",
]
__version__ = "0.2.0"
