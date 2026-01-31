from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class RegionBBox:
    lat_min: float  # degrees
    lat_max: float  # degrees
    lon_min: float  # degrees
    lon_max: float  # degrees


@dataclass(frozen=True)
class ProjectConfig:
    region: RegionBBox
    grid_deg: float = 1.0
    monthly: bool = True

    shark_species: tuple[str, ...] = ()
    prey_groups: tuple[str, ...] = ()

    collapse_threshold_quantile: float = 0.1

    mc_trajectories: int = 200
    random_seed: int = 42

    delta_scenarios: tuple[float, ...] = (-0.2, -0.1, 0.0, 0.1, 0.2)


def as_tuple(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(str(v).strip() for v in values if str(v).strip())
