"""Unit-of-measure normalization and conversion.

ALL unit conversion in SupplyRadar lives in this module. The conversion table is
explicit; anything outside it raises UOMError — no silent guessing.

Contract:
- normalize_uom: any spelling/casing of a known unit -> canonical ('ton'|'kg'|'pc').
- convert: mass<->mass via kg factors; mass<->pc requires the item's unit_wt_kg.
- Unknown units, or mass<->pc without a positive unit weight, raise UOMError.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml


class UOMError(ValueError):
    """Raised on unknown units or impossible conversions."""

# Built-in defaults; config/uom.yaml (same content) can extend them via load_uom_config.
_ALIASES: dict[str, str] = {
    "ton": "ton", "tons": "ton", "t": "ton", "mt": "ton",
    "kg": "kg", "kgs": "kg", "k": "kg",
    "pc": "pc", "pcs": "pc", "piece": "pc", "pieces": "pc", "ea": "pc",
    "p": "pc",
}
_MASS_TO_KG: dict[str, float] = {"ton": 1000.0, "kg": 1.0}

def load_uom_config(path: str | Path) -> None:
    """Extend the alias/mass tables from a YAML file (see config/uom.yaml)."""
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    for alias, canon in (cfg.get("aliases") or {}).items():
        _ALIASES[str(alias).strip().lower()] = str(canon).strip().lower()
    for unit, factor in (cfg.get("mass_to_kg") or {}).items():
        _MASS_TO_KG[str(unit).strip().lower()] = float(factor)

def normalize_uom(uom: str) -> str:
    """Return the canonical unit for any known spelling. Raises UOMError if unknown."""
    if uom is None or (isinstance(uom, float) and np.isnan(uom)):
        raise UOMError("uom is missing")
    key = str(uom).strip().lower()
    if key not in _ALIASES:
        raise UOMError(f"unknown uom {uom!r}")
    return _ALIASES[key]

def is_mass(uom: str) -> bool:
    """True if the canonical unit is a mass unit."""
    return normalize_uom(uom) in _MASS_TO_KG

def convert(
    qty: float | np.ndarray,
    from_uom: str,
    to_uom: str,
    *,
    unit_wt_kg: float | np.ndarray | None = None,
) -> float | np.ndarray:
    """Convert qty between canonical units.

    mass -> mass  : via kg factors.
    mass <-> pc   : requires unit_wt_kg > 0 (kg per piece), else UOMError.
    pc -> pc      : identity.
    Works elementwise on numpy arrays (unit_wt_kg may be scalar or aligned array).
    """
    f, t = normalize_uom(from_uom), normalize_uom(to_uom)
    if f == t:
        return qty
    f_mass, t_mass = f in _MASS_TO_KG, t in _MASS_TO_KG
    if f_mass and t_mass:
        return qty * (_MASS_TO_KG[f] / _MASS_TO_KG[t])
    if unit_wt_kg is None:
        raise UOMError(f"conversion {f} -> {t} requires unit_wt_kg")
    wt = np.asarray(unit_wt_kg, dtype=float)
    if f_mass and t == "pc":
        # division: a piece of unknown/zero weight cannot be counted from mass
        if np.any(~(wt > 0)):
            raise UOMError(f"conversion {f} -> pc requires unit_wt_kg > 0")
        return qty * _MASS_TO_KG[f] / wt
    if f == "pc" and t_mass:
        # multiplication: a missing/zero piece weight yields zero mass — this is
        # the source data's own convention for weightless pc items (validation
        # reports these items); it must not block the projection.
        wt = np.where(np.isfinite(wt) & (wt > 0), wt, 0.0)
        result = qty * wt / _MASS_TO_KG[t]
        return float(result) if np.isscalar(qty) else result
    raise UOMError(f"no conversion path {f} -> {t}")
