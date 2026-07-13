import numpy as np
import pytest

from supplyradar.core.uom import UOMError, convert, is_mass, normalize_uom


def test_normalize_known_spellings():
    assert normalize_uom(" Ton ") == "ton"
    assert normalize_uom("PC") == "pc"
    assert normalize_uom("Kg") == "kg"


def test_unknown_uom_raises():
    with pytest.raises(UOMError):
        normalize_uom("bucket")


def test_mass_to_mass():
    assert convert(2.0, "ton", "kg") == 2000.0
    assert convert(500.0, "kg", "ton") == 0.5
    assert is_mass("ton") and not is_mass("pc")


def test_mass_pc_roundtrip():
    assert convert(1.0, "ton", "pc", unit_wt_kg=250.0) == 4.0
    assert convert(4.0, "pc", "ton", unit_wt_kg=250.0) == 1.0


def test_mass_to_pc_requires_positive_weight():
    with pytest.raises(UOMError):
        convert(1.0, "ton", "pc", unit_wt_kg=0.0)
    with pytest.raises(UOMError):
        convert(1.0, "ton", "pc")


def test_pc_to_mass_zero_weight_is_zero_mass():
    # the source data's own convention for weightless pc items
    assert convert(5.0, "pc", "ton", unit_wt_kg=0.0) == 0.0
    out = convert(np.array([1.0, 2.0]), "pc", "ton",
                  unit_wt_kg=np.array([np.nan, 100.0]))
    assert out.tolist() == [0.0, 0.2]


# --- the three confirmed std_cons_rate_uom branches, hand-computed ---------

def test_branch_kg_per_ton():
    # 2 kg per ton of steel, 1000 t produced -> 2 t consumed
    rate, produced_t = 2.0, 1000.0
    cons_ton = rate * produced_t / 1000.0
    assert cons_ton == 2.0
    assert convert(cons_ton, "ton", "ton") == 2.0  # base uom ton


def test_branch_ton_per_day():
    # flat 0.5 ton per calendar day regardless of production
    rate = 0.5
    assert convert(rate, "ton", "kg") == 500.0


def test_branch_pc_per_heat():
    # 0.4 pc per heat, 5 heats -> 2 pc; at 20 kg/pc -> 0.04 t
    base = 0.4 * 5
    assert base == 2.0
    assert convert(base, "pc", "ton", unit_wt_kg=20.0) == pytest.approx(0.04)
