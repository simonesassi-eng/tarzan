"""Synthetic identity oracles: receipt time is not market observation time."""

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
import pytest

from tarzan.contracts.material import MATERIAL_FACTS_VERSION, material_value
from tarzan.runtime.session import canonical_analysis_id


def _digest(value):
    return canonical_analysis_id({"facts": material_value(value)})


@pytest.mark.parametrize("field", [
    "attempt_id", "recorded_at", "fetch_time", "fetch_timestamp",
    "latency_ms", "age_seconds", "freshness_age_seconds",
])
def test_declared_operational_fields_are_excluded_at_every_container_boundary(field):
    first = {
        "nested": {
            field: "first-receipt",
            "prices": pd.DataFrame({
                "price": [106.0], field: ["first-receipt"],
            }),
        },
    }
    second = {
        "nested": {
            field: "second-receipt",
            "prices": pd.DataFrame({
                "price": [106.0], field: ["second-receipt"],
            }),
        },
    }
    assert _digest(first) == _digest(second)
    assert material_value(first)["nested"]["prices"]["columns"] == ["price"]


@pytest.mark.parametrize("field", [
    "price", "quantity", "cost_basis_eur", "source_observed_at", "known_at",
    "quote_date", "effective_date", "availability", "policy_version", "currency",
])
def test_financial_observation_and_authority_fields_remain_material(field):
    assert _digest({field: "first"}) != _digest({field: "second"})
    assert _digest(pd.DataFrame({field: ["first"]})) != (
        _digest(pd.DataFrame({field: ["second"]}))
    )


def test_version_declares_the_changed_hash_contract():
    assert MATERIAL_FACTS_VERSION == "3"


def test_exact_wide_integer_is_not_rounded_by_adjacent_float_column():
    wide = 2**53 + 1
    table = pd.DataFrame({
        "quantity": pd.Series([wide], dtype=np.int64),
        "price": [106.0],
        "fetch_timestamp": ["first-receipt"],
    })
    facts = material_value(table)
    assert facts["rows"] == [[wide, 106.0]]
    assert type(facts["rows"][0][0]) is int
    changed = table.copy()
    changed["quantity"] = np.int64(wide + 1)
    assert _digest(table) != _digest(changed)


def test_duplicate_financial_columns_are_positional_not_collapsed():
    table = pd.DataFrame(
        [[106.0, "receipt", 107.0]],
        columns=["price", "fetch_timestamp", "price"],
    )
    facts = material_value(table)
    assert facts["columns"] == ["price", "price"]
    assert facts["rows"] == [[106.0, 107.0]]
    changed = table.copy()
    changed.iloc[0, 2] = 108.0
    assert _digest(table) != _digest(changed)


def test_operational_only_and_zero_column_tables_preserve_row_identity():
    only = pd.DataFrame({"fetch_timestamp": ["first", "second"]}, index=["a", "b"])
    zero = pd.DataFrame(index=["a", "b"])
    assert material_value(only) == material_value(zero)
    assert material_value(zero)["rows"] == [[], []]
    assert _digest(zero) != _digest(zero.iloc[::-1])


def test_financial_row_column_and_order_sequence_order_is_not_canonicalized_away():
    frame = pd.DataFrame({"price": [106.0, 107.0], "quantity": [2, 3]})
    assert _digest(frame) != _digest(frame.iloc[::-1])
    assert _digest(frame) != _digest(frame[["quantity", "price"]])
    fills = [
        {"date": "2025-01-02", "type": "BUY", "quantity": 2.0},
        {"date": "2025-01-02", "type": "SELL", "quantity": 1.0},
    ]
    assert _digest(fills) != _digest(fills[::-1])


def test_enum_field_names_follow_the_existing_scalar_mapping_contract():
    class Field(Enum):
        FETCH = "fetch_timestamp"
        PRICE = "price"

    assert material_value({Field.FETCH: "receipt", Field.PRICE: 106.0}) == {
        "price": 106.0,
    }


def test_dataclass_operational_boundary_and_nullable_values_still_work():
    @dataclass
    class Quote:
        price: float
        source_observed_at: str
        fetch_timestamp: str

    assert _digest(Quote(106.0, "2025-01-02", "receipt-one")) == (
        _digest(Quote(106.0, "2025-01-02", "receipt-two"))
    )
    assert _digest(Quote(106.0, "2025-01-02", "receipt-one")) != (
        _digest(Quote(106.0, "2025-01-03", "receipt-one"))
    )
    frame = pd.DataFrame({
        "price": pd.Series([pd.NA, 106], dtype="Int64"),
        "quantity": [float("nan"), float("inf")],
    })
    assert material_value(frame)["rows"] == [[None, None], [106, None]]


def test_table_labels_are_not_recursively_treated_as_field_names():
    # A compound financial label is not an operational scalar field.
    columns = pd.MultiIndex.from_tuples([("price", "fetch_timestamp")])
    table = pd.DataFrame([[106.0]], columns=columns)
    assert material_value(table)["columns"] == [["price", "fetch_timestamp"]]
    assert material_value(table)["rows"] == [[106.0]]


def test_unknown_material_values_are_not_silently_hidden():
    with pytest.raises(TypeError, match="unsupported material fact"):
        material_value({"source_observed_at": object()})
    # Operational payloads are deliberately irrelevant even if nonserializable.
    assert material_value({"fetch_timestamp": object(), "price": 106.0}) == {
        "price": 106.0,
    }


@pytest.mark.parametrize("scalar", [
    np.float16(1.25), np.float32(1.25), np.float64(1.25),
    np.int8(-7), np.int16(-7), np.int32(-7), np.int64(2**53 + 1),
    np.uint8(7), np.uint16(7), np.uint32(7), np.uint64(2**64 - 1),
    np.bool_(False), np.bool_(True),
])
def test_native_numpy_scalars_keep_their_lossless_python_material(scalar):
    assert material_value(scalar) == scalar.item()
    assert type(material_value(scalar)) is type(scalar.item())


@pytest.mark.parametrize("container", ["scalar", "mapping", "list", "series", "frame"])
@pytest.mark.parametrize("dtype", [np.longdouble, np.clongdouble])
def test_extended_numpy_scalars_fail_boundedly_without_precision_loss(container, dtype):
    scalar = dtype("1.0000000000000000001")
    if not isinstance(scalar.item(), np.generic):
        pytest.skip("this NumPy platform supplies a native Python scalar")
    value = {
        "scalar": scalar,
        "mapping": {"price": scalar},
        "list": [scalar],
        "series": pd.Series([scalar], dtype=dtype),
        "frame": pd.DataFrame({"price": pd.Series([scalar], dtype=dtype)}),
    }[container]
    with pytest.raises(TypeError, match="unsupported material fact"):
        material_value(value)


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_extended_nonfinite_real_values_are_unavailable_not_recursion(value):
    assert material_value(np.longdouble(value)) is None


def test_a_nonconverting_numpy_item_is_not_retried_or_silently_rounded():
    calls = []

    class Nonconverting(np.float64):
        def item(self, *args):
            calls.append(True)
            return self

    with pytest.raises(TypeError, match="unsupported material fact"):
        material_value(Nonconverting(106.0))
    assert calls == [True]
