"""
Automated tests for app.py using streamlit.testing.v1.AppTest.

Run with:  py -3.13 -m pytest test_app.py -v
"""

import datetime
import json
import pytest
import joblib

from streamlit.testing.v1 import AppTest
from app import build_features, predict, band

APP_PATH = "app.py"
TIMEOUT = 30


# ══════════════════════════════════════════════════════════════════════════════
# Test 1 & 2: App loads without exceptions, shows 3 metrics, and form submit works
# ══════════════════════════════════════════════════════════════════════════════

def test_app_load_and_metrics():
    """App must load and render with default inputs without exceptions, showing 3 metrics."""
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.run()

    assert not at.exception, f"App raised an exception on load: {at.exception}"
    assert len(at.metric) >= 3, f"Expected at least 3 metrics, got {len(at.metric)}"


def test_change_lead_time_and_submit():
    """Changing lead time and clicking predict should run without exceptions."""
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception

    # Find lead_time input
    lead_input = None
    for ni in at.number_input:
        if "lead" in ni.label.lower():
            lead_input = ni
            break

    assert lead_input is not None, "Could not find lead time input"
    lead_input.set_value(180)

    # Submit form by clicking the primary button
    at.button[0].click().run()

    assert not at.exception, f"Exception after submitting form: {at.exception}"
    assert len(at.metric) >= 3


# ══════════════════════════════════════════════════════════════════════════════
# Test 3: Sanity check: 7 days vs 300 days lead time
# ══════════════════════════════════════════════════════════════════════════════

def test_sanity_lead_time():
    """A booking made 7 days before arrival should score clearly lower than 300 days."""
    with open("app_metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    model = joblib.load("app_model.joblib")
    calibrator = joblib.load("app_calibrator.joblib")

    base = dict(
        hotel="City Hotel",
        arrival=datetime.date.today() + datetime.timedelta(days=60),
        nights=3,
        adr=99.0,
        adults=2,
        children=0.0,
        babies=0,
        is_repeated_guest=0,
        previous_cancellations=0,
        previous_bookings_not_canceled=0,
        customer_type="Transient",
        market_segment="Online TA",
        distribution_channel="TA/TO",
        deposit_type="No Deposit",
        meal="BB",
        reserved_room_type="A",
        country="PRT",
        agent="None",
        has_company=0,
        num_cols=meta["num_cols"],
        cat_cols=meta["cat_cols"],
    )

    X_7 = build_features(**{**base, "lead_time": 7})
    X_300 = build_features(**{**base, "lead_time": 300})

    p_7 = predict(model, calibrator, X_7)
    p_300 = predict(model, calibrator, X_300)

    assert p_300 > p_7, f"Expected p(300d) > p(7d), got p(300d)={p_300:.4f} and p(7d)={p_7:.4f}"


# ══════════════════════════════════════════════════════════════════════════════
# Test 4: Sanity check: Previous cancellations increase risk
# ══════════════════════════════════════════════════════════════════════════════

def test_sanity_previous_cancellations():
    """A guest with previous cancellations should score higher."""
    with open("app_metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    model = joblib.load("app_model.joblib")
    calibrator = joblib.load("app_calibrator.joblib")

    base = dict(
        hotel="City Hotel",
        arrival=datetime.date.today() + datetime.timedelta(days=60),
        nights=3,
        lead_time=60,
        adr=99.0,
        adults=2,
        children=0.0,
        babies=0,
        is_repeated_guest=0,
        previous_bookings_not_canceled=0,
        customer_type="Transient",
        market_segment="Online TA",
        distribution_channel="TA/TO",
        deposit_type="No Deposit",
        meal="BB",
        reserved_room_type="A",
        country="PRT",
        agent="None",
        has_company=0,
        num_cols=meta["num_cols"],
        cat_cols=meta["cat_cols"],
    )

    X_clean = build_features(**{**base, "previous_cancellations": 0})
    X_cancel = build_features(**{**base, "previous_cancellations": 3})

    p_clean = predict(model, calibrator, X_clean)
    p_cancel = predict(model, calibrator, X_cancel)

    assert p_cancel > p_clean, f"Expected previous cancellations to raise risk: p(clean)={p_clean:.4f}, p(cancel)={p_cancel:.4f}"


# ══════════════════════════════════════════════════════════════════════════════
# Test 5: Odd inputs do not crash
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("overrides,desc", [
    ({"children": 0.0}, "0 children"),
    ({"nights": 60}, "60 nights"),
    ({"arrival": datetime.date(2020, 1, 1)}, "past arrival date"),
    ({"country": "Unknown"}, "Other country"),
    ({"adr": 0.0}, "zero ADR"),
])
def test_odd_inputs_no_crash(overrides, desc):
    """Test odd inputs without crashing."""
    with open("app_metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    model = joblib.load("app_model.joblib")
    calibrator = joblib.load("app_calibrator.joblib")

    base = dict(
        hotel="City Hotel",
        arrival=datetime.date.today() + datetime.timedelta(days=60),
        nights=3,
        lead_time=60,
        adr=99.0,
        adults=2,
        children=0.0,
        babies=0,
        is_repeated_guest=0,
        previous_cancellations=0,
        previous_bookings_not_canceled=0,
        customer_type="Transient",
        market_segment="Online TA",
        distribution_channel="TA/TO",
        deposit_type="No Deposit",
        meal="BB",
        reserved_room_type="A",
        country="PRT",
        agent="None",
        has_company=0,
        num_cols=meta["num_cols"],
        cat_cols=meta["cat_cols"],
    )

    inputs = {**base, **overrides}
    X = build_features(**inputs)
    prob = predict(model, calibrator, X)
    assert 0.0 <= prob <= 1.0, f"Probability out of range for {desc}: {prob}"


# ══════════════════════════════════════════════════════════════════════════════
# Test 6: Risk band threshold checks
# ══════════════════════════════════════════════════════════════════════════════

def test_band_thresholds():
    """Risk band mapping must strictly respect metadata thresholds."""
    with open("app_metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    thr = meta["thresholds"]
    assert band(0.0, thr) == "Low"
    assert band(thr["low"] - 1e-5, thr) == "Low"
    assert band(thr["low"], thr) == "Medium"
    assert band(thr["high"] - 1e-5, thr) == "Medium"
    assert band(thr["high"], thr) == "High"
    assert band(1.0, thr) == "High"
