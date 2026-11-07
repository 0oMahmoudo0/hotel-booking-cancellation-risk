"""
Hotel Booking Cancellation Risk — Streamlit App
================================================
Loads a trained scikit-learn pipeline + isotonic calibrator and, for a single
booking entered through a form, displays:
  • Cancellation probability (%)
  • Risk level (Low / Medium / High)
  • Suggested deposit policy
  • A what-if table showing how five alternative scenarios change the score

Designed for non-technical hotel revenue managers.
Never crashes on bad input.
"""

import datetime
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "app_model.joblib"
CALIBRATOR_PATH = BASE_DIR / "app_calibrator.joblib"
METADATA_PATH = BASE_DIR / "app_metadata.json"

# ── Country friendly names (ISO-3166 alpha-3 → display name) ────────────────
COUNTRY_NAMES = {
    "AGO": "Angola",
    "AUS": "Australia",
    "AUT": "Austria",
    "BEL": "Belgium",
    "BRA": "Brazil",
    "CHE": "Switzerland",
    "CHN": "China",
    "CN": "China (CN)",
    "DEU": "Germany",
    "DNK": "Denmark",
    "ESP": "Spain",
    "FIN": "Finland",
    "FRA": "France",
    "GBR": "United Kingdom",
    "HUN": "Hungary",
    "IRL": "Ireland",
    "ITA": "Italy",
    "LUX": "Luxembourg",
    "MAR": "Morocco",
    "NLD": "Netherlands",
    "NOR": "Norway",
    "POL": "Poland",
    "PRT": "Portugal",
    "ROU": "Romania",
    "RUS": "Russia",
    "SWE": "Sweden",
    "TUR": "Turkey",
    "USA": "United States",
    "Unknown": "Unknown",
}


# ══════════════════════════════════════════════════════════════════════════════
# Artifact loading — cached so they are read only once per session
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading model…")
def load_artifacts():
    """
    Load model, calibrator, and metadata exactly once.
    Shows a user-friendly error (not a traceback) if any file is missing or
    the library versions do not match.
    """
    import joblib

    try:
        model = joblib.load(MODEL_PATH)
        calibrator = joblib.load(CALIBRATOR_PATH)
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return model, calibrator, meta
    except Exception as exc:
        st.error(
            f"**Could not load model artifacts.**\n\n"
            f"Make sure these three files are in the same folder as `app.py`:\n"
            f"- `app_model.joblib`\n"
            f"- `app_calibrator.joblib`\n"
            f"- `app_metadata.json`\n\n"
            f"Required library versions: "
            f"`pandas==2.3.3`, `numpy==2.1.3`, `scikit-learn==1.6.1`, "
            f"`xgboost==3.2.0`, `joblib==1.6.0`\n\n"
            f"Error detail: `{exc}`"
        )
        st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# Feature engineering — must match training exactly
# ══════════════════════════════════════════════════════════════════════════════

def build_features(
    *,
    hotel: str,
    arrival: datetime.date,
    nights: int,
    lead_time: int,
    adr: float,
    adults: int,
    children: float,
    babies: int,
    is_repeated_guest: int,
    previous_cancellations: int,
    previous_bookings_not_canceled: int,
    customer_type: str,
    market_segment: str,
    distribution_channel: str,
    deposit_type: str,
    meal: str,
    reserved_room_type: str,
    country: str,
    agent: str,
    has_company: int,
    num_cols: list,
    cat_cols: list,
) -> pd.DataFrame:
    """
    Derive all 36 model columns from raw form inputs and return a one-row
    DataFrame with columns in exactly num_cols + cat_cols order.
    """
    # ── Date-based derivations ────────────────────────────────────────────
    arrival_month = arrival.month
    arrival_day = arrival.day
    arrival_week = arrival.isocalendar()[1]
    arrival_dow = arrival.weekday()

    # Count weekend nights (weekday >= 5 is Saturday/Sunday)
    weekend_nights = sum(
        1 for i in range(nights)
        if (arrival + datetime.timedelta(days=i)).weekday() >= 5
    )
    week_nights = nights - weekend_nights

    # Booking date = arrival - lead_time days
    booking_date = arrival - datetime.timedelta(days=lead_time)
    booking_month = booking_date.month
    booking_dow = booking_date.weekday()

    # ── Guest totals ──────────────────────────────────────────────────────
    total_guests = adults + children + babies
    total_guests_safe = total_guests if total_guests > 0 else 1
    adr_safe = max(adr, 0.0)

    adr_per_person = adr_safe / total_guests_safe
    weekend_night_share = weekend_nights / nights if nights > 0 else 0.0

    # ── Boolean flags ─────────────────────────────────────────────────────
    is_domestic = 1 if country == "PRT" else 0
    is_weekend_arrival = 1 if arrival_dow >= 5 else 0

    # ── Log / ratio features ──────────────────────────────────────────────
    lead_time_log = math.log1p(lead_time)
    prev_cancel_ratio = previous_cancellations / (
        previous_cancellations + previous_bookings_not_canceled + 1
    )

    # ── Cyclic encoding of arrival month ─────────────────────────────────
    arrival_month_sin = math.sin(2 * math.pi * arrival_month / 12)
    arrival_month_cos = math.cos(2 * math.pi * arrival_month / 12)

    # ── Assemble numeric row ──────────────────────────────────────────────
    num_values = {
        "lead_time": lead_time,
        "arrival_month": arrival_month,
        "arrival_date_week_number": arrival_week,
        "arrival_date_day_of_month": arrival_day,
        "arrival_dow": arrival_dow,
        "stays_in_weekend_nights": weekend_nights,
        "stays_in_week_nights": week_nights,
        "total_nights": nights,
        "adults": adults,
        "children": float(children),
        "babies": babies,
        "total_guests": total_guests,
        "is_repeated_guest": is_repeated_guest,
        "previous_cancellations": previous_cancellations,
        "previous_bookings_not_canceled": previous_bookings_not_canceled,
        "adr": adr_safe,
        "has_company": has_company,
        "adr_per_person": adr_per_person,
        "is_domestic": is_domestic,
        "is_weekend_arrival": is_weekend_arrival,
        "weekend_night_share": weekend_night_share,
        "lead_time_log": lead_time_log,
        "prev_cancel_ratio": prev_cancel_ratio,
        "booking_month": booking_month,
        "booking_dow": booking_dow,
        "arrival_month_sin": arrival_month_sin,
        "arrival_month_cos": arrival_month_cos,
    }

    # ── Assemble categorical row ──────────────────────────────────────────
    cat_values = {
        "hotel": str(hotel),
        "meal": str(meal),
        "country": str(country),
        "market_segment": str(market_segment),
        "distribution_channel": str(distribution_channel),
        "reserved_room_type": str(reserved_room_type),
        "deposit_type": str(deposit_type),
        "customer_type": str(customer_type),
        "agent": str(agent),
    }

    # ── Build DataFrame in exact column order ─────────────────────────────
    row = {col: num_values[col] for col in num_cols}
    row.update({col: cat_values[col] for col in cat_cols})
    return pd.DataFrame([row])


# ══════════════════════════════════════════════════════════════════════════════
# Prediction helpers
# ══════════════════════════════════════════════════════════════════════════════

def predict(model, calibrator, X: pd.DataFrame) -> float:
    """
    Return calibrated cancellation probability for a one-row DataFrame.
    """
    raw_proba = model.predict_proba(X)[:, 1]
    return float(calibrator.predict(raw_proba)[0])


def band(p: float, thresholds: dict) -> str:
    """Map probability to Low / Medium / High using metadata thresholds."""
    if p < thresholds["low"]:
        return "Low"
    elif p < thresholds["high"]:
        return "Medium"
    return "High"


# ══════════════════════════════════════════════════════════════════════════════
# What-if scenario definitions
# ══════════════════════════════════════════════════════════════════════════════

def build_scenarios(meta: dict) -> list[dict]:
    """
    Return a list of scenario dicts. Each has:
      label   – human-readable name shown in the table
      overrides – {column_name: new_value} dict applied to the base form inputs
    """
    return [
        {
            "label": "Lead time 7 days",
            "overrides": {"lead_time": 7},
        },
        {
            "label": "Lead time 180 days",
            "overrides": {"lead_time": 180},
        },
        {
            "label": "No earlier cancellations",
            "overrides": {"previous_cancellations": 0},
        },
        {
            "label": "Booked directly",
            "overrides": {
                "market_segment": "Direct",
                "distribution_channel": "Direct",
            },
            "requires_in_meta": {
                "market_segment": "Direct",
                "distribution_channel": "Direct",
            },
        },
        {
            "label": "Booked via online travel agency",
            "overrides": {
                "market_segment": "Online TA",
                "distribution_channel": "TA/TO",
            },
            "requires_in_meta": {
                "market_segment": "Online TA",
                "distribution_channel": "TA/TO",
            },
        },
        {
            "label": "The other hotel",
            "overrides": {"hotel": "__other__"},
            "requires_in_meta": {"hotel": None},
        },
    ]


# ══════════════════════════════════════════════════════════════════════════════
# Streamlit Application
# ══════════════════════════════════════════════════════════════════════════════

def run_app():
    st.set_page_config(
        page_title="Hotel Booking Cancellation Risk",
        page_icon="🏨",
        layout="wide",
    )

    model, calibrator, meta = load_artifacts()

    # Warn if prediction_moment != 'at_booking'
    if meta.get("prediction_moment") != "at_booking":
        st.warning(
            f"⚠️ Metadata `prediction_moment` is `{meta.get('prediction_moment')}` "
            f"(expected `at_booking`). Predictions may include post-booking leakage."
        )

    num_cols = meta["num_cols"]
    cat_cols = meta["cat_cols"]
    thresholds = meta["thresholds"]
    policy = meta["policy"]
    categories = meta["categories"]
    num_ranges = meta["numeric_ranges"]

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("🏨 Risk Bands & Policies")
        thr_low = thresholds["low"]
        thr_high = thresholds["high"]

        st.success(f"🟢 **Low** (p < {thr_low:.1%})\n\n*{policy.get('Low', 'No deposit')}*")
        st.warning(f"🟡 **Medium** ({thr_low:.1%} ≤ p < {thr_high:.1%})\n\n*{policy.get('Medium', 'Partial deposit')}*")
        st.error(f"🔴 **High** (p ≥ {thr_high:.1%})\n\n*{policy.get('High', 'Full deposit')}*")

        st.divider()
        with st.expander("About the model and its limits"):
            st.markdown(
                """
                - **Features:** Only information available at the time of booking is used.
                - **Future drift:** Accuracy on future bookings is naturally lower than on historical data.
                - **Thresholds:** Derived from assumed business cost trade-offs (missed cancel vs. false alarm).
                - **Decision support:** This tool informs human judgment and does not guarantee outcomes.
                """
            )

    # ── Main Header ──────────────────────────────────────────────────────────
    st.title("Hotel Booking Cancellation Risk")
    st.markdown(
        "Enter booking details below to estimate cancellation probability, "
        "evaluate risk levels, and receive recommended deposit policies tailored "
        "for hotel revenue management."
    )
    st.info(
        "ℹ️ **Decision Support Notice:** This prediction serves as decision support, "
        "not an absolute guarantee. Final revenue decisions remain with hotel management."
    )

    # ── Prepare options from metadata categories ─────────────────────────────
    hotels = categories.get("hotel", ["City Hotel", "Resort Hotel"])
    meals = categories.get("meal", ["BB"])
    market_segs = categories.get("market_segment", ["Direct"])
    dist_channels = categories.get("distribution_channel", ["Direct"])
    room_types = categories.get("reserved_room_type", ["A"])
    deposit_types = categories.get("deposit_type", ["No Deposit"])
    customer_types = categories.get("customer_type", ["Transient"])

    # Country options with friendly names
    country_codes = [c for c in categories.get("country", []) if c != "ISR"]
    def country_label(code: str) -> str:
        name = COUNTRY_NAMES.get(code, code)
        return f"{name} [{code}]" if code != "Unknown" else "Unknown"

    country_options_display = [country_label(c) for c in country_codes if c != "Unknown"]
    country_options_display.append("Other / not listed")
    country_map = {country_label(c): c for c in country_codes if c != "Unknown"}
    country_map["Other / not listed"] = "Unknown"

    default_country_display = country_label("PRT") if "PRT" in country_codes else country_options_display[0]

    # Agent options
    agent_codes = categories.get("agent", [])
    def agent_label(code: str) -> str:
        return "No agent" if code == "None" else f"Agent {code}"

    agent_options_display = [agent_label(c) for c in agent_codes]
    agent_map = {agent_label(c): c for c in agent_codes}
    default_agent_display = "No agent" if "None" in agent_codes else agent_options_display[0]

    # Defaults from numeric ranges
    default_adr = float(num_ranges.get("adr", [0.0, 261.0, 99.0])[2])
    default_nights = 3
    default_lead_time = 60
    default_arrival = datetime.date.today() + datetime.timedelta(days=default_lead_time)

    # ── Form Layout ──────────────────────────────────────────────────────────
    with st.form("booking_form"):
        st.subheader("Booking Information")
        col_stay, col_guests, col_booking = st.columns(3)

        with col_stay:
            st.markdown("### 🛏️ Stay")
            hotel = st.selectbox("Hotel", hotels)
            arrival = st.date_input(
                "Arrival date",
                value=default_arrival,
                min_value=datetime.date(2000, 1, 1),
            )
            nights = st.number_input(
                "Nights",
                min_value=1,
                max_value=365,
                value=default_nights,
                step=1,
            )
            adr = st.number_input(
                "Price per night (ADR €)",
                min_value=0.0,
                max_value=10000.0,
                value=default_adr,
                step=1.0,
                format="%.2f",
            )
            meal = st.selectbox("Meal plan", meals)
            reserved_room_type = st.selectbox("Reserved room type", room_types)

        with col_guests:
            st.markdown("### 👥 Guests")
            adults = st.number_input("Adults", min_value=1, max_value=50, value=2, step=1)
            children = st.number_input("Children", min_value=0, max_value=20, value=0, step=1)
            babies = st.number_input("Babies", min_value=0, max_value=10, value=0, step=1)
            country_display = st.selectbox(
                "Country",
                country_options_display,
                index=country_options_display.index(default_country_display) if default_country_display in country_options_display else 0,
            )
            is_repeated_guest = st.checkbox("Repeat guest", value=False)
            previous_cancellations = st.number_input(
                "Previous cancellations",
                min_value=0,
                max_value=100,
                value=0,
                step=1,
            )
            previous_stays = st.number_input(
                "Previous stays not cancelled",
                min_value=0,
                max_value=100,
                value=0,
                step=1,
            )

        with col_booking:
            st.markdown("### 📋 Booking")
            lead_time = st.number_input(
                "Lead time (days)",
                min_value=0,
                max_value=730,
                value=default_lead_time,
                step=1,
            )
            customer_type = st.selectbox("Customer type", customer_types)
            market_segment = st.selectbox(
                "Market segment",
                market_segs,
                index=market_segs.index("Online TA") if "Online TA" in market_segs else 0,
            )
            distribution_channel = st.selectbox(
                "Distribution channel",
                dist_channels,
                index=dist_channels.index("TA/TO") if "TA/TO" in dist_channels else 0,
            )
            deposit_type = st.selectbox("Deposit type", deposit_types)
            agent_display = st.selectbox(
                "Agent",
                agent_options_display,
                index=agent_options_display.index(default_agent_display) if default_agent_display in agent_options_display else 0,
            )
            has_company = st.checkbox("Booked through a company", value=False)

        submit_btn = st.form_submit_button("Predict cancellation risk", type="primary")

    # Map form display inputs to categorical string codes
    country_code = country_map.get(country_display, "Unknown")
    agent_code = agent_map.get(agent_display, "None")

    form_inputs = dict(
        hotel=hotel,
        arrival=arrival,
        nights=int(nights),
        lead_time=int(lead_time),
        adr=float(adr),
        adults=int(adults),
        children=float(children),
        babies=int(babies),
        is_repeated_guest=int(is_repeated_guest),
        previous_cancellations=int(previous_cancellations),
        previous_bookings_not_canceled=int(previous_stays),
        customer_type=customer_type,
        market_segment=market_segment,
        distribution_channel=distribution_channel,
        deposit_type=deposit_type,
        meal=meal,
        reserved_room_type=reserved_room_type,
        country=country_code,
        agent=agent_code,
        has_company=int(has_company),
    )

    # ── Predict on load or form submit ───────────────────────────────────────
    try:
        X = build_features(**form_inputs, num_cols=num_cols, cat_cols=cat_cols)
        prob = predict(model, calibrator, X)
        risk = band(prob, thresholds)
        policy_sugg = policy.get(risk, "No deposit")
    except Exception as exc:
        st.error(f"Prediction could not be completed. Error: {exc}")
        return

    # ── Results Metrics and Banner ───────────────────────────────────────────
    st.divider()
    st.subheader("Risk Assessment")

    booking_value = float(adr) * int(nights)
    expected_risk_value = prob * booking_value

    m1, m2, m3 = st.columns(3)
    m1.metric("Cancellation Probability", f"{prob:.1%}")
    m2.metric("Booking Value", f"€{booking_value:,.2f}")
    m3.metric("Expected Value at Risk", f"€{expected_risk_value:,.2f}")

    st.progress(float(min(max(prob, 0.0), 1.0)))

    banner_msg = f"{risk} risk. Suggested policy: {policy_sugg}"
    if risk == "Low":
        st.success(f"🟢 **{banner_msg}**")
    elif risk == "Medium":
        st.warning(f"🟡 **{banner_msg}**")
    else:
        st.error(f"🔴 **{banner_msg}**")

    # ── What-If Table ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("What-If Scenarios")

    scenarios = build_scenarios(meta)
    hotels_in_meta = categories.get("hotel", [])
    other_hotel = next((h for h in hotels_in_meta if h != hotel), None)

    what_if_list = []
    for s in scenarios:
        overrides = dict(s["overrides"])
        requires = s.get("requires_in_meta", {})

        # Resolve other hotel sentinel
        if overrides.get("hotel") == "__other__":
            if not other_hotel:
                continue
            overrides["hotel"] = other_hotel

        # Check required categories exist in metadata
        missing_cat = False
        for cat_name, cat_val in requires.items():
            if cat_val is not None and cat_val not in categories.get(cat_name, []):
                missing_cat = True
                break
        if missing_cat:
            continue

        # Skip if override matches current form value
        matches_current = True
        for k, v in overrides.items():
            if str(form_inputs.get(k, "")) != str(v):
                matches_current = False
                break
        if matches_current:
            continue

        scenario_inputs = {**form_inputs, **overrides}
        try:
            X_s = build_features(**scenario_inputs, num_cols=num_cols, cat_cols=cat_cols)
            prob_s = predict(model, calibrator, X_s)
            diff = prob_s - prob
            what_if_list.append({
                "Scenario": s["label"],
                "Probability": f"{prob_s:.1%}",
                "Change vs. current": f"{diff:+.1%}",
                "_abs_diff": abs(diff),
            })
        except Exception:
            continue

    if what_if_list:
        what_if_list.sort(key=lambda x: x["_abs_diff"], reverse=True)
        top5 = what_if_list[:5]
        df_what_if = pd.DataFrame([
            {
                "Scenario": item["Scenario"],
                "Probability": item["Probability"],
                "Change vs. current": item["Change vs. current"],
            }
            for item in top5
        ])
        st.dataframe(df_what_if, hide_index=True, width="stretch")
        st.caption(
            "Shows how the model reacts, not proven cause and effect."
        )
    else:
        st.info("No what-if scenarios currently differ from your selected inputs.")


if __name__ == "__main__":
    run_app()
