
import os

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh


# -----------------------------
# Configuration
# -----------------------------

st.set_page_config(
    page_title="Network Security Monitor",
    page_icon="🛡️",
    layout="wide"
)

# Automatically refresh every 3 seconds
st_autorefresh(
    interval=3000,
    key="network_dashboard_refresh"
)

LOG_FILE = os.path.join("data", "network_events.csv")

st.title("🛡️ Network Anomaly Detection & Security Monitoring")

st.caption(
    "Cloud-ready monitoring prototype | "
    "Current mode: Synthetic network traffic simulation"
)


# -----------------------------
# Load event records
# -----------------------------

@st.cache_data(ttl=2)
def load_events():
    if not os.path.exists(LOG_FILE):
        return pd.DataFrame()

    try:
        return pd.read_csv(LOG_FILE)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


df = load_events()


# -----------------------------
# Dashboard metrics
# -----------------------------

if df.empty:

    st.warning(
        "No events found. Start the monitoring process first."
    )

    st.info(
        "Run `python -m backend.monitor` in a separate terminal "
        "to generate traffic records."
    )

else:

    # Ensure expected columns exist
    expected_columns = [
        "timestamp",
        "packet_count",
        "bytes_transferred",
        "connection_duration",
        "failed_connections",
        "status",
        "anomaly_score",
        "risk_level",
        "reason"
    ]

    for column in expected_columns:
        if column not in df.columns:
            df[column] = "Unknown"

    df["status"] = df["status"].astype(str).str.strip()
    df["risk_level"] = df["risk_level"].fillna("Unknown").astype(str)
    df["reason"] = df["reason"].fillna("Not recorded").astype(str)

    # Convert numeric columns safely
    numeric_columns = [
        "packet_count",
        "bytes_transferred",
        "connection_duration",
        "failed_connections",
        "anomaly_score"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    # -----------------------------
    # Summary statistics
    # -----------------------------

    total_records = len(df)

    anomaly_count = int(
        (df["status"] == "Anomaly").sum()
    )

    normal_count = int(
        (df["status"] == "Normal").sum()
    )

    anomaly_percentage = (
        anomaly_count / total_records * 100
        if total_records else 0
    )

    known_risk = df[
        df["risk_level"].isin(["Low", "Medium", "High"])
    ]

    high_risk_count = int(
        (known_risk["risk_level"] == "High").sum()
    )

    medium_risk_count = int(
        (known_risk["risk_level"] == "Medium").sum()
    )

    low_risk_count = int(
        (known_risk["risk_level"] == "Low").sum()
    )

    unknown_risk_count = int(
        (df["risk_level"] == "Unknown").sum()
    )

    # -----------------------------
    # Main metrics
    # -----------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Traffic Records",
        total_records
    )

    col2.metric(
        "Anomalies Detected",
        anomaly_count
    )

    col3.metric(
        "Normal Records",
        normal_count
    )

    col4.metric(
        "Anomaly Percentage",
        f"{anomaly_percentage:.1f}%"
    )

    st.divider()

    # -----------------------------
    # Risk level metrics
    # -----------------------------

    st.subheader("🚦 Risk Level Summary")

    risk1, risk2, risk3, risk4 = st.columns(4)

    risk1.metric("🔴 High Risk", high_risk_count)
    risk2.metric("🟠 Medium Risk", medium_risk_count)
    risk3.metric("🟢 Low Risk", low_risk_count)
    risk4.metric("⚪ Unknown Risk", unknown_risk_count)

    st.caption(
        "Historical records may show Unknown because the earlier "
        "detector did not record risk levels."
    )

    st.divider()

    # -----------------------------
    # Traffic visualizations
    # -----------------------------

    left, right = st.columns(2)

    with left:

        st.subheader("Traffic Classification")

        classification_counts = (
            df["status"]
            .value_counts()
            .reindex(
                ["Normal", "Anomaly"],
                fill_value=0
            )
        )

        st.bar_chart(classification_counts)

    with right:

        st.subheader("Risk Level Distribution")

        risk_counts = (
            df["risk_level"]
            .value_counts()
            .reindex(
                ["High", "Medium", "Low", "Unknown"],
                fill_value=0
            )
        )

        st.bar_chart(risk_counts)

    # -----------------------------
    # Packet count trend
    # -----------------------------

    st.subheader("📈 Packet Count Over Time")

    chart_data = df.dropna(
        subset=["timestamp", "packet_count"]
    ).copy()

    if not chart_data.empty:

        chart_data = chart_data.sort_values("timestamp")
        chart_data = chart_data.set_index("timestamp")

        st.line_chart(
            chart_data["packet_count"]
        )

    else:
        st.info("No valid timestamp or packet-count data is available.")

    st.divider()

    # -----------------------------
    # Anomaly events
    # -----------------------------

    st.subheader("🚨 Detected Anomaly Events")

    anomalies = df[
        df["status"].str.casefold() == "anomaly"
    ].copy()

    if anomalies.empty:

        st.success(
            "No anomalies have been flagged in the current log."
        )

    else:

        anomaly_columns = [
            "timestamp",
            "packet_count",
            "bytes_transferred",
            "connection_duration",
            "failed_connections",
            "anomaly_score",
            "risk_level",
            "reason"
        ]

        anomalies = anomalies.sort_values(
            "timestamp",
            ascending=False,
            na_position="last"
        )

        st.dataframe(
            anomalies[anomaly_columns],
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    # -----------------------------
    # Filterable event log
    # -----------------------------

    st.subheader("📋 Network Traffic Event Log")

    available_statuses = sorted(
        df["status"].dropna().unique().tolist()
    )

    selected_statuses = st.multiselect(
        "Filter by classification",
        options=available_statuses,
        default=available_statuses
    )

    available_risks = sorted(
        df["risk_level"].dropna().unique().tolist()
    )

    selected_risks = st.multiselect(
        "Filter by risk level",
        options=available_risks,
        default=available_risks
    )

    filtered_df = df[
        df["status"].isin(selected_statuses)
        & df["risk_level"].isin(selected_risks)
    ].copy()

    filtered_df = filtered_df.sort_values(
        "timestamp",
        ascending=False,
        na_position="last"
    )

    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        f"Displaying {len(filtered_df)} of {total_records} records."
    )

    # -----------------------------
    # Download reports
    # -----------------------------

    st.subheader("📥 Export Event Records")

    st.download_button(
        label="Download All Events (CSV)",
        data=df.to_csv(index=False),
        file_name="network_events.csv",
        mime="text/csv"
    )

    st.download_button(
        label="Download Filtered Events (CSV)",
        data=filtered_df.to_csv(index=False),
        file_name="filtered_network_events.csv",
        mime="text/csv"
    )


# -----------------------------
# Refresh controls
# -----------------------------

st.divider()

if st.button("🔄 Refresh Dashboard"):
    st.cache_data.clear()
    st.rerun()

st.caption(
    "Note: Anomaly classifications are model predictions, "
    "not confirmed attacks. Traffic is currently simulated. "
    "Risk thresholds have not been validated against real network attacks."
)