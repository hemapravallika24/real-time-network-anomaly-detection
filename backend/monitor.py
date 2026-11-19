
import csv
import os
import time
from collections import deque
from datetime import datetime

import pandas as pd

from backend.detector import NetworkAnomalyDetector
from backend.traffic_collector import generate_traffic


# Project paths
DATA_DIR = "data"
LOG_FILE = os.path.join(DATA_DIR, "network_events.csv")

FEATURES = [
    "packet_count",
    "bytes_transferred",
    "connection_duration",
    "failed_connections"
]

# Persistence configuration
PERSISTENCE_WINDOW = 5
PERSISTENCE_REQUIRED = 3

RISK_PRIORITY = {
    "Low": 1,
    "Medium": 2,
    "High": 3
}


def create_detector():
    """
    Train the detector using synthetic normal traffic
    that matches the traffic collector's normal ranges.
    """

    import random

    rng = random.Random(42)

    training_records = []

    for _ in range(500):
        training_records.append({
            "packet_count": rng.randint(80, 150),
            "bytes_transferred": rng.randint(4000, 6000),
            "connection_duration": rng.randint(5, 15),
            "failed_connections": rng.randint(0, 2)
        })

    training_data = pd.DataFrame(training_records)

    detector = NetworkAnomalyDetector()
    detector.train(training_data)

    print("Detector trained using 500 synthetic normal baseline records.")

    return detector


def apply_persistence(raw_result, history):
    """
    Raise a final anomaly alert when at least 3 of the
    latest 5 observations are classified as anomalous.
    """

    # Store the current raw detector result.
    history.append(raw_result.copy())

    # Count raw anomaly classifications in the rolling window.
    anomaly_results = [
        item for item in history
        if item["status"] == "Anomaly"
    ]

    anomaly_count = len(anomaly_results)

    final_result = raw_result.copy()

    # Confirm a persistent anomaly.
    if anomaly_count >= PERSISTENCE_REQUIRED:

        final_result["status"] = "Anomaly"

        # Retain the highest risk level among anomalous
        # observations in the current window.
        final_result["risk_level"] = max(
            (
                item.get("risk_level", "Low")
                for item in anomaly_results
            ),
            key=lambda risk: RISK_PRIORITY.get(risk, 0)
        )

        # Combine the available anomaly reasons.
        reasons = list(dict.fromkeys(
            item.get("reason", "")
            for item in anomaly_results
            if item.get("reason")
        ))

        final_result["reason"] = (
            "; ".join(reasons)
            if reasons
            else "Persistent anomaly confirmed"
        )

    else:
        # Do not raise an alert before persistence is confirmed.
        final_result["status"] = "Normal"
        final_result["risk_level"] = (
            raw_result.get("risk_level", "Low")
            if raw_result["status"] == "Anomaly"
            else "Low"
            )
        if raw_result["status"] == "Anomaly":
            final_result["reason"] = (
                f"Raw anomaly awaiting persistence confirmation "
                f"({anomaly_count}/{PERSISTENCE_WINDOW})"
            )
        else:
            final_result["reason"] = (
                f"No persistent anomaly "
                f"({anomaly_count}/{PERSISTENCE_WINDOW})"
            )

    return final_result, anomaly_count


def save_event(record, result):
    """Append traffic records and detection results to the CSV."""

    os.makedirs(DATA_DIR, exist_ok=True)

    file_exists = (
        os.path.exists(LOG_FILE)
        and os.path.getsize(LOG_FILE) > 0
    )

    # Keep the existing CSV schema for dashboard compatibility.
    fieldnames = [
        "timestamp",
        *FEATURES,
        "status",
        "anomaly_score",
        "risk_level",
        "reason"
    ]

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "timestamp": record["timestamp"],
            **{
                feature: record[feature]
                for feature in FEATURES
            },
            "status": result["status"],
            "anomaly_score": result["anomaly_score"],
            "risk_level": result["risk_level"],
            "reason": result["reason"]
        })


def main():
    detector = create_detector()

    # Rolling history automatically keeps only the latest 5 results.
    history = deque(maxlen=PERSISTENCE_WINDOW)

    print("=" * 60)
    print("REAL-TIME NETWORK ANOMALY MONITOR")
    print("=" * 60)
    print("Mode: SYNTHETIC TRAFFIC SIMULATION")
    print("Detection: 3-out-of-5 persistence")
    print("Monitoring interval: 2 seconds")
    print("Press Ctrl+C to stop.\n")

    total_records = 0
    total_raw_anomalies = 0
    total_persistent_alerts = 0

    try:
        while True:

            # Generate synthetic traffic.
            record = generate_traffic()

            # Extract required features.
            features = {
                feature: record[feature]
                for feature in FEATURES
            }

            # Run the original detector.
            raw_result = detector.predict(features)

            if raw_result["status"] == "Anomaly":
                total_raw_anomalies += 1

            # Apply the persistence mechanism.
            result, anomaly_count = apply_persistence(
                raw_result,
                history
            )

            # Save the final classification.
            save_event(record, result)

            total_records += 1

            if result["status"] == "Anomaly":
                total_persistent_alerts += 1

            # Display monitoring information.
            print(f"Timestamp: {record['timestamp']}")

            print(
                f"Packets: {record['packet_count']} | "
                f"Bytes: {record['bytes_transferred']} | "
                f"Duration: {record['connection_duration']}s | "
                f"Failed connections: {record['failed_connections']}"
            )

            print(
                f"Raw detection: {raw_result['status']} | "
                f"Final detection: {result['status']}"
            )

            print(
                f"Persistence: {anomaly_count}/"
                f"{PERSISTENCE_WINDOW} | "
                f"Required: {PERSISTENCE_REQUIRED}"
            )

            print(
                f"Risk: {result['risk_level']} | "
                f"Score: {result['anomaly_score']}"
            )

            print(f"Reason: {result['reason']}")

            print(
                f"Records processed: {total_records} | "
                f"Raw anomalies: {total_raw_anomalies} | "
                f"Persistent alert records: {total_persistent_alerts}"
            )

            print("-" * 60)

            time.sleep(2)

    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
        print(f"Total records processed: {total_records}")
        print(f"Raw anomalies: {total_raw_anomalies}")
        print(
            f"Persistent alert records: {total_persistent_alerts}"
        )
        print(f"Event log saved to: {LOG_FILE}")


if __name__ == "__main__":
    main()
