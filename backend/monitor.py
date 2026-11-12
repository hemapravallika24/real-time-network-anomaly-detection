
import csv
import os
import time
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



def create_detector():
    """
    Train the detector using synthetic normal traffic
    that matches the traffic collector's normal ranges.
    """

    import random

    rng = random.Random(42)

    training_records = []

    # Generate 500 synthetic normal-traffic records.
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



def save_event(record, result):
    """Append traffic records and detection results to the CSV."""

    os.makedirs(DATA_DIR, exist_ok=True)

    file_exists = (
        os.path.exists(LOG_FILE)
        and os.path.getsize(LOG_FILE) > 0
    )

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
            **{feature: record[feature] for feature in FEATURES},
            "status": result["status"],
            "anomaly_score": result["anomaly_score"],
            "risk_level": result["risk_level"],
            "reason": result["reason"]
        })

def main():
    detector = create_detector()

    print("=" * 55)
    print("REAL-TIME NETWORK ANOMALY MONITOR")
    print("=" * 55)
    print("Mode: SYNTHETIC TRAFFIC SIMULATION")
    print("Press Ctrl+C to stop.\n")

    total_records = 0
    total_anomalies = 0

    try:
        while True:
            # Generate one synthetic traffic record.
            record = generate_traffic()

            # Extract only the model's required features.
            features = {
                feature: record[feature]
                for feature in FEATURES
            }

            # Classify the traffic.
            result = detector.predict(features)

            # Save the record and its classification.
            save_event(record, result)

            total_records += 1

            if result["status"] == "Anomaly":
                total_anomalies += 1

            # Display the result.
            print(f"Timestamp: {record['timestamp']}")
            print(
                f"Packets: {record['packet_count']} | "
                f"Bytes: {record['bytes_transferred']} | "
                f"Duration: {record['connection_duration']}s | "
                f"Failed connections: {record['failed_connections']}"
            )
            print(
                f"Detection: {result['status']} | "
                f"Risk: {result['risk_level']} | "
                f"Score: {result['anomaly_score']}"
                )
            print(f"Reason: {result['reason']}")
            print(
                f"Records processed: {total_records} | "
                f"Anomalies detected: {total_anomalies}"
            )
            print("-" * 55)

            time.sleep(2)

    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
        print(f"Total records processed: {total_records}")
        print(f"Total anomalies detected: {total_anomalies}")
        print(f"Event log saved to: {LOG_FILE}")


if __name__ == "__main__":
    main()