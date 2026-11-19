
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

FEATURES = [
    "packet_count",
    "bytes_transferred",
    "connection_duration",
    "failed_connections"
]

rng = np.random.default_rng(2035)


def generate_normal_baseline(size, seed):
    random = np.random.default_rng(seed)

    return pd.DataFrame({
        "packet_count": random.integers(80, 151, size),
        "bytes_transferred": random.integers(4000, 6001, size),
        "connection_duration": random.integers(5, 16, size),
        "failed_connections": random.integers(0, 3, size)
    })


def generate_benign_profiles():
    profiles = {}

    profiles["Baseline"] = pd.DataFrame({
        "packet_count": rng.integers(80, 151, 180),
        "bytes_transferred": rng.integers(4000, 6001, 180),
        "connection_duration": rng.integers(5, 16, 180),
        "failed_connections": rng.integers(0, 3, 180)
    })

    profiles["High packet activity"] = pd.DataFrame({
        "packet_count": rng.integers(130, 221, 180),
        "bytes_transferred": rng.integers(4000, 6501, 180),
        "connection_duration": rng.integers(5, 21, 180),
        "failed_connections": rng.integers(0, 3, 180)
    })

    profiles["Large data transfers"] = pd.DataFrame({
        "packet_count": rng.integers(80, 181, 180),
        "bytes_transferred": rng.integers(5000, 7501, 180),
        "connection_duration": rng.integers(5, 26, 180),
        "failed_connections": rng.integers(0, 3, 180)
    })

    profiles["Long sessions"] = pd.DataFrame({
        "packet_count": rng.integers(80, 161, 180),
        "bytes_transferred": rng.integers(4000, 6501, 180),
        "connection_duration": rng.integers(15, 36, 180),
        "failed_connections": rng.integers(0, 3, 180)
    })

    profiles["Temporary retry bursts"] = pd.DataFrame({
        "packet_count": rng.integers(80, 171, 180),
        "bytes_transferred": rng.integers(4000, 6501, 180),
        "connection_duration": rng.integers(5, 26, 180),
        "failed_connections": rng.integers(0, 6, 180)
    })

    return profiles


def generate_injected_anomalies():
    records = []

    for _ in range(25):
        records.append({
            "packet_count": int(rng.integers(250, 501)),
            "bytes_transferred": int(rng.integers(4000, 6001)),
            "connection_duration": int(rng.integers(5, 16)),
            "failed_connections": int(rng.integers(0, 3))
        })

        records.append({
            "packet_count": int(rng.integers(80, 151)),
            "bytes_transferred": int(rng.integers(9000, 20001)),
            "connection_duration": int(rng.integers(5, 16)),
            "failed_connections": int(rng.integers(0, 3))
        })

        records.append({
            "packet_count": int(rng.integers(80, 151)),
            "bytes_transferred": int(rng.integers(4000, 6001)),
            "connection_duration": int(rng.integers(40, 91)),
            "failed_connections": int(rng.integers(0, 3))
        })

        records.append({
            "packet_count": int(rng.integers(80, 151)),
            "bytes_transferred": int(rng.integers(4000, 6001)),
            "connection_duration": int(rng.integers(5, 16)),
            "failed_connections": int(rng.integers(6, 16))
        })

    return pd.DataFrame(records)


def main():
    print("=" * 75)
    print("BENIGN NETWORK TRAFFIC VARIABILITY EXPERIMENT")
    print("=" * 75)

    # Train using baseline normal traffic
    training_data = generate_normal_baseline(500, 42)

    thresholds = {
        feature: np.percentile(training_data[feature], 99.5)
        for feature in FEATURES
    }

    print("\n99.5th percentile thresholds:")
    for feature, threshold in thresholds.items():
        print(f"{feature:25}: {threshold:.2f}")

    # Generate five different benign profiles
    profiles = generate_benign_profiles()

    normal_parts = []
    profile_labels = []

    for name, data in profiles.items():
        normal_parts.append(data)
        profile_labels.extend([name] * len(data))

    normal_data = pd.concat(normal_parts, ignore_index=True)

    # Generate 100 labelled synthetic anomaly scenarios
    anomaly_data = generate_injected_anomalies()

    normal_data["actual"] = 0
    anomaly_data["actual"] = 1

    normal_data["profile"] = profile_labels
    anomaly_data["profile"] = "Injected anomaly"

    evaluation_data = pd.concat(
        [normal_data, anomaly_data],
        ignore_index=True
    )

    # Apply statistical threshold rules
    evaluation_data["predicted"] = (
        evaluation_data[FEATURES]
        .gt(pd.Series(thresholds))
        .any(axis=1)
        .astype(int)
    )

    y_true = evaluation_data["actual"]
    y_pred = evaluation_data["predicted"]

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    print("\nOVERALL RESULTS")
    print("-" * 60)
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1 Score:  {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"TN: {tn} | FP: {fp} | FN: {fn} | TP: {tp}")

    print("\nFALSE POSITIVES BY BENIGN TRAFFIC PROFILE")
    print("-" * 60)

    profile_results = []

    for name in profiles:
        subset = evaluation_data[
            evaluation_data["profile"] == name
        ]

        false_positives = int(subset["predicted"].sum())

        profile_results.append({
            "profile": name,
            "records": len(subset),
            "false_positives": false_positives,
            "false_positive_rate": false_positives / len(subset)
        })

        print(
            f"{name:25}: "
            f"{false_positives:3}/{len(subset)} "
            f"({false_positives / len(subset):.2%})"
        )

    print("\nINJECTED ANOMALY DETECTION")
    print("-" * 60)

    anomaly_results = evaluation_data[
        evaluation_data["actual"] == 1
    ]

    print(
        f"Detected: {anomaly_results['predicted'].sum()}/"
        f"{len(anomaly_results)}"
    )

    evaluation_data.to_csv(
        "benign_variability_results.csv",
        index=False
    )

    pd.DataFrame(profile_results).to_csv(
        "benign_profile_summary.csv",
        index=False
    )

    print("\nSaved:")
    print("1. benign_variability_results.csv")
    print("2. benign_profile_summary.csv")

    print("\nEXPERIMENT COMPLETED")


if __name__ == "__main__":
    main()
