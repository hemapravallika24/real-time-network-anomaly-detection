
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
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

PROFILE_NAMES = [
    "Baseline",
    "High packet activity",
    "Large data transfers",
    "Long sessions",
    "Temporary retry bursts"
]


def generate_profile(name, size, rng):
    if name == "Baseline":
        data = {
            "packet_count": rng.integers(80, 151, size),
            "bytes_transferred": rng.integers(4000, 6001, size),
            "connection_duration": rng.integers(5, 16, size),
            "failed_connections": rng.integers(0, 3, size)
        }

    elif name == "High packet activity":
        data = {
            "packet_count": rng.integers(130, 221, size),
            "bytes_transferred": rng.integers(4000, 6501, size),
            "connection_duration": rng.integers(5, 21, size),
            "failed_connections": rng.integers(0, 3, size)
        }

    elif name == "Large data transfers":
        data = {
            "packet_count": rng.integers(80, 181, size),
            "bytes_transferred": rng.integers(5000, 7501, size),
            "connection_duration": rng.integers(5, 26, size),
            "failed_connections": rng.integers(0, 3, size)
        }

    elif name == "Long sessions":
        data = {
            "packet_count": rng.integers(80, 161, size),
            "bytes_transferred": rng.integers(4000, 6501, size),
            "connection_duration": rng.integers(15, 36, size),
            "failed_connections": rng.integers(0, 3, size)
        }

    else:
        data = {
            "packet_count": rng.integers(80, 171, size),
            "bytes_transferred": rng.integers(4000, 6501, size),
            "connection_duration": rng.integers(5, 26, size),
            "failed_connections": rng.integers(0, 6, size)
        }

    return pd.DataFrame(data)


def generate_anomalies(rng):
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


def calculate_metrics(y_true, predictions):
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }


def main():
    print("=" * 75)
    print("DIVERSE NORMAL TRAINING EXPERIMENT")
    print("=" * 75)

    # Step 1: Generate diverse training data
    train_rng = np.random.default_rng(2040)

    training_parts = [
        generate_profile(name, 500, train_rng)
        for name in PROFILE_NAMES
    ]

    training_data = pd.concat(
        training_parts, ignore_index=True
    )

    print("\nTraining records:", len(training_data))
    print("Training profiles:", len(PROFILE_NAMES))

    # Step 2: Calculate thresholds from diverse training data
    thresholds = {
        feature: np.percentile(
            training_data[feature], 99.5
        )
        for feature in FEATURES
    }

    print("\nDiverse training thresholds:")
    for feature, threshold in thresholds.items():
        print(f"{feature:25}: {threshold:.2f}")

    # Step 3: Train Isolation Forest
    model = IsolationForest(
        n_estimators=300,
        contamination=0.01,
        random_state=42
    )

    model.fit(training_data[FEATURES])

    # Step 4: Generate unseen evaluation records
    test_rng = np.random.default_rng(2041)

    evaluation_parts = []

    for name in PROFILE_NAMES:
        part = generate_profile(name, 180, test_rng)
        part["actual"] = 0
        part["profile"] = name
        evaluation_parts.append(part)

    anomaly_data = generate_anomalies(test_rng)
    anomaly_data["actual"] = 1
    anomaly_data["profile"] = "Injected anomaly"

    evaluation = pd.concat(
        evaluation_parts + [anomaly_data],
        ignore_index=True
    )

    X_test = evaluation[FEATURES]

    # Step 5: Generate predictions
    exceed_count = X_test.gt(
        pd.Series(thresholds)
    ).sum(axis=1)

    predictions = {}

    predictions["Isolation Forest"] = (
        model.predict(X_test) == -1
    ).astype(int)

    predictions["Rules - Any feature"] = (
        exceed_count >= 1
    ).astype(int)

    predictions["Rules - Two or more features"] = (
        exceed_count >= 2
    ).astype(int)

    predictions["Hybrid - IF OR two features"] = (
        (predictions["Isolation Forest"] == 1)
        | (predictions["Rules - Two or more features"] == 1)
    ).astype(int)

    predictions["Hybrid - IF AND two features"] = (
        (predictions["Isolation Forest"] == 1)
        & (predictions["Rules - Two or more features"] == 1)
    ).astype(int)

    # Step 6: Display overall results
    y_true = evaluation["actual"]

    print("\nOVERALL MODEL COMPARISON")
    print("-" * 75)

    summary = []

    for name, prediction in predictions.items():
        metrics = calculate_metrics(y_true, prediction)

        summary.append({
            "method": name,
            **metrics
        })

        print(f"\n{name}")
        print("-" * 55)

        for metric in [
            "accuracy", "precision", "recall", "f1"
        ]:
            print(f"{metric.capitalize():12}: {metrics[metric]:.4f}")

        print(
            f"TN: {metrics['TN']} | FP: {metrics['FP']} | "
            f"FN: {metrics['FN']} | TP: {metrics['TP']}"
        )

    # Step 7: Compare false positives by profile
    print("\nFALSE POSITIVE RATE BY BENIGN PROFILE")
    print("-" * 75)

    profile_summary = []

    for profile in PROFILE_NAMES:
        subset = evaluation[
            evaluation["profile"] == profile
        ]

        row = {"profile": profile}

        print(f"\n{profile}")

        for name, prediction in predictions.items():
            profile_predictions = prediction[subset.index]
            false_positives = int(profile_predictions.sum())
            rate = false_positives / len(subset)

            row[name] = rate

            print(
                f"{name:35}: "
                f"{false_positives}/{len(subset)} ({rate:.2%})"
            )

        profile_summary.append(row)

    # Step 8: Save results
    for name, prediction in predictions.items():
        evaluation[name] = prediction

    pd.DataFrame(summary).to_csv(
        "diverse_training_model_comparison.csv",
        index=False
    )

    pd.DataFrame(profile_summary).to_csv(
        "diverse_training_profile_results.csv",
        index=False
    )

    evaluation.to_csv(
        "diverse_training_evaluation.csv",
        index=False
    )

    print("\nSaved files:")
    print("1. diverse_training_model_comparison.csv")
    print("2. diverse_training_profile_results.csv")
    print("3. diverse_training_evaluation.csv")

    print("\nEXPERIMENT COMPLETED")


if __name__ == "__main__":
    main()
