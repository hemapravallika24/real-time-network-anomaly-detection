
import random

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from evaluate_model import generate_evaluation_data


def generate_training_data():
    """Generate the same 500-record synthetic normal baseline."""

    rng = random.Random(42)

    records = []

    for _ in range(500):
        records.append({
            "packet_count": rng.randint(80, 150),
            "bytes_transferred": rng.randint(4000, 6000),
            "connection_duration": rng.randint(5, 15),
            "failed_connections": rng.randint(0, 2),
        })

    return pd.DataFrame(records)


def main():

    print("=" * 75)
    print("ISOLATION FOREST - CONTAMINATION COMPARISON")
    print("=" * 75)

    features = [
        "packet_count",
        "bytes_transferred",
        "connection_duration",
        "failed_connections",
    ]

    # Generate identical training and evaluation data.
    training_data = generate_training_data()

    records, actual_labels = generate_evaluation_data()
    evaluation_data = pd.DataFrame(records)[features]

    contamination_values = [0.01, 0.02, 0.05, 0.10]

    print("\nTraining records:", len(training_data))
    print("Evaluation records:", len(evaluation_data))

    print("\nRESULTS")
    print("-" * 75)

    print(
        f"{'Contamination':<16}"
        f"{'Accuracy':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1-score':<12}"
        f"{'FPR':<12}"
    )

    for value in contamination_values:

        model = IsolationForest(
            n_estimators=200,
            contamination=value,
            random_state=42,
        )

        model.fit(training_data)

        raw_predictions = model.predict(evaluation_data)

        predictions = [
            1 if result == -1 else 0
            for result in raw_predictions
        ]

        tn, fp, fn, tp = confusion_matrix(
            actual_labels,
            predictions,
            labels=[0, 1],
        ).ravel()

        false_positive_rate = fp / (fp + tn) if (fp + tn) else 0

        print(
            f"{value:<16.2f}"
            f"{accuracy_score(actual_labels, predictions):<12.4f}"
            f"{precision_score(actual_labels, predictions, zero_division=0):<12.4f}"
            f"{recall_score(actual_labels, predictions, zero_division=0):<12.4f}"
            f"{f1_score(actual_labels, predictions, zero_division=0):<12.4f}"
            f"{false_positive_rate:<12.4f}"
        )

        print(
            f"  Confusion Matrix: TN={tn}, FP={fp}, "
            f"FN={fn}, TP={tp}"
        )

    print("\n" + "=" * 75)
    print("CONTAMINATION COMPARISON COMPLETED")
    print("Results apply only to the synthetic evaluation dataset.")
    print("=" * 75)


if __name__ == "__main__":
    main()
