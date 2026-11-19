
import random

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from backend.monitor import create_detector


def generate_evaluation_data():
    """Generate reproducible synthetic traffic with known labels."""

    rng = random.Random(2026)
    dataset = []

    # Generate 900 normal traffic records.
    for _ in range(900):
        traffic = {
            "packet_count": rng.randint(80, 150),
            "bytes_transferred": rng.randint(4000, 6000),
            "connection_duration": rng.randint(5, 15),
            "failed_connections": rng.randint(0, 2),
        }

        dataset.append((traffic, 0))

    # Generate 100 anomalous traffic records.
    for _ in range(100):
        traffic = {
            "packet_count": rng.randint(2000, 10000),
            "bytes_transferred": rng.randint(100000, 900000),
            "connection_duration": rng.randint(100, 300),
            "failed_connections": rng.randint(10, 100),
        }

        dataset.append((traffic, 1))

    rng.shuffle(dataset)

    records = [item[0] for item in dataset]
    labels = [item[1] for item in dataset]

    return records, labels


def display_metrics(name, actual, predicted):
    """Calculate and display classification metrics."""

    tn, fp, fn, tp = confusion_matrix(
        actual, predicted, labels=[0, 1]
    ).ravel()

    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0

    print(f"\n{name}")
    print("-" * 55)

    print(f"Accuracy:           {accuracy_score(actual, predicted):.4f}")
    print(f"Precision:          {precision_score(actual, predicted, zero_division=0):.4f}")
    print(f"Recall:             {recall_score(actual, predicted, zero_division=0):.4f}")
    print(f"F1-score:           {f1_score(actual, predicted, zero_division=0):.4f}")
    print(f"False Positive Rate: {false_positive_rate:.4f}")

    print("\nConfusion Matrix:")
    print(f"True Negatives:  {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives:  {tp}")


def main():
    print("=" * 60)
    print("NETWORK ANOMALY DETECTOR - MODEL EVALUATION")
    print("=" * 60)

    # Generate labeled evaluation data.
    records, actual_labels = generate_evaluation_data()

    print(f"\nTotal evaluation records: {len(records)}")
    print(f"Normal records: {actual_labels.count(0)}")
    print(f"Anomalous records: {actual_labels.count(1)}")

    # Train detector using the existing synthetic normal baseline.
    detector = create_detector()

    # Evaluate Isolation Forest alone.
    feature_data = pd.DataFrame(records)[detector.features]

    model_results = detector.model.predict(feature_data)

    model_predictions = [
        1 if result == -1 else 0
        for result in model_results
    ]

    display_metrics(
        "ISOLATION FOREST ONLY",
        actual_labels,
        model_predictions,
    )

    # Evaluate the complete detector:
    # Isolation Forest + configured anomaly rules.
    combined_predictions = []

    for record in records:
        result = detector.predict(record)

        prediction = 1 if result["status"] == "Anomaly" else 0
        combined_predictions.append(prediction)

    display_metrics(
        "COMBINED DETECTOR (MODEL + RULES)",
        actual_labels,
        combined_predictions,
    )

    print("\n" + "=" * 60)
    print("Evaluation completed.")
    print("Note: Results apply only to this synthetic dataset.")
    print("=" * 60)


if __name__ == "__main__":
    main()
