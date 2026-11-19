
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


FEATURES = [
    "packet_count",
    "bytes_transferred",
    "connection_duration",
    "failed_connections",
]


def generate_training_data():
    """Generate 500 normal baseline records."""

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


def generate_evaluation_data():
    """Generate normal and subtle anomalous traffic."""

    rng = random.Random(2027)
    records = []
    labels = []

    # 900 normal records.
    for _ in range(900):
        records.append({
            "packet_count": rng.randint(80, 150),
            "bytes_transferred": rng.randint(4000, 6000),
            "connection_duration": rng.randint(5, 15),
            "failed_connections": rng.randint(0, 2),
        })
        labels.append(0)

    # 100 subtle anomalous records.
    for _ in range(100):
        records.append({
            "packet_count": rng.randint(160, 250),
            "bytes_transferred": rng.randint(6500, 10000),
            "connection_duration": rng.randint(16, 30),
            "failed_connections": rng.randint(3, 8),
        })
        labels.append(1)

    combined = list(zip(records, labels))
    rng.shuffle(combined)

    records = [item[0] for item in combined]
    labels = [item[1] for item in combined]

    return pd.DataFrame(records)[FEATURES], labels


def evaluate_model(model, evaluation_data, actual_labels, name):

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

    print(f"\n{name}")
    print("-" * 60)

    print(f"Accuracy:            {accuracy_score(actual_labels, predictions):.4f}")
    print(f"Precision:           {precision_score(actual_labels, predictions, zero_division=0):.4f}")
    print(f"Recall:              {recall_score(actual_labels, predictions, zero_division=0):.4f}")
    print(f"F1-score:            {f1_score(actual_labels, predictions, zero_division=0):.4f}")
    print(f"False Positive Rate: {false_positive_rate:.4f}")

    print("\nConfusion Matrix:")
    print(f"True Negatives:  {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives:  {tp}")


def main():

    print("=" * 65)
    print("SUBTLE NETWORK ANOMALY EVALUATION")
    print("=" * 65)

    training_data = generate_training_data()
    evaluation_data, actual_labels = generate_evaluation_data()

    print("\nTraining records:", len(training_data))
    print("Evaluation records:", len(evaluation_data))
    print("Normal records:", actual_labels.count(0))
    print("Subtle anomalous records:", actual_labels.count(1))

    for contamination in [0.01, 0.05]:

        model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42,
        )

        model.fit(training_data)

        evaluate_model(
            model,
            evaluation_data,
            actual_labels,
            f"ISOLATION FOREST - CONTAMINATION {contamination}",
        )

    print("\n" + "=" * 65)
    print("SUBTLE ANOMALY EVALUATION COMPLETED")
    print("Results apply only to this synthetic dataset.")
    print("=" * 65)


if __name__ == "__main__":
    main()
