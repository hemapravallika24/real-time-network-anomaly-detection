
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
    rng = random.Random(2028)
    dataset = []

    # 900 normal records.
    for _ in range(900):
        traffic = {
            "packet_count": rng.randint(80, 150),
            "bytes_transferred": rng.randint(4000, 6000),
            "connection_duration": rng.randint(5, 15),
            "failed_connections": rng.randint(0, 2),
        }
        dataset.append((traffic, 0, "Normal"))

    scenarios = [
        (
            "High packet count",
            lambda: {
                "packet_count": rng.randint(300, 600),
                "bytes_transferred": rng.randint(4000, 6000),
                "connection_duration": rng.randint(5, 15),
                "failed_connections": rng.randint(0, 2),
            },
        ),
        (
            "High data transfer",
            lambda: {
                "packet_count": rng.randint(80, 150),
                "bytes_transferred": rng.randint(15000, 30000),
                "connection_duration": rng.randint(5, 15),
                "failed_connections": rng.randint(0, 2),
            },
        ),
        (
            "Long connection duration",
            lambda: {
                "packet_count": rng.randint(80, 150),
                "bytes_transferred": rng.randint(4000, 6000),
                "connection_duration": rng.randint(40, 80),
                "failed_connections": rng.randint(0, 2),
            },
        ),
        (
            "High failed connections",
            lambda: {
                "packet_count": rng.randint(80, 150),
                "bytes_transferred": rng.randint(4000, 6000),
                "connection_duration": rng.randint(5, 15),
                "failed_connections": rng.randint(5, 12),
            },
        ),
    ]

    for category, generator in scenarios:
        for _ in range(25):
            dataset.append((generator(), 1, category))

    rng.shuffle(dataset)

    records = [item[0] for item in dataset]
    labels = [item[1] for item in dataset]
    categories = [item[2] for item in dataset]

    return pd.DataFrame(records)[FEATURES], labels, categories


def main():
    print("=" * 70)
    print("CATEGORY-WISE NETWORK ANOMALY EVALUATION")
    print("=" * 70)

    training_data = generate_training_data()
    evaluation_data, actual_labels, categories = generate_evaluation_data()

    print("\nTraining records:", len(training_data))
    print("Evaluation records:", len(evaluation_data))

    for contamination in [0.01, 0.05]:

        model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
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

        print(f"\nCONTAMINATION: {contamination}")
        print("-" * 70)

        print(f"Accuracy:  {accuracy_score(actual_labels, predictions):.4f}")
        print(f"Precision: {precision_score(actual_labels, predictions, zero_division=0):.4f}")
        print(f"Recall:    {recall_score(actual_labels, predictions, zero_division=0):.4f}")
        print(f"F1-score:  {f1_score(actual_labels, predictions, zero_division=0):.4f}")

        print("\nCATEGORY-WISE DETECTION")
        print("-" * 70)

        for category in [
            "High packet count",
            "High data transfer",
            "Long connection duration",
            "High failed connections",
        ]:
            indices = [
                i for i, item in enumerate(categories)
                if item == category
            ]

            detected = sum(predictions[i] for i in indices)
            total = len(indices)

            print(
                f"{category:<30} "
                f"{detected}/{total} detected "
                f"({detected / total * 100:.2f}%)"
            )

        normal_indices = [
            i for i, item in enumerate(categories)
            if item == "Normal"
        ]

        normal_false_positives = sum(
            predictions[i] for i in normal_indices
        )

        print("\nNORMAL TRAFFIC")
        print(
            f"False positives: {normal_false_positives}/"
            f"{len(normal_indices)}"
        )

        print(
            f"Confusion Matrix: TN={tn}, FP={fp}, "
            f"FN={fn}, TP={tp}"
        )

    print("\n" + "=" * 70)
    print("CATEGORY-WISE EVALUATION COMPLETED")
    print("Results apply only to this synthetic dataset.")
    print("=" * 70)


if __name__ == "__main__":
    main()
