
import random
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


def generate_normal_record(rng):
    return {
        "packet_count": rng.randint(80, 150),
        "bytes_transferred": rng.randint(4000, 6000),
        "connection_duration": rng.randint(5, 15),
        "failed_connections": rng.randint(0, 2)
    }


def show_metrics(labels, predictions, title):
    tn, fp, fn, tp = confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()

    print(f"\n{title}")
    print("-" * 60)
    print(f"Accuracy:  {accuracy_score(labels, predictions):.4f}")
    print(f"Precision: {precision_score(labels, predictions, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(labels, predictions, zero_division=0):.4f}")
    print(f"F1 Score:  {f1_score(labels, predictions, zero_division=0):.4f}")
    print(f"TN: {tn} | FP: {fp} | FN: {fn} | TP: {tp}")


def main():
    print("=" * 70)
    print("FEATURE-SPECIFIC THRESHOLD OPTIMIZATION")
    print("=" * 70)

    # 1. Generate normal training data.
    train_rng = random.Random(42)

    training_data = pd.DataFrame([
        generate_normal_record(train_rng)
        for _ in range(500)
    ])

    # 2. Calculate different thresholds using training data only.
    threshold_sets = {
        "Mean + 3 Std": (
            training_data[FEATURES].mean()
            + 3 * training_data[FEATURES].std()
        ),
        "95th Percentile": training_data[FEATURES].quantile(0.95),
        "97.5th Percentile": training_data[FEATURES].quantile(0.975),
        "99th Percentile": training_data[FEATURES].quantile(0.99)
    }

    print("\nCALCULATED FEATURE THRESHOLDS")
    print("-" * 70)

    for method, thresholds in threshold_sets.items():
        print(f"\n{method}")
        for feature in FEATURES:
            print(f"{feature:25}: {thresholds[feature]:.2f}")

    # 3. Train Isolation Forest.
    model = IsolationForest(
        n_estimators=200,
        contamination=0.01,
        random_state=42
    )
    model.fit(training_data[FEATURES])

    # 4. Generate the same style of borderline evaluation data.
    eval_rng = random.Random(2030)

    records = []
    labels = []
    categories = []

    # Normal records.
    for _ in range(900):
        records.append(generate_normal_record(eval_rng))
        labels.append(0)
        categories.append("Normal")

    # Borderline anomaly records: 25 per feature.
    # Their values are generated around the Mean + 3 Std threshold.
    reference_thresholds = threshold_sets["Mean + 3 Std"]

    for feature in FEATURES:
        for _ in range(25):
            record = generate_normal_record(eval_rng)

            threshold = reference_thresholds[feature]

            lower = max(0, int(threshold * 0.80))
            upper = max(lower, int(threshold * 1.20))

            record[feature] = eval_rng.randint(lower, upper)

            records.append(record)
            labels.append(1)
            categories.append(feature)

    evaluation_data = pd.DataFrame(records)
    y_true = labels

    # 5. Evaluate Isolation Forest alone.
    if_predictions = (
        model.predict(evaluation_data[FEATURES]) == -1
    ).astype(int).tolist()

    show_metrics(
        y_true,
        if_predictions,
        "ISOLATION FOREST ONLY - 0.01"
    )

    # 6. Evaluate each threshold strategy and its hybrid.
    for method, thresholds in threshold_sets.items():

        rule_predictions = (
            evaluation_data[FEATURES] > thresholds
        ).any(axis=1).astype(int).tolist()

        hybrid_predictions = [
            int(if_result == 1 or rule_result == 1)
            for if_result, rule_result
            in zip(if_predictions, rule_predictions)
        ]

        show_metrics(
            y_true,
            rule_predictions,
            f"STATISTICAL RULES - {method}"
        )

        show_metrics(
            y_true,
            hybrid_predictions,
            f"HYBRID - {method}"
        )

        print("\nCategory-wise rule detection:")

        for category in FEATURES:
            indices = [
                i for i, value in enumerate(categories)
                if value == category
            ]

            detected = sum(
                rule_predictions[i] for i in indices
            )

            print(
                f"{category:25}: "
                f"{detected}/25 "
                f"({detected / 25 * 100:.2f}%)"
            )

    print("\n" + "=" * 70)
    print("FEATURE THRESHOLD EXPERIMENT COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
