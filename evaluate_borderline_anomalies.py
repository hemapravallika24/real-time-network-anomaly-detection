
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


def evaluate(y_true, predictions, title):
    print(f"\n{title}")
    print("-" * 60)

    print(f"Accuracy:  {accuracy_score(y_true, predictions):.4f}")
    print(f"Precision: {precision_score(y_true, predictions, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_true, predictions, zero_division=0):.4f}")
    print(f"F1 Score:  {f1_score(y_true, predictions, zero_division=0):.4f}")

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    print(f"TN: {tn} | FP: {fp} | FN: {fn} | TP: {tp}")


def main():
    print("=" * 65)
    print("BORDERLINE NETWORK ANOMALY EVALUATION")
    print("=" * 65)

    # Step 1: Generate synthetic normal training data.
    train_rng = random.Random(42)

    training_data = pd.DataFrame([
        generate_normal_record(train_rng)
        for _ in range(500)
    ])

    # Step 2: Calculate statistical thresholds.
    thresholds = {}

    for feature in FEATURES:
        thresholds[feature] = (
            training_data[feature].mean()
            + 3 * training_data[feature].std()
        )

    print("\nCalculated thresholds:")
    for feature, threshold in thresholds.items():
        print(f"{feature:25}: {threshold:.2f}")

    # Step 3: Train Isolation Forest.
    models = {}

    for contamination in [0.01, 0.05]:
        model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42
        )
        model.fit(training_data[FEATURES])
        models[contamination] = model

    # Step 4: Generate evaluation data.
    eval_rng = random.Random(2030)

    records = []
    labels = []
    categories = []

    # 900 normal records.
    for _ in range(900):
        records.append(generate_normal_record(eval_rng))
        labels.append(0)
        categories.append("Normal")

    # 100 borderline anomaly records: 25 per feature.
    for feature in FEATURES:
        for _ in range(25):
            record = generate_normal_record(eval_rng)

            threshold = thresholds[feature]

            # Generate values around the threshold,
            # including values below and above it.
            lower = max(0, int(threshold * 0.80))
            upper = max(lower, int(threshold * 1.20))

            record[feature] = eval_rng.randint(lower, upper)

            records.append(record)
            labels.append(1)
            categories.append(feature)

    evaluation_data = pd.DataFrame(records)
    y_true = labels

    # Step 5: Statistical rules.
    rule_predictions = (
        evaluation_data[FEATURES] >
        pd.Series(thresholds)
    ).any(axis=1).astype(int).tolist()

    evaluate(
        y_true,
        rule_predictions,
        "STATISTICAL RULES ONLY"
    )

    # Step 6: Compare Isolation Forest and hybrid detection.
    for contamination, model in models.items():

        if_predictions = (
            model.predict(evaluation_data[FEATURES]) == -1
        ).astype(int).tolist()

        hybrid_predictions = [
            int(if_result == 1 or rule_result == 1)
            for if_result, rule_result
            in zip(if_predictions, rule_predictions)
        ]

        evaluate(
            y_true,
            if_predictions,
            f"ISOLATION FOREST ONLY - {contamination}"
        )

        evaluate(
            y_true,
            hybrid_predictions,
            f"HYBRID DETECTOR - {contamination}"
        )

        print("\nCategory-wise detection:")
        for category in FEATURES:
            indices = [
                i for i, value in enumerate(categories)
                if value == category
            ]

            detected = sum(
                hybrid_predictions[i] for i in indices
            )

            print(
                f"{category:25}: "
                f"{detected}/25 "
                f"({detected / 25 * 100:.2f}%)"
            )


if __name__ == "__main__":
    main()
