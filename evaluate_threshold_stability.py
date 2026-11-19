
import random
import statistics
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

EVALUATION_SEEDS = [2030, 2031, 2032, 2033, 2034]


def generate_normal_record(rng):
    return {
        "packet_count": rng.randint(80, 150),
        "bytes_transferred": rng.randint(4000, 6000),
        "connection_duration": rng.randint(5, 15),
        "failed_connections": rng.randint(0, 2)
    }


def generate_evaluation_data(seed, reference_thresholds):
    rng = random.Random(seed)

    records = []
    labels = []
    categories = []

    # 900 normal records.
    for _ in range(900):
        records.append(generate_normal_record(rng))
        labels.append(0)
        categories.append("Normal")

    # 100 borderline records, 25 per feature.
    for feature in FEATURES:
        for _ in range(25):
            record = generate_normal_record(rng)

            threshold = reference_thresholds[feature]
            lower = max(0, int(threshold * 0.80))
            upper = max(lower, int(threshold * 1.20))

            record[feature] = rng.randint(lower, upper)

            records.append(record)
            labels.append(1)
            categories.append(feature)

    return pd.DataFrame(records), labels, categories


def calculate_metrics(labels, predictions):
    tn, fp, fn, tp = confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()

    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(
            labels, predictions, zero_division=0
        ),
        "recall": recall_score(
            labels, predictions, zero_division=0
        ),
        "f1": f1_score(labels, predictions, zero_division=0),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }


def print_metrics(name, totals, run_f1):
    tn = totals["tn"]
    fp = totals["fp"]
    fn = totals["fn"]
    tp = totals["tp"]

    total = tn + fp + fn + tp

    accuracy = (tn + tp) / total
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    print(f"\n{name}")
    print("-" * 65)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"TN: {tn} | FP: {fp} | FN: {fn} | TP: {tp}")
    print(f"Average per-run F1: {statistics.mean(run_f1):.4f}")
    print(f"F1 standard deviation: {statistics.stdev(run_f1):.4f}")


def main():
    print("=" * 75)
    print("THRESHOLD STABILITY AND FALSE-POSITIVE ANALYSIS")
    print("=" * 75)

    # Train one baseline using a fixed seed.
    train_rng = random.Random(42)

    training_data = pd.DataFrame([
        generate_normal_record(train_rng)
        for _ in range(500)
    ])

    # Calculate thresholds using training data only.
    threshold_sets = {
        "Mean + 3 Std": (
            training_data[FEATURES].mean()
            + 3 * training_data[FEATURES].std()
        ),
        "95th Percentile": training_data[FEATURES].quantile(0.95),
        "97.5th Percentile": training_data[FEATURES].quantile(0.975),
        "99th Percentile": training_data[FEATURES].quantile(0.99)
    }

    print("\nTRAINING THRESHOLDS")
    print("-" * 75)

    for method, thresholds in threshold_sets.items():
        print(f"\n{method}")
        for feature in FEATURES:
            print(f"{feature:25}: {thresholds[feature]:.2f}")

    # Train Isolation Forest once.
    model = IsolationForest(
        n_estimators=200,
        contamination=0.01,
        random_state=42
    )
    model.fit(training_data[FEATURES])

    # Store results across all five evaluation runs.
    strategies = [
        "Isolation Forest",
        "Mean + 3 Std",
        "95th Percentile",
        "97.5th Percentile",
        "99th Percentile",
        "Hybrid 99th Percentile"
    ]

    totals = {
        name: {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
        for name in strategies
    }

    run_f1 = {name: [] for name in strategies}

    category_totals = {
        name: {feature: [0, 0] for feature in FEATURES}
        for name in strategies
    }

    # Evaluate across independent random seeds.
    for seed in EVALUATION_SEEDS:
        print(f"\nRunning evaluation seed: {seed}")

        evaluation_data, labels, categories = (
            generate_evaluation_data(
                seed,
                threshold_sets["Mean + 3 Std"]
            )
        )

        if_predictions = (
            model.predict(evaluation_data[FEATURES]) == -1
        ).astype(int).tolist()

        predictions_by_strategy = {
            "Isolation Forest": if_predictions
        }

        for method, thresholds in threshold_sets.items():
            rule_predictions = (
                evaluation_data[FEATURES] > thresholds
            ).any(axis=1).astype(int).tolist()

            predictions_by_strategy[method] = rule_predictions

            if method == "99th Percentile":
                hybrid_predictions = [
                    int(a == 1 or b == 1)
                    for a, b in zip(
                        if_predictions,
                        rule_predictions
                    )
                ]

                predictions_by_strategy[
                    "Hybrid 99th Percentile"
                ] = hybrid_predictions

        # Accumulate confusion matrices and category results.
        for name, predictions in predictions_by_strategy.items():
            metrics = calculate_metrics(labels, predictions)

            for key in ["tn", "fp", "fn", "tp"]:
                totals[name][key] += metrics[key]

            run_f1[name].append(metrics["f1"])

            for category in FEATURES:
                indices = [
                    i for i, value in enumerate(categories)
                    if value == category
                ]

                detected = sum(predictions[i] for i in indices)

                category_totals[name][category][0] += detected
                category_totals[name][category][1] += len(indices)

    # Display aggregate results.
    print("\n" + "=" * 75)
    print("AGGREGATED RESULTS ACROSS FIVE EVALUATION RUNS")
    print("=" * 75)

    for name in strategies:
        print_metrics(name, totals[name], run_f1[name])

    # Category-wise detection rates.
    print("\n" + "=" * 75)
    print("CATEGORY-WISE DETECTION ACROSS FIVE RUNS")
    print("=" * 75)

    for name in strategies:
        print(f"\n{name}")

        for feature in FEATURES:
            detected, total = category_totals[name][feature]
            percentage = detected / total * 100

            print(
                f"{feature:25}: "
                f"{detected}/{total} ({percentage:.2f}%)"
            )

    print("\n" + "=" * 75)
    print("THRESHOLD STABILITY EXPERIMENT COMPLETED")
    print("=" * 75)


if __name__ == "__main__":
    main()