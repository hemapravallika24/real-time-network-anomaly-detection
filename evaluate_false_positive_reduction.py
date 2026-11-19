
import random
import statistics
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix

FEATURES = [
    "packet_count",
    "bytes_transferred",
    "connection_duration",
    "failed_connections"
]

SEEDS = [2030, 2031, 2032, 2033, 2034]


def generate_normal_record(rng):
    return {
        "packet_count": rng.randint(80, 150),
        "bytes_transferred": rng.randint(4000, 6000),
        "connection_duration": rng.randint(5, 15),
        "failed_connections": rng.randint(0, 2)
    }


def generate_evaluation_data(seed, thresholds):
    rng = random.Random(seed)

    records = []
    labels = []
    categories = []

    for _ in range(900):
        records.append(generate_normal_record(rng))
        labels.append(0)
        categories.append("Normal")

    reference = thresholds["Mean + 3 Std"]

    for feature in FEATURES:
        for _ in range(25):
            record = generate_normal_record(rng)

            lower = max(0, int(reference[feature] * 0.80))
            upper = max(lower, int(reference[feature] * 1.20))

            record[feature] = rng.randint(lower, upper)

            records.append(record)
            labels.append(1)
            categories.append(feature)

    return pd.DataFrame(records), labels, categories


def get_metrics(labels, predictions):
    tn, fp, fn, tp = confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()

    accuracy = (tn + tp) / (tn + fp + fn + tp)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }


def main():
    print("=" * 75)
    print("FALSE-POSITIVE REDUCTION EXPERIMENT")
    print("=" * 75)

    # Generate training data.
    rng = random.Random(42)

    training_data = pd.DataFrame([
        generate_normal_record(rng)
        for _ in range(500)
    ])

    # Calculate thresholds.
    threshold_sets = {
        "Mean + 3 Std": (
            training_data[FEATURES].mean()
            + 3 * training_data[FEATURES].std()
        ),
        "99th Percentile": training_data[FEATURES].quantile(0.99),
        "99.5th Percentile": training_data[FEATURES].quantile(0.995)
    }

    print("\nTHRESHOLD VALUES")

    for name, thresholds in threshold_sets.items():
        print(f"\n{name}")
        for feature in FEATURES:
            print(f"{feature:25}: {thresholds[feature]:.2f}")

    # Train Isolation Forest.
    model = IsolationForest(
        n_estimators=200,
        contamination=0.01,
        random_state=42
    )

    model.fit(training_data[FEATURES])

    strategies = [
        "Isolation Forest",
        "Rules 99",
        "Rules 99.5",
        "Hybrid 99 OR",
        "Hybrid 99 AND",
        "Hybrid 99.5 OR",
        "Hybrid 99.5 AND"
    ]

    totals = {
        name: {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
        for name in strategies
    }

    run_f1 = {name: [] for name in strategies}

    # Run five independent evaluation sets.
    for seed in SEEDS:

        print(f"\nEvaluating seed: {seed}")

        data, labels, categories = generate_evaluation_data(
            seed, threshold_sets
        )

        if_pred = (
            model.predict(data[FEATURES]) == -1
        ).astype(int).tolist()

        predictions = {
            "Isolation Forest": if_pred
        }

        for percentile, strategy_name in [
            ("99th Percentile", "99"),
            ("99.5th Percentile", "99.5")
        ]:

            rule_pred = (
                data[FEATURES] > threshold_sets[percentile]
            ).any(axis=1).astype(int).tolist()

            predictions[f"Rules {strategy_name}"] = rule_pred

            predictions[f"Hybrid {strategy_name} OR"] = [
                int(a == 1 or b == 1)
                for a, b in zip(if_pred, rule_pred)
            ]

            predictions[f"Hybrid {strategy_name} AND"] = [
                int(a == 1 and b == 1)
                for a, b in zip(if_pred, rule_pred)
            ]

        for name, pred in predictions.items():
            result = get_metrics(labels, pred)

            for key in ["tn", "fp", "fn", "tp"]:
                totals[name][key] += result[key]

            run_f1[name].append(result["f1"])

    # Display aggregate results.
    print("\n" + "=" * 75)
    print("AGGREGATED RESULTS")
    print("=" * 75)

    summary = []

    for name in strategies:
        values = totals[name]

        tn = values["tn"]
        fp = values["fp"]
        fn = values["fn"]
        tp = values["tp"]

        accuracy = (tn + tp) / (tn + fp + fn + tp)
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0
        )

        std = statistics.stdev(run_f1[name])

        print(f"\n{name}")
        print("-" * 60)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"TN: {tn} | FP: {fp} | FN: {fn} | TP: {tp}")
        print(f"Average run F1: {statistics.mean(run_f1[name]):.4f}")
        print(f"F1 Std Dev: {std:.4f}")

        summary.append({
            "Strategy": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "False Positives": fp,
            "False Negatives": fn,
            "Average Run F1": statistics.mean(run_f1[name]),
            "F1 Std Dev": std
        })

    # Save results.
    pd.DataFrame(summary).to_csv(
        "false_positive_results.csv",
        index=False
    )

    print("\nResults saved to false_positive_results.csv")

    print("\n" + "=" * 75)
    print("FALSE-POSITIVE REDUCTION EXPERIMENT COMPLETED")
    print("=" * 75)


if __name__ == "__main__":
    main()
