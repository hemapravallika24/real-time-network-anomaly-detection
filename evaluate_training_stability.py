
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

TRAINING_SEEDS = [42, 43, 44, 45, 46]
EVALUATION_SEEDS = [2030, 2031, 2032, 2033, 2034]

STRATEGIES = [
    "Isolation Forest",
    "Rules 97.5",
    "Rules 99",
    "Rules 99.5",
    "Hybrid 99.5 OR"
]


def generate_normal_record(rng):
    return {
        "packet_count": rng.randint(80, 150),
        "bytes_transferred": rng.randint(4000, 6000),
        "connection_duration": rng.randint(5, 15),
        "failed_connections": rng.randint(0, 2)
    }


def generate_training_data(seed):
    rng = random.Random(seed)

    return pd.DataFrame([
        generate_normal_record(rng)
        for _ in range(500)
    ])


def generate_evaluation_data(seed, reference_thresholds):
    rng = random.Random(seed)

    records = []
    labels = []

    # 900 normal records.
    for _ in range(900):
        records.append(generate_normal_record(rng))
        labels.append(0)

    # 100 borderline anomalies: 25 per feature.
    for feature in FEATURES:
        for _ in range(25):
            record = generate_normal_record(rng)

            threshold = reference_thresholds[feature]

            lower = max(0, int(threshold * 0.80))
            upper = max(lower, int(threshold * 1.20))

            record[feature] = rng.randint(lower, upper)

            records.append(record)
            labels.append(1)

    return pd.DataFrame(records), labels


def get_metrics(labels, predictions):
    tn, fp, fn, tp = confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": (tn + tp) / (tn + fp + fn + tp),
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def main():
    print("=" * 75)
    print("INDEPENDENT TRAINING STABILITY EXPERIMENT")
    print("=" * 75)

    # 1. Generate a reference baseline.
    # This defines the same borderline scenarios for every comparison.
    reference_training = generate_training_data(42)

    reference_thresholds = (
        reference_training[FEATURES].mean()
        + 3 * reference_training[FEATURES].std()
    )

    # 2. Prepare evaluation datasets once.
    evaluation_sets = {}

    for seed in EVALUATION_SEEDS:
        evaluation_sets[seed] = generate_evaluation_data(
            seed, reference_thresholds
        )

    # 3. Prepare result storage.
    totals = {
        name: {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
        for name in STRATEGIES
    }

    run_f1 = {name: [] for name in STRATEGIES}

    threshold_records = []

    # 4. Train five independent models.
    for train_seed in TRAINING_SEEDS:

        print(f"\nTraining dataset seed: {train_seed}")

        training_data = generate_training_data(train_seed)

        thresholds = {
            "97.5": training_data[FEATURES].quantile(0.975),
            "99": training_data[FEATURES].quantile(0.99),
            "99.5": training_data[FEATURES].quantile(0.995)
        }

        print("Calculated thresholds:")

        for feature in FEATURES:
            values = {
                name: round(thresholds[name][feature], 2)
                for name in thresholds
            }

            print(f"{feature:25}: {values}")

            threshold_records.append({
                "Training Seed": train_seed,
                "Feature": feature,
                "97.5 Percentile": thresholds["97.5"][feature],
                "99 Percentile": thresholds["99"][feature],
                "99.5 Percentile": thresholds["99.5"][feature]
            })

        # Train a fresh Isolation Forest for this training sample.
        model = IsolationForest(
            n_estimators=200,
            contamination=0.01,
            random_state=42
        )

        model.fit(training_data[FEATURES])

        # Evaluate this trained model against all five evaluation sets.
        for eval_seed in EVALUATION_SEEDS:

            data, labels = evaluation_sets[eval_seed]

            if_predictions = (
                model.predict(data[FEATURES]) == -1
            ).astype(int).tolist()

            predictions = {
                "Isolation Forest": if_predictions
            }

            for percentile in ["97.5", "99", "99.5"]:

                rule_predictions = (
                    data[FEATURES] > thresholds[percentile]
                ).any(axis=1).astype(int).tolist()

                predictions[f"Rules {percentile}"] = rule_predictions

                if percentile == "99.5":
                    predictions["Hybrid 99.5 OR"] = [
                        int(a == 1 or b == 1)
                        for a, b in zip(
                            if_predictions, rule_predictions
                        )
                    ]

            # Accumulate results across all 25 combinations.
            for name, pred in predictions.items():

                result = get_metrics(labels, pred)

                for key in ["tn", "fp", "fn", "tp"]:
                    totals[name][key] += result[key]

                run_f1[name].append(result["f1"])

    # 5. Display aggregated metrics.
    print("\n" + "=" * 75)
    print("AGGREGATED RESULTS - 25 TRAINING/EVALUATION COMBINATIONS")
    print("=" * 75)

    summary = []

    for name in STRATEGIES:

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

        average_f1 = statistics.mean(run_f1[name])
        std_f1 = statistics.stdev(run_f1[name])

        print(f"\n{name}")
        print("-" * 60)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"TN: {tn} | FP: {fp} | FN: {fn} | TP: {tp}")
        print(f"Average run F1: {average_f1:.4f}")
        print(f"F1 standard deviation: {std_f1:.4f}")

        summary.append({
            "Strategy": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "False Positives": fp,
            "False Negatives": fn,
            "Average Run F1": average_f1,
            "F1 Std Dev": std_f1
        })

    # 6. Save experiment results.
    pd.DataFrame(summary).to_csv(
        "training_stability_results.csv",
        index=False
    )

    pd.DataFrame(threshold_records).to_csv(
        "training_threshold_variation.csv",
        index=False
    )

    print("\nSaved files:")
    print("1. training_stability_results.csv")
    print("2. training_threshold_variation.csv")

    print("\n" + "=" * 75)
    print("INDEPENDENT TRAINING STABILITY EXPERIMENT COMPLETED")
    print("=" * 75)


if __name__ == "__main__":
    main()
