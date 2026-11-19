
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from evaluate_single_feature_anomalies import (
    FEATURES,
    generate_training_data,
    generate_evaluation_data,
)


def calculate_thresholds(training_data):
    """Calculate an upper statistical threshold for each feature."""

    thresholds = {}

    for feature in FEATURES:
        mean = training_data[feature].mean()
        std = training_data[feature].std()

        thresholds[feature] = mean + (3 * std)

    return thresholds


def predict_using_rules(evaluation_data, thresholds):
    """Flag records that exceed any individual feature threshold."""

    predictions = []

    for _, record in evaluation_data.iterrows():
        is_anomaly = any(
            record[feature] > thresholds[feature]
            for feature in FEATURES
        )

        predictions.append(1 if is_anomaly else 0)

    return predictions


def display_metrics(name, actual, predicted, categories):

    tn, fp, fn, tp = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1],
    ).ravel()

    print(f"\n{name}")
    print("-" * 65)

    print(f"Accuracy:  {accuracy_score(actual, predicted):.4f}")
    print(f"Precision: {precision_score(actual, predicted, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(actual, predicted, zero_division=0):.4f}")
    print(f"F1-score:  {f1_score(actual, predicted, zero_division=0):.4f}")

    print("\nConfusion Matrix:")
    print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

    print("\nCategory-wise detection:")

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

        detected = sum(predicted[i] for i in indices)

        print(
            f"{category:<30} "
            f"{detected}/{len(indices)} detected "
            f"({detected / len(indices) * 100:.2f}%)"
        )


def main():

    print("=" * 70)
    print("HYBRID NETWORK ANOMALY DETECTION EVALUATION")
    print("=" * 70)

    training_data = generate_training_data()

    evaluation_data, actual_labels, categories = (
        generate_evaluation_data()
    )

    thresholds = calculate_thresholds(training_data)

    print("\nCalculated feature thresholds:")
    for feature, threshold in thresholds.items():
        print(f"{feature:<25}: {threshold:.2f}")

    # Evaluate statistical rules independently.
    rule_predictions = predict_using_rules(
        evaluation_data,
        thresholds,
    )

    display_metrics(
        "STATISTICAL RULES ONLY",
        actual_labels,
        rule_predictions,
        categories,
    )

    # Compare Isolation Forest and hybrid detection.
    for contamination in [0.01, 0.05]:

        model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42,
        )

        model.fit(training_data)

        raw_predictions = model.predict(evaluation_data)

        model_predictions = [
            1 if result == -1 else 0
            for result in raw_predictions
        ]

        hybrid_predictions = [
            1 if model_predictions[i] == 1
            or rule_predictions[i] == 1
            else 0
            for i in range(len(actual_labels))
        ]

        display_metrics(
            f"ISOLATION FOREST ONLY - {contamination}",
            actual_labels,
            model_predictions,
            categories,
        )

        display_metrics(
            f"HYBRID DETECTOR - {contamination}",
            actual_labels,
            hybrid_predictions,
            categories,
        )

    print("\n" + "=" * 70)
    print("HYBRID EVALUATION COMPLETED")
    print("Results apply only to this synthetic dataset.")
    print("=" * 70)


if __name__ == "__main__":
    main()
