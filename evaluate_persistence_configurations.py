
import numpy as np
import pandas as pd

FEATURES = [
    "packet_count",
    "bytes_transferred",
    "connection_duration",
    "failed_connections"
]

PROFILES = [
    "Baseline",
    "High packet activity",
    "Large data transfers",
    "Long sessions",
    "Temporary retry bursts"
]

CONFIGURATIONS = [
    (2, 3),
    (3, 5),
    (4, 5),
    (5, 5)
]


def generate_profile(name, size, rng):
    ranges = {
        "Baseline": [(80,151),(4000,6001),(5,16),(0,3)],
        "High packet activity": [(130,221),(4000,6501),(5,21),(0,3)],
        "Large data transfers": [(80,181),(5000,7501),(5,26),(0,3)],
        "Long sessions": [(80,161),(4000,6501),(15,36),(0,3)],
        "Temporary retry bursts": [(80,171),(4000,6501),(5,26),(0,6)]
    }

    return pd.DataFrame({
        feature: rng.integers(low, high, size)
        for feature, (low, high) in zip(
            FEATURES, ranges[name]
        )
    })


def persistence_alerts(flags, window, required):
    alerts = np.zeros(len(flags), dtype=int)

    for i in range(window - 1, len(flags)):
        recent = flags[i-window+1:i+1]

        if recent.sum() >= required:
            alerts[i] = 1

    return alerts


def main():
    print("=" * 75)
    print("PERSISTENCE CONFIGURATION COMPARISON")
    print("=" * 75)

    rng = np.random.default_rng(2060)

    training = pd.concat(
        [
            generate_profile(name, 500, rng)
            for name in PROFILES
        ],
        ignore_index=True
    )

    thresholds = {
        feature: np.percentile(training[feature], 99.5)
        for feature in FEATURES
    }

    print("\nThresholds:")
    for feature, value in thresholds.items():
        print(f"{feature:25}: {value:.2f}")

    benign_sequences = []
    attack_sequences = []

    # Benign sequences: two isolated spikes per sequence
    for profile in PROFILES:
        for sequence_id in range(20):
            data = generate_profile(profile, 100, rng)

            positions = rng.choice(
                np.arange(5, 100),
                size=2,
                replace=False
            )

            for position in positions:
                feature = rng.choice(FEATURES)
                data[feature] = data[feature].astype(float)
                data.loc[position, feature] = (
                    thresholds[feature] + 1
                )

            flags = (
                data[FEATURES]
                .gt(pd.Series(thresholds))
                .any(axis=1)
                .astype(int)
                .to_numpy()
            )

            benign_sequences.append({
                "profile": profile,
                "sequence_id": sequence_id,
                "flags": flags
            })

    # Attack sequences: sustained anomalies lasting 5 records
    for feature in FEATURES:
        for sequence_id in range(25):
            data = generate_profile("Baseline", 100, rng)

            start = int(rng.integers(10, 90))
            end = start + 5

            data[feature] = data[feature].astype(float)
            data.loc[start:end-1, feature] = (
                thresholds[feature] * 1.5 + 1
            )

            flags = (
                data[FEATURES]
                .gt(pd.Series(thresholds))
                .any(axis=1)
                .astype(int)
                .to_numpy()
            )

            attack_sequences.append({
                "feature": feature,
                "sequence_id": sequence_id,
                "flags": flags,
                "start": start,
                "end": end
            })

    results = []

    print("\nCONFIGURATION RESULTS")
    print("-" * 75)

    for required, window in CONFIGURATIONS:
        benign_alert_sequences = 0
        attack_detected = 0
        benign_alert_records = 0
        attack_alert_records = 0

        for item in benign_sequences:
            alerts = persistence_alerts(
                item["flags"], window, required
            )

            benign_alert_records += int(alerts.sum())

            if alerts.any():
                benign_alert_sequences += 1

        for item in attack_sequences:
            alerts = persistence_alerts(
                item["flags"], window, required
            )

            start = item["start"]
            end = item["end"]

            if alerts[start:end].any():
                attack_detected += 1

            attack_alert_records += int(
                alerts[start:end].sum()
            )

        benign_rate = (
            benign_alert_sequences / len(benign_sequences)
        )

        attack_rate = (
            attack_detected / len(attack_sequences)
        )

        results.append({
            "configuration": f"{required}-of-{window}",
            "benign_sequences_alerted": benign_alert_sequences,
            "benign_sequence_rate": benign_rate,
            "attack_sequences_detected": attack_detected,
            "attack_detection_rate": attack_rate,
            "benign_alert_records": benign_alert_records,
            "attack_alert_records": attack_alert_records
        })

        print(f"\n{required}-out-of-{window}")
        print(
            f"Benign sequences alerted: "
            f"{benign_alert_sequences}/{len(benign_sequences)}"
        )
        print(
            f"Attack sequences detected: "
            f"{attack_detected}/{len(attack_sequences)}"
        )
        print(
            f"Benign alert records: {benign_alert_records}"
        )
        print(
            f"Attack alert records: {attack_alert_records}"
        )

    pd.DataFrame(results).to_csv(
        "persistence_configuration_results.csv",
        index=False
    )

    print("\nSaved: persistence_configuration_results.csv")
    print("\nEXPERIMENT COMPLETED")


if __name__ == "__main__":
    main()
