
import numpy as np
import pandas as pd

FEATURES = [
    "packet_count",
    "bytes_transferred",
    "connection_duration",
    "failed_connections"
]

PROFILE_NAMES = [
    "Baseline",
    "High packet activity",
    "Large data transfers",
    "Long sessions",
    "Temporary retry bursts"
]

WINDOW = 5
REQUIRED_ALERTS = 3


def generate_profile(name, size, rng):
    if name == "Baseline":
        ranges = [(80, 151), (4000, 6001), (5, 16), (0, 3)]
    elif name == "High packet activity":
        ranges = [(130, 221), (4000, 6501), (5, 21), (0, 3)]
    elif name == "Large data transfers":
        ranges = [(80, 181), (5000, 7501), (5, 26), (0, 3)]
    elif name == "Long sessions":
        ranges = [(80, 161), (4000, 6501), (15, 36), (0, 3)]
    else:
        ranges = [(80, 171), (4000, 6501), (5, 26), (0, 6)]

    return pd.DataFrame({
        feature: rng.integers(low, high, size)
        for feature, (low, high) in zip(FEATURES, ranges)
    })


def apply_persistence(raw_flags):
    alerts = np.zeros(len(raw_flags), dtype=int)

    for i in range(len(raw_flags)):
        start = max(0, i - WINDOW + 1)
        recent = raw_flags[start:i + 1]

        if len(recent) == WINDOW:
            if recent.sum() >= REQUIRED_ALERTS:
                alerts[i] = 1

    return alerts


def add_spike(data, feature, threshold):
    data.loc[:, feature] = data[feature].copy()
    data.loc[data.index[0], feature] = threshold + 1


def main():
    print("=" * 75)
    print("TEMPORAL NETWORK ANOMALY DETECTION EXPERIMENT")
    print("=" * 75)

    rng = np.random.default_rng(2050)

    # Train a diverse normal baseline
    training_data = pd.concat(
        [
            generate_profile(name, 500, rng)
            for name in PROFILE_NAMES
        ],
        ignore_index=True
    )

    thresholds = {
        feature: np.percentile(
            training_data[feature], 99.5
        )
        for feature in FEATURES
    }

    print("\nTraining thresholds:")
    for feature, threshold in thresholds.items():
        print(f"{feature:25}: {threshold:.2f}")

    benign_sequence_results = []
    attack_sequence_results = []

    # Create benign sequences
    for profile in PROFILE_NAMES:
        for sequence_id in range(20):
            sequence = generate_profile(profile, 100, rng)

            # Insert two isolated benign spikes
            spike_positions = rng.choice(
                np.arange(5, 100),
                size=2,
                replace=False
            )

            for position in spike_positions:
                feature = rng.choice(FEATURES)
                sequence.loc[position, feature] = (
                    thresholds[feature] + 1
                )

            raw_flags = (
                sequence[FEATURES]
                .gt(pd.Series(thresholds))
                .any(axis=1)
                .astype(int)
                .to_numpy()
            )

            persistent_alerts = apply_persistence(raw_flags)

            benign_sequence_results.append({
                "profile": profile,
                "sequence_id": sequence_id,
                "raw_alerts": int(raw_flags.sum()),
                "persistent_alerts": int(persistent_alerts.sum()),
                "persistent_sequence_alert": int(
                    persistent_alerts.any()
                )
            })

    # Create sustained single-feature anomaly episodes
    for category, feature in enumerate(FEATURES):
        for sequence_id in range(25):
            sequence = generate_profile(
                "Baseline", 100, rng
            )

            # Choose a random episode lasting 5 consecutive records
            start = int(rng.integers(10, 90))
            end = start + 5

            # Sustained anomaly in one feature
            # Convert the selected feature to float before inserting
# decimal anomaly values.
            sequence[feature] = sequence[feature].astype(float)
            sequence.loc[start:end - 1, feature] = (
    thresholds[feature] * 1.5 + 1
)

            raw_flags = (
                sequence[FEATURES]
                .gt(pd.Series(thresholds))
                .any(axis=1)
                .astype(int)
                .to_numpy()
            )

            persistent_alerts = apply_persistence(raw_flags)

            attack_sequence_results.append({
                "feature": feature,
                "sequence_id": sequence_id,
                "raw_alerts": int(raw_flags.sum()),
                "persistent_alerts": int(
                    persistent_alerts.sum()
                ),
                "raw_detected": int(raw_flags.any()),
                "persistent_detected": int(
                    persistent_alerts[start:end].any()
                )
            })

    benign = pd.DataFrame(benign_sequence_results)
    attacks = pd.DataFrame(attack_sequence_results)

    print("\nBENIGN SEQUENCE RESULTS")
    print("-" * 60)

    print("Total benign sequences:", len(benign))
    print(
        "Sequences with raw alerts:",
        int((benign["raw_alerts"] > 0).sum())
    )
    print(
        "Sequences with persistent alerts:",
        int(benign["persistent_sequence_alert"].sum())
    )

    print("\nBENIGN PROFILE BREAKDOWN")
    print("-" * 60)

    for profile, group in benign.groupby("profile"):
        print(
            f"{profile:25}: "
            f"raw alerts in {(group['raw_alerts'] > 0).sum()}/"
            f"{len(group)} sequences | "
            f"persistent alerts in "
            f"{group['persistent_sequence_alert'].sum()}/"
            f"{len(group)} sequences"
        )

    print("\nSUSTAINED ANOMALY RESULTS")
    print("-" * 60)

    print("Total attack sequences:", len(attacks))
    print(
        "Detected by immediate rules:",
        int(attacks["raw_detected"].sum())
    )
    print(
        "Detected by persistence rules:",
        int(attacks["persistent_detected"].sum())
    )

    print("\nANOMALY CATEGORY BREAKDOWN")
    print("-" * 60)

    for feature, group in attacks.groupby("feature"):
        print(
            f"{feature:25}: "
            f"raw {group['raw_detected'].sum()}/{len(group)} | "
            f"persistent "
            f"{group['persistent_detected'].sum()}/{len(group)}"
        )

    benign.to_csv(
        "temporal_benign_results.csv",
        index=False
    )

    attacks.to_csv(
        "temporal_attack_results.csv",
        index=False
    )

    print("\nSaved:")
    print("1. temporal_benign_results.csv")
    print("2. temporal_attack_results.csv")

    print("\nEXPERIMENT COMPLETED")


if __name__ == "__main__":
    main()
