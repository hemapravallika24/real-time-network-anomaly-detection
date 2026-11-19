
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

ATTACK_DURATIONS = [1, 2, 3, 5]


def generate_profile(name, size, rng):
    ranges = {
        "Baseline": [(80, 151), (4000, 6001), (5, 16), (0, 3)],
        "High packet activity": [(130, 221), (4000, 6501), (5, 21), (0, 3)],
        "Large data transfers": [(80, 181), (5000, 7501), (5, 26), (0, 3)],
        "Long sessions": [(80, 161), (4000, 6501), (15, 36), (0, 3)],
        "Temporary retry bursts": [(80, 171), (4000, 6501), (5, 26), (0, 6)]
    }

    return pd.DataFrame({
        feature: rng.integers(low, high, size)
        for feature, (low, high) in zip(FEATURES, ranges[name])
    })


def persistence_alerts(flags, window, required):
    alerts = np.zeros(len(flags), dtype=int)

    for i in range(window - 1, len(flags)):
        recent = flags[i - window + 1:i + 1]

        if recent.sum() >= required:
            alerts[i] = 1

    return alerts


def create_benign_sequences(rng, thresholds):
    sequences = []

    for profile in PROFILES:
        for sequence_id in range(20):
            data = generate_profile(profile, 100, rng)

            # Preserve the previous experiment's two isolated spikes.
            positions = rng.choice(
                np.arange(5, 100),
                size=2,
                replace=False
            )

            for position in positions:
                feature = rng.choice(FEATURES)
                data[feature] = data[feature].astype(float)
                data.loc[position, feature] = thresholds[feature] + 1

            flags = (
                data[FEATURES]
                .gt(pd.Series(thresholds))
                .any(axis=1)
                .astype(int)
                .to_numpy()
            )

            sequences.append({
                "profile": profile,
                "sequence_id": sequence_id,
                "flags": flags
            })

    return sequences


def create_attack_sequences(rng, thresholds):
    attack_groups = {}

    for duration in ATTACK_DURATIONS:
        sequences = []

        for feature in FEATURES:
            for sequence_id in range(25):
                data = generate_profile("Baseline", 100, rng)

                start = int(rng.integers(10, 90))
                end = start + duration

                data[feature] = data[feature].astype(float)
                data.loc[start:end - 1, feature] = (
                    thresholds[feature] * 1.5 + 1
                )

                flags = (
                    data[FEATURES]
                    .gt(pd.Series(thresholds))
                    .any(axis=1)
                    .astype(int)
                    .to_numpy()
                )

                sequences.append({
                    "feature": feature,
                    "sequence_id": sequence_id,
                    "flags": flags,
                    "start": start,
                    "end": end,
                    "duration": duration
                })

        attack_groups[duration] = sequences

    return attack_groups


def main():
    print("=" * 78)
    print("SHORT AND SUSTAINED ANOMALY EPISODE EVALUATION")
    print("=" * 78)

    rng = np.random.default_rng(2060)

    # Same diverse training profiles and seed as the previous experiment.
    training = pd.concat(
        [
            generate_profile(profile, 500, rng)
            for profile in PROFILES
        ],
        ignore_index=True
    )

    thresholds = {
        feature: np.percentile(training[feature], 99.5)
        for feature in FEATURES
    }

    print("\nTRAINING THRESHOLDS")
    print("-" * 78)

    for feature, value in thresholds.items():
        print(f"{feature:25}: {value:.2f}")

    benign_sequences = create_benign_sequences(rng, thresholds)
    attack_groups = create_attack_sequences(rng, thresholds)

    results = []

    print("\nBENIGN TRAFFIC RESULTS")
    print("-" * 78)

    for required, window in CONFIGURATIONS:
        alerted_sequences = 0
        alert_records = 0

        for item in benign_sequences:
            alerts = persistence_alerts(
                item["flags"], window, required
            )

            alert_records += int(alerts.sum())

            if alerts.any():
                alerted_sequences += 1

        print(
            f"{required}-of-{window}: "
            f"{alerted_sequences}/{len(benign_sequences)} sequences alerted; "
            f"{alert_records} alert records"
        )

        for duration in ATTACK_DURATIONS:
            results.append({
                "configuration": f"{required}-of-{window}",
                "attack_duration": duration,
                "benign_sequences_alerted": alerted_sequences,
                "benign_sequence_rate": (
                    alerted_sequences / len(benign_sequences)
                ),
                "benign_alert_records": alert_records
            })

    print("\nATTACK EPISODE RESULTS")
    print("-" * 78)

    for required, window in CONFIGURATIONS:
        for duration in ATTACK_DURATIONS:
            sequences = attack_groups[duration]

            detected = 0
            alert_records = 0
            detection_delays = []

            for item in sequences:
                alerts = persistence_alerts(
                    item["flags"], window, required
                )

                start = item["start"]
                end = item["end"]

                episode_alerts = np.where(
                    alerts[start:end] == 1
                )[0]

                if len(episode_alerts) > 0:
                    detected += 1
                    detection_delays.append(
                        int(episode_alerts[0]) + 1
                    )

                alert_records += int(
                    alerts[start:end].sum()
                )

            detection_rate = detected / len(sequences)

            average_delay = (
                float(np.mean(detection_delays))
                if detection_delays else None
            )

            results.append({
                "configuration": f"{required}-of-{window}",
                "attack_duration": duration,
                "attack_sequences_detected": detected,
                "attack_detection_rate": detection_rate,
                "attack_alert_records": alert_records,
                "average_detection_delay": average_delay
            })

            delay_text = (
                f"{average_delay:.2f} observations"
                if average_delay is not None
                else "Not detected"
            )

            print(
                f"{required}-of-{window} | "
                f"{duration} observations | "
                f"Detected: {detected}/{len(sequences)} | "
                f"Detection rate: {detection_rate:.2%} | "
                f"Average delay: {delay_text}"
            )

    output = pd.DataFrame(results)

    output.to_csv(
        "short_episode_detection_results.csv",
        index=False
    )

    print("\nSaved: short_episode_detection_results.csv")
    print("\nEXPERIMENT COMPLETED")


if __name__ == "__main__":
    main()
