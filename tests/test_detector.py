import pandas as pd

from backend.monitor import create_detector


def main():
    detector = create_detector()

    test_cases = [
        {
            "name": "Ordinary traffic",
            "packet_count": 110,
            "bytes_transferred": 5000,
            "connection_duration": 10,
            "failed_connections": 0
        },
        {
            "name": "High packet count",
            "packet_count": 8000,
            "bytes_transferred": 200000,
            "connection_duration": 250,
            "failed_connections": 80
        },
        {
            "name": "High data transfer",
            "packet_count": 120,
            "bytes_transferred": 800000,
            "connection_duration": 10,
            "failed_connections": 0
        },
        {
            "name": "Many failed connections",
            "packet_count": 120,
            "bytes_transferred": 5000,
            "connection_duration": 10,
            "failed_connections": 80
        }
    ]

    print("\nNETWORK ANOMALY DETECTOR TEST")
    print("=" * 65)

    for case in test_cases:
        name = case["name"]

        features = {
            key: value
            for key, value in case.items()
            if key != "name"
        }

        result = detector.predict(features)

        print(f"\nTest: {name}")
        print(f"Features: {features}")
        print(f"Prediction: {result['status']}")
        print(f"Anomaly score: {result['anomaly_score']}")

    print("\nTest completed.")


if __name__ == "__main__":
    main()