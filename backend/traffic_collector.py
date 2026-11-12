
import random
import time
from datetime import datetime


def generate_traffic():
    """
    Generate one synthetic network traffic record.

    These records are simulated for development and testing.
    They are not captured from a real network.
    """

    is_anomaly = random.random() < 0.10

    if is_anomaly:
        traffic = {
            "packet_count": random.randint(2000, 10000),
            "bytes_transferred": random.randint(100000, 900000),
            "connection_duration": random.randint(100, 300),
            "failed_connections": random.randint(10, 100)
        }
    else:
        traffic = {
            "packet_count": random.randint(80, 150),
            "bytes_transferred": random.randint(4000, 6000),
            "connection_duration": random.randint(5, 15),
            "failed_connections": random.randint(0, 2)
        }

    traffic["timestamp"] = datetime.now().isoformat(
        timespec="seconds"
    )

    return traffic


if __name__ == "__main__":
    print("SYNTHETIC NETWORK TRAFFIC GENERATOR")
    print("-" * 40)
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            record = generate_traffic()
            print(record)
            time.sleep(2)

    except KeyboardInterrupt:
        print("\nTraffic generation stopped.")
