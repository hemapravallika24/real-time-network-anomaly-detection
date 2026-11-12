import math

import pandas as pd
from sklearn.ensemble import IsolationForest


class NetworkAnomalyDetector:

    def __init__(self):
        self.model = IsolationForest(
            n_estimators=200,
            contamination=0.05,
            random_state=42
        )

        self.features = [
            "packet_count",
            "bytes_transferred",
            "connection_duration",
            "failed_connections"
        ]

        self.is_trained = False

    def train(self, traffic_data):
        """Train the model using representative baseline traffic."""

        missing = set(self.features) - set(traffic_data.columns)

        if missing:
            raise ValueError(
                f"Missing required features: {sorted(missing)}"
            )

        if traffic_data.empty:
            raise ValueError("Training data cannot be empty.")

        data = traffic_data[self.features].copy()

        for column in self.features:
            data[column] = pd.to_numeric(
                data[column],
                errors="raise"
            )

        if data.isnull().any().any():
            raise ValueError("Training data contains missing values.")

        for column in self.features:
            if not data[column].map(
                lambda value: math.isfinite(float(value))
                and value >= 0
            ).all():
                raise ValueError(
                    "Training features must be finite and non-negative."
                )

        self.model.fit(data)
        self.is_trained = True

    def predict(self, traffic_record):
        """Classify traffic and assign a preliminary risk level."""

        if not self.is_trained:
            raise RuntimeError(
                "The detector must be trained before prediction."
            )

        missing = set(self.features) - set(traffic_record.keys())

        if missing:
            raise ValueError(
                f"Missing required features: {sorted(missing)}"
            )

        record = pd.DataFrame(
            [{
                feature: traffic_record[feature]
                for feature in self.features
            }]
        )

        for column in self.features:
            record[column] = pd.to_numeric(
                record[column],
                errors="raise"
            )

        if record.isnull().any().any():
            raise ValueError("Traffic record contains missing values.")

        for column in self.features:
            if not record[column].map(
                lambda value: math.isfinite(float(value))
                and value >= 0
            ).all():
                raise ValueError(
                    "Traffic features must be finite and non-negative."
                )

        result = self.model.predict(record)[0]
        score = float(self.model.decision_function(record)[0])

        # Start with the machine-learning prediction.
        reasons = []

        if result == -1:
            reasons.append("Isolation Forest identified unusual traffic")

        # Preliminary rules for extreme simulated traffic.
        if traffic_record["packet_count"] > 1000:
            reasons.append("Unusually high packet count")

        if traffic_record["bytes_transferred"] > 100000:
            reasons.append("Unusually high data transfer")

        if traffic_record["connection_duration"] > 60:
            reasons.append("Unusually long connection duration")

        if traffic_record["failed_connections"] > 10:
            reasons.append("Many failed connections")

        is_anomaly = bool(reasons)

        if not is_anomaly:
            status = "Normal"
            risk_level = "Low"
        elif (
            traffic_record["packet_count"] > 5000
            or traffic_record["bytes_transferred"] > 500000
            or traffic_record["failed_connections"] > 50
        ):
            status = "Anomaly"
            risk_level = "High"
        else:
            status = "Anomaly"
            risk_level = "Medium"

        return {
            "status": status,
            "anomaly_score": round(score, 4),
            "risk_level": risk_level,
            "reason": "; ".join(reasons) if reasons else "No configured anomaly indicators"
        }