
from collections import deque

from backend.monitor import apply_persistence


def make_result(status="Normal", risk="Low"):
    return {
        "status": status,
        "risk_level": risk,
        "anomaly_score": -0.05,
        "reason": "Test anomaly" if status == "Anomaly" else "Normal"
    }


def test_three_anomalies_in_five_trigger_alert():
    history = deque(maxlen=5)

    sequence = [
        "Anomaly",
        "Normal",
        "Anomaly",
        "Anomaly",
        "Normal"
    ]

    result = None

    for status in sequence:
        result, count = apply_persistence(
            make_result(status),
            history
        )

    assert count == 3
    assert result["status"] == "Anomaly"


def test_isolated_anomaly_does_not_trigger_alert():
    history = deque(maxlen=5)

    sequence = [
        "Anomaly",
        "Normal",
        "Normal",
        "Normal",
        "Normal"
    ]

    result = None

    for status in sequence:
        result, count = apply_persistence(
            make_result(status),
            history
        )

    assert count == 1
    assert result["status"] == "Normal"


def test_old_anomaly_expires_from_window():
    history = deque(maxlen=5)

    sequence = [
        "Anomaly",
        "Normal",
        "Anomaly",
        "Anomaly",
        "Normal"
    ]

    for status in sequence:
        result, count = apply_persistence(
            make_result(status),
            history
        )

    assert result["status"] == "Anomaly"

    result, count = apply_persistence(
        make_result("Normal"),
        history
    )

    assert count == 2
    assert result["status"] == "Normal"


def test_pending_anomaly_preserves_risk_level():
    history = deque(maxlen=5)

    result, count = apply_persistence(
        make_result("Anomaly", "High"),
        history
    )

    assert result["status"] == "Normal"
    assert result["risk_level"] == "High"
    assert count == 1
