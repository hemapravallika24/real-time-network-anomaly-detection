
import pytest

from backend.monitor import create_detector


@pytest.fixture
def detector():
    """Create a trained detector for testing."""
    return create_detector()


def force_model_normal(detector, monkeypatch):
    """
    Make Isolation Forest return a normal prediction.
    This isolates the configured anomaly-rule tests.
    """
    monkeypatch.setattr(
        detector.model,
        "predict",
        lambda data: [1]
    )

    monkeypatch.setattr(
        detector.model,
        "decision_function",
        lambda data: [0.1]
    )


def test_ordinary_traffic(detector, monkeypatch):
    """Ordinary traffic should receive a Low risk classification."""

    force_model_normal(detector, monkeypatch)

    traffic = {
        "packet_count": 110,
        "bytes_transferred": 5000,
        "connection_duration": 10,
        "failed_connections": 0
    }

    result = detector.predict(traffic)

    assert result["status"] == "Normal"
    assert result["risk_level"] == "Low"


@pytest.mark.parametrize(
    "traffic, expected_risk",
    [
        (
            {
                "packet_count": 8000,
                "bytes_transferred": 200000,
                "connection_duration": 250,
                "failed_connections": 80
            },
            "High"
        ),
        (
            {
                "packet_count": 120,
                "bytes_transferred": 800000,
                "connection_duration": 10,
                "failed_connections": 0
            },
            "High"
        ),
        (
            {
                "packet_count": 120,
                "bytes_transferred": 5000,
                "connection_duration": 10,
                "failed_connections": 80
            },
            "High"
        )
    ]
)
def test_configured_anomaly_rules(
    detector,
    monkeypatch,
    traffic,
    expected_risk
):
    """Configured anomaly indicators should identify unusual traffic."""

    force_model_normal(detector, monkeypatch)

    result = detector.predict(traffic)

    assert result["status"] == "Anomaly"
    assert result["risk_level"] == expected_risk
    assert result["reason"] != "No configured anomaly indicators"
