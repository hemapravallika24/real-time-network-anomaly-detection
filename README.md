# Real-Time Network Anomaly Detection and Security Monitoring Through Cloud Computing and DevOps

A cloud-oriented network monitoring prototype that uses Machine Learning, statistical anomaly indicators, and temporal persistence-based alerting to identify unusual network traffic patterns.

The project combines **Python, Isolation Forest, Streamlit, Docker, Docker Compose, and automated testing** to demonstrate a containerized network anomaly monitoring workflow.

> **Important:** The current implementation uses synthetic network traffic. It does not capture live packets from a real network. Anomaly classifications are experimental indicators, not confirmed cyberattacks.

## 1. Project Overview

Network traffic can contain unusual patterns that may indicate potential security issues. Monitoring these patterns helps administrators understand network behavior and identify suspicious activity.

This project implements a prototype that:

- Generates synthetic normal and anomalous network traffic.
- Uses the Isolation Forest algorithm for unsupervised anomaly detection.
- Applies configured feature-based anomaly indicators.
- Uses a 3-out-of-5 temporal persistence mechanism to reduce alerts from isolated observations.
- Assigns risk levels to detected events.
- Stores monitoring events in CSV format.
- Displays network activity through an interactive Streamlit dashboard.
- Runs monitoring and dashboard services using Docker Compose.
- Includes automated tests and synthetic evaluation experiments.

## 2. Key Features

### Synthetic Traffic Generation
Generates simulated network traffic using four features:

- Packet count
- Bytes transferred
- Connection duration
- Failed connections

### Machine Learning Detection
Uses the Isolation Forest algorithm to identify unusual traffic patterns.

### Feature-Based Anomaly Indicators
Applies configured thresholds to identify unusual feature values alongside model predictions.

### Temporal Persistence-Based Alerting
Uses a rolling window of five observations.

- An alert is confirmed when at least three of the latest five observations are anomalous.
- Isolated anomaly observations remain pending until the persistence condition is satisfied.
- The highest risk level among anomalous observations in the window is retained.

This mechanism helps reduce alerts caused by isolated unusual observations, while introducing a detection delay for sustained anomalies.

### Risk Classification
Categorizes events into:

- Low
- Medium
- High

### Event Logging
Stores timestamps, traffic features, anomaly scores, classifications, risk levels, and detection reasons in CSV format.

### Interactive Dashboard
Provides:

- Total traffic records
- Normal and anomalous event statistics
- Risk-level summaries
- Traffic classification charts
- Packet-count trends
- Anomaly event tables
- Filterable event logs
- CSV export
- Automatic dashboard refresh

### Docker and Docker Compose
Runs the monitoring process and dashboard as separate containerized services with a shared data directory.

## 3. Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Machine Learning | Scikit-learn, Isolation Forest |
| Data Processing | Pandas |
| Dashboard | Streamlit |
| Data Storage | CSV |
| Testing | Pytest |
| Containerization | Docker |
| Orchestration | Docker Compose |
| Version Control | Git, GitHub |

## 4. System Architecture

The current prototype follows this workflow:

1. The traffic collector generates synthetic network traffic.
2. The monitoring service extracts the required traffic features.
3. The Isolation Forest detector evaluates the traffic.
4. Configured feature-based rules identify additional anomaly indicators.
5. The persistence mechanism evaluates the latest five observations.
6. A final classification and risk level are generated.
7. The monitoring service records the event in a CSV file.
8. The Streamlit dashboard reads the shared event log and displays monitoring information.

### Architecture Flow

```text
Synthetic Traffic Generator
           |
           v
     Traffic Monitor
           |
           v
   Feature Extraction
           |
           v
   Isolation Forest Model
           |
           v
  Configured Anomaly Rules
           |
           v
  3-out-of-5 Persistence
           |
           v
   Risk Classification
           |
           v
      CSV Event Log
           |
           v
   Streamlit Dashboard
```

## 5. Project Structure

```text
network-anomaly-detection/
|
|-- backend/
|   |-- detector.py
|   |-- monitor.py
|   |-- traffic_collector.py
|
|-- data/
|   |-- network_events.csv
|
|-- models/
|
|-- tests/
|   |-- test_detector.py
|   |-- test_monitor_persistence.py
|
|-- dashboard.py
|-- migrate_csv.py
|
|-- evaluate_model.py
|-- compare_contamination.py
|-- evaluate_subtle_anomalies.py
|-- evaluate_single_feature_anomalies.py
|-- evaluate_hybrid_detector.py
|-- evaluate_borderline_anomalies.py
|-- evaluate_feature_thresholds.py
|-- evaluate_threshold_stability.py
|-- evaluate_false_positive_reduction.py
|-- evaluate_training_stability.py
|-- evaluate_benign_variability.py
|-- evaluate_diverse_training.py
|-- evaluate_temporal_detection.py
|-- evaluate_persistence_configurations.py
|-- evaluate_short_episode_detection.py
|
|-- requirements.txt
|-- Dockerfile
|-- compose.yaml
|-- .dockerignore
|-- .gitignore
|-- README.md
```

The evaluation scripts and their corresponding CSV result files are used for experimental analysis.

## 6. Getting Started

### Prerequisites

- Python 3.13 or compatible Python environment
- Git
- Docker Desktop
- Docker Compose

### Step 1: Clone the Repository

```bash
git clone https://github.com/hemapravallika24/real-time-network-anomaly-detection.git

cd real-time-network-anomaly-detection
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

Activate on Windows:

```cmd
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Traffic Monitor

Open a terminal in the project directory:

```bash
python -m backend.monitor
```

The monitor generates synthetic traffic and records classifications every two seconds.

### Step 5: Run the Dashboard

Open a second terminal and activate the virtual environment.

Run:

```bash
streamlit run dashboard.py
```

Open the dashboard:

http://localhost:8501

## 7. Running with Docker Compose

Make sure Docker Desktop is running.

### Build and Start Services

```bash
docker compose up --build -d
```

### Check Container Status

```bash
docker compose ps
```

### View Monitor Logs

```bash
docker compose logs -f monitor
```

### View Dashboard Logs

```bash
docker compose logs -f dashboard
```

### Access the Dashboard

http://localhost:8501

### Stop Services

```bash
docker compose down
```

The Compose configuration shares the local `data` directory between the monitor and dashboard containers, allowing both services to access the same event log.

**Note:** Avoid running the local monitor and Docker monitor simultaneously because both write to the same CSV file when using the shared directory.

## 8. Automated Testing

The project uses Pytest to validate detector behavior and temporal persistence logic.

Run all tests:

```bash
python -m pytest -v
```

### Test Coverage

**Detector tests:**
- Ordinary traffic classification
- Configured anomaly rules
- High-risk classification for unusual traffic patterns

**Persistence tests:**
- Three anomalies in five observations trigger an alert.
- An isolated anomaly does not trigger an alert.
- Older anomalies expire from the rolling window.
- Pending anomalies preserve their risk level.

Latest verified result:

```text
8 passed in 6.62s
```

## 9. Experimental Evaluation

The project includes synthetic evaluation experiments covering:

- Isolation Forest contamination settings
- Subtle and single-feature anomalies
- Hybrid detection approaches
- Feature threshold comparisons
- Threshold stability across random seeds
- False-positive reduction
- Training-data stability
- Benign traffic variability
- Diverse normal training profiles
- Temporal anomaly detection
- Persistence configuration comparisons
- Short-duration anomaly episodes

The experiments helped investigate the trade-offs between anomaly detection, false positives, and alert confirmation delay.

**Experimental limitation:** Results are based on generated datasets and controlled scenarios. They do not establish real-world detection accuracy or guarantee performance on actual network traffic.

## 10. Current Development Status

- [x] Synthetic network traffic generation
- [x] Isolation Forest anomaly detection
- [x] Configured feature-based anomaly indicators
- [x] Risk classification
- [x] CSV-based event logging
- [x] 3-out-of-5 temporal persistence mechanism
- [x] Interactive Streamlit dashboard
- [x] Automatic dashboard refresh
- [x] Docker containerization
- [x] Docker Compose orchestration
- [x] Automated detector and persistence tests
- [x] Synthetic evaluation experiments
- [ ] Real network traffic collection
- [ ] Cloud deployment
- [ ] CI/CD pipeline and DevSecOps integration
- [ ] Centralized cloud storage
- [ ] Validation using real-world network datasets

## 11. Future Enhancements

- Integration with real network traffic collection tools.
- Cloud-based deployment and centralized event storage.
- CI/CD automation for testing and deployment.
- DevSecOps practices and automated security validation.
- Evaluation using real-world labeled network datasets.
- Enhanced monitoring, scalability, and alert management.

## 12. Project Information

**Project Title:** Real-Time Network Anomaly Detection and Security Monitoring Through Cloud Computing and DevOps

**Domain:** Machine Learning, Network Monitoring, Cloud Computing, and DevOps

**Project Type:** Academic Major Project

**Implementation Status:** Containerized prototype with automated tests and synthetic evaluation experiments.

**GitHub Repository:**
https://github.com/hemapravallika24/real-time-network-anomaly-detection

---

*This project is developed for academic and learning purposes. The current implementation is an experimental prototype based on synthetic network traffic.*