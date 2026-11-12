# Real-Time Network Anomaly Detection and Security Monitoring

A cloud-oriented network monitoring system that uses Machine Learning to identify unusual network traffic patterns and provides real-time visibility through an interactive Streamlit dashboard.

The project combines **Machine Learning, Python, Docker, and DevOps practices** to explore automated network traffic monitoring and anomaly detection.

## Project Overview

Network traffic can contain unusual patterns that may indicate potential security issues. Monitoring these patterns helps administrators understand network behavior and identify suspicious activity.

This project implements a prototype that:
- Generates synthetic network traffic records.
- Uses the Isolation Forest algorithm to identify unusual traffic patterns.
- Assigns anomaly classifications and risk levels.
- Stores monitoring events in a CSV file.
- Displays traffic statistics and detected anomalies through a Streamlit dashboard.
- Uses Docker and Docker Compose to run the monitoring and dashboard services.

**Note:** The current prototype uses simulated network traffic for development and testing. It does not capture live traffic from a real network. Anomaly predictions are experimental indicators, not confirmed security attacks.

## Key Features

- **Traffic Generation:** Generates synthetic normal and anomalous network traffic records.
- **Machine Learning Detection:** Implements Isolation Forest for unsupervised anomaly detection.
- **Risk Classification:** Categorizes detected events into Low, Medium, and High risk levels using configured rules.
- **Event Logging:** Stores traffic records, anomaly scores, classifications, and detection reasons in CSV format.
- **Interactive Dashboard:** Displays network activity, anomaly statistics, risk summaries, and event logs.
- **Automatic Refresh:** Periodically refreshes the dashboard to display newly recorded events.
- **Containerization:** Packages the application using Docker.
- **Service Orchestration:** Uses Docker Compose to run the monitoring and dashboard services together.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Machine Learning | Scikit-learn, Isolation Forest |
| Data Processing | Pandas |
| Dashboard | Streamlit |
| Data Storage | CSV |
| Containerization | Docker |
| Orchestration | Docker Compose |
| Testing | Pytest |
| Version Control | Git and GitHub |

## System Architecture

The current prototype follows this workflow:

1. The traffic collector generates synthetic network traffic.
2. The monitor sends traffic features to the anomaly detection model.
3. The Isolation Forest model evaluates unusual traffic patterns.
4. Configured rules determine additional anomaly indicators and risk levels.
5. Detection results are recorded in a CSV file.
6. The Streamlit dashboard reads the recorded events and displays monitoring information.

## Project Structure

```text
network-anomaly-detection/
│
├── backend/
│   ├── detector.py
│   ├── monitor.py
│   └── traffic_collector.py
│
├── data/
│   └── network_events.csv
│
├── models/
│
├── tests/
│   └── test_detector.py
│
├── dashboard.py
├── migrate_csv.py
├── requirements.txt
├── Dockerfile
├── compose.yaml
├── .dockerignore
└── .gitignore
```

## Getting Started

### Prerequisites

- Python 3.13 or compatible Python environment
- Git
- Docker Desktop with Docker Compose

### 1. Clone the Repository

```bash
git clone https://github.com/hemapravallika24/real-time-network-anomaly-detection.git
cd real-time-network-anomaly-detection
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```cmd
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Dashboard Locally

```bash
streamlit run dashboard.py
```

Open the application at:

http://localhost:8501

For continuous monitoring, run the traffic monitor in a separate terminal:

```bash
python -m backend.monitor
```

## Run Using Docker Compose

Make sure Docker Desktop is running.

Build and start both services:

```bash
docker compose up --build -d
```

Check the running services:

```bash
docker compose ps
```

View monitor logs:

```bash
docker compose logs -f monitor
```

Access the dashboard:

http://localhost:8501

Stop the services:

```bash
docker compose down
```

The Compose configuration shares the local `data` directory between the monitor and dashboard containers so that recorded events remain accessible on the host.

## Testing

Run the available automated tests using:

```bash
pytest -v
```

The tests cover core anomaly detection behavior and input validation.

## Current Development Status

- [x] Synthetic network traffic generation
- [x] Isolation Forest anomaly detection
- [x] Configurable anomaly indicators and risk classification
- [x] CSV-based event logging
- [x] Interactive Streamlit monitoring dashboard
- [x] Automatic dashboard refresh
- [x] Docker containerization
- [x] Docker Compose orchestration
- [x] Initial automated tests
- [ ] Quantitative evaluation using labeled synthetic test data
- [ ] Cloud deployment
- [ ] CI/CD pipeline and DevSecOps integration
- [ ] Advanced monitoring and centralized cloud storage

## Future Enhancements

- Integration with real network traffic collection.
- Cloud-based deployment and centralized event storage.
- CI/CD automation for testing and deployment.
- DevSecOps practices and security validation.
- Model evaluation using precision, recall, F1-score, and false-positive rate.
- Enhanced monitoring, visualization, and scalability.

## Project Information

**Project Title:** Real-Time Network Anomaly Detection and Security Monitoring Through Cloud Computing and DevOps

**Domain:** Machine Learning, Network Monitoring, Cloud Computing, and DevOps

**Status:** Ongoing Academic Project

**GitHub Repository:** https://github.com/hemapravallika24/real-time-network-anomaly-detection

---

*This project is being developed for academic and learning purposes. The current implementation is an experimental prototype based on synthetic network traffic.*