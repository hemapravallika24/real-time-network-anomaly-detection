# Use a lightweight Python image
FROM python:3.13-slim

# Set the working directory inside the container
WORKDIR /app

# Install project dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files into the container
COPY . .

# Streamlit dashboard port
EXPOSE 8501

# Start the dashboard
CMD ["streamlit", "run", "dashboard.py", "--server.address=0.0.0.0", "--server.port=8501"]