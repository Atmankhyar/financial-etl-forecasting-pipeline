FROM apache/airflow:2.7.1-python3.8

# Copy requirements
COPY requirements.txt /requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /requirements.txt

# Set working directory
WORKDIR /opt/airflow
