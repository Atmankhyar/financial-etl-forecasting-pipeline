FROM python:3.8-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install pytest pytest-cov

# Copy source code
COPY src/ ./src/
COPY tests/ ./tests/
COPY dags/ ./dags/

# Create directory for test results
RUN mkdir -p /app/test-results

# Run tests
CMD ["pytest", "--cov=src", "--cov-report=xml:/app/test-results/coverage.xml", "tests/"]
