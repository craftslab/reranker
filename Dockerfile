FROM python:3.13-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY reranker.py .

# Expose the port the app runs on
EXPOSE 8080

# Command to run the application
CMD ["python", "reranker.py"]
