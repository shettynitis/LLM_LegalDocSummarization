# Base image
FROM python:3.11-slim-buster

# Create and set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy in the rest of your application code
COPY . .

# Environment variables
ENV FLASK_APP=run.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=8000
ENV FLASK_ENV=production

# Expose the port—match FLASK_RUN_PORT
EXPOSE 8000

# Launch the app
CMD ["flask", "run"]