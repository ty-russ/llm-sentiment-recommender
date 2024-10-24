# Base image
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Install necessary dependencies including Java 17 and procps for PySpark
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    libglib2.0-0 \
    python3-dev \
    openjdk-17-jdk-headless \
    procps  # Provides the ps command needed by PySpark

# Set JAVA_HOME environment variable for PySpark to locate Java
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

# Debug: Check Java installation
RUN echo "JAVA_HOME is set to $JAVA_HOME" && \
    java -version && \
    echo "Java path is: $(which java)"

# Copy the requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Expose the port the app will run on (Note: This is ignored with host network mode)
EXPOSE 7070

# Run the Flask app
CMD ["python", "app.py"]
