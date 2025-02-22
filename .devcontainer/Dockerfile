# Use a base Python image
FROM mcr.microsoft.com/devcontainers/python:3.11

# Set the working directory inside the container
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files into the container
COPY . ./

# Default command
CMD ["python"]
