# Use an official Python runtime as a parent image
FROM python:3.10-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Install any dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files into the container at /app
COPY . /app/

# Set environment variables (if needed)
# ENV DATABASE_URL="your_database_url_here"

# Define the command to run when the container starts
CMD ["python", "main.py"]