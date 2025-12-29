# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and ML artifacts
COPY ./python-services/ /app/python-services/
COPY ./app/dataset/ /app/app/dataset/

# Expose the port the app runs on
EXPOSE 8000

# Define the command to run the application
CMD ["uvicorn", "python-services.main:app", "--host", "0.0.0.0", "--port", "8000"]
