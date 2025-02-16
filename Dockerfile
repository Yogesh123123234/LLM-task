# Use an official Python runtime as a parent image
FROM python:3.10-slim


# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install Node.js and Prettier
RUN apt-get update && apt-get install -y curl gnupg && \
    curl -sL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs && \
    npm install -g prettier@3.4.2 && \
    apt-get update && apt-get install -y git

# Set environment variables for the API key
ENV OPENAI_API_KEY=eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIxZjIwMDE1NzNAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.2lju_qamt1QHOQfj0deRcWTngQI0Gom8feKmmUXBuno

# Expose the port
EXPOSE 8000

# Start the FastAPI application and run the evaluation script
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8000"]