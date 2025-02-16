# DataWorks Automation Agent

This project automates routine tasks for DataWorks Solutions using FastAPI.

## Setup

1. Clone the repository.
2. Create a virtual environment and activate it.
3. Install dependencies using `pip install -r requirements.txt`.
4. Run the application using `uvicorn main:app --reload`.

## Endpoints

- `POST /run?task=<task description>`: Executes a plain-English task.
- `GET /read?path=<file path>`: Returns the content of the specified file.

## Docker

1. Build the Docker image: `docker build -t dataworks-automation .`
2. Run the Docker container: `docker run -e AIPROXY_TOKEN=$AIPROXY_TOKEN -p 8000:8000 dataworks-automation`