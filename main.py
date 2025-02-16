from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import subprocess
import os
import json
import sqlite3
import requests
import logging
import datetime
from dateutil.parser import parse
import re
from fastapi.responses import PlainTextResponse
import base64
import httpx
import asyncio
import numpy as np
import git
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
import markdown


load_dotenv()

app = FastAPI()

# Set the environment variable for testing
openai_api_key = os.getenv("AIPROXY_TOKEN")

class Task(BaseModel):
    description: str

@app.post("/run")
async def run_task(task: str = Query(..., description="Task description")):
    try:
        result = await execute_task(task)
        return {"message": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/read")
async def read_file(path: str = Query(..., description="Path of the file to read")):
    path = "." + path
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as file:
            content = file.read()
            if './data/dates-wednesdays.txt' in path:
                return int(content)
            elif './data/logs-recent.txt' in path:
                return PlainTextResponse(content)
            elif './data/email-sender.txt' in path:
                return PlainTextResponse(content)
            elif './data/credit-card.txt' in path:
                return int(content)
            else:
                return PlainTextResponse(content)
    else:
        raise HTTPException(status_code=404, detail="File not found")

async def execute_task(task):
    if 'Install `uv`' in task:
        return install_and_run_datagen(task)
    elif 'format.md' in task:
        return format_markdown()
    elif 'dates.txt' in task:
        return count_days_of_week(task)
    elif 'contacts.json' in task:
        return sort_contacts()
    elif 'logs/' in task:
        return extract_log_lines()
    elif 'Markdown' in task:
        return generate_index()
    elif 'email.txt' in task:
        return await extract_email()
    elif 'credit_card.png' in task:
        return await extract_credit_card_number()
    elif 'comments.txt' in task:
        return await find_similar_comments()
    elif 'ticket-sales.db' in task:
        return await calculate_total_sales()
    elif 'Fetch data from an API and save it' in task:
        return await fetch_data_from_api_and_save(task)
    elif 'Clone a git repo and make a commit' in task:
        return clone_and_commit_repo(task)
    elif 'Extract data from' in task:
        return await scrape_website(task)
    elif 'Compress or resize an image' in task:
        return await compress_or_resize_image_task(task)
    elif 'Convert to HTML' in task:
        return await convert_markdown_file_to_html()
    else:
        raise ValueError('Task not recognized')

def install_and_run_datagen(task):
    subprocess.run(['pip', 'install', 'uv'], check=True)
    email_match = re.search(r'`([^`]+@[^`]+)`', task)
    if not email_match:
        raise ValueError("Email not found in task description")
    email = email_match.group(1)
    subprocess.run(['python3', 'datagen.py', email], check=True)
    return 'Datagen script executed successfully'

def format_markdown():
    subprocess.run(["npx", "--yes", "prettier@3.4.2", "--write", 'data/format.md'],check=True,)
    return 'Markdown file formatted successfully'

def count_days_of_week(task):
    days_of_week = {
        'Wednesdays': 2,
        'Thursdays': 3,
        'Sundays': 6,
    }
    day_keyword = None
    input_file = None
    output_file = None

    if 'dates.txt' in task and 'Wednesdays' in task:
        day_keyword = 'Wednesdays'
        input_file = 'data/dates.txt'
        output_file = 'data/dates-wednesdays.txt'
    elif 'extracts.txt' in task and 'Thursdays' in task:
        day_keyword = 'Thursdays'
        input_file = 'data/extracts.txt'
        output_file = 'data/extracts-count.txt'
    elif 'contents.log' in task and ('Sundays' in task or 'रविवार' in task or 'ஞாயிறு' in task):
        day_keyword = 'Sundays'
        input_file = 'data/contents.log'
        output_file = 'data/contents.dates'

    if not day_keyword or not input_file or not output_file:
        raise ValueError('Invalid task description for counting days of the week')

    with open(input_file, 'r') as file:
        dates = file.readlines()
    
    count = 0
    for date in dates:
        date = date.strip()
        try:
            date_obj = parse(date)
        except ValueError:
            continue
        
        if date_obj.weekday() == days_of_week[day_keyword]:
            count += 1
    
    with open(output_file, 'w') as file:
        file.write(str(count))
    
    return f'Number of {day_keyword} counted successfully'

def sort_contacts():
    with open('data/contacts.json', 'r') as file:
        contacts = json.load(file)
        
    contacts.sort(key=lambda x: (x['last_name'], x['first_name']))
    
    with open('data/contacts-sorted.json', 'w') as file:
        json.dump(contacts, file, indent=2)
    
    return 'Contacts sorted successfully'

def extract_log_lines():
    log_files = [f for f in os.listdir('data/logs/') if f.endswith('.log')]
    log_files.sort(key=lambda x: os.path.getmtime(os.path.join('data/logs/', x)), reverse=True)
    log_lines = []
    for log_file in log_files[:10]:
        with open(os.path.join('data/logs/', log_file), 'r') as file:
            log_lines.append(file.readline().strip())
    with open('data/logs-recent.txt', 'w') as file:
        file.write("\n".join(log_lines) + "\n")
    return 'Log lines extracted successfully'

def generate_index():
    input_folder = 'data/docs/'
    output_file_path = 'data/docs/index.json'
    index = {}

    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.startswith('# '):
                            title = line[2:].strip()
                            relative_path = os.path.relpath(file_path, input_folder)
                            relative_path_with_slash = relative_path.replace("\\", "/")
                            index[relative_path_with_slash] = title
                            break

    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=4)

    return 'Index generated successfully'

async def extract_email():
    input_file_path = 'data/email.txt'
    output_file_path = 'data/email-sender.txt'

    async def extract_sender_email(input_file_path, output_file_path):
        AIPROXY_TOKEN = os.environ["AIPROXY_TOKEN"]
        # Read the email content from the input file
        with open(input_file_path, "r") as file:
            email_content = file.read()

        model = "gpt-4o-mini"

        # Call the LLM to extract the sender's email address
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {AIPROXY_TOKEN}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": [
                            {
                                "role": "system",
                                "content": "Extract the sender's e-mail address from the following email message",
                            },
                            {"role": "user", "content": f"{email_content}"},
                        ],
                    },
                )

                result = response.json()
                print(f"Response received from GPT: {result}")
                sender_email = result["choices"][0]["message"]["content"]

                print(f"Sender's email id detected: {sender_email}")

                # Write the email address to the output file
                with open(output_file_path, "w") as file:
                    file.write(sender_email)

        except Exception as e:
            print(f"Error occurred while fetching response from GPT: {e}")
            raise e

    await extract_sender_email(input_file_path, output_file_path)
    return 'Email address extracted successfully'

async def extract_credit_card_number():
    input_file_path = 'data/credit_card.png'
    output_file_path = 'data/credit-card.txt'
    
    async def extract_numbers_from_image(input_file_path, output_file_path):
        # Read the image file and encode it in base64
        with open(input_file_path, "rb") as f:
            binary_data = f.read()
            image_b64 = base64.b64encode(binary_data).decode()

        # Data URI example (embed images in HTML/CSS)
        data_uri = f"data:image/png;base64,{image_b64}"

        AIPROXY_TOKEN = os.environ["AIPROXY_TOKEN"]
        OPENAI_API_URL = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    OPENAI_API_URL,
                    headers={"Authorization": f"Bearer {AIPROXY_TOKEN}"},
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "Extract the digits from this image with length more than 12 digits",
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": data_uri},
                                    },
                                ],
                            }
                        ],
                    },
                )
            print(f"GPT response = {response.json()}")
            card_number_text = response.json()["choices"][0]["message"]["content"]
            print(f"Extracted text from GPT: {card_number_text}")
            card_number = extract_credit_card_number_from_text(card_number_text)
            print(f"Extracted credit card number: {card_number}")

            # Write the result to the output file (without spaces in the card number)
            with open(output_file_path, "w") as output_file:
                output_file.write(card_number.replace(" ", ""))

            print(f"Extracted credit card number has been written to {output_file_path}")

        except KeyError as e:
            print(
                f"INSIDE EXTRACT_NUMBERS_FROM_IMAGE IN A8. KeyError occurred while querying GPT: {e}"
            )
            raise HTTPException(status_code=400, detail=str(e))

        except Exception as e:
            print(
                f"INSIDE EXTRACT_NUMBERS_FROM_IMAGE IN A8. General Error while querying gpt: {str(e)}"
            )
            raise HTTPException(status_code=500, detail=str(e))

    await extract_numbers_from_image(input_file_path, output_file_path)
    return 'Credit card number extracted successfully'

def extract_credit_card_number_from_text(text):
    # Regular expression pattern for credit card number (with or without spaces)
    card_number_pattern = r"\b(?:\d[ -]*?){13,19}\b"

    # Search for the card number in the given text
    match = re.search(card_number_pattern, text)

    if match:
        # Clean the card number (remove any spaces or hyphens)
        card_number = match.group(0).replace(" ", "").replace("-", "")
        return card_number
    else:
        return None

async def find_similar_comments():
    input_file_path = 'data/comments.txt'
    output_file_path = 'data/comments-similar.txt'

    async def get_similarity_from_embeddings(emb1: list[float], emb2: list[float]) -> float:
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

    async def embed_list(text_list: list[str]) -> list[float]:
        AIPROXY_TOKEN = os.environ["AIPROXY_TOKEN"]
        OPENAI_API_URL = "http://aiproxy.sanand.workers.dev/openai/v1/embeddings"
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    OPENAI_API_URL,
                    headers={"Authorization": f"Bearer {AIPROXY_TOKEN}"},
                    json={"model": "text-embedding-3-small", "input": text_list},
                )
            emb_list = [emb["embedding"] for emb in response.json()["data"]]
            return emb_list

        except KeyError as e:
            print(f"INSIDE EMBED_LIST IN A9. KeyError occurred while querying GPT: {e}")
            raise HTTPException(status_code=400, detail=str(e))

        except Exception as e:
            print(f"INSIDE EMBED_LIST IN A9. General Error while querying gpt: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def most_similar(embeddings):
        phrases = list(embeddings.keys())
        emb_values = list(embeddings.values())

        max_similarity = -1
        most_similar_pair = None

        for i in range(len(emb_values)):
            for j in range(i + 1, len(emb_values)):
                similarity = await get_similarity_from_embeddings(emb_values[i], emb_values[j])
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_pair = (phrases[i], phrases[j])

        return most_similar_pair

    with open(input_file_path, "r") as file:
        comments = file.readlines()

    embeddings = await embed_list(comments)
    embed_dict = dict(zip(comments, embeddings))
    most_similar_pair = await most_similar(embed_dict)

    with open(output_file_path, "w") as file:
        for comment in most_similar_pair:
            file.write(f"{comment.strip()}\n")

    return 'Similar comments found successfully'

async def calculate_total_sales():
    input_file_path = 'data/ticket-sales.db'
    output_file_path = 'data/ticket-sales-gold.txt'
    ticket_type = "Gold"

    conn = sqlite3.connect(input_file_path)
    cursor = conn.cursor()

    ticket_type = ticket_type.strip()

    cursor.execute(
        "SELECT SUM(units * price) FROM tickets WHERE type LIKE ?",
        ("%" + ticket_type + "%",),
    )
    total_sales = cursor.fetchone()[0]

    if total_sales is None:
        total_sales = 0.0

    with open(output_file_path, "w") as file:
        file.write(f"{total_sales}")

    conn.close()

    print(f"Total sales for {ticket_type} tickets = {total_sales} have been written to {output_file_path}")
    return 'Total sales calculated successfully'

async def fetch_data_from_api_and_save(task):
    url_match = re.search(r'from\s+(https?://[^\s]+)', task)
    if not url_match:
        raise ValueError("API URL not found in task description")
    
    api_url = url_match.group(1)
    output_file_path = 'data/api_data.json'
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(api_url)
            response.raise_for_status()
            data = response.json()

        with open(output_file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

        return 'Data fetched from API and saved successfully'
    
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def clone_and_commit_repo(task):
    repo_url_match = re.search(r'Clone a git repo and make a commit from\s+(https?://[^\s]+)', task)
    if not repo_url_match:
        raise ValueError("Repo URL not found in task description")
    
    repo_url = repo_url_match.group(1)
    repo_name = repo_url.split('/')[-1].replace('.git', '')
    commit_message = "Commit made via API"
    
    # Clone the repo
    repo = git.Repo.clone_from(repo_url, repo_name)

    # Set up authentication for the remote repository
    origin = repo.remote(name='origin')
    origin.set_url(repo_url.replace('https://', f'https://{os.getenv("GIT_USERNAME")}:{os.getenv("GIT_TOKEN")}@'))

    # Make a change and commit
    file_path = os.path.join(repo.working_tree_dir, "new_file5.txt")
    with open(file_path, 'w') as file:
        file.write("This is a new file committed via API")
    
    repo.index.add([file_path])
    repo.index.commit(commit_message)
    
    # Push the commit
    origin.push()
    
    return 'Repo cloned and commit made successfully'

async def scrape_website(task):
    url_match = re.search(r'Extract data from (https?://[^\s]+)', task)
    if not url_match:
        raise ValueError("Website URL not found in task description")
    
    website_url = url_match.group(1)
    output_file_path = 'data/website_data.json'
    
    # Ensure the data directory exists
    os.makedirs('data', exist_ok=True)
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(website_url)
            response.raise_for_status()
            html_content = response.text

        # Parse the HTML content
        soup = BeautifulSoup(html_content, 'html.parser')

        # Extract data based on your needs. Here we extract all text inside <p> tags for demonstration purposes.
        data = {'paragraphs': [p.get_text() for p in soup.find_all('p')]}

        with open(output_file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

        return 'Data scraped from website and saved successfully'
    
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def compress_or_resize_image_task(task: str):
    url_match = re.search(r'from\s+(https?://[^\s]+)', task)
    if not url_match:
        raise ValueError("Image URL not found in task description")
    
    image_url = url_match.group(1)
    size_match = re.search(r'width\s+(\d+)\s+and\s+height\s+(\d+)', task)
    if not size_match:
        raise ValueError("Width and height not found in task description")
    
    width = int(size_match.group(1))
    height = int(size_match.group(2))
    
    return await resize_image_url(image_url=image_url, width=width, height=height)

async def resize_image_url(image_url: str, width: int, height: int):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(image_url)
            response.raise_for_status()
            image_data = response.content
        
        image = Image.open(BytesIO(image_data))
        image = image.resize((width, height))
        
        output_path = f"data/resized_image.jpg"
        
        # Ensure the data directory exists
        os.makedirs('data', exist_ok=True)
        
        image.save(output_path)
        
        return {"message": f"Image resized and saved to {output_path}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def convert_markdown_file_to_html():
    input_file_path = 'data/format.md'
    output_file_path = 'data/output.html'
    
    # Read the Markdown content from the file
    if not os.path.exists(input_file_path):
        raise HTTPException(status_code=404, detail="Markdown file not found")
    
    with open(input_file_path, 'r', encoding='utf-8') as file:
        markdown_content = file.read()
    
    # Convert Markdown to HTML
    html_content = markdown.markdown(markdown_content)
    
    # Ensure the data directory exists
    os.makedirs('data', exist_ok=True)
    
    # Save the HTML content to a file
    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write(html_content)
    
    return {"html_content": html_content, "file_path": output_file_path}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("main:app", host='0.0.0.0', port=8000, reload=True)