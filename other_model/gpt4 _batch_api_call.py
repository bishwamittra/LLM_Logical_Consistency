import argparse
import os
import requests
from openai import OpenAI
import pandas as pd
import json
from IPython.display import Image, display


# Set up argument parsing
parser = argparse.ArgumentParser(description="Process a file for OpenAI API batch request.")

parser.add_argument("file_path", type=str, help="Path to the input file in csv format")
parser.add_argument("api_key", type=str, help="Your openai key")

args = parser.parse_args()
file_path = args.file_path
api_key = args.api_key

"""# Read data"""
df=pd.read_csv(file_path)
df.head()

# Commented out IPython magic to ensure Python compatibility.
# make sure you have the latest version of the SDK available to use the Batch API
# %pip install openai --upgrade

api_key= api_key

# Initializing OpenAI client
client = OpenAI(api_key=api_key)

"""# Filtering the data"""

df=df[df['use_context']==True]
df=df.head(5000)
df.head()

System='You are a helpful assistant. Always follow the instructions precisely and output the response exactly in the requested format.'

# convert llama to gpt like
def convert_to_gpt_prompt(llama_prompt):
    return (
        llama_prompt.split('[INST]')[-1]
        .replace("<s><<SYS>>", "System: ")
        .replace("<</SYS>>", "")
        .replace("[NEWLINE]", "\n")
        .replace("[INST]", "Instruction:")
        .replace("[/INST]", "")
    )


df['gpt_prompt_base_query'] = df['prompt_base_query'].apply(convert_to_gpt_prompt)


print(df['gpt_prompt_base_query'][1])

df['gpt_prompt_negation_query'] = df['prompt_negation_query'].apply(convert_to_gpt_prompt)

len(df)

# Apply conversion
df['gpt_prompt_base_query'] = df['prompt_base_query'].apply(convert_to_gpt_prompt)
df['gpt_prompt_negation_query'] = df['prompt_negation_query'].apply(convert_to_gpt_prompt)

# Display results
print(df['gpt_prompt_base_query'][1])

"""# API call

# Edited api call
"""

categorize_system_prompt='You are a helpful assistant. Always follow the instructions precisely and output the response exactly in the requested format.'

tasks = []

for index , row in df.iterrows():

    description = row['gpt_prompt_base_query']

    task = {
        "custom_id": f"task-{index}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            # This is what you would have in your Chat Completions API call
            "model": "gpt-4o",
            "temperature": 0,
            #"response_format": {
                #"type": "json_object"
            #},
        "messages":[
            {"role": "system", "content": System},
            {"role": "user", "content": description}
        ],


    #"output_file": "output_results.jsonl",  # Specify desired output file
    #"error_file": "error_logs.jsonl"       # Specify desired error file
        }
    }


    tasks.append(task)

"""# Without batch dividing"""

# Creating the file

file_name = "/content/mydata/data/batch_tasks_freebase.jsonl"

with open(file_name, 'w') as file:
    for obj in tasks:
        file.write(json.dumps(obj) + '\n')

# Uploading the file

batch_file = client.files.create(
  file=open(file_name, "rb"),
  purpose="batch"
)

batch_file.id

# Creating the job

batch_job = client.batches.create(
  input_file_id=batch_file.id,
  endpoint="/v1/chat/completions",
  completion_window="24h",
)

batch_job

batch_job = client.batches.retrieve(batch_job.id)
print(batch_job.status)

print(batch_job)

print(batch_job.status)

"""# Divding the tasks into batches"""

import time

# split tasks
def split_into_batches(tasks, batch_size=125):
    return [tasks[i:i + batch_size] for i in range(0, len(tasks), batch_size)]

# apply func
batches = split_into_batches(tasks)


for batch_index, batch in enumerate(batches):
    # create a new file for each batch
    batch_file_name = f"batch_{batch_index + 1}.jsonl"
    file_name = file_path + batch_file_name

    # Write the current batch to a file
    with open(file_name, 'w') as file:
        for obj in batch:
            file.write(json.dumps(obj) + '\n')

    # Upload the file
    batch_file = client.files.create(
        file=open(file_name, "rb"),
        purpose="batch"
    )

    # Create the batch job for this batch
    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )

    print(f"Batch {batch_index + 1} submitted with file {batch_file_name}.")

    time.sleep(1)

batch_id = "batch_673b10a6cc18819081f8873729d79f1e"

# check batch status
batch_info = client.batches.retrieve(batch_id)

print(f"Batch Status: {batch_info.status}")
print(f"Output File ID: {batch_info.output_file_id}")
print(f"Error File ID: {batch_info.error_file_id}")

#URL for the file info endpoint


file_id='file-Y8GjUuztA4XdjHh0f0Mj7UIW'
url = f"https://api.openai.com/v1/files/{file_id}"

# set headers
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# make a get request to retrieve file info
response = requests.get(url, headers=headers)

# check if the request was successful
if response.status_code == 200:
    file_info = response.json()
    print("File Info:", file_info)
else:
    print("Error:", response.status_code, response.text)

import requests


file_id='file-Y8GjUuztA4XdjHh0f0Mj7UIW'

# URL to download the file
url = f"https://api.openai.com/v1/files/{file_id}/content"

# set headers
headers = {
    "Authorization": f"Bearer {api_key}"
}

# send a get request to download the file
response = requests.get(url, headers=headers)

# check if the request was successful
if response.status_code == 200:
    # save the file content locally
    with open("error_file.jsonl", "wb") as f:
        f.write(response.content)
    print("File downloaded successfully.")
else:
    print("Error downloading file:", response.status_code, response.text)

import json

# open and read the JSONL file
with open("error_file.jsonl", "r") as file:
    for line in file:
        # Each line is a JSON object
        try:
            data = json.loads(line)
            print(data)  # process the JSON dat
        except json.JSONDecodeError:
            print("Error decoding line:", line)

import json

# open and read the json file
with open("error_file.jsonl", "r") as file:
    for line in file:
        # Each line is a json object
        try:
            data = json.loads(line)
            print(data)  # process the json data (you can handle it accordingly)
        except json.JSONDecodeError:
            print("Error decoding line:", line)

import requests

# Your OpenAI API key
#api_key = 'your_openai_api_key'

# Define the URL
url = 'https://api.openai.com/v1/batches'

# Set up the headers, including your API key
headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json'
}

# Send a GET request to the API
response = requests.get(url, headers=headers)

# Check the response status code
if response.status_code == 200:
    # If successful, print the JSON response
    print(response.json()['data'])
else:
    # If there was an error, print the status code and error message
    print(f"Error {response.status_code}: {response.text}")

for res in (response.json()['data']):
  batch_id=res['status']
  print(batch_id)
  #url='https://api.openai.com/v1/batches/{batch_id}/cancel'
  response = requests.get(url, headers=headers)
