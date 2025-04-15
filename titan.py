import boto3
import json
import base64
import os
import time

prompt_data = """
Create a colorful and artistic landscape featuring beautiful buildings, ancient structures, and a peaceful surrounding.
"""

bedrock = boto3.client(service_name='bedrock-runtime') # service name

payload = {
    "textToImageParams": {
        "text": prompt_data.strip()  # cleaned prompt text
    },
    "taskType": "TEXT_IMAGE",
    "imageGenerationConfig": {
        "cfgScale": 6,
        "seed": 0,
        "quality": "standard",
        "width": 1024,
        "height": 1024
    }
}

body = json.dumps(payload)
response = bedrock.invoke_model(
    modelId="amazon.titan-image-generator-v1",
    contentType="application/json",
    accept="application/json",
    body=body
)

response_body = json.loads(response['body'].read())
#print(json.dumps(response_body, indent=2))  # optional — for debugging

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Get a timestamp for unique naming
timestamp = int(time.time())

# Loop through each image and save
for idx, image_data in enumerate(response_body['images']):
    image_bytes = base64.b64decode(image_data)
    file_name = f"{output_dir}/generated_image_{timestamp}_{idx+1}.png"
    with open(file_name, "wb") as f:
        f.write(image_bytes)
    print(f"Image {idx+1} saved successfully in {output_dir} as {file_name}/")