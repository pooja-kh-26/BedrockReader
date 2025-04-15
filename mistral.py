import boto3
import json

prompt_data = """
Act as Shakespeare and write a poem on Me.
"""

bedrock = boto3.client(service_name='bedrock-runtime') # service name

payload = {
    "prompt": "[INST]" + prompt_data + "[/INST]",
    "max_tokens": 1000,
    "temperature": 0.7,
    "top_p": 0.9
}

body=json.dumps(payload)
model_id = "mistral.mistral-large-2402-v1:0"
response=bedrock.invoke_model(
    body=body,
    modelId=model_id,
    accept="application/json",
    contentType="application/json",
)

response_body = json.loads(response['body'].read())

response_text = response_body['outputs'][0]['text']
print(response_text)
