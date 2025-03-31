import json
from bedrock_client import get_bedrock_client

# Initialize AWS Bedrock client
bedrock = get_bedrock_client()

# Define the model and input parameters
model_id = "anthropic.claude-v2"  # or "amazon.titan-text-express-v1"
prompt = "Write a short introduction about cloud computing."

# Prepare the request payload
payload = {
    "prompt": prompt,
    "max_tokens_to_sample": 200,
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.9
}

# Invoke the model
response = bedrock.invoke_model(
    modelId=model_id,
    contentType="application/json",
    accept="application/json",
    body=json.dumps(payload)
)

# Parse and print the response
response_body = json.loads(response["body"].read())
print(response_body)
