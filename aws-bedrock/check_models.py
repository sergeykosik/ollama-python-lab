import boto3
import json

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve AWS credentials from environment variables
aws_access_key = os.getenv("AWS_BEDROCK_API_KEY")
aws_secret_key = os.getenv("AWS_BEDROCK_API_SECRET")

bedrock_models = boto3.client(
    service_name="bedrock",
    region_name="us-east-1",
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key
)

response = bedrock_models.list_foundation_models()
for model in response['modelSummaries']:
    if 'claude' in model['modelId'].lower():
        print(model['modelId'])


# anthropic.claude-3-sonnet-20240229-v1:0:200k