import boto3
import json

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve AWS credentials from environment variables
aws_access_key = os.getenv("AWS_BEDROCK_API_KEY")
aws_secret_key = os.getenv("AWS_BEDROCK_API_SECRET")

""" bedrock_models = boto3.client(
    service_name="bedrock",
    region_name="us-east-1",
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key
)

response = bedrock_models.list_foundation_models()
for model in response['modelSummaries']:
    if 'claude' in model['modelId'].lower():
        print(model['modelId']) """


# anthropic.claude-3-sonnet-20240229-v1:0:200k


import boto3
from botocore.exceptions import ClientError

def check_bedrock_access(region='us-east-1'):
    bedrock = boto3.client(
        service_name="bedrock",
        region_name=region,
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key
    )
    
    print(f"Checking Bedrock access in region: {region}")
    print("=" * 50)
    
    # Check foundation models
    try:
        models_response = bedrock.list_foundation_models()
        anthropic_models = [
            model for model in models_response['modelSummaries'] 
            if model['providerName'] == 'Anthropic'
        ]
        
        print(f"Found {len(anthropic_models)} Anthropic models:")
        for model in anthropic_models:
            print(f"  - {model['modelId']}")
            
    except Exception as e:
        print(f"Error listing models: {e}")
    
    print("\n" + "=" * 50)
    
    # Check inference profiles
    try:
        profiles_response = bedrock.list_inference_profiles()
        print(f"Found {len(profiles_response['inferenceProfileSummaries'])} inference profiles:")
        
        for profile in profiles_response['inferenceProfileSummaries']:
            profile_id = profile['inferenceProfileId']
            if 'anthropic' in profile_id.lower() or 'claude' in profile_id.lower():
                print(f"  - {profile_id}")
                
    except Exception as e:
        print(f"Inference profiles not available: {e}")

# Run the check
check_bedrock_access()