import boto3
import os
import json
from dotenv import load_dotenv

def get_bedrock_client():
    """
    Initialize and return an AWS Bedrock runtime client using credentials from .env file
    
    Returns:
        boto3.client: Configured Bedrock runtime client
    """
    # Load environment variables from .env file
    load_dotenv()
    
    # Retrieve AWS credentials from environment variables
    aws_access_key = os.getenv("AWS_BEDROCK_API_KEY")
    aws_secret_key = os.getenv("AWS_BEDROCK_API_SECRET")
    
    # Set the region (default to us-east-1 if not specified)
    region_name = os.getenv("AWS_BEDROCK_REGION", "us-east-1")
    
    # Initialize AWS Bedrock client
    bedrock = boto3.client(
        service_name="bedrock-runtime",
        region_name=region_name,
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key
    )
    
    return bedrock


def generate_message(bedrock_runtime, model_id, system_prompt, messages, max_tokens, temperature=0.7, top_p=0.9, top_k=50):
    """
    Generate a response from Claude using the messages API format
    
    Args:
        bedrock_runtime: Bedrock runtime client
        model_id: Claude model ID to use
        system_prompt: System instructions for Claude
        messages: List of message objects with role and content
        max_tokens: Maximum number of tokens in the response
        temperature: Controls randomness (0.0-1.0)
        top_p: Controls diversity via nucleus sampling (0.0-1.0)
        top_k: Controls diversity by limiting vocabulary (1-100)
        
    Returns:
        Response body from Claude
    """
    body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k
        }
    )
    
    response = bedrock_runtime.invoke_model(
        body=body, 
        modelId=model_id
    )
    
    response_body = json.loads(response.get('body').read())
    
    return response_body


def print_response(response):
    if "content" in response:
        print("\nPrompt Response:")
        for content_block in response["content"]:
            if content_block["type"] == "text":
                print(content_block["text"])


def generate_message_streaming(bedrock_runtime, model_id, system_prompt, messages, max_tokens, temperature=0.7, top_p=0.9, top_k=50):
    """
    Generate a streaming response from Claude using the messages API format
    
    Args:
        bedrock_runtime: Bedrock runtime client
        model_id: Claude model ID to use
        system_prompt: System instructions for Claude
        messages: List of message objects with role and content
        max_tokens: Maximum number of tokens in the response
        temperature: Controls randomness (0.0-1.0)
        top_p: Controls diversity via nucleus sampling (0.0-1.0)
        top_k: Controls diversity by limiting vocabulary (1-100)
        
    Returns:
        Streaming response from Claude
    """
    body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k
        }
    )
    
    # Enable streaming by adding the appropriate parameters
    response = bedrock_runtime.invoke_model_with_response_stream(
        body=body, 
        modelId=model_id
    )
    
    return response.get('body')


def process_streaming_response(stream):
    """
    Process a streaming response from Claude
    
    Args:
        stream: Streaming response from Claude
        
    Returns:
        Full response text
    """
    full_response = ""
    
    # Iterate through the streaming chunks
    for event in stream:
        chunk = event.get('chunk')
        if chunk:
            chunk_data = json.loads(chunk.get('bytes').decode())
            
            # Check if there's any text content in the chunk
            if 'type' in chunk_data:
                # This is for the first message with metadata
                if chunk_data.get('type') == 'message_start':
                    print("[Stream started]")
                elif chunk_data.get('type') == 'content_block_start':
                    print("[Content block started]")
                elif chunk_data.get('type') == 'content_block_delta':
                    # Extract and print the text delta
                    if 'delta' in chunk_data and 'text' in chunk_data['delta']:
                        text_delta = chunk_data['delta']['text']
                        print(text_delta, end='', flush=True)
                        full_response += text_delta
                elif chunk_data.get('type') == 'message_delta':
                    if 'delta' in chunk_data and 'stop_reason' in chunk_data['delta']:
                        print(f"\n[Stopped: {chunk_data['delta']['stop_reason']}]")
                elif chunk_data.get('type') == 'message_stop':
                    print("\n[Stream ended]")
    
    return full_response