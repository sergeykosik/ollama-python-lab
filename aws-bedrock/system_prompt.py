import json
from bedrock_client import get_bedrock_client, generate_message, print_response

def main():
    # Get the Bedrock client from the separate module
    bedrock = get_bedrock_client()
    
    # Define the system and user messages
    system_prompt = "You are a helpful AI assistant. Answer in a step-by-step manner."
    user_message = {"role": "user", "content": "Explain how blockchain works in simple terms."}
    
    # Create messages array with just the user message
    messages = [user_message]
    
    # Model ID for Claude 3 Sonnet
    model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
    
    # Generate response
    response = generate_message(
        bedrock_runtime=bedrock,
        model_id=model_id,
        system_prompt=system_prompt,
        messages=messages,
        max_tokens=300,
        temperature=0.7,
        top_p=0.9,
        top_k=50
    )
    
    # Print full response
    print("Full response:")
    print(json.dumps(response, indent=2))
    
    print_response(response)

if __name__ == "__main__":
    main()