from bedrock_client import get_bedrock_client, generate_message, print_response, process_streaming_response, generate_message_streaming

'''

I've provided two more powerful prompting techniques that work well with Claude 3:
1. Few-Shot Prompting
This example demonstrates how to teach Claude to perform a task by showing examples:

Key Elements:

Provides multiple examples of input and the expected output format
Establishes a clear pattern through demonstration rather than explicit instructions
Creates a conversation history with example inputs and outputs
Shows Claude exactly how to format its response through examples
Uses a moderate temperature (0.3) to maintain consistency with examples



Few-shot prompting is excellent for:

Teaching Claude specific response formats
Demonstrating desired reasoning patterns
Classification and categorization tasks
Sentiment analysis and emotion detection
Creating consistent output templates

2. Role Prompting
This example shows how to have Claude adopt a specific expert persona:

Key Elements:

Defines a detailed professional role with specific expertise
Includes years of experience and areas of specialization
Provides clear guidelines for how the expert would approach questions
Establishes expectations for response format and priorities
Uses a moderate temperature (0.5) to allow for some creativity while maintaining expertise



Role prompting is valuable for:

Getting domain-specific expertise
Technical advice and consulting
Professional-level analysis
Industry-specific recommendations
Responses that require specialized knowledge

Both approaches can be combined with other techniques. For example, you could use:

Few-shot examples within a role-based prompt
Chain-of-thought reasoning within a few-shot framework
Role-based prompting with structured output requirements

The power of these techniques comes from how they shape Claude's approach to the problem, giving it clear patterns to follow without having to explicitly program every detail of the desired behavior.

'''

def few_shot_prompting_example():
    # Get the Bedrock client from the separate module
    bedrock = get_bedrock_client()
    
    # Define the system prompt
    system_prompt = """
    You are a helpful AI assistant that specializes in sentiment analysis and emotion detection in customer feedback.
    """
    
    # Create messages with few-shot examples
    messages = [
        {"role": "user", "content": "Can you classify the following customer feedback by sentiment (positive, negative, neutral) and detect the primary emotion?"},
        {"role": "assistant", "content": "I'd be happy to analyze customer feedback for sentiment and emotion. Please share the feedback you'd like me to classify."},
        
        {"role": "user", "content": "The delivery was much faster than expected and the product quality exceeded my expectations!"},
        {"role": "assistant", "content": """
Sentiment: Positive
Primary Emotion: Satisfaction/Delight
Analysis: The feedback contains strongly positive language ("much faster than expected", "exceeded my expectations") indicating high satisfaction with both delivery speed and product quality.
"""},
        
        {"role": "user", "content": "I've been waiting for 3 weeks and customer service hasn't responded to any of my emails."},
        {"role": "assistant", "content": """
Sentiment: Negative
Primary Emotion: Frustration
Analysis: The feedback expresses clear dissatisfaction with the waiting period ("3 weeks") and lack of communication ("hasn't responded"), indicating customer frustration with service experience.
"""},
        
        {"role": "user", "content": "The product works as described. Shipping took 5 days."},
        {"role": "assistant", "content": """
Sentiment: Neutral
Primary Emotion: Satisfaction (mild)
Analysis: The feedback is factual and doesn't contain strong positive or negative language. There's a mild satisfaction that the product functions as expected, but the tone is predominantly neutral.
"""},
        
        # Now provide the actual feedback to classify
        {"role": "user", "content": "I'm absolutely thrilled with how easy the software was to install, but the monthly subscription fee seems a bit high compared to competitors."}
    ]
    
    # Model ID for Claude 3 Sonnet
    model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
    
    # Generate response
    response = generate_message(
        bedrock_runtime=bedrock,
        model_id=model_id,
        system_prompt=system_prompt,
        messages=messages,
        max_tokens=500,
        temperature=0.3  # Lower temperature for more consistent analysis
    )
    
    print_response(response)
                
    return response


def role_prompting_example():
    """
    Demonstrates role prompting technique with Claude
    """
    # Get the Bedrock client from the separate module
    bedrock = get_bedrock_client()
    
    # Role-based system prompt
    system_prompt = """
    You are an expert cybersecurity consultant with specialization in network security.
    You have 15+ years of experience in:
    - Enterprise network architecture
    - Threat detection and prevention
    - Security policy development
    - Incident response
    
    When answering questions:
    - Focus on practical, actionable advice
    - Consider both security and usability
    - Highlight relevant standards and best practices (NIST, ISO, etc.)
    - Prioritize recommendations by impact and implementation difficulty
    """
    
    # User question
    user_question = """
    Our small business (25 employees) needs to implement a secure remote work policy. 
    We use Microsoft 365, have a mix of company and personal devices, and handle some sensitive customer data. 
    What are the essential security measures we should implement right away?
    """
    
    # Create messages
    messages = [
        {"role": "user", "content": user_question}
    ]
    
    # Model ID for Claude 3 Sonnet
    model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
    
    # Generate response
    stream_response = generate_message_streaming(
        bedrock_runtime=bedrock,
        model_id=model_id,
        system_prompt=system_prompt,
        messages=messages,
        max_tokens=1000,
        temperature=0.5
    )
    
    full_response = process_streaming_response(stream_response)
                
    return full_response

if __name__ == "__main__":
    print("Running Few-Shot Prompting Example...")
    few_shot_prompting_example()
    
    print("\n\n" + "="*80 + "\n\n")
    
    print("Running Role Prompting Example...")
    role_prompting_example()