from bedrock_client import get_bedrock_client, generate_message, generate_message_streaming, process_streaming_response, print_response

'''

I've provided two examples of open-ended prompting techniques that encourage Claude to generate more exploratory, creative, and divergent responses:
Open-Ended Philosophical Exploration
The first example demonstrates how to prompt Claude for deeper, multi-perspective thinking on complex topics:

Key Elements:

System prompt that encourages exploring topics from multiple angles
Invites consideration of diverse disciplines and unexpected connections
Explicitly values reflection over definitive answers
Acknowledges nuance and complexity
Uses a higher temperature setting (0.8) to encourage more diverse thinking
Increased top_p (0.95) and top_k (250) values for greater language diversity



This approach is excellent for:

Philosophical discussions
Future speculation and forecasting
Ethical considerations
Complex social questions
Scenario exploration
Educational contexts that encourage critical thinking

Creative Brainstorming
The second example shows how to use Claude as a creative ideation partner:

Key Elements:

System prompt that establishes a "no judgment" brainstorming environment
Encourages generating diverse ideas without self-censoring
Promotes building upon concepts in unexpected ways
Uses very high temperature (0.9) to maximize creative variability
Explicitly suspends criticism during the idea generation phase
Focuses on expanding thinking beyond conventional boundaries



This approach is valuable for:

Product ideation
Design challenges
Creative problem-solving
Marketing ideation
Educational activities
Innovation workshops

Implementation Notes
Both examples use higher temperature settings (0.8-0.9) than the previous examples, which increases the randomness and diversity in Claude's responses. They also use:

Higher top_p value (0.95) which samples from a wider range of token probabilities
Increased top_k (250) which allows for more vocabulary diversity
Larger max_tokens (1500) to give Claude room to explore ideas more fully

These parameter adjustments work together with the open-ended system prompts to create an environment where Claude feels "permitted" to explore more widely and creatively, rather than converging on the most likely or conventional responses.
The key difference between this approach and more structured prompting techniques is that you're intentionally creating space for exploration rather than directing Claude toward a specific format or answer.


'''

def open_ended_prompting_example():
    # Get the Bedrock client from the separate module
    bedrock = get_bedrock_client()
    
    # Open-ended system prompt that encourages exploratory thinking
    system_prompt = """
    You are a thoughtful and creative AI assistant who excels at exploring complex topics from multiple perspectives.
    
    When responding to open-ended questions:
    - Consider the topic from various angles and disciplines
    - Draw from diverse fields of knowledge
    - Explore unexpected connections between ideas
    - Offer thoughtful reflections rather than definitive answers
    - Be willing to acknowledge nuance and complexity
    
    Your goal is not to provide a single "correct" answer, but to stimulate creative thinking and deeper exploration of the topic.
    """
    
    # Open-ended question
    open_ended_question = """
    How might the widespread adoption of artificial intelligence transform the relationship between humans and technology over the next few decades? What unexpected consequences might emerge?
    """
    
    # Create messages
    messages = [
        {"role": "user", "content": open_ended_question}
    ]
    
    # Model ID for Claude 3 Sonnet
    model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
    
    # Generate response
    stream_response = generate_message_streaming(
        bedrock_runtime=bedrock,
        model_id=model_id,
        system_prompt=system_prompt,
        messages=messages,
        max_tokens=1500,
        temperature=0.8,  # Higher temperature for more creative and diverse responses
        top_p=0.95,  # Higher top_p for more diverse sampling
        top_k=250  # Increased top_k for more vocabulary diversity
    )

    full_response = process_streaming_response(stream_response)

    return full_response

def brainstorming_prompt_example():
    # Get the Bedrock client from the separate module
    bedrock = get_bedrock_client()
    
    # System prompt for creative ideation
    system_prompt = """
    You are a creative ideation partner who helps generate innovative solutions through free-flowing brainstorming.
    
    When brainstorming:
    - Generate a diverse range of ideas without self-censoring
    - Build upon concepts in unexpected ways
    - Explore unusual combinations and connections
    - Consider both practical and imaginative approaches
    - Avoid criticizing or evaluating ideas during the generative phase
    
    Your goal is to help expand thinking beyond conventional boundaries and discover novel possibilities.
    """
    
    # Brainstorming challenge
    brainstorming_challenge = """
    I'm designing a community garden in an urban neighborhood with limited space (about half an acre). 
    I want it to be multi-functional - providing food, beauty, and social connection.
    Can you help me brainstorm innovative features or elements that could make this garden unique?
    """
    
    # Create messages
    messages = [
        {"role": "user", "content": brainstorming_challenge}
    ]
    
    # Model ID for Claude 3 Sonnet
    model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
    
    # Generate response
    response = generate_message(
        bedrock_runtime=bedrock,
        model_id=model_id,
        system_prompt=system_prompt,
        messages=messages,
        max_tokens=1500,
        temperature=0.9,  # Very high temperature for maximum creativity
        top_p=0.95,  # Higher top_p for more diverse sampling
        top_k=250  # Increased top_k for more vocabulary diversity
    )
    
    print_response(response)
                
    return response


if __name__ == "__main__":
    print("Running Open-Ended Prompting Example...")
    open_ended_prompting_example()
    
    print("\n\n" + "="*80 + "\n\n")
    
    print("Running Brainstorming Prompt Example...")
    brainstorming_prompt_example()