from bedrock_client import get_bedrock_client, generate_message, print_response

'''

This example demonstrates how to use Claude 3 with a chain of thought (CoT) approach for solving problems that benefit from step-by-step reasoning.
The key elements of this chain of thought implementation:

System Prompt Design:

Explicitly instructs Claude to "think step-by-step"
Provides a specific reasoning framework (identify variables, set up equations, etc.)
Guides the model to show its work before providing an answer


Problem Selection:

Uses a math problem that benefits from structured reasoning
The problem involves algebraic relationships and constraints that are best approached systematically


Parameter Adjustments:

Uses a lower temperature (0.2) to reduce randomness and focus on logical reasoning
Increases max_tokens to 1000 to allow space for detailed explanations
This gives the model "room to think" and explain its process



When you run this code, Claude will solve the warehouse inventory problem by breaking down its reasoning into clear steps, showing how it:

Defines variables for the three sections
Transforms the text description into algebraic equations
Creates a system of equations
Solves for the number of items in each section
Verifies the solution matches the original constraints

Chain of thought prompting is particularly effective for:

Mathematical problems
Logic puzzles
Multi-step reasoning tasks
Decision-making scenarios
Any problem where showing work is valuable

You can adapt this approach to other domains by modifying the system prompt to guide the specific type of reasoning needed for your use case.

'''

def chain_of_thought_example():
    # Get the Bedrock client from the separate module
    bedrock = get_bedrock_client()
    
    # Define a problem that benefits from step-by-step reasoning
    math_problem = """
    A warehouse has 3 sections. Section A contains twice as many items as Section B. 
    Section C contains 50 more items than Section B. 
    If the warehouse contains a total of 500 items, how many items are in each section?
    """
    
    # Chain of thought system prompt - explicitly requesting step-by-step reasoning
    system_prompt = """
    You are a helpful AI assistant that excels at problem-solving. 
    Think step-by-step when solving problems. Break down your reasoning process clearly:
    1. First, identify the variables and what you're looking for
    2. Set up the relevant equations
    3. Solve the equations systematically
    4. Verify your solution makes sense
    5. Provide the final answer clearly
    """
    
    # User message containing the problem
    user_message = {"role": "user", "content": math_problem}
    
    # Create messages array with just the user message
    messages = [user_message]
    
    # Model ID for Claude 3 Sonnet
    model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
    
    # Generate response with higher max tokens to allow for detailed reasoning
    response = generate_message(
        bedrock_runtime=bedrock,
        model_id=model_id,
        system_prompt=system_prompt,
        messages=messages,
        max_tokens=1000,
        temperature=0.2  # Lower temperature for more deterministic reasoning
    )
    
    print_response(response)
                
    return response


if __name__ == "__main__":
    chain_of_thought_example()