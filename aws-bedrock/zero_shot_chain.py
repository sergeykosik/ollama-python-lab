import json
from bedrock_client import get_bedrock_client, generate_message

'''
I've provided two different advanced prompting techniques that are particularly effective with Claude 3:
1. Zero-Shot Chain of Reasoning (ZSCoR)
This example guides Claude through analyzing a complex business decision involving marketing strategies:

Key Elements:

Provides a clear reasoning structure in the system prompt
Breaks analysis into discrete steps (facts → analysis → comparison → recommendation)
Sets explicit expectations for output format and organization
Lower temperature (0.3) for consistency in analytical reasoning



This approach is excellent for:

Complex business decisions
Multi-variable analysis
Situations requiring cost-benefit comparisons
Policy analysis and recommendations
Risk assessment scenarios

2. Structured Output Format
This example demonstrates how to get Claude to return data in a specific format (JSON):

Key Elements:

Provides an exact JSON schema in the system prompt
Very low temperature (0.1) for highly deterministic outputs
Includes validation of the returned JSON
Defines specific data types for each field



This approach is valuable for:

Integration with other systems
Data extraction and transformation
Creating API-like interactions
Building structured dashboards
Generating machine-readable outputs

Both examples function independently, so you can run either one based on your needs. The code includes functionality to run both examples sequentially, showing the versatility of different prompting techniques with the same underlying Claude 3 model.
These prompting strategies can dramatically improve the quality and usefulness of Claude's responses for specific use cases. They help the model structure its thinking and output in ways that are more directly usable for your specific applications.'

'''



def zero_shot_chain_of_reasoning_example():
    # Get the Bedrock client from the separate module
    bedrock = get_bedrock_client()
    
    # Define a complex scenario that requires analysis
    scenario = """
    A startup is deciding between three marketing strategies:
    
    Strategy A: Social media campaign costing $50,000 with estimated 5% conversion rate
    Strategy B: Influencer partnerships costing $75,000 with estimated 7% conversion rate
    Strategy C: Content marketing costing $40,000 with estimated 3% conversion rate
    
    Each converted customer has an average lifetime value of $200.
    The company expects to reach 100,000 potential customers with any strategy.
    The company has a marketing budget of $60,000.
    
    Which marketing strategy should the company choose and why?
    """
    
    # Zero-Shot Chain of Reasoning system prompt - using a structured format for thinking
    system_prompt = """
    You are a strategic business consultant with expertise in marketing ROI analysis.
    
    When analyzing complex business decisions:
    1. Extract and organize all relevant facts, constraints, and variables
    2. Calculate key metrics for each option (ROI, expected value, etc.)
    3. Compare options objectively against constraints
    4. Consider qualitative factors beyond the numbers
    5. Formulate a clear recommendation with justification
    
    Your analysis should follow this format:
    - Facts: List all relevant information and constraints
    - Analysis: Calculate metrics for each option
    - Comparison: Directly compare options against requirements
    - Recommendation: Provide a clear, justified choice
    """
    
    # User message containing the scenario
    user_message = {"role": "user", "content": scenario}
    
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
        max_tokens=1500,
        temperature=0.3  # Lower temperature for more consistent analysis
    )
    
    # Extract and print just the text content
    if "content" in response:
        print("\nZero-Shot Chain of Reasoning Response:")
        for content_block in response["content"]:
            if content_block["type"] == "text":
                print(content_block["text"])
                
    return response

def structured_output_example():
    # Get the Bedrock client from the separate module
    bedrock = get_bedrock_client()
    
    # Input data for company analysis
    company_data = """
    Company: TechInnovate Inc.
    Industry: Software as a Service (SaaS)
    Founded: 2018
    Employees: 120
    Revenue (2023): $8.5 million
    Growth Rate: 35% year-over-year
    Key Products: AI-assisted document processing, intelligent workflow automation
    Competitors: DocuAI, FlowTech, Automator Pro
    Recent News: Just released a new mobile app version, considering Series B funding
    """
    
    # Structured output system prompt - requesting JSON format
    system_prompt = """
    You are a business analyst who provides structured analysis in JSON format.
    
    When analyzing a company:
    1. Extract all key information
    2. Perform SWOT analysis (Strengths, Weaknesses, Opportunities, Threats)
    3. Provide strategic recommendations
    
    Your response must be valid JSON following this exact schema:
    {
      "company_profile": {
        "name": "string",
        "industry": "string",
        "founding_year": number,
        "employee_count": number,
        "annual_revenue": number,
        "growth_rate": number
      },
      "swot_analysis": {
        "strengths": ["string", "string", ...],
        "weaknesses": ["string", "string", ...],
        "opportunities": ["string", "string", ...],
        "threats": ["string", "string", ...]
      },
      "strategic_recommendations": ["string", "string", ...],
      "investment_outlook": {
        "risk_level": "string", // "low", "medium", or "high"
        "potential_return": "string", // "low", "medium", or "high" 
        "time_horizon": "string" // "short-term", "medium-term", or "long-term"
      }
    }
    """
    
    # User message requesting structured analysis
    user_message = {"role": "user", "content": f"Provide a structured analysis of this company:\n\n{company_data}"}
    
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
        max_tokens=1500,
        temperature=0.1  # Very low temperature for consistent structured output
    )
    
    # Extract and print just the text content
    if "content" in response:
        print("\nStructured Output Response (JSON):")
        for content_block in response["content"]:
            if content_block["type"] == "text":
                print(content_block["text"])
                
                # Parse the JSON to verify it's valid and show formatted
                try:
                    json_data = json.loads(content_block["text"])
                    print("\nParsed JSON successfully!")
                except json.JSONDecodeError as e:
                    print(f"\nError parsing JSON: {e}")
                
    return response


if __name__ == "__main__":
    print("Running Zero-Shot Chain of Reasoning Example...")
    zero_shot_chain_of_reasoning_example()
    
    print("\n\n" + "="*80 + "\n\n")
    
    print("Running Structured Output Example...")
    structured_output_example()