from bedrock_client import get_bedrock_client, generate_message, print_response

'''

This example demonstrates instructional prompting, a powerful technique for getting highly specific and structured responses from Claude. Here's what makes this approach effective:
Instructional Prompting Technique
Instructional prompting uses detailed, procedural directions to guide Claude's response format, content, and approach. It's like providing Claude with a detailed rubric or checklist.
Key Elements:

Explicit Structure Requirements

Defines exact sections to include (Overall Impression, Key Strengths, etc.)
Specifies the precise order of these sections
Sets clear constraints on length (e.g., "3 bullet points maximum")


Tone and Style Guidelines

Defines the voice and approach ("constructive but direct")
Sets boundaries on what type of feedback to provide
Ensures consistency in communication style


Content Specifications

Prioritizes what to focus on (substance over formatting)
Directs attention to specific aspects (achievement statements)
Provides criteria for evaluation (metrics, red flags, keywords)


Analysis Parameters

Sets the depth and scope of analysis
Establishes contextual considerations (role/industry relevance)
Provides perspective guidelines (human readers vs. ATS systems)



When to Use Instructional Prompting:
This technique is particularly effective for:

Professional document reviews (resumes, cover letters, reports)
Structured feedback requiring specific formats
Analytical tasks needing consistent methodology
Complex evaluations with multiple considerations
Technical writing with strict guidelines

The power of instructional prompting comes from its precision and detail. By providing Claude with exact guidelines for both content and format, you get highly consistent, structured responses tailored to your specific requirements.
This approach helps Claude understand not just what to analyze, but exactly how you want the analysis presented, making it ideal for professional and technical applications where format and methodology matter as much as the content itself.

'''

def instructional_prompting_example():
    # Get the Bedrock client from the separate module
    bedrock = get_bedrock_client()
    
    # Complex instructional system prompt
    system_prompt = """
    You are a professional resume reviewer and career coach specializing in tech industry applications.
    
    INSTRUCTIONS FOR RESUME ANALYSIS:
    
    1. FORMAT: Structure your feedback in the following sections, in this exact order:
       - Overall Impression (1-2 sentences)
       - Key Strengths (3 bullet points maximum)
       - Areas for Improvement (3 bullet points maximum)
       - Impact Enhancement (specific suggestions to quantify achievements)
       - ATS Optimization Tips (keyword recommendations based on industry)
       
    2. TONE REQUIREMENTS:
       - Be constructive but direct
       - Provide specific, actionable feedback
       - Use a professional tone appropriate for career coaching
       - Avoid generic advice; all suggestions must be tailored to the specific resume
       
    3. CONTENT GUIDELINES:
       - Focus on substance over formatting issues
       - Emphasize achievement statements over job descriptions
       - Suggest specific metrics where quantification is missing
       - Identify any red flags or gaps that might concern recruiters
       - Recommend industry-specific keywords that would improve ATS performance
       
    4. ANALYSIS DEPTH:
       - Relate feedback to the specific role/industry mentioned
       - Consider both human readers and ATS systems in your review
       - If a target position is mentioned, tailor advice to that specific role
    """
    
    # User message with resume to review
    user_message = {
        "role": "user", 
        "content": """Please review my resume for a Senior Software Engineer position:

JANE SMITH
Seattle, WA | janesmith@email.com | (555) 123-4567 | github.com/janesmith

PROFESSIONAL SUMMARY
Software engineer with 6 years of experience in full-stack development. Skilled in Python, JavaScript, and cloud technologies. Strong problem-solving abilities and team collaboration skills.

EXPERIENCE
Software Engineer, TechCorp Inc., Seattle, WA
July 2020 - Present
- Developed and maintained web applications using React and Node.js
- Implemented CI/CD pipelines for automated testing and deployment
- Collaborated with product managers to define and implement new features
- Participated in code reviews and mentored junior developers

Junior Software Engineer, StartApp LLC, Portland, OR
June 2018 - June 2020
- Assisted in the development of RESTful APIs using Django
- Fixed bugs and implemented minor features in existing applications
- Wrote unit tests to ensure code quality

EDUCATION
Bachelor of Science in Computer Science
University of Washington, Seattle, WA
Graduated: June 2018

SKILLS
Languages: Python, JavaScript, HTML, CSS, SQL
Frameworks: React, Node.js, Django
Tools: Git, Docker, AWS, Jenkins
Soft Skills: Team collaboration, problem-solving, communication
"""
    }
    
    # Create messages
    messages = [user_message]
    
    # Model ID for Claude 3 Sonnet
    model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
    
    # Generate response
    response = generate_message(
        bedrock_runtime=bedrock,
        model_id=model_id,
        system_prompt=system_prompt,
        messages=messages,
        max_tokens=1200,
        temperature=0.4,  # Moderate temperature for professional feedback
        top_p=0.95,  # Higher top_p for more diverse sampling
        top_k=250  # Increased top_k for more vocabulary diversity
    )
    
    print_response(response)
                
    return response

if __name__ == "__main__":
    instructional_prompting_example()