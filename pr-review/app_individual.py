import streamlit as st
import os
from github_pr_client import GithubPRClient
from ollama_client_individual import OllamaClient
from config import DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT, DEFAULT_OLLAMA_MODEL
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    if 'pr_files' not in st.session_state:
        st.session_state.pr_files = None
    if 'pr_details' not in st.session_state:
        st.session_state.pr_details = None
    if 'github_client' not in st.session_state:
        github_token = os.getenv('GITHUB_TOKEN')
        if not github_token:
            st.error("GitHub token not found in .env file!")
        else:
            st.session_state.github_client = GithubPRClient(github_token)
    if 'review_status' not in st.session_state:
        st.session_state.review_status = {}
    if 'system_prompt' not in st.session_state:
        st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
    if 'user_prompt' not in st.session_state:
        st.session_state.user_prompt = DEFAULT_USER_PROMPT

def format_pr_summary():
    """Format PR summary for display."""
    if not st.session_state.pr_details or not st.session_state.pr_files:
        return ""
    
    pr = st.session_state.pr_details
    files = st.session_state.pr_files
    
    total_additions = sum(file['additions'] for file in files)
    total_deletions = sum(file['deletions'] for file in files)
    
    return f"""
    #### Pull Request Details
    - **Title**: {pr['title']}
    - **Author**: {pr['user']['login']}
    - **Base Branch**: {pr['base']['ref']}
    - **Head Branch**: {pr['head']['ref']}
    - **Total Changes**: +{total_additions} -{total_deletions}
    - **Files Modified**: {len(files)}
    """

def display_file_list():
    """Display the list of modified files with checkboxes."""
    if not st.session_state.pr_files:
        return []
    
    selected_files = []
    for file in st.session_state.pr_files:
        changes = f"+{file['additions']} -{file['deletions']}"
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            if st.checkbox(file['filename'], key=f"file_{file['filename']}"):
                selected_files.append(file)
        with col2:
            st.text(file['status'])
        with col3:
            st.text(changes)
    
    return selected_files

def fetch_pr_data(pr_url):
    """Fetch PR data using the GitHub client."""
    try:
        # Parse PR URL and get data
        pr_info = st.session_state.github_client.parse_github_url(pr_url)
        
        with st.spinner('Fetching PR details...'):
            pr_details = st.session_state.github_client.get_pr_info(
                pr_info['owner'],
                pr_info['repo'],
                pr_info['pr_number']
            )
            st.session_state.pr_details = pr_details
            
            files = st.session_state.github_client.get_pr_files(
                pr_info['owner'],
                pr_info['repo'],
                pr_info['pr_number']
            )
            st.session_state.pr_files = files
        
        return True
    except Exception as e:
        st.error(f"Error fetching PR data: {str(e)}")
        return False

def generate_prompt(prompt_template, **kwargs):
    """Generate prompt by filling in the template."""
    try:
        return prompt_template.format(**kwargs)
    except KeyError as e:
        st.error(f"Invalid prompt template: missing key {e}")
        return None
    except Exception as e:
        st.error(f"Error generating prompt: {str(e)}")
        return None

def review_files(files_to_review):
    """Review selected files using Ollama."""
    ollama_client = OllamaClient(model=DEFAULT_OLLAMA_MODEL)
    
    for file in files_to_review:
        if file['status'] != 'removed':
            try:
                # Check if file was already reviewed
                if file['filename'] in st.session_state.review_status:
                    continue
                
                st.markdown(f"### Reviewing: {file['filename']}")
                status_placeholder = st.empty()
                review_placeholder = st.empty()
                
                status_placeholder.text("Fetching file content...")
                
                # Get file content and diff
                pr_info = st.session_state.github_client.parse_github_url(st.session_state.pr_url)
                file_data = st.session_state.github_client.get_file_content(
                    pr_info['owner'],
                    pr_info['repo'],
                    pr_info['pr_number'],
                    file['filename']
                )

                if file_data['content']:
                    status_placeholder.text("Generating review...")
                    
                    # Generate user prompt from template
                    user_prompt = generate_prompt(
                        st.session_state.user_prompt,
                        filename=file['filename'],
                        content=file_data['content'],
                        diff=file_data['diff']
                    )
                    
                    if not user_prompt:
                        continue
                    
                    response = ollama_client.review_code(
                        file['filename'],
                        file_data['content'],
                        file_data['diff'],
                        st.session_state.system_prompt
                    )

                    if response:
                        review_text = ""
                        for chunk in response:
                            if hasattr(chunk.choices[0].delta, 'content'):
                                content = chunk.choices[0].delta.content
                                if content:
                                    review_text += content
                                    review_placeholder.markdown(review_text)
                        
                        st.session_state.review_status[file['filename']] = review_text
                        status_placeholder.empty()
                    else:
                        status_placeholder.error("Error generating review")
                else:
                    status_placeholder.error("Could not retrieve file content")

            except Exception as e:
                st.error(f"Error processing {file['filename']}: {str(e)}")

def main():
    st.set_page_config(page_title="PR Review Assistant", layout="wide")
    st.title("Pull Request Review Assistant")
    
    initialize_session_state()
    
    # Only proceed if GitHub token is available
    if not st.session_state.github_client:
        st.error("Please add your GitHub token to the .env file!")
        st.stop()
    
    # Sidebar for configuration and prompts
    with st.sidebar:
        st.header("Configuration")
        
        st.subheader("PR URL")
        pr_url = st.text_input("Enter GitHub PR URL", value=st.session_state.get('pr_url', ''))
        
        if st.button("Fetch PR Data"):
            if not pr_url:
                st.error("Please provide a PR URL")
            else:
                st.session_state.pr_url = pr_url
                st.session_state.review_status = {}  # Reset review status
                if fetch_pr_data(pr_url):
                    st.success("PR data fetched successfully!")
        
        st.subheader("Prompts")
        st.session_state.system_prompt = st.text_area(
            "System Prompt",
            value=st.session_state.system_prompt,
            height=200
        )
        
        st.session_state.user_prompt = st.text_area(
            "User Prompt Template",
            value=st.session_state.user_prompt,
            height=200,
            help="Available variables: {filename}, {content}, {diff}"
        )
    
    # Main content
    if st.session_state.pr_details and st.session_state.pr_files:
        st.markdown(format_pr_summary())
        
        st.markdown("### Select Files to Review")
        review_option = st.radio(
            "Review options:",
            ["Review all files", "Select specific files"]
        )
        
        files_to_review = []
        if review_option == "Review all files":
            files_to_review = st.session_state.pr_files
        else:
            files_to_review = display_file_list()
        
        if st.button("Start Review", disabled=not bool(files_to_review)):
            review_files(files_to_review)
        
        # Display saved reviews
        if st.session_state.review_status:
            st.markdown("### Completed Reviews")
            for filename, review in st.session_state.review_status.items():
                with st.expander(filename):
                    st.markdown(review)

if __name__ == "__main__":
    main()