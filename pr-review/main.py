import os
from dotenv import load_dotenv
from github_pr_client import GithubPRClient
from ollama_client import OllamaClient
from utils import print_pr_summary, prompt_for_files
from config import SYSTEM_PROMPT, DEFAULT_OLLAMA_MODEL

def main():
    # Load environment variables
    load_dotenv()
    github_token = os.getenv('GITHUB_TOKEN')
    pr_url = os.getenv('PR_URL')

    if not github_token or not pr_url:
        raise ValueError("Please ensure GITHUB_TOKEN and PR_URL are set in .env file")

    # Initialize clients
    github_client = GithubPRClient(github_token)
    ollama_client = OllamaClient(model=DEFAULT_OLLAMA_MODEL)

    try:
        # Parse PR URL and get files
        pr_info = github_client.parse_github_url(pr_url)
        
        # Get PR details and files
        pr_details = github_client.get_pr_info(
            pr_info['owner'],
            pr_info['repo'],
            pr_info['pr_number']
        )
        
        files = github_client.get_pr_files(
            pr_info['owner'],
            pr_info['repo'],
            pr_info['pr_number']
        )

        # Print PR summary and modified files
        print_pr_summary(pr_details, files)
        
        # Let user select files to review
        files_to_review = prompt_for_files(files)
        
        if not files_to_review:
            print("No files selected for review.")
            return

        print(f"\nProceeding with review of {len(files_to_review)} files...")

        # Process selected files
        for file in files_to_review:
            print(f"\nAnalyzing: {file['filename']}")
            print(f"Status: {file['status']}")
            print(f"Changes: +{file['additions']} -{file['deletions']}")
            print("-" * 80)

            if file['status'] != 'removed':
                try:
                    # Get file content and diff
                    file_data = github_client.get_file_content(
                        pr_info['owner'],
                        pr_info['repo'],
                        pr_info['pr_number'],
                        file['filename']
                    )

                    if file_data['content']:
                        print("Generating code review...")
                        # Get code review from Ollama
                        response = ollama_client.review_code(
                            file['filename'],
                            file_data['content'],
                            file_data['diff'],
                            SYSTEM_PROMPT
                        )

                        if response:
                            print("\nCode Review:")
                            for chunk in response:
                                print(chunk.choices[0].delta.content, end="", flush=True)
                            print("\n")
                    else:
                        print("Could not retrieve file content")

                except Exception as e:
                    print(f"Error processing file: {str(e)}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()