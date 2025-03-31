import os
import base64
import requests
from dotenv import load_dotenv
from urllib.parse import urlparse

# Load environment variables
load_dotenv()

def parse_github_url(pr_url):
    """Parse GitHub PR URL to extract owner, repo, and PR number."""
    # Example URL: https://github.com/owner/repo/pull/123
    parts = urlparse(pr_url).path.split('/')
    return {
        'owner': parts[1],
        'repo': parts[2],
        'pr_number': int(parts[4])
    }

def get_pr_files(owner, repo, pr_number, github_token):
    """Retrieve list of files modified in the PR."""
    headers = {
        'Authorization': f'token {github_token}',
        'Accept': 'application/vnd.github.v3+json'
    }
    
    url = f'https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/files'
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    return response.json()

def get_file_content_from_pr(owner, repo, pr_number, filename, github_token):
    """Retrieve content of a specific file from the PR."""
    headers = {
        'Authorization': f'token {github_token}',
        'Accept': 'application/vnd.github.v3+json'
    }
    
    # Get the PR details to find the head SHA
    pr_url = f'https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}'
    pr_response = requests.get(pr_url, headers=headers)
    pr_response.raise_for_status()
    pr_data = pr_response.json()
    head_sha = pr_data['head']['sha']

    # Try to get the raw file content directly from the PR's head reference
    raw_headers = {
        'Authorization': f'token {github_token}',
        'Accept': 'application/vnd.github.v3.raw'
    }
    raw_url = f'https://raw.githubusercontent.com/{owner}/{repo}/{head_sha}/{filename}'
    response = requests.get(raw_url, headers=raw_headers)
    
    if response.status_code == 200:
        return response.text
    
    # If raw content fails, try the blob API
    blob_url = f'https://api.github.com/repos/{owner}/{repo}/contents/{filename}?ref={head_sha}'
    blob_response = requests.get(blob_url, headers=headers)
    
    if blob_response.status_code == 200:
        content_data = blob_response.json()
        if 'content' in content_data:
            return base64.b64decode(content_data['content']).decode('utf-8')
    
    # If both methods fail, try getting the content from the patch
    patch_headers = {
        'Authorization': f'token {github_token}',
        'Accept': 'application/vnd.github.v3.patch'
    }
    patch_url = f'https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}'
    patch_response = requests.get(patch_url, headers=patch_headers)
    patch_response.raise_for_status()
    
    # Return the patch content
    return patch_response.text

def get_file_diff(owner, repo, pr_number, filename, github_token):
    """Retrieve the diff for a specific file in the PR."""
    headers = {
        'Authorization': f'token {github_token}',
        'Accept': 'application/vnd.github.v3.diff'
    }
    
    url = f'https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/files'
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    # Find the specific file's patch
    for file in response.json():
        if file['filename'] == filename:
            return file.get('patch', 'No changes found in diff')
    
    return 'File not found in PR'

def main():
    # Get environment variables
    github_token = os.getenv('GITHUB_TOKEN')
    pr_url = os.getenv('PR_URL')
    
    if not github_token or not pr_url:
        raise ValueError("Please ensure GITHUB_TOKEN and PR_URL are set in .env file")
    
    # Parse PR URL
    pr_info = parse_github_url(pr_url)
    
    try:
        # Get list of files modified in PR
        files = get_pr_files(
            pr_info['owner'],
            pr_info['repo'],
            pr_info['pr_number'],
            github_token
        )
        
        print(f"Found {len(files)} modified files in PR #{pr_info['pr_number']}")
        
        # Process each file
        for file in files:
            print(f"\nFile: {file['filename']}")
            print(f"Status: {file['status']}")
            print(f"Changes: +{file['additions']} -{file['deletions']}")
            
            if file['status'] != 'removed':
                try:
                    # Get file content
                    content = get_file_content_from_pr(
                        pr_info['owner'],
                        pr_info['repo'],
                        pr_info['pr_number'],
                        file['filename'],
                        github_token
                    )
                    print("Content retrieved successfully")
                    
                    # Get file diff
                    diff = get_file_diff(
                        pr_info['owner'],
                        pr_info['repo'],
                        pr_info['pr_number'],
                        file['filename'],
                        github_token
                    )
                    print("\nFile changes:")
                    print(diff)
                    
                    # Create output directory if it doesn't exist
                    os.makedirs('pr_files', exist_ok=True)
                    
                    # Create subdirectories if needed
                    base_path = os.path.join('pr_files', file['filename'])
                    os.makedirs(os.path.dirname(base_path), exist_ok=True)
                    
                    # Save file content
                    with open(base_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"Content saved to: {base_path}")
                    
                    # Save diff to a separate file
                    diff_path = f"{base_path}.diff"
                    with open(diff_path, 'w', encoding='utf-8') as f:
                        f.write(diff)
                    print(f"Diff saved to: {diff_path}")
                    
                except requests.exceptions.RequestException as e:
                    print(f"Error retrieving content: {e}")
                except Exception as e:
                    print(f"Error processing file: {str(e)}")
    
    except requests.exceptions.RequestException as e:
        print(f"Error accessing GitHub API: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()