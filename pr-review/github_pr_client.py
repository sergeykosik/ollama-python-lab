from urllib.parse import urlparse
import requests
from typing import Dict

class GithubPRClient:
    def __init__(self, github_token: str):
        self.github_token = github_token
        self.headers = {
            'Authorization': f'token {github_token}',
            'Accept': 'application/vnd.github.v3+json'
        }

    def parse_github_url(self, pr_url: str) -> Dict[str, any]:
        parts = urlparse(pr_url).path.split('/')
        return {
            'owner': parts[1],
            'repo': parts[2],
            'pr_number': int(parts[4])
        }

    def get_pr_info(self, owner: str, repo: str, pr_number: int) -> Dict:
        url = f'https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}'
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def get_pr_files(self, owner: str, repo: str, pr_number: int) -> list:
        url = f'https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/files'
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def get_file_content(self, owner: str, repo: str, pr_number: int, filename: str) -> Dict[str, str]:
        # Get PR details
        pr_url = f'https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}'
        pr_response = requests.get(pr_url, headers=self.headers)
        pr_response.raise_for_status()
        pr_data = pr_response.json()
        head_sha = pr_data['head']['sha']

        # Get raw content
        raw_headers = {**self.headers, 'Accept': 'application/vnd.github.v3.raw'}
        raw_url = f'https://raw.githubusercontent.com/{owner}/{repo}/{head_sha}/{filename}'
        response = requests.get(raw_url, headers=raw_headers)

        content = None
        if response.status_code == 200:
            content = response.text

        # Get diff
        diff_url = f'https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/files'
        diff_response = requests.get(diff_url, headers=self.headers)
        diff_response.raise_for_status()
        
        diff = None
        for file in diff_response.json():
            if file['filename'] == filename:
                diff = file.get('patch', '')
                break

        return {
            'content': content,
            'diff': diff
        }