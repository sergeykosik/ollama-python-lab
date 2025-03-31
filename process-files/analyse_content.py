from file_walker import walk_files
from ollama_client import OllamaClient
from pathlib import Path
from typing import List, Union

def process_files_with_ollama(
    directory_path: Union[str, Path],
    include_patterns: List[str] = None,
    exclude_patterns: List[str] = None,
    system_prompt: str = "You are a code review assistant. Analyze the following code and provide insights:"
):
    ollama = OllamaClient()
    
    try:
        for file_info in walk_files(directory_path, include_patterns, exclude_patterns):
            print(f"\nProcessing file: {file_info['name']}")
            print("-" * 80)
            
            prompt = f"Review this file: {file_info['name']}\n\nContent:\n{file_info['content']}"
            
            response = ollama.process_content(prompt, system_prompt)
            if response:
                print("Analysis:")
                for chunk in response:
                    print(chunk.choices[0].delta.content, end="", flush=True)
                print("\n")
            
    except Exception as e:
        print(f"Error processing files: {str(e)}")

if __name__ == "__main__":
    # Example usage
    directory_path = "/workspaces/AccountsPrep/systems/AccountsPrep/src/AccountsPrep.WebUI/Views"
    
    include_patterns = ['BankAccounts.cshtml', 'BankReconciliation.cshtml', 'BankStatementReport.cshtml']
    exclude_patterns = ['*.cs']
    
    system_prompt = """You are a code review assistant specialized in C# and ASP.NET.
    Analyze each file and provide:
    1. Brief summary of the file's purpose
    2. Potential improvements
    3. Any security concerns
    4. Best practices recommendations
    Keep your responses concise and focused."""
    
    process_files_with_ollama(
        directory_path,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        system_prompt=system_prompt
    )