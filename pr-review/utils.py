import textwrap
from typing import Dict, List

def print_pr_summary(pr_info: Dict, files: list):
    """Print a summary of the PR and its files."""
    print("\n=== Pull Request Summary ===")
    print(f"Title: {pr_info['title']}")
    print(f"Author: {pr_info['user']['login']}")
    print(f"Base Branch: {pr_info['base']['ref']}")
    print(f"Head Branch: {pr_info['head']['ref']}")
    
    total_additions = sum(file['additions'] for file in files)
    total_deletions = sum(file['deletions'] for file in files)
    
    print(f"\nTotal Changes: +{total_additions} -{total_deletions}")
    print(f"Number of Files Modified: {len(files)}")
    
    print("\n=== Modified Files ===")
    print(f"{'File Path':<70} | {'Status':<10} | {'Changes':>15}")
    print("-" * 100)
    
    for file in files:
        changes = f"+{file['additions']} -{file['deletions']}"
        print(f"{textwrap.shorten(file['filename'], width=69):<70} | {file['status']:<10} | {changes:>15}")

def prompt_for_files(files: list) -> list:
    """Prompt user to select which files to review."""
    print("\nWould you like to review:")
    print("1. All files")
    print("2. Select specific files")
    
    choice = input("\nEnter your choice (1 or 2): ").strip()
    
    if choice == "1":
        return files
    elif choice == "2":
        print("\nEnter file numbers to review (comma-separated) or 'q' to quit:")
        for i, file in enumerate(files, 1):
            print(f"{i}. {file['filename']}")
        
        selections = input("\nFiles to review: ").strip()
        if selections.lower() == 'q':
            return []
        
        try:
            selected_indices = [int(i.strip()) - 1 for i in selections.split(',')]
            return [files[i] for i in selected_indices if 0 <= i < len(files)]
        except (ValueError, IndexError):
            print("Invalid selection. Reviewing all files.")
            return files
    else:
        print("Invalid choice. Reviewing all files.")
        return files