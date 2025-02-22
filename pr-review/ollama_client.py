from openai import OpenAI
from typing import Generator, List, Dict, Tuple
import tiktoken

class OllamaClient:
    def __init__(self, model: str = "qwen2.5-coder:7b", max_tokens: int = 32000):
        self.client = OpenAI(api_key="ollama", base_url="http://host.docker.internal:11434/v1/")
        self.model = model
        self.max_tokens = max_tokens
        # Use GPT-4 tokenizer as an approximation
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text."""
        return len(self.tokenizer.encode(text))

    def create_batch_prompt(self, files_data: List[Dict]) -> Tuple[str, int]:
        """Create a prompt for a batch of files and return the total token count."""
        files_content = []
        for file_data in files_data:
            file_section = f"""File: {file_data['filename']}

Changes:
```diff
{file_data['diff'] or 'No changes detected'}
```

Full content:
```
{file_data['content']}
```
"""
            files_content.append(file_section)

        batch_header = f"Review the following files that are part of the same PR. Consider their relationships and overall impact.\n\nNumber of files in this batch: {len(files_data)}\n\n"
        files_section = "\n\n" + "-" * 80 + "\n\n".join(files_content)
        review_instructions = """

Please provide a comprehensive review that includes:
1. Overall assessment of the changes in this batch
2. File-specific analysis for each file
3. Inter-file relationships and impacts
4. Common patterns or issues across files
5. Suggestions for improvement"""

        full_prompt = batch_header + files_section + review_instructions
        token_count = self.count_tokens(full_prompt)
        
        return full_prompt, token_count

    def estimate_tokens_for_file(self, file_data: Dict) -> int:
        """Estimate tokens for a single file's content and diff."""
        file_template = f"""File: {file_data['filename']}

Changes:
```diff
{file_data['diff'] or 'No changes detected'}
```

Full content:
```
{file_data['content']}
```"""
        return self.count_tokens(file_template)

    def calculate_batch_token_details(self, batch: List[Dict], system_prompt: str) -> Dict:
        """Calculate detailed token information for a batch."""
        prompt_tokens = self.count_tokens(system_prompt)
        file_tokens = sum(self.estimate_tokens_for_file(file) for file in batch)
        template_tokens = self.count_tokens("""Review the following files that are part of the same PR...
1. Overall assessment...
5. Suggestions for improvement""")
        
        return {
            'system_prompt_tokens': prompt_tokens,
            'file_tokens': file_tokens,
            'template_tokens': template_tokens,
            'total_tokens': prompt_tokens + file_tokens + template_tokens,
            'num_files': len(batch)
        }

    def create_batches(self, files_data: List[Dict], system_prompt: str) -> List[List[Dict]]:
        """Split files into batches that fit within token limits."""
        batches = []
        current_batch = []
        current_tokens = self.count_tokens(system_prompt)
        
        # Add template tokens
        template = """Review the following files that are part of the same PR...
1. Overall assessment...
5. Suggestions for improvement"""
        template_tokens = self.count_tokens(template)
        current_tokens += template_tokens
        
        for file_data in files_data:
            file_tokens = self.estimate_tokens_for_file(file_data)
            
            # Add padding for separators and safety margin
            total_tokens = current_tokens + file_tokens + 500
            
            if total_tokens > self.max_tokens and current_batch:
                batches.append(current_batch)
                current_batch = [file_data]
                current_tokens = self.count_tokens(system_prompt) + template_tokens + file_tokens
            else:
                current_batch.append(file_data)
                current_tokens = total_tokens
        
        if current_batch:
            batches.append(current_batch)
        
        # Calculate and print token details for each batch
        print("\nBatch Token Information:")
        for i, batch in enumerate(batches, 1):
            details = self.calculate_batch_token_details(batch, system_prompt)
            print(f"\nBatch {i}:")
            print(f"Number of files: {details['num_files']}")
            print(f"System prompt tokens: {details['system_prompt_tokens']}")
            print(f"File content tokens: {details['file_tokens']}")
            print(f"Template tokens: {details['template_tokens']}")
            print(f"Total tokens: {details['total_tokens']}")
            print(f"Percentage of max tokens: {(details['total_tokens'] / self.max_tokens * 100):.1f}%")
        
        return batches

    def review_batch(self, files_data: List[Dict], system_prompt: str) -> Generator:
        """Review a single batch of files."""
        prompt, token_count = self.create_batch_prompt(files_data)
        
        # Print token usage for this review
        print(f"\nCurrent batch token usage:")
        print(f"Prompt tokens: {token_count}")
        print(f"System prompt tokens: {self.count_tokens(system_prompt)}")
        total_tokens = token_count + self.count_tokens(system_prompt)
        print(f"Total tokens: {total_tokens}")
        print(f"Percentage of max tokens: {(total_tokens / self.max_tokens * 100):.1f}%")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                stream=True,
            )
            return response
        except Exception as e:
            print(f"Error processing content with Ollama: {str(e)}")
            return None