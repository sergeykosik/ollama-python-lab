from openai import OpenAI
from typing import Generator
from config import DEFAULT_OLLAMA_MODEL

class OllamaClient:
    def __init__(self, model: str = None):
        if model is None:
            model = DEFAULT_OLLAMA_MODEL
        self.client = OpenAI(api_key="ollama", base_url="http://host.docker.internal:11434/v1/")
        self.model = model

    def review_code(self, filename: str, content: str, diff: str, system_prompt: str) -> Generator:
        prompt = f"""Review this file: {filename}

Full content:
```
{content}
```

Changes made in this PR:
```diff
{diff}
```
"""
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