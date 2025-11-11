DEFAULT_SYSTEM_PROMPT = """You are an expert code reviewer. Analyze the provided code and changes, focusing on:
1. Understanding of the changes made in this PR
2. Potential bugs or issues
3. Security concerns
4. Performance implications
5. Code style and best practices
6. Suggestions for improvement

Format your response in clear sections. Be specific and reference the actual code where relevant."""

DEFAULT_USER_PROMPT = """Review this file: {filename}

Full content:
```
{content}
```

Changes made in this PR:
```diff
{diff}
```"""

import os
from dotenv import load_dotenv

load_dotenv()

# Use model from .env or fallback to qwen3:8b
DEFAULT_OLLAMA_MODEL = os.getenv("LLM_MODEL", "qwen3:8b")

# Use max tokens from .env or fallback to 32000
DEFAULT_MAX_TOKENS = int(os.getenv("MAX_TOKENS", "32000"))