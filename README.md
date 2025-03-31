## Setup
Create .env file based on env_sample.

### Updated requirements

Once the requirements.txt is updated run:

`pip install -r requirements.txt`

## How to run

Note: make sure the ollama is running.
Open powershell and run:

`ollama serve`

if the port is already taken, then ollama is running. Close it from Task popup (desktop).

### DeepSeek chat:

from terminal:

`python deepseek/deepseek.py`

### Code assistant:

from terminal:

`streamlit run deepseek/code-assistant.py`


### Document assistant:

from terminal:

`streamlit run deepseek/research-assistant.py`


### Multi assistant:

from terminal:

`streamlit run deepseek/multi-assistant.py`


### PR Review

from terminal:

`streamlit run pr-review/app.py`

Individual file processing

`streamlit run pr-review/app_individual.py`


### AI & LLM Engineering Course

https://github.com/pdichone/ai-llm-engineer-course 

RAG:
https://github.com/pdichone/vector-databases-course

https://docs.pinecone.io/guides/get-started/quickstart
https://docs.trychroma.com/docs/overview/getting-started