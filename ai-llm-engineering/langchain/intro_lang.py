from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Import the correct model class for Ollama
from langchain_ollama import ChatOllama

# Initialize the model properly
model = ChatOllama(model="llama3.2", base_url="http://host.docker.internal:11434")

# Simple chat example
messages = [
    SystemMessage(content="Translate the following from English into Italian"),
    HumanMessage(content="hi!"),
]

response = model.invoke(messages)
print(response)

# Using prompt templates
from langchain_core.prompts import ChatPromptTemplate

system_template = "Translate the following from English into {language}"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

prompt = prompt_template.invoke({"language": "Portuguese", "text": "Ola!"})
print(prompt)

response = model.invoke(prompt)
print(response.content)