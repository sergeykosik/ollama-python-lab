import json
from typing import Dict, List
import ollama
import sys
import os
from dotenv import load_dotenv

load_dotenv()


def create_initial_messages() -> List[Dict[str, str]]:
    """Create the initial messages for the context memory."""
    return [{"role": "system", "content": "You are a helpful assistant."}]


def chat(user_input: str, messages: List[Dict[str, str]], model_name: str) -> str:
    """Handle user input and generate responses using Ollama with streaming output."""
    messages.append({"role": "user", "content": user_input})

    try:
        client = ollama.Client(host='http://host.docker.internal:11434')
        response_stream = client.chat(model=model_name, messages=messages, stream=True)
        assistant_response = ""

        print("\nAssistant: ", end="", flush=True)
        for chunk in response_stream:
            text = chunk.get("message", {}).get("content", "")
            assistant_response += text
            print(text, end="", flush=True)

        print()  # Move to the next line after streaming
        messages.append({"role": "assistant", "content": assistant_response})

        return assistant_response
    except Exception as e:
        print(f"\nError with Ollama API: {str(e)}")
        return ""


def summarize_messages(messages: List[Dict[str, str]], model_name: str) -> List[Dict[str, str]]:
    """Summarize older messages using Ollama with streaming output."""
    
    recent_messages = messages[-5:]
    old_messages = messages[:-5]

    if not old_messages:
        return recent_messages

    old_text = "\n".join([f"{m['role']}: {m['content']}" for m in old_messages])

    try:
        print("\nSummarizing conversation...", end="", flush=True)
        client = ollama.Client(host='http://host.docker.internal:11434')
        response_stream = client.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": "Summarize the following conversation while preserving key details."},
                {"role": "user", "content": old_text}
            ],
            stream=True
        )

        summary = "Summary: "
        for chunk in response_stream:
            text = chunk.get("message", {}).get("content", "")
            summary += text
            print(text, end="", flush=True)

        print()  # Move to the next line after streaming
    except Exception as e:
        summary = f"Error generating summary: {str(e)}"

    return [{"role": "system", "content": summary}] + recent_messages


def save_conversation(messages: List[Dict[str, str]], filename: str = "conversation.json"):
    """Save conversation to a file."""
    with open(filename, "w") as f:
        json.dump(messages, f)


def load_conversation(filename: str = "conversation.json") -> List[Dict[str, str]]:
    """Load conversation from a file."""
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"No conversation file found at {filename}")
        return create_initial_messages()


def main():
    model_name = os.getenv("LLM_MODEL")  # Change this if needed

    messages = create_initial_messages()

    print(f"\nUsing Ollama model '{model_name}'. Type 'quit' to exit.")
    print("Available commands:")
    print("- 'save': Save conversation")
    print("- 'load': Load conversation")
    print("- 'summary': Summarize conversation")

    while True:
        user_input = input("\nYou: ")

        if user_input.lower() == "quit":
            break
        elif user_input.lower() == "save":
            save_conversation(messages)
            print("Conversation saved!")
            continue
        elif user_input.lower() == "load":
            messages = load_conversation()
            print("Conversation loaded!")
            continue
        elif user_input.lower() == "summary":
            messages = summarize_messages(messages, model_name)
            print("Conversation summarized!")
            continue

        chat(user_input, messages, model_name)

        if len(messages) > 10:
            messages = summarize_messages(messages, model_name)
            print("\n(Conversation automatically summarized)")


if __name__ == "__main__":
    main()
