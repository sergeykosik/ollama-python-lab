import json
from typing import Dict, List
import ollama
import sys
from dotenv import load_dotenv

load_dotenv()


def create_initial_messages() -> List[Dict[str, str]]:
    """Create the initial messages for the context memory."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
    ]


def chat(user_input: str, messages: List[Dict[str, str]], model_name: str) -> str:
    """Handle user input and generate responses using Ollama."""
    messages.append({"role": "user", "content": user_input})

    try:
        client = ollama.Client(host='http://host.docker.internal:11434')
        response = client.chat(model=model_name, messages=messages)
        assistant_response = response["message"]["content"]
        messages.append({"role": "assistant", "content": assistant_response})
        return assistant_response
    except Exception as e:
        return f"Error with Ollama API: {str(e)}"


def summarize_messages(messages: List[Dict[str, str]], model_name: str) -> List[Dict[str, str]]:
    """Summarize older messages using Ollama to save tokens."""
    
    recent_messages = messages[-5:]
    old_messages = messages[:-5]

    if not old_messages:
        return recent_messages

    old_text = "\n".join([f"{m['role']}: {m['content']}" for m in old_messages])

    try:
        client = ollama.Client(host='http://host.docker.internal:11434')
        response = client.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": "Summarize the following conversation while preserving key details."},
                {"role": "user", "content": old_text}
            ]
        )
        summary = response["message"]["content"]
    except Exception as e:
        summary = f"Error generating summary: {str(e)}"

    return [{"role": "system", "content": "Summary: " + summary}] + recent_messages


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
    model_name = "llama3.2:latest"  # Change this to another Ollama model if needed

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

        response = chat(user_input, messages, model_name)
        print(f"\nAssistant: {response}")

        if len(messages) > 10:
            messages = summarize_messages(messages, model_name)
            print("\n(Conversation automatically summarized)")


if __name__ == "__main__":
    main()
