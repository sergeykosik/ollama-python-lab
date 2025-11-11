from typing import Annotated, TypedDict

# Import the Ollama chat model from langchain-community
from langchain_community.chat_models import ChatOllama

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

# --- Ollama Model Setup ---
# Make sure Ollama is running and you have pulled the model you want to use.
# Example: `ollama pull llama3` in your terminal.
# By default, ChatOllama connects to http://localhost:11434.
llm_name = "llama3.2:latest"  # Or use "mistral", "phi3", etc.
model = ChatOllama(model=llm_name, base_url='http://host.docker.internal:11434')


# STEP 1: Define the state for our graph
class State(TypedDict):
    """
    The state of our graph is a list of messages.
    The `add_messages` function in the annotation defines how this state
    is updated. It appends messages to the list, rather than overwriting them.
    """
    messages: Annotated[list, add_messages]


# STEP 2: Define the nodes for our graph
def bot(state: State):
    """
    This is the primary node of our graph. It takes the current state
    (the list of messages) and invokes the Ollama model to get a response.
    The response is then added to the state.
    """
    print("Node 'bot': Getting response from Ollama...")
    # The `state` parameter is a dictionary with a "messages" key
    # state["messages"] is the list of messages
    return {"messages": [model.invoke(state["messages"])]}


# STEP 3: Build the graph
graph_builder = StateGraph(State)

# Add our "bot" node to the graph
graph_builder.add_node("bot", bot)

# Set the entry point and finish point of the graph
# For this simple chatbot, the "bot" node is both the beginning and the end.
graph_builder.set_entry_point("bot")
graph_builder.set_finish_point("bot")

# STEP 4: Compile the graph into a runnable object
graph = graph_builder.compile()


# STEP 5: Run the chatbot
print("Chatbot is ready! Type 'quit', 'exit', or 'q' to end the conversation.")
while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    # Use the graph's stream method to get real-time outputs
    for event in graph.stream({"messages": ("user", user_input)}):
        # The event dictionary contains the output of the node that just ran
        for value in event.values():
            # The "messages" key contains a list of AIMessage objects
            # We print the content of the last message
            print("Assistant:", value["messages"][-1].content)