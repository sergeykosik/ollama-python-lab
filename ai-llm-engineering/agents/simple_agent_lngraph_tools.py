import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv

# --- OLLAMA MODEL IMPORT ---
from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables from .env file (for the TAVILY_API_KEY)
load_dotenv()

# --- API KEY and MODEL SETUP ---
# We only need the Tavily key now. The OpenAI key and client are removed.
tavily_api_key = os.getenv("TAVILY_API_KEY")
if not tavily_api_key:
    raise ValueError("TAVILY_API_KEY not found in .env file. Please add it.")

# Instantiate the Ollama model.
# Make sure the model name matches one you have pulled with `ollama pull <model_name>`
# Llama 3 and Phi-3 are good choices as they are trained for tool/function calling.
llm_name = "llama3.2:latest"  # or "phi3"
model = ChatOllama(model=llm_name, base_url='http://host.docker.internal:11434')

# The rest of the setup is largely the same!

# STEP 1: Define the agent's state
class State(TypedDict):
    # The `add_messages` function appends messages to the list, rather than overwriting them
    messages: Annotated[list, add_messages]


# STEP 2: Define the tools and nodes
# Create the Tavily search tool
tool = TavilySearchResults(max_results=2, api_key=tavily_api_key)
tools = [tool]

# Bind the tools to the Ollama model. This allows the model to "see" the tools
# and decide when to use them. This works because ChatOllama supports the
# standard LangChain tool-calling interface.
model_with_tools = model.bind_tools(tools)

# The `bot` node calls the model with the current state.
# The model can either respond directly or generate a tool_call.
def bot(state: State):
    print("---CALLING MODEL---")
    messages = state["messages"]
    print("Messages so far:", [msg.pretty_repr() for msg in messages])
    response = model_with_tools.invoke(messages)
    # We return a list, because we want to add it to the state
    return {"messages": [response]}

# The `tool_node` executes the tools that the model has decided to call.
tool_node = ToolNode(tools=tools)

# STEP 3: Build the graph
graph_builder = StateGraph(State)

# Define the nodes
graph_builder.add_node("bot", bot)
graph_builder.add_node("tools", tool_node)

# Define the edges
graph_builder.set_entry_point("bot")

# The `tools_condition` function checks the last message in the state.
# If it contains `tool_calls`, it routes to the `tools` node.
# Otherwise, it ends the conversation turn.
graph_builder.add_conditional_edges(
    "bot",
    tools_condition,
    # The "tools" path routes to the `tools` node.
    # The "__end__" path signifies the end of the graph run.
    {"tools": "tools", "__end__": "__end__"}
)

# Any time the `tools` node is called, it should route back to the `bot`
# node to let the bot process the tool's output.
graph_builder.add_edge("tools", "bot")

# STEP 4: Add memory and compile the graph
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# --- DEMONSTRATION OF MEMORY AND TOOL USE ---
# We'll use a unique thread_id to keep the conversation state separate.
config = {"configurable": {"thread_id": "my-thread-1"}}

# --- First interaction: Introduce name and ask a question that requires a tool ---
user_input_1 = "Hi there! My name is Bond, and I live in the same city as the Eiffel Tower. What is the weather like there?"
print(f"\n--- User Input 1 ---\n{user_input_1}\n")

# The config is the **second positional argument** to stream() or invoke()!
events = graph.stream(
    {"messages": [("user", user_input_1)]}, config, stream_mode="values"
)

for event in events:
    # `event` is the output of a node, containing a dictionary with the key "messages"
    event["messages"][-1].pretty_print()

# --- Second interaction: Ask a follow-up question to test memory ---
user_input_2 = "Great, thanks! By the way, do you remember my name?"
print(f"\n--- User Input 2 ---\n{user_input_2}\n")

events = graph.stream(
    {"messages": [("user", user_input_2)]}, config, stream_mode="values"
)

for event in events:
    event["messages"][-1].pretty_print()

# --- You can inspect the final state of the graph for this thread ---
print("\n--- Final State ---")
snapshot = graph.get_state(config)
snapshot.values["messages"][-1].pretty_print()