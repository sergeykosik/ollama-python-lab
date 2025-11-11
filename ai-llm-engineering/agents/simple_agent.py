import re
from ollama import Client

# Make sure you have an Ollama model installed, e.g., by running `ollama pull llama3`
# Update this to the model you want to use
llm_name = "llama3.2:latest"

# Initialize the Ollama client
client = Client(host='http://host.docker.internal:11434')

# The initial test call (now updated for Ollama)
# response = client.chat(
#     model=llm_name,
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "who is Nelson Mandela?"},
#     ],
# )
# print(response['message']['content'])
# exit()


# Create our own simple agent
class Agent:
    def __init__(self, system=""):
        self.system = system
        self.messages = []
        if system:
            self.messages.append({"role": "system", "content": system})

    def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self):
        response = client.chat(
            model=llm_name,
            messages=self.messages,
            options={'temperature': 0.0}
        )
        return response['message']['content']


# --- FIX 1: IMPROVE THE PROMPT ---
# The example now explicitly shows the LLM that it should only use the
# numerical values in the `calculate` action, and then format the final answer.
# This is the most critical fix.
prompt = """
You run in a loop of Thought, Action, PAUSE, Observation.
At theend of the loop you output a final Answer.
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.

Your available actions are:

calculate:
e.g. calculate: 4 * 7 / 3
Runs a calculation and returns the number. This uses Python, so use '*' for multiplication.
IMPORTANT: Only use the numerical part of the values you have. Do not include text or units like '× 10^24 kg'.

planet_mass:
e.g. planet_mass: Earth
returns the mass of a planet in the solar system

Example session:

Question: What is the combined mass of Earth and Mars?
Thought: I need to find the mass of Earth and Mars individually and then add them.
Action: planet_mass: Earth
PAUSE

You will be called again with this:

Observation: Earth has a mass of 5.972 × 10^24 kg

You then output:

Thought: I have the mass of Earth (5.972). Now I need the mass of Mars.
Action: planet_mass: Mars
PAUSE

You will be called again with this:

Observation: Mars has a mass of 0.64171 × 10^24 kg

You then output:

Thought: I have the mass of Earth (5.972) and Mars (0.64171). I need to add these two numbers.
Action: calculate: 5.972 + 0.64171
PAUSE

You will be called again with this:

Observation: 6.61371

You then output:

Answer: The combined mass of Earth and Mars is 6.61371 × 10^24 kg.
""".strip()


# --- FIX 2: MAKE THE CALCULATE TOOL MORE ROBUST ---
# This function will now replace common non-python math symbols before evaluating.
def calculate(what):
    """
    A safer calculate function that replaces common symbols and evaluates the expression.
    """
    # Replace user-friendly multiplication symbol with Python's asterisk
    what = what.replace("×", "*")
    # Replace caret for exponent with Python's double-asterisk (less common but good practice)
    what = what.replace("^", "**")
    # Remove any lingering text (though the prompt should prevent this)
    what = re.sub(r'[a-zA-Z\s]', '', what)
    
    try:
        # Use a restricted eval for safety in a real application
        # For this example, the standard eval is fine.
        return eval(what)
    except Exception as e:
        return f"Error evaluating expression: {e}"


def planet_mass(name):
    name = name.strip().capitalize()
    masses = {
        "Mercury": 0.33011,
        "Venus": 4.8675,
        "Earth": 5.972,
        "Mars": 0.64171,
        "Jupiter": 1898.19,
        "Saturn": 568.34,
        "Uranus": 86.813,
        "Neptune": 102.413,
    }
    if name in masses:
        return f"{name} has a mass of {masses[name]} × 10^24 kg"
    else:
        return f"Unknown planet: {name}"


known_actions = {"calculate": calculate, "planet_mass": planet_mass}

# Regex to find "Action: tool: input"
action_re = re.compile(r"^Action: (\w+): (.*)$", re.MULTILINE)

# --- FIX 3: REFINE THE INTERACTIVE LOOP LOGIC ---
def query_interactive():
    bot = Agent(prompt)
    next_prompt = None

    while True:
        try:
            if next_prompt is None:
                next_prompt = input("You: ")
            if next_prompt.lower() in ["exit", "quit"]:
                print("Exiting...")
                break

            result = bot(next_prompt)
            print(f"\nBot:\n{result}")

            # First, check if the bot has given a final answer.
            if "Answer:" in result:
                next_prompt = None # Reset for the next user question
                continue

            # If no answer, look for an action to perform.
            actions = action_re.findall(result)
            if actions:
                action, action_input = actions[0]
                if action in known_actions:
                    print(f"--- Running Action: {action} {action_input} ---")
                    observation = known_actions[action](action_input)
                    print(f"Observation: {observation}")
                    next_prompt = f"Observation: {observation}"
                else:
                    print(f"--- Error: Unknown action: {action} ---")
                    next_prompt = None # Reset on error
            else:
                # Bot didn't provide an Answer or an Action. It might be stuck.
                print("--- Bot did not provide a valid next step. Please rephrase or guide it. ---")
                next_prompt = None # Reset for new user input

        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

if __name__ == "__main__":
    print("Starting interactive agent. Type 'exit' or 'quit' to end.")
    query_interactive()