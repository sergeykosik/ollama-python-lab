module_name = "ollama"  # Replace with the module you want to check

try:
    __import__(module_name)
    print(f"{module_name} is installed")
except ImportError:
    print(f"{module_name} is NOT installed")