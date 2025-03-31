import ollama

client = ollama.Client(host='http://host.docker.internal:11434')
response = client.list()
print(response)