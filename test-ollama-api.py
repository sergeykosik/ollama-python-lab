import requests
import json
import os
import platform

def get_ollama_url():
    """
    Determine the appropriate Ollama URL based on environment
    """
    # Check if we're running in a container
    in_container = os.path.exists('/.dockerenv')
    
    # Default URLs to try
    urls = [
        "http://host.docker.internal:11434",  # Docker Desktop on Windows/Mac
        "http://172.17.0.1:11434",           # Default Docker bridge network
        "http://localhost:11434",             # Direct localhost
        "http://host.containers.internal:11434"  # Newer Docker versions
    ]
    
    for url in urls:
        try:
            response = requests.get(f"{url}/api/tags", timeout=5)
            if response.status_code == 200:
                print(f"Successfully connected using: {url}")
                return url
        except requests.exceptions.RequestException:
            continue
    
    raise ConnectionError("Could not connect to Ollama API using any known URLs")

def test_ollama_connection():
    """
    Test connection to Ollama API with detailed debugging information
    """
    try:
        # Get system information
        system_info = {
            "platform": platform.platform(),
            "in_container": os.path.exists('/.dockerenv'),
            "python_version": platform.python_version()
        }
        
        # Try to get the working URL
        base_url = get_ollama_url()
        
        # Make the actual API call
        response = requests.get(f"{base_url}/api/tags")
        response.raise_for_status()
        
        return {
            "status": "success",
            "system_info": system_info,
            "api_url": base_url,
            "response": response.json()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "system_info": system_info,
            "error": str(e),
            "error_type": type(e).__name__
        }

def main():
    result = test_ollama_connection()
    print("\nConnection Test Results:")
    print(json.dumps(result, indent=2))
    
    if result["status"] == "success":
        print("\nAvailable Ollama models:")
        for model in result["response"]["models"]:
            print(f"- {model['name']}")
    else:
        print("\nTroubleshooting steps:")
        print("1. Verify Ollama is running on the host machine")
        print("2. Check if the dev container has network access to the host")
        print("3. Try adding 'extra_hosts: host.docker.internal:host-gateway' to your docker-compose.yml")
        print("4. Ensure no firewalls are blocking the connection")

if __name__ == "__main__":
    main()