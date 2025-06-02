import base64
import requests
import json
import sys

def encode_image_to_base64(image_path):
    """Encode an image file to base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: Image file '{image_path}' not found.")
        return None
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

def send_ollama_chat_request(image_path, model="gemma3", prompt="Do you see a car(s) exists other than Parked cars on street? short answer less than 5 words"):
    """Send a chat request to Ollama API with an image."""
    # Encode the image
    base64_image = encode_image_to_base64(image_path)
    if not base64_image:
        return None

    # Prepare the request payload
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [base64_image]
            }
        ],
        "stream": False
    }

    # Send the request to Ollama API
    try:
        response = requests.post("http://localhost:11434/api/chat", json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to Ollama API at http://localhost:11434")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        return None
    except Exception as e:
        print(f"Error sending request: {e}")
        return None

def main():
    # Example usage
    if len(sys.argv) < 2:
        print("Usage: python gollama.py <image_path>")
        return
    image_path = sys.argv[1]
    response = send_ollama_chat_request(image_path) 
    
    if response:
        # Extract and print the model's response
        try:
            message_content = response.get("message", {}).get("content", "No content in response")
            print("Model response:", message_content)
        except Exception as e:
            print(f"Error parsing response: {e}")
    else:
        print("Failed to get a response from the Ollama API.")

if __name__ == "__main__":
    main()