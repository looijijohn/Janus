import cv2
import base64
import requests
import json
import time

def encode_frame_to_base64(frame):
    """Encode an OpenCV frame to base64 string."""
    try:
        _, buffer = cv2.imencode('.png', frame)
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        print(f"Error encoding frame: {e}")
        return None

def send_ollama_chat_request(base64_image, model="gemma3:12b", prompt="Do you see a car(s) exists other than parked cars on street? Short answer less than 5 words"):
    """Send a chat request to Ollama API with a frame."""
    if not base64_image:
        return None
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
    try:
        response = requests.post("http://localhost:11434/api/chat", json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error sending request: {e}")
        return None

def main():
    # Path to the video file
    video_path = "v/Motorway.mp4"
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video or failed to read frame.")
                break

            # Encode frame to base64
            base64_image = encode_frame_to_base64(frame)
            if not base64_image:
                continue

            # Send request to Ollama API
            response = send_ollama_chat_request(base64_image)
            if response:
                try:
                    message_content = response.get("message", {}).get("content", "No response")
                    print("Model response:", message_content)
                except Exception as e:
                    print(f"Error parsing response: {e}")

            # Display the frame
            cv2.imshow('Video Stream', frame)

            # Process every 2 seconds to avoid overwhelming the API
            time.sleep(2)

            # Break on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()