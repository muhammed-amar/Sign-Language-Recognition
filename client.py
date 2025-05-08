import cv2
import requests
import base64
import time
from modules.sign_processor import SignProcessor

def encode_image_to_base64(frame):
    """Convert image to base64 string"""
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

def main():
    # Initialize the sign processor
    processor = SignProcessor()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # API endpoint
    API_URL = "https://7ff0-45-244-172-52.ngrok-free.app/ws"
    
    try:
        while True:
            # Capture frame from webcam
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Process frame locally
            local_response = processor.process_frame(frame)
            
            # Prepare image for API
            image_base64 = encode_image_to_base64(frame)
            
            # Send to API
            try:
                response = requests.post(
                    API_URL,
                    json={
                        "image": image_base64
                    }
                )
                
                if response.status_code == 200:
                    api_response = response.json()
                    print("\nLocal Processing:")
                    print(f"Text: {local_response['current_text']}")
                    print(f"Action: {local_response['action_feedback']}")
                    print(f"FPS: {local_response['fps']:.1f}")
                    print("\nAPI Response:")
                    print(f"Text: {api_response['current_text']}")
                    print(f"Action: {api_response['action_feedback']}")
                    print(f"FPS: {api_response['fps']:.1f}")
                else:
                    print(f"API Error: {response.status_code}")
                    
            except Exception as e:
                print(f"Error sending to API: {str(e)}")
            
            # Add a small delay to prevent overwhelming the system
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nStopping the program...")
    finally:
        # Clean up
        cap.release()
        processor.close()

if __name__ == "__main__":
    main() 