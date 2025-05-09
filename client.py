import cv2
import requests
import base64
import time
from modules.sign_processor import SignProcessor

def encode_image_to_base64(frame):
    """Convert image to base64 format"""
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

def main():
    # Setup sign language processor
    processor = SignProcessor()
    
    # Setup webcam
    cap = cv2.VideoCapture(0)
    
    # API configuration
    API_URL = "https://7ff0-45-244-172-52.ngrok-free.app/ws"
    
    try:
        while True:
            # Get webcam frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Local processing
            local_response = processor.process_frame(frame)
            
            # Convert frame to base64
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
            
            # Rate limiting
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nStopping the program...")
    finally:
        # Cleanup resources
        cap.release()
        processor.close()

if __name__ == "__main__":
    main() 