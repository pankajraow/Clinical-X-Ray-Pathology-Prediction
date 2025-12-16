import requests
import sys

def test_api():
    url = "http://127.0.0.1:5000/api/predict"
    # We need a sample image.
    # checking if we have one, if not download a dummy xray or create one
    image_path = "sample_xray.jpg"
    
    # Create valid dummy image if not exists
    import numpy as np
    from PIL import Image
    if not os.path.exists(image_path):
        # Create a random noise image (simulating xray)
        arr = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
        img = Image.fromarray(arr)
        img.save(image_path)
    
    print(f"Sending {image_path} to {url}...")
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(url, files=files)
        
        print(f"Status Code: {response.status_code}")
        print("Response JSON:")
        print(response.json())
        
        if response.status_code == 200:
            print("PASS: API responded correctly.")
        else:
            print("FAIL: API returned error.")
            
    except Exception as e:
        print(f"FAIL: Request failed - {e}")

if __name__ == "__main__":
    import os
    test_api()
