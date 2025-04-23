import requests
import json

# The UCI Wine dataset has 13 features
# This is a sample from the first class (0)
sample_features = {
    "features": [
        14.23, 1.71, 2.43, 15.6, 127, 2.8, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065
    ]
}

def test_prediction():
    url = "http://127.0.0.1:8000/predict"
    
    try:
        response = requests.post(url, json=sample_features)
        print("Request:")
        print(json.dumps(sample_features, indent=2))
        
        print("\nResponse:")
        print(json.dumps(response.json(), indent=2))
        
        if response.status_code == 200:
            print("\nTest successful!")
        else:
            print(f"\nTest failed with status code: {response.status_code}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_prediction()