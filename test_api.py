import requests

url = 'http://127.0.0.1:8000/api/predict'
files = {'file': ('test_face.jpg', open('test_face.jpg', 'rb'), 'image/jpeg')}

try:
    response = requests.post(url, files=files)
    print("Status Code:", response.status_code)
    print("Response:", response.json())
except Exception as e:
    print("Error:", e)
