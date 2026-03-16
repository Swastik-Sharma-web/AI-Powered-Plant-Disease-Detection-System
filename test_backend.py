import requests

url = 'http://127.0.0.1:8000/api/predict'
# Download a temp image of a dog to test
response = requests.get('https://images.dog.ceo/breeds/retriever-golden/n02099601_3004.jpg')
with open('temp_dog.jpg', 'wb') as f:
    f.write(response.content)

files = {'file': ('temp_dog.jpg', open('temp_dog.jpg', 'rb'), 'image/jpeg')}

try:
    response = requests.post(url, files=files)
    print("Status Code:", response.status_code)
    print("Response:", response.json())
except Exception as e:
    print("Error:", e)
