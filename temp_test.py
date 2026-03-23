import requests
import os

url = 'http://127.0.0.1:5000/prediction'
test_file = 'static/uploads/test_blue.jpg'
if not os.path.exists(test_file):
    print("Test file not found")
else:
    with open(test_file, 'rb') as f:
        files = {'file': ('test_blue.jpg', f, 'image/jpeg')}
        response = requests.post(url, files=files)
        print("Status code:", response.status_code)
        
        if 'error-card' in response.text:
            idx = response.text.find('<div class="error-card')
            print("ERROR found in HTML:")
            print(response.text[idx:idx+200])
        else:
            idx = response.text.find('<img class="img-enhanced"')
            print("Success HTML snippet:")
            print(response.text[idx:idx+300])
