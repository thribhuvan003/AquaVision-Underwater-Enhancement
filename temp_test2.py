import traceback
from app import app
import sys

app.testing = True
client = app.test_client()

with open('error.txt', 'w') as out:
    try:
        with open('static/uploads/test_blue.jpg', 'rb') as f:
            response = client.post('/prediction', data={'file': (f, 'test_blue.jpg')})
            out.write(f"Status Code: {response.status_code}\n")
            if response.status_code == 500:
                out.write(response.data.decode('utf-8'))
    except Exception as e:
        out.write(traceback.format_exc())
