import requests
endpoint = 'http://127.0.0.1:8000/predict'
text_data = "This is a sample text for prediction"  
res = requests.post(endpoint, json={'text':text_data})  # Use json here
print(res.status_code, res.text)
