import requests
import json

# Set your token and endpoint
server_token = 'aea3003157a777ce862ca7f42832f792'
# Endpoint URL
endpoint = "https://us-central1-lucid-box-387617.cloudfunctions.net/watermark-costar-detection-v1"

# Prepare the headers and data payload for the request
headers = {
    'Accept': 'application/json',
    'Authorization': 'test',
    'Content-Type': 'application/json'
}

data = {
    "instances": [
        {
            "image_url": "https://costar.brightspotcdn.com/dims4/default/f3d3a97/2147483647/strip/true/crop/5460x3640+4+0/resize/750x500!/quality/90/?url=http%3A%2F%2Fcostar-brightspot.s3.us-east-1.amazonaws.com%2F00%2F76%2F942ecd7f44d681f0c5fac42f3279%2Fprimaryphoto-8.jpg"
        }
    ]
}

# Make the POST request
response = requests.post(endpoint, headers=headers, data=json.dumps(data))

# Check the response
if response.status_code == 200:
    # Process the response data
    print(response.json())
else:
    print(f'Failed to retrieve data: {response.status_code}, {response.text}')
