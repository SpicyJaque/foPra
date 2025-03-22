params = {
    'auth_key': api_key_deepl,
    'text': 'Hello, world!',
    'target_lang': 'DE'  # Translate to German
}

# Make the API request
response = requests.post(url, data=params)

# Check if the request was successful
if response.status_code == 200:
    result = response.json()
    translated_text = result['translations'][0]['text']
    print(f'Translated text: {translated_text}')
else:
    print(f'Error: {response.status_code}')
    print(response.text)