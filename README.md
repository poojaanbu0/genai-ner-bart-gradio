## Development of a Named Entity Recognition (NER) Prototype Using a Fine-Tuned BART Model and Gradio Framework

### AIM:
To design and develop a prototype application for Named Entity Recognition (NER) by leveraging a fine-tuned BART model and deploying the application using the Gradio framework for user interaction and evaluation.

### PROBLEM STATEMENT:
The challenge is to build an NER system capable of identifying named entities (e.g., people, organizations, locations) in text, using a pre-trained BART model fine-tuned for this task. The system should be interactive, allowing users to input text and see the recognized entities in real-time.

### DESIGN STEPS:

#### STEP 1: Fine-tune the BART model
Start by fine-tuning the BART model for NER tasks. This involves training the model on a labeled NER dataset with text data that contains named entities (e.g., people, places, organizations).

#### STEP 2: Create an API for NER model inference
Develop an API endpoint that takes input text and returns the recognized entities using the fine-tuned BART model.

#### STEP 3: Integrate the API with Gradio
Build a Gradio interface that takes user input, passes it to the NER model via the API, and displays the results as highlighted text with identified entities.

### PROGRAM:
```python
import os

# Set the Hugging Face API key directly
os.environ["HF_API_KEY"] = "NER_BART"  # Replace with your actual API key

# Retrieve and print the API key to ensure it's set correctly
hf_api_key = os.getenv("HF_API_KEY")
print(hf_api_key)  # Should print the API key

import os

# Set the API URL directly
os.environ["API_URL"] = "https://api-inference.huggingface.co/models/dslim/bert-base-NER"
# Retrieve API URL from environment variables
API_URL = os.getenv("API_URL")
print(API_URL)

import os
import io
from IPython.display import Image, display, HTML
from PIL import Image
import base64 
from dotenv import load_dotenv, find_dotenv
import requests
import json
import gradio as gr

# Load environment variables
_ = load_dotenv(find_dotenv())  # Read local .env file

# Retrieve the API key and URL from environment variables
hf_api_key = os.getenv("HF_API_KEY")
API_URL = os.getenv("API_URL")

# Verify that the environment variables are loaded
print(f"API Key: {hf_api_key}")
print(f"API URL: {API_URL}")

# Helper function to send API requests
def get_completion(inputs, parameters=None, ENDPOINT_URL=API_URL):
    headers = {
        "Authorization": f"Bearer {hf_api_key}",
        "Content-Type": "application/json"
    }
    data = {"inputs": inputs}
    if parameters is not None:
        data.update({"parameters": parameters})
    response = requests.post(ENDPOINT_URL, headers=headers, data=json.dumps(data))

    print(f"Response status code: {response.status_code}")
    if response.status_code == 200:
        print(f"Response content: {response.content.decode('utf-8')}")
    else:
        print("Error in API call")

    return json.loads(response.content.decode("utf-8"))


# Function to merge subword tokens (e.g., "San" and "Francisco" into "San Francisco")
def merge_tokens(tokens):
    merged_tokens = []
    for token in tokens:
        print(f"Token: {token}")
        if merged_tokens and token['entity'].startswith('I-') and merged_tokens[-1]['entity'].endswith(token['entity'][2:]):
            last_token = merged_tokens[-1]
            last_token['word'] += token['word'].replace('##', '')
            last_token['end'] = token['end']
            last_token['score'] = (last_token['score'] + token['score']) / 2
        else:
            merged_tokens.append(token)
    return merged_tokens


# NER function to process input and call the API
def ner(input):
    try:
        output = get_completion(input, parameters=None, ENDPOINT_URL=API_URL)
        merged_tokens = merge_tokens(output)
        return {"text": input, "entities": merged_tokens}
    except Exception as e:
        print(f"Error: {e}")
        return {"text": input, "entities": [{"word": "Error", "entity": str(e), "score": 1.0}]}


# Initialize Gradio interface
gr.close_all()
demo = gr.Interface(
    fn=ner,
    inputs=[gr.Textbox(label="Text to find entities", lines=2)],
    outputs=[gr.HighlightedText(label="Text with entities")],
    title="NER with dslim/bert-base-NER",
    description="Find entities using the `dslim/bert-base-NER` model under the hood!",
    allow_flagging="never",
    examples=[
        "My name is Andrew, I'm building DeeplearningAI and I live in California",
        "My name is Poli, I live in Vienna and work at HuggingFace"
    ]
)

# Launch the Gradio interface
demo.launch()
```

### OUTPUT:

![389215445-4b275e08-4e20-4c71-b1c9-d650f3186112](https://github.com/user-attachments/assets/8df72317-a0ec-4354-b943-4452533dd31f)

### RESULT:
Thus, developed an NER prototype application with user interaction and evaluation features, using a fine-tuned BART model deployed through the Gradio framework.
