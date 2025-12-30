import pickle
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer


import ollama

# Load SBERT encoder model
print("Loading SentenceTransformer encoder...")
encoder_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load classification model from pickle
classification_model_path = Path('./pickle_files/sbert_classification_model.pkl')
if not classification_model_path.exists():
    raise FileNotFoundError(f"Classification model not found: {classification_model_path}")

print("Loading classification model...")
with open(classification_model_path, 'rb') as f:
    classification_model = pickle.load(f)

# Load intent encodings (class ID -> intent name mapping)
encodings_path = Path('./utils/result_encodings.json')
if not encodings_path.exists():
    raise FileNotFoundError(f"Encodings file not found: {encodings_path}")

print("Loading intent encodings...")
with open(encodings_path, 'r', encoding='utf-8') as f:
    intent_mappings = json.load(f)

# Test string
test_string = input('Enter your instruction: \n')
print(f"\nTest input: '{test_string}'")

# Encode text using SBERT
print("Encoding text with SentenceTransformer...")
text_encoding = encoder_model.encode(test_string, convert_to_numpy=True)

# Reshape to 2D array for classifier (expects shape: (n_samples, n_features))
text_encoding_2d = np.asarray(text_encoding).reshape(1, -1)

# Predict class
print("Predicting intent class...")
predicted_class = classification_model.predict(text_encoding_2d)[0]
print(f"Predicted class: {predicted_class}")

# Map class to intent name
predicted_intent = intent_mappings.get(str(predicted_class), f"Unknown class {predicted_class}")
print(f"Predicted intent: {predicted_intent}")

response = ollama.chat(
    model='gemma3:4b',
    messages=[
        {'role': 'user', 'content': f'''You are an intelligent IDE assistant.

                                    You will receive:
                                    - A user instruction (natural language)
                                    - An intent label provided by the system

                                    Rules:
                                    - The intent is for internal reasoning only.
                                    - NEVER mention, explain, or reveal the intent in your response.
                                    - Respond only to the user instruction.
                                    - Assume the instruction is valid and permitted.
                                    - Reply in a concise, professional, IDE-style tone.
                                    - Acknowledge the instruction and confirm the action clearly.
                                    - Do not ask follow-up questions unless absolutely necessary.
                                    - Do not include explanations unless the instruction explicitly asks for them.

                                    Your goal is to confirm and follow the instruction as an IDE assistant would.
                                    Instruction:
                                    {test_string}

                                    Intent (for internal analysis only):
                                    {predicted_intent}

                                '''}
    ]
)

print(f'The response is : {response.message.content}')


