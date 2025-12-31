import json
import pickle
from pathlib import Path
import numpy as np

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import ollama

def load_models():
    try:
        from sentence_transformers import SentenceTransformer
        sbert_encoding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print(f"\nLoaded SBERT ENCODING model ...\n")
    except Exception as e:
        print(f"\nError loading SBERT ENCODING model: {e}\n")
        sbert_encoding_model = None
        
    try:
        balanced_sbert_data_classification_model = pickle.load(open('./pickle_files/balanced_sbert_data_classification_model.pkl', 'rb'))
        balanced_word2vec_data_classification_model = pickle.load(open('./pickle_files/balanced_word2vec_data_classification_model.pkl', 'rb'))
        sbert_data_classification_model = pickle.load(open('./pickle_files/sbert_data_classification_model.pkl', 'rb'))
        word2vec_data_classification_model = pickle.load(open('./pickle_files/word2vec_data_classification_model.pkl', 'rb'))
        word2vec_model = pickle.load(open('./pickle_files/word2vec_model.pkl', 'rb'))

        # print models that are empty
        if balanced_sbert_data_classification_model is None:
            print(f"\nBalanced SBERT Classification Model is empty.\n")
        if balanced_word2vec_data_classification_model is None:
            print(f"\nBalanced Word2Vec Classification Model is empty.\n")
        if sbert_data_classification_model is None:
            print(f"\nSBERT Classification Model is empty.\n")
        if word2vec_data_classification_model is None:
            print(f"\nWord2Vec Classification Model is empty.\n")
        if word2vec_model is None:
            print(f"\nWord2Vec Model is empty.\n")

        print(f"\nLoaded classification models ...\n")
        return (
            sbert_encoding_model,
            balanced_sbert_data_classification_model,
            balanced_word2vec_data_classification_model,
            sbert_data_classification_model,
            word2vec_data_classification_model,
            word2vec_model
            )
    except Exception as e:
        print(f"\nError loading classification models: {e}\n")
    
        

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens
    
def load_intents(file_path):
    try:
        with open(file_path, 'r') as file:
            intents = json.load(file)
            if intents:
                print(f"\nloaded the intent...\n")
                return intents
            else:
                print(f"\nNo intents found in the file.\n")
                return None
        file.close()
    except Exception as e:
        print(f"\nError loading intents from {file_path}: {e}\n")
        return None
    
def convert_to_word2vec_vector(processed_text=None,word2vec_model=None):
    vecs = np.array([word2vec_model.wv[w] for w in processed_text if w in word2vec_model.wv])
    if len(vecs) == 0:
        input_vector = np.zeros(word2vec_model.vector_size * 3)
    else:
        input_vector = np.concatenate([
            vecs.mean(axis=0),
            vecs.max(axis=0),
            vecs.min(axis=0)
        ])
    return input_vector

def convert_to_sbert_vector(text,model):
    return model.encode([text])[0]

def ollama_response(string,intent):
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
                    {string}

                    Intent (for internal analysis only):
                    {intent}
                '''}
            ]
    )
    return response.message.content
 
    
def main():

    models = load_models()
    
    sbert_encoding_model = models[0]
    balanced_sbert_data_classification_model = models[1]
    balanced_word2vec_data_classification_model = models[2]
    sbert_data_classification_model = models[3]
    word2vec_data_classification_model = models[4]
    word2vec_model = models[5]    
    
    BASE_DIR = Path(__file__).resolve().parent.parent
    result_encoding_file = BASE_DIR / "utils" / "result_encodings.json"
    intent = load_intents(result_encoding_file)
    
    input_text = input("Enter your text: \n")
    
    processed_text = preprocess_text(input_text)
    
    print(f"\nProcessed Text: {processed_text}\n")
    
    word2vec_vector = convert_to_word2vec_vector(processed_text, word2vec_model)
    sbert_vector = convert_to_sbert_vector(input_text, sbert_encoding_model)
    
    balanced_word2vec_pred = balanced_word2vec_data_classification_model.predict([word2vec_vector])[0]
    balanced_sbert_pred = balanced_sbert_data_classification_model.predict([sbert_vector])[0]
    word2vec_pred = word2vec_data_classification_model.predict([word2vec_vector])[0]
    sbert_pred = sbert_data_classification_model.predict([sbert_vector])[0]
    
    print(f'\nBalanced Word2Vec Prediction: {intent[str(balanced_word2vec_pred)]}\n')
    print(f'\nBalanced SBERT Prediction: {intent[str(balanced_sbert_pred)]}\n')
    print(f'\nWord2Vec Prediction: {intent[str(word2vec_pred)]}\n')
    print(f'\nSBERT Prediction: {intent[str(sbert_pred)]}\n')
    
    # since sbert predictions are better, we use sbert predictions
    predicted_intent = intent[str(balanced_sbert_pred)]
    print(ollama_response(input_text,predicted_intent))

if __name__ == "__main__":
    main()