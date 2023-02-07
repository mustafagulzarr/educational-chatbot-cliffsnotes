#this script uses the questgen api to render questions which will be used in training the chatbot
#first we will use summarize_bot to generate cliffnotes
#this will then be used as input for the questgen to generate question answer pairs and then be used to train the chatbot
#these Q/A pairs can be used to create a FAQ page later on

from pprint import pprint
import nltk
import json
nltk.download('stopwords')
from Questgen import main
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


def questionGeneratorWhole():
    f = open('payload.json')
    payload = json.load(f)

    payload = {
        "input_text": payload['text']
    }

    qg = main.QGen()
    output = qg.predict_shortq(payload)
    pprint(output)


def preprocess_text():

    f = open('payload.json')
    payload = json.load(f)


    text = payload['text']

    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Tokenize
    tokens = nltk.word_tokenize(text)
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Stem tokens
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    
    # Rejoin tokens
    text = ' '.join(tokens)
    print(text)
    return text

preprocess_text()

