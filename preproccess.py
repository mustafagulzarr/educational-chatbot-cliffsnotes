#this script uses the questgen api to render questions which will be used in training the chatbot
#first we will use summarize_bot to generate cliffnotes
#this will then be used as input for the questgen to generate question answer pairs and then be used to train the chatbot
#these Q/A pairs can be used to create a FAQ page later on

from pprint import pprint
import nltk
import json
nltk.download('stopwords')
from Questgen import main

f = open('payload.json')
payload = json.load(f)

payload = {
    "input_text": payload['text']
}

qg = main.QGen()
output = qg.predict_shortq(payload)
pprint(output)