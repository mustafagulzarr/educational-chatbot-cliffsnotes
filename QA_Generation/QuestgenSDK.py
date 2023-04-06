import nltk
nltk.download('stopwords')
from Questgen import main

def questionGeneratorWhole(contents):
    output=[]
    payload = {
    "input_text": contents
    }
    qg = main.QGen()
    output.append(qg.predict_shortq(payload, 10))
    result = []
    for i in range(len(output[0]['questions'])):
        result.append(output[0]['questions'][i]['Question'])
        result.append(output[0]['questions'][i]['Answer'])
    return result


