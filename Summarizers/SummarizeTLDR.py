import json
import requests

def summarizer_tldr():
    url = "https://tldrthis.p.rapidapi.com/v1/model/extractive/summarize-text/"
    f = open('payload.json')
    payload = json.load(f)
    text = open('Data/input.txt', 'r')
    contents = text.read()
    payload['text'] = contents

    headers = {
        "content-type": "application/json",
        "X-RapidAPI-Key": "3ffe54c5d1msh5a8dc30d86bec87p18fa4djsn7a1456607adc",
        "X-RapidAPI-Host": "tldrthis.p.rapidapi.com"
    }

    response = requests.request("POST", url, json=payload, headers=headers)
    result = response.json()['summary']
    extract = ''.join(result)


    return extract