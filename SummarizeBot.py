# import requests
# import json

# def summarizer():
#     url = "https://tldrthis.p.rapidapi.com/v1/model/extractive/summarize-text/"
#     f = open('payload.json')
#     payload = json.load(f)

#     headers = {
#         "content-type": "application/json",
#         "X-RapidAPI-Key": "3ffe54c5d1msh5a8dc30d86bec87p18fa4djsn7a1456607adc",
#         "X-RapidAPI-Host": "tldrthis.p.rapidapi.com"
#     }

#     response = requests.request("POST", url, json=payload, headers=headers)

#     print(response.json()['summary'])

import requests

def get_book_summary(isbn):
    base_url = "https://openlibrary.org/api/books?bibkeys=ISBN:"
    url = base_url + isbn + "&format=json&jscmd=data"
    # url = base_url + isbn 
    response = requests.get(url)
    # data = response.json()
    # try:
    #     summary = data["ISBN:" + isbn]["description"][0]["value"]
    # except KeyError:
    #     summary = "Summary not available."
    return response.json()["ISBN:" + isbn].keys()

isbn = input("Enter the ISBN of the book: ")
print("Book Summary:", get_book_summary(isbn))


