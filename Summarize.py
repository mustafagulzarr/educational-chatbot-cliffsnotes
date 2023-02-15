import requests
import json
import openai
import PyPDF2
def summarizer_tldr():
    url = "https://tldrthis.p.rapidapi.com/v1/model/extractive/summarize-text/"
    f = open('payload.json')
    payload = json.load(f)

    headers = {
        "content-type": "application/json",
        "X-RapidAPI-Key": "3ffe54c5d1msh5a8dc30d86bec87p18fa4djsn7a1456607adc",
        "X-RapidAPI-Host": "tldrthis.p.rapidapi.com"
    }

    response = requests.request("POST", url, json=payload, headers=headers)

    print(response.json()['summary'])

def summarizer_openai():
    openai.api_key = "your_api_key"

    # Load the PDF file and extract the text
    pdf_path = "path/to/your/book.pdf"
    text = ""
    with open(pdf_path, "rb") as f:
        pdf_reader = PyPDF2.PdfReader(f)
        for page in pdf_reader.pages:
            text += page.extract_text()

    # Generate the summary using GPT-3
    model = "text-davinci-002"
    prompt = "Please summarize the following book: " + text
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=1024,
        temperature=0.5,
        n=1,
        stop=None,
    )

    # Extract the summary from the GPT-3 response
    summary = response.choices[0].text

    # Print the summary
    print(summary)




