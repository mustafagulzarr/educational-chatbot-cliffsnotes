#this file is going to be used to webscrape cliffsnotes.com to fetch the summaries of the book that will be input
import requests
from bs4 import BeautifulSoup


def get_book_summary():
    book_name = input('enter book name ')
    # Format the book name for use in the URL
    formatted_book_name = book_name.replace(" ", "-")
    initial = formatted_book_name[0]
    print(formatted_book_name)

    # Make a request to the website
    response = requests.get(f"https://www.cliffsnotes.com/literature/{initial}/{formatted_book_name}/book-summary")

    print(response)
    # Parse the HTML content of the page
    soup = BeautifulSoup(response.content, "html.parser")

    # # Find the summary text
    summary_text = soup.findAll("p", class_="litNoteText").get_text()

    return summary_text

get_book_summary()