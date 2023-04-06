# README

This README file provides information about the project educational-chatbot-cliffsnotes and how to use it. 

## Project Description

The project is an educational chatbot that takes input a text, and allows you to ask questions about its content

## Installation

To use the project, follow these steps:

1. Clone the files to your local machine

2. Navigate into the directory of the project:
    cd educational-chatbot-cliffsnotes

3. Set the `PYTHONPATH` environment variable to the current directory by running the following command:
    export PYTHONPATH=$(pwd)

4. Install the requirements:
    pip install --no-deps -r requirements.txt

5. Navigate to the `app/` directory:
    cd app/

6. Run the Flask application by executing the following command:
    flask run

7. Open a web browser and go to `http://localhost:5000/`.

8. Enter your text in the text area provided and wait for the application to process your input (this can take 10-15 minutes depending on the length of the input).

9. Once the input has been processed, ask your question to the chatbot by typing it into the chatbot interface (this can take 1-2 minutes depending on the complexity of the question and the length of the input).



