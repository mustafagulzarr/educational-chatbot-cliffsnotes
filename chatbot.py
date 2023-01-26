from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer, ListTrainer
# from pprint import pprint
# import nltk
# import json
# nltk.download('stopwords')
# from Questgen import main

# f = open('payload.json')
# payload = json.load(f)

# payload = {
#     "input_text": payload['text']
# }

# qg = main.QGen()
# output = qg.predict_shortq(payload)
# pprint(output)

# Create a new chatbot
chatbot = ChatBot("Educational Chatbot")

# Create a new trainer
corpus_trainer = ChatterBotCorpusTrainer(chatbot)

# Now let's train the bot on the english corpus
corpus_trainer.train("chatterbot.corpus.english.greetings",
              "chatterbot.corpus.english",
              "chatterbot.corpus.english.conversations")

list_trainer = ListTrainer(chatbot)
list_trainer.train([
    "What is the main theme of the book?", "The main theme of the book is X",
    "Who is the author of the book?", "The author of the book is Y",
    "When was the book published?", "The book was published in Z",
    "Hello, how are you today?", "I am really W, how about K"
    ])
# Get a response to an input statement
response = chatbot.get_response("Hello, how are you today?")
print(response)