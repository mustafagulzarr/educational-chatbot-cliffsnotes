from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer, ListTrainer
import gensim
import warcio
from io import BytesIO

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
exit_conditions = (":q", " quit", " exit")
# Get a response to an input statement

print('The exit commands are: ' + exit_conditions[0] + exit_conditions[1] + exit_conditions[2])
while True:
    
    query = input("> ")
    if query in exit_conditions:
        break
    else:
        print(f"BOT: {chatbot.get_response(query)}")