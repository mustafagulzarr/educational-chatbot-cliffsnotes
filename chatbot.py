from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer

# Create a new chatbot
chatbot = ChatBot("Educational Chatbot")

# Create a new trainer
trainer = ChatterBotCorpusTrainer(chatbot)

# Now let's train the bot on the english corpus
trainer.train("chatterbot.corpus.english.greetings",
              "chatterbot.corpus.english.conversations")

# Get a response to an input statement
response = chatbot.get_response("Hello, how are you today?")
print(response)