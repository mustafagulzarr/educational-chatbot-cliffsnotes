from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer, ListTrainer, UbuntuCorpusTrainer
from io import BytesIO
import preproccesser

# Create a new chatbot
chatbot = ChatBot(name='Educational Chatbot', read_only=True, logic_adapters=['chatterbot.logic.BestMatch'])

# Create a new trainer
corpus_trainer = ChatterBotCorpusTrainer(chatbot)
ubuntu_trainer = UbuntuCorpusTrainer(chatbot)
ubuntu_trainer.train()

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

whole_text = preproccesser.questionGeneratorWhole('payload.json')
list_trainer.train(whole_text)

preproccess_text = preproccesser.preprocess_text('payload.json')
# corpus_trainer.train(preproccess_text)
################################################################################################

exit_conditions = (":q", " quit", " exit")
# Get a response to an input statement

print('The exit commands are: ' + exit_conditions[0] + exit_conditions[1] + exit_conditions[2])
while True:
    
    query = input("> ")
    if query in exit_conditions:
        break
    else:
        print(f"BOT: {chatbot.get_response(query)}")