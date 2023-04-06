from flask import Flask, render_template, redirect, url_for, request, session, jsonify
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer, ListTrainer
from Summarizers import SummarizeBart, SummarizeLSA
from QA_Generation import QuestgenDeberta, QuestgenSDK

bot = ChatBot(name='Educational Chatbot', read_only=True, logic_adapters=['chatterbot.logic.BestMatch'], storage_adapter="chatterbot.storage.SQLStorageAdapter")

# Create a new trainer
corpus_trainer = ChatterBotCorpusTrainer(bot)
list_trainer =  ListTrainer(bot)

# Now let's train the bot on the english corpus
corpus_trainer.train("chatterbot.corpus.english.greetings",
            "chatterbot.corpus.english.conversations")

app = Flask(__name__, template_folder="templates")
app.secret_key = "my_secret_key"

QuestgenSDK_flag = False

@app.route("/")
def home():
    session.clear()
    return render_template("index.html")

@app.route("/upload")
def upload():
    return redirect(url_for("upload_text"))

@app.route("/upload_text", methods=["GET", "POST"])
def upload_text():
    if request.method == "POST":
        text = request.form['text']
        file = open('../Data/input.txt', 'w')
        file.write(text)
        train()
        return redirect(url_for("chatbot"))
    
    return render_template("upload_text.html")

@app.route("/chatbot", methods=["GET", "POST"])
def chatbot():
    all_messages = []
    user_messages = session.get("user_messages", [])
    messages = session.get("messages", [])
    if request.method == "POST":
        # Get the user input
        user_input = request.form["user-input"]
        user_input = "You: " + user_input
        user_messages.append(user_input)
        session["user_messages"] = user_messages
        messages.append(("You", user_input))
        session["messages"] = messages

        if 'summar' in user_input or QuestgenSDK_flag==True:
            bot_response = bot.get_response(user_input)
        else:
            file = open('../Data/input.txt', 'r')
            text = file.read()
            bot_response = QuestgenDeberta.generate_answer(user_input, text)
            list_trainer.train(user_input,bot_response)
            all_messages.append(user_input)
        bot_response_text = "Bot: " + str(bot_response)
        user_messages.append(bot_response_text)
        session["user_messages"] = user_messages
        messages.append(("Bot", bot_response_text))
        session["messages"] = messages


    return render_template("chatbot.html", messages=messages, user_message_list=user_messages)


def train():
    file = open('../Data/input.txt', 'r')
    text = file.read()
    abstract_summary = SummarizeBart.generate_summary(text)
    extractive_summary = SummarizeLSA.summarize_text(text, 50)
    lst = [
        'Give me the abstractive summary',
        f'{abstract_summary}'
        'Give me the extractive summary',
        f'{extractive_summary}'
    ]
    list_trainer.train(lst)

def trainQuestgenSDK():
    lst = QuestgenSDK.questionGeneratorWhole()
    list_trainer.train(lst)
    QuestgenSDK_flag==True
    print(lst)


if __name__ == "__main__":
    app.run(debug=True)
