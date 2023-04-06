from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import heapq
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
from rouge_score import rouge_scorer

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    wordnet_lemmatizer = WordNetLemmatizer()

    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalnum() and word not in stop_words]
    words = [wordnet_lemmatizer.lemmatize(word) for word in words]

    return ' '.join(words)

def summarize_text(text, num_sentences):
    sentences = sent_tokenize(text)
    preprocessed_sentences = [preprocess_text(sentence) for sentence in sentences]

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(preprocessed_sentences)

    svd = TruncatedSVD(n_components=1)
    X_svd = svd.fit_transform(X)

    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        sentence_scores[sentence] = X_svd[i][0]

    summarized_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    summary = ' '.join(summarized_sentences)


    return summary



# # Replace the text below with the content of your book chapter.
# book_chapter = open('Data/input.txt','r').read()

# # Set the number of sentences you want in the summary.
# num_summary_sentences = 200

# summary = summarize_text(book_chapter, num_summary_sentences)
# print(summary)



