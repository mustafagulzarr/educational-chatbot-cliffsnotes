# Import necessary modules
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import heapq

# Preprocesses text by converting to lowercase, tokenizing, removing stop words and lemmatizing words
def preprocess_text(text):
    # Set stop words and lemmatizer
    stop_words = set(stopwords.words('english'))
    wordnet_lemmatizer = WordNetLemmatizer()

    # Tokenize words and remove stop words and non-alphanumeric characters
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalnum() and word not in stop_words]
    
    # Lemmatize words and join them into a string
    words = [wordnet_lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# Summarizes text by extracting the most important sentences
def summarize_text(text, num_sentences):
    # Tokenize text into sentences and preprocess each sentence
    sentences = sent_tokenize(text)
    preprocessed_sentences = [preprocess_text(sentence) for sentence in sentences]

    # Create a document-term matrix
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(preprocessed_sentences)

    # Perform dimensionality reduction using TruncatedSVD
    svd = TruncatedSVD(n_components=1)
    X_svd = svd.fit_transform(X)

    # Compute sentence scores based on singular values
    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        sentence_scores[sentence] = X_svd[i][0]

    # Extract the top 'num_sentences' sentences
    summarized_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    summary = ' '.join(summarized_sentences)

    return summary
