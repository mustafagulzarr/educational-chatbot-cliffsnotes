import re
import numpy as np
from collections import defaultdict
from math import log
from operator import itemgetter

def preprocess_book(book_text):
    # Split book text into individual sentences
    sentences = re.split(r'[.?!]+', book_text)
    
    # Remove unwanted characters and split into words
    processed_sentences = []
    for sentence in sentences:
        sentence = re.sub(r'[^a-zA-Z\s]', '', sentence)
        sentence = sentence.lower().split()
        processed_sentences.append(sentence)
        
    return processed_sentences

def calculate_tf_idf(sentences):
    # Calculate term frequency for each sentence
    term_frequency = [defaultdict(int) for _ in sentences]
    for i, sentence in enumerate(sentences):
        for word in sentence:
            term_frequency[i][word] += 1
    
    # Calculate inverse document frequency for each word
    inverse_document_frequency = defaultdict(int)
    for sentence in sentences:
        for word in set(sentence):
            inverse_document_frequency[word] += 1
    num_documents = len(sentences)
    for word in inverse_document_frequency:
        inverse_document_frequency[word] = log(num_documents / inverse_document_frequency[word])
    
    # Calculate tf-idf score for each word in each sentence
    tf_idf = np.zeros((len(sentences), len(inverse_document_frequency)))
    for i, sentence in enumerate(sentences):
        sentence_length = len(sentence)
        for word, frequency in term_frequency[i].items():
            tf = frequency / sentence_length
            idf = inverse_document_frequency[word]
            j = list(inverse_document_frequency.keys()).index(word)
            tf_idf[i,j] = tf * idf
    
    return tf_idf

def apply_lsa(tf_idf, sentences, num_topics=50):
    # Perform LSA on the tf-idf matrix
    u, s, vt = np.linalg.svd(tf_idf, full_matrices=False)
    s[num_topics:] = 0
    reduced_tf_idf = np.dot(np.dot(u, np.diag(s)), vt)

    # Normalize the matrix
    norms = np.linalg.norm(reduced_tf_idf, axis=1)
    norms[norms == 0] = 1e-9  # Avoid division by zero
    normalized_tf_idf = reduced_tf_idf / norms[:, None]

    # Calculate the cosine similarity between each pair of sentences
    similarity_matrix = np.dot(normalized_tf_idf, normalized_tf_idf.T)

    # Identify the most important sentences
    sentence_scores = list(enumerate(similarity_matrix.sum(axis=1)))
    sentence_scores = sorted(sentence_scores, key=itemgetter(1), reverse=True)
    important_sentences = [sentences[i] for i, score in sentence_scores[:num_topics]]

    return important_sentences


def generate_summary(book_text):
    # Preprocess the book into individual sentences
    sentences = preprocess_book(book_text)
    
    # Calculate tf-idf scores for each word in each sentence
    tf_idf = calculate_tf_idf(sentences)
    
    # Apply LSA to identify important sentences
    important_sentences = apply_lsa(tf_idf, sentences)
    
    # Concatenate the important sentences to form the summary
    summary = ' '.join([' '.join(sentence) for sentence in important_sentences])
    print(summary)
    return summary

