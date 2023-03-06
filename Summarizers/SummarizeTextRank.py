import nltk
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import re

def read_article(text):
    article = text.split(". ")
    sentences = []
    for sentence in article:
        sentences.append(re.findall(r'\b\w+\b', sentence))
    sentences.pop()
    return sentences

# def sentence_similarity(sent1, sent2, stopwords=None):
#     if stopwords is None:
#         stopwords = []
#     sent1 = [w.lower() for w in sent1]
#     sent2 = [w.lower() for w in sent2]
#     all_words = list(set(sent1 + sent2))
#     vector1 = [0] * len(all_words)
#     vector2 = [0] * len(all_words)
#     for w in sent1:
#         if w in stopwords:
#             continue
#         vector1[all_words.index(w)] += 1
#     for w in sent2:
#         if w in stopwords:
#             continue
#         vector2[all_words.index(w)] += 1
#     return cosine_similarity([vector1], [vector2])[0][0]
def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
    
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
    
    # Identify and replace personal pronouns with a standard token
    tokens1 = nltk.word_tokenize(' '.join(sent1))  # Join the list of tokens into a string
    tokens2 = nltk.word_tokenize(' '.join(sent2))  # Join the list of tokens into a string
    pos_tags1 = nltk.pos_tag(tokens1)
    pos_tags2 = nltk.pos_tag(tokens2)
    chunked1 = nltk.ne_chunk(pos_tags1, binary=False)
    chunked2 = nltk.ne_chunk(pos_tags2, binary=False)
    pronoun_token = "PRONOUN"
    
    for i in range(len(chunked1)):
        if type(chunked1[i]) == nltk.tree.Tree and chunked1[i].label() == 'PERSON':
            for j in range(len(chunked1[i])):
                if chunked1[i][j][1] == 'PRP':
                    sent1 = sent1.replace(chunked1[i][j][0], pronoun_token)
                    
    for i in range(len(chunked2)):
        if type(chunked2[i]) == nltk.tree.Tree and chunked2[i].label() == 'PERSON':
            for j in range(len(chunked2[i])):
                if chunked2[i][j][1] == 'PRP':
                    sent2 = sent2.replace(chunked2[i][j][0], pronoun_token)

    all_words = list(set(sent1 + sent2))
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
        
    return cosine_similarity([vector1], [vector2])[0][0]

def build_similarity_matrix(sentences, stop_words):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:
                continue
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)
    return similarity_matrix

def generate_summary(text, top_n=50):
    stop_words = stopwords.words('english')
    summarize_text = []

    # Step 1 - Read text and tokenize
    sentences = read_article(text)

    # Step 2 - Generate Similary Martix across sentences
    sentence_similarity_matrix = build_similarity_matrix(sentences, stop_words)

    # Step 3 - Rank sentences in similarity matrix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph)

    # Step 4 - Sort the rank and pick top sentences
    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
    for i in range(top_n):
        summarize_text.append(" ".join(ranked_sentences[i][1]))

    # Step 5 - Output the summarized text
    summary = ". ".join(summarize_text)
    with open("output.txt", "w") as f:
        # Write a line to the file
        f.write(summary)
    return summary
