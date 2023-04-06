import nltk
import numpy as np
from scipy.spatial.distance import cosine as cosine_distance
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

def generate_summary(text, max_sentences=200):
    # Load the BART model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("slauw87/bart_summarisation")

    model = AutoModelForSeq2SeqLM.from_pretrained("slauw87/bart_summarisation") 

    # Load the sentence-transformers model for sentence embeddings
    embedder = SentenceTransformer('sentence-transformers/paraphrase-mpnet-base-v2')

    # Split the input text into chunks of at most 1024 tokens
    text_chunks = [text[i:i + 1024] for i in range(0, len(text), 1024)]

    # Generate the summary output for each text chunk using the BART model
    summary_chunks = []
    for chunk in text_chunks:
        inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)
        summary_ids = model.generate(inputs['input_ids'], 
                                      max_length=256, 
                                      num_beams=10, 
                                      length_penalty=0.8, 
                                      early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summary_chunks.append(summary)

    # Combine the summaries using Hierarchical Agglomerative Clustering
    sentences = [nltk.sent_tokenize(summary) for summary in summary_chunks]
    sentences = [sent for sublist in sentences for sent in sublist]

    if len(sentences) == 0:
        return ""

    # Compute the sentence embeddings
    sentence_embeddings = embedder.encode(sentences)

    # Compute the similarity matrix
    sim_mat = np.array([[cosine_distance(sentence_embeddings[i], sentence_embeddings[j])
                         for j in range(len(sentences))] for i in range(len(sentences))])
    if sim_mat.shape != (len(sentences), len(sentences)):
        raise ValueError("Expected a square 2D array for sim_mat, but got an array of shape %s" % (sim_mat.shape,))

    # Apply Hierarchical Agglomerative Clustering to group the most similar sentences
    hac = AgglomerativeClustering(n_clusters=None, distance_threshold=1.0, linkage='complete')
    hac.fit(sim_mat)
    clusters = {}
    for i, cluster_label in enumerate(hac.labels_):
        if cluster_label not in clusters:
            clusters[cluster_label] = []
        clusters[cluster_label].append(i)
    cluster_centers = [np.mean(sentence_embeddings[cluster], axis=0) for cluster in clusters.values()]

    # Select the most representative sentence from each cluster
    ranked_sentences = []
    for i, cluster in clusters.items():
        cluster_similarities = [1 - cosine_distance(sentence_embeddings[s], cluster_centers[i])
                               for s in cluster]
        most_similar_sentence = cluster[np.argmax(cluster_similarities)]
        ranked_sentences.append((most_similar_sentence, sentences[most_similar_sentence]))

    # Sort the selected sentences by their position in the original text
    ranked_sentences = sorted(ranked_sentences, key=lambda x: x[0])

    # Join the selected sentences to form the summary
    summary = ' '.join([sent for _, sent in ranked_sentences][:max_sentences])


    return summary