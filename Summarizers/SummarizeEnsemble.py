from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import numpy as np
import nltk
import torch
from nltk.cluster.util import cosine_distance

nltk.download('punkt')

def generate_summary(text):
    # Load the Pegasus model and tokenizer
    model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-xsum')
    tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-xsum')

    # Split the input text into chunks of at most 1024 tokens
    text_chunks = [text[i:i + 1024] for i in range(0, len(text), 1024)]

    # Generate the summary output for each text chunk using the Pegasus model
    summary_chunks = []
    for chunk in text_chunks:
        # Tokenize the input text chunk and append the special tokens
        inputs = tokenizer.encode("summarize: " + chunk, return_tensors="pt")

        # Generate the summary output using the Pegasus model
        summary_ids = model.generate(
            input_ids=inputs, 
            max_length=128, 
            num_beams=5, 
            length_penalty=0.8, 
            early_stopping=True
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summary_chunks.append(summary)

    # Combine the two summaries using sentence clustering
    sentences = [nltk.sent_tokenize(summary) for summary in summary_chunks]
    sentences = [sent for sublist in sentences for sent in sublist]

    embedding_dim = 1024
    sentence_embeddings = np.zeros((len(sentences), embedding_dim))
    for i, sent in enumerate(sentences):
        with torch.no_grad():
            inputs = tokenizer.encode(sent, return_tensors='pt', truncation=True, padding='max_length', max_length=1024)
            model_output = model(inputs, decoder_input_ids=inputs, return_dict=True)
            cls_token = model_output.last_hidden_state[:, 0, :]
            sentence_embeddings[i] = cls_token

    sim_mat = np.zeros([len(sentences), len(sentences)])
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim_mat[i][j] = cosine_distance(sentence_embeddings[i].reshape(1, embedding_dim),
                                                 sentence_embeddings[j].reshape(1, embedding_dim))

    import networkx as nx

    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)

    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    top_sentences = [sent for _, sent in ranked_sentences[:5]]

    summary = ' '.join(top_sentences)
    print(summary)
    return summary

