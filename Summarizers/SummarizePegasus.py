from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# Load the BART model and tokenizer

tokenizer = AutoTokenizer.from_pretrained("google/bigbird-pegasus-large-arxiv")

model = AutoModelForSeq2SeqLM.from_pretrained("google/bigbird-pegasus-large-arxiv")

def generate_summary(text):


    # Split the input text into chunks of at most 1024 tokens (BART has a maximum input length of 1024 tokens)
    text_chunks = [text[i:i + 1024] for i in range(0, len(text), 1024)]

    # Generate the summary output for each text chunk using the BART model
    summary_chunks = []
    for chunk in text_chunks:
        # Tokenize the input text chunk and append the special tokens
        inputs = tokenizer.encode(chunk, return_tensors="pt")

        # Generate the summary output using the BART model with adjusted parameters
        summary_ids = model.generate(inputs, 
                                      max_length=256, 
                                      num_beams=10, 
                                      length_penalty=0.8, 
                                      early_stopping=True)

        # Decode the summary and append to the list of summary chunks
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summary_chunks.append(summary)
    

    # Join the summary chunks into a single summary
    summary = ' '.join(summary_chunks)

    return summary
