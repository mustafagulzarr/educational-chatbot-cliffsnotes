import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

def generate_summary(text):
    # Load the T5 model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained('t5-base')
    tokenizer = T5Tokenizer.from_pretrained('t5-base')

    # Tokenize the input text and append the special tokens
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt")

    # Generate the summary output using the T5 model
    summary_ids = model.generate(inputs, max_length=150, num_beams=2, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary