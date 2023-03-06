# import torch
# from transformers import PegasusForConditionalGeneration, PegasusTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
# def PegasusFineTune():
#     # Load the Pegasus model and tokenizer
#     model_name = 'google/pegasus-xsum'
#     model = PegasusForConditionalGeneration.from_pretrained(model_name)
#     tokenizer = PegasusTokenizer.from_pretrained(model_name)

#     # Load the book summary dataset
#     dataset_path = 'Data/booksummaries.txt'
#     with open(dataset_path, 'r') as f:
#         summaries = f.readlines()

#     # Tokenize the summaries and generate the corresponding input sequences
#     input_sequences = []
#     count = 0 
#     for summary in summaries:
#         input_sequence = tokenizer.encode(summary.strip(), truncation=True, padding='max_length', max_length=512)
#         input_sequences.append(input_sequence)
#         count = count + 1
#         if count == 1000:
#             break

#     # Define the training arguments
#     training_args = TrainingArguments(
#         output_dir='./results',          # output directory
#         num_train_epochs=1,              # total number of training epochs
#         per_device_train_batch_size=2,   # batch size per device during training
#         per_device_eval_batch_size=2,    # batch size for evaluation
#         warmup_steps=100,                # number of warmup steps for learning rate scheduler
#         learning_rate=1e-5,              # learning rate
#         weight_decay=0.01,               # strength of weight decay
#         logging_dir='./logs',            # directory for storing logs
#         logging_steps=10,
#         save_steps=50,                   # number of steps to save the model
#         save_total_limit=1               # limit on the total number of checkpoints to save
#     )

#     # Define the data collator
#     data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

#     # Instantiate a Trainer
#     trainer = Trainer(
#         model=model,                       # the instantiated 🤗 Transformers model to be trained
#         args=training_args,                # training arguments, defined above
#         train_dataset=input_sequences,     # training dataset
#         data_collator=data_collator
#     )

#     # Fine-tune the model
#     trainer.train()

#     # Save the fine-tuned model
#     model.save_pretrained('./fine_tuned_pegasus')
#     tokenizer.save_pretrained('./fine_tuned_pegasus')



import torch
from transformers import BartForConditionalGeneration, BartTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from rouge_score import rouge_scorer

def generate_summary(text):
    # Load the BART model and tokenizer
    model_name = "facebook/bart-large-cnn"
    model = BartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = BartTokenizer.from_pretrained(model_name)

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
        print(summary)
    

    # Join the summary chunks into a single summary
    summary = ' '.join(summary_chunks)
    # Calculate ROUGE score between generated summary and original text
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(summary, text)

    # Print ROUGE scores
    print('ROUGE-1:', scores['rouge1'].fmeasure)
    print('ROUGE-2:', scores['rouge2'].fmeasure)
    print('ROUGE-L:', scores['rougeL'].fmeasure)

    return summary




# PegasusFineTune()



