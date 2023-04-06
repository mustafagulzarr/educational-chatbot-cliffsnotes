#below is the code to fine tune the model on a particular dataset to align the model more towards the book summarization use case. In this project, the model could not be fine-tuned due to hardware restrictions as a strong GPU is needed to compute the fine-tuned model in reasonable.



# import torch
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
# def BartFineTune():
#     # Load the BART model and tokenizer
# tokenizer = AutoTokenizer.from_pretrained("slauw87/bart_summarisation")

# model = AutoModelForSeq2SeqLM.from_pretrained("slauw87/bart_summarisation")

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
#         model=model,                 
#         args=training_args,               
#         train_dataset=input_sequences,     
#         data_collator=data_collator
#     )

#     # Fine-tune the model
#     trainer.train()

#     # Save the fine-tuned model
    # model.save_pretrained('./fine_tuned_bart')
    # tokenizer.save_pretrained('./fine_tuned_bart')



from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score

# Load the BART model and tokenizer

tokenizer = AutoTokenizer.from_pretrained("slauw87/bart_summarisation")

model = AutoModelForSeq2SeqLM.from_pretrained("slauw87/bart_summarisation")

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



