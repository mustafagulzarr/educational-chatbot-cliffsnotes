# from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AdamW, get_linear_schedule_with_warmup
# import torch
# from torch.utils.data import DataLoader
# import numpy as np
# from datasets import load_dataset
# from transformers import Trainer, TrainingArguments


# def QA_fine_tune():
#     # Set the device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Load the pre-trained model and tokenizer
#     model_name = "bert-base-uncased"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)

#     # Set the hyperparameters
#     batch_size = 16
#     learning_rate = 2e-5
#     num_epochs = 3
#     warmup_steps = 100
#     max_seq_length = 384

#     # Load the data
#     dataset = load_dataset('narrativeqa', 'simplified_qa', split='train')

#     def load_data(dataset):
#         examples = []
#         for example in dataset:
#             context = example["document"]["text"]
#             question = example["question"]["text"]
#             answer = example["answers"][0]["text"]
#             examples.append((context, question, answer))
#         return examples

#     train_examples = load_data(dataset)

#     # Convert the examples to features
#     def convert_examples_to_features(examples, tokenizer, max_seq_length):
#         features = []
#         for example in examples:
#             context = example[0]
#             question = example[1]
#             answer = example[2]
#             encoding = tokenizer.encode_plus(question, context, max_length=max_seq_length, padding="max_length", truncation=True)
#             input_ids = encoding["input_ids"]
#             attention_mask = encoding["attention_mask"]
#             start_positions = None
#             end_positions = None
#             if answer in context:
#                 start_index = context.index(answer)
#                 end_index = start_index + len(answer) - 1
#                 start_positions = np.zeros(len(input_ids), dtype=np.int32)
#                 end_positions = np.zeros(len(input_ids), dtype=np.int32)
#                 for i, (input_id, attention_mask_bit) in enumerate(zip(input_ids, attention_mask)):
#                     if i == 0:
#                         continue
#                     if attention_mask_bit == 0:
#                         continue
#                     if input_id == tokenizer.sep_token_id:
#                         break
#                     start_positions[i] = int(start_index <= i)
#                     end_positions[i] = int(end_index >= i)
#             features.append((input_ids, attention_mask, start_positions, end_positions))
#         return features

#     train_features = convert_examples_to_features(train_examples, tokenizer, max_seq_length)

#     # Define the dataset and data loader
#     class QADataset(torch.utils.data.Dataset):
#         def __init__(self, features):
#             self.features = features
        
#         def __len__(self):
#             return len(self.features)
        
#         def __getitem__(self, index):
#             input_ids = torch.tensor(self.features[index][0], dtype=torch.long)
#             attention_mask = torch.tensor(self.features[index][1], dtype=torch.long)
#             start_positions = torch.tensor(self.features[index][2], dtype=torch.float)
#             end_positions = torch.tensor(self.features[index][3], dtype=torch.float)
#             return input_ids, attention_mask, start_positions, end_positions

#     train_dataset = QADataset(train_features)
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

#     # Set the optimizer and scheduler
#     num_training_steps = len(train_loader) * num_epochs
#     optimizer = AdamW(model.parameters(), lr=learning_rate)
#     scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)

#     # Set up the Trainer object
#     training_args = TrainingArguments(
#         output_dir='./results',  # output directory
#         num_train_epochs=num_epochs,  # total number of training epochs
#         per_device_train_batch_size=batch_size,  # batch size per device during training
#         warmup_steps=warmup_steps,  # number of warmup steps for learning rate scheduler
#         weight_decay=0.01,  # strength of weight decay
#         logging_dir='./logs',  # directory for storing logs
#         logging_steps=10,
#         save_total_limit=3, # number of total save checkpoints
#         save_steps=1000 # after every 1000 steps, save checkpoint
#     )

#     def compute_metrics(pred):
#         predictions, label_ids = pred
#         preds = np.argmax(predictions, axis=1)
#         accuracy = (preds == label_ids).mean()
#         return {'accuracy': accuracy}

#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         data_collator=lambda data: {
#             'input_ids': torch.stack([f[0] for f in data]),
#             'attention_mask': torch.stack([f[1] for f in data]),
#             'start_positions': torch.stack([f[2] for f in data]),
#             'end_positions': torch.stack([f[3] for f in data])
#         },
#         compute_metrics=compute_metrics,
#         optimizer=optimizer,
#         scheduler=scheduler
#     )

#     # Start training
#     trainer.train()


# Define a function to generate question-answer pairs


import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

tokenizer = AutoTokenizer.from_pretrained("deepset/bert-large-uncased-whole-word-masking-squad2")

model = AutoModelForQuestionAnswering.from_pretrained("deepset/bert-large-uncased-whole-word-masking-squad2")

def generate_answer(question, context):
    # Split the input sequence into smaller segments that are less than 512 tokens each
    max_length = 512 - len(tokenizer.encode(question, add_special_tokens=False)) - 3
    segments = [context[i:i+max_length] for i in range(0, len(context), max_length)]

    # Generate an answer for each segment using the BERT model
    answers = []
    for segment in segments:
        inputs = tokenizer.encode_plus(question, segment, add_special_tokens=True, return_tensors="pt")
        answer_start_scores, answer_end_scores = model(**inputs).values()
        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))
        if answer != '[CLS]':
            if answer != '':
                score = torch.max(torch.softmax(answer_start_scores, dim=1)) * torch.max(torch.softmax(answer_end_scores, dim=1))
                answers.append((answer, score.item()))

    # Concatenate the answers to get the final answer for the entire input sequence
    answers = sorted(answers, key=lambda x: x[1], reverse=True)

    if len(answers) == 0 or answers[0][0]=='':
        return 'Sorry I do not know the answer, kindly rephrase your question'
    return answers[0][0]








