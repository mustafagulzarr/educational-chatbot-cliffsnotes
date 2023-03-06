from transformers import pipeline, AutoTokenizer
import re
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AdamW, get_linear_schedule_with_warmup
import torch
from torch.utils.data import DataLoader
import json
import numpy as np

def QA_fine_tune():
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the pre-trained model and tokenizer
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)

    # Set the hyperparameters
    batch_size = 16
    learning_rate = 2e-5
    num_epochs = 3
    warmup_steps = 100
    max_seq_length = 384

    # Load the data
    # Here, we assume that you have a file named "train.json" that contains your training data
    # Each example in the file should be a JSON object with "context", "question", and "answer" fields
    def load_data(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
        examples = []
        for example in data:
            context = example["context"]
            question = example["question"]
            answer = example["answer"]
            examples.append((context, question, answer))
        return examples

    train_examples = load_data("train.json")

    # Convert the examples to features
    def convert_examples_to_features(examples, tokenizer, max_seq_length):
        features = []
        for example in examples:
            context = example[0]
            question = example[1]
            answer = example[2]
            encoding = tokenizer.encode_plus(question, context, max_length=max_seq_length, padding="max_length", truncation=True)
            input_ids = encoding["input_ids"]
            attention_mask = encoding["attention_mask"]
            start_positions = None
            end_positions = None
            if answer in context:
                start_index = context.index(answer)
                end_index = start_index + len(answer) - 1
                start_positions = np.zeros(len(input_ids), dtype=np.int32)
                end_positions = np.zeros(len(input_ids), dtype=np.int32)
                for i, (input_id, attention_mask_bit) in enumerate(zip(input_ids, attention_mask)):
                    if i == 0:
                        continue
                    if attention_mask_bit == 0:
                        continue
                    if input_id == tokenizer.sep_token_id:
                        break
                    start_positions[i] = int(start_index <= i)
                    end_positions[i] = int(end_index >= i)
            features.append((input_ids, attention_mask, start_positions, end_positions))
        return features

    train_features = convert_examples_to_features(train_examples, tokenizer, max_seq_length)

    # Define the dataset and data loader
    class QADataset(torch.utils.data.Dataset):
        def __init__(self, features):
            self.features = features
        
        def __len__(self):
            return len(self.features)
        
        def __getitem__(self, index):
            input_ids = torch.tensor(self.features[index][0], dtype=torch.long)
            attention_mask = torch.tensor(self.features[index][1], dtype=torch.long)
            start_positions = torch.tensor(self.features[index][2], dtype=torch.float)
            end_positions = torch.tensor(self.features[index][3], dtype=torch.float)
            return input_ids, attention_mask, start_positions, end_positions

    train_dataset = QADataset(train_features)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Set the optimizer and scheduler
    # Set the optimizer and scheduler
    num_training_steps = len(train_loader) * num_epochs
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)

    # Define the loss function
    def loss_fn(start_logits, end_logits, start_positions, end_positions):
        start_loss = torch.nn.functional.binary_cross_entropy_with_logits(start_logits, start_positions)
        end_loss = torch.nn.functional.binary_cross_entropy_with_logits(end_logits, end_positions)
        total_loss = (start_loss + end_loss) / 2
        return total_loss

    # Train the model
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in train_loader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask, start_positions, end_positions = batch
            outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
            loss = loss_fn(outputs.start_logits, outputs.end_logits, start_positions, end_positions)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1} loss: {epoch_loss / len(train_loader):.4f}")

    # Save the trained model
    model.save_pretrained("qa_model")
    tokenizer.save_pretrained("qa_model") 
def generate_qa_pairs(text):
    # clean up text
    text = re.sub(r"\n", " ", text)
    
    # initialize pipeline
    qa_generator = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

    # generate question-answer pairs
    pairs = []
    for sentence in text.split("."):
        sentence = sentence.strip()
        if sentence:
            example = {
                "question": f"What is {sentence.split(' is ')[0]}?",
                "context": sentence
            }
            result = qa_generator(example)
            if result["score"] > 0.5:
                pairs.append((example["question"], result["answer"]))
    return pairs

# example usage
# file = open('book.txt','r')
# text = file.read()
# pairs = generate_qa_pairs(text)
# print(pairs)
QA_fine_tune()