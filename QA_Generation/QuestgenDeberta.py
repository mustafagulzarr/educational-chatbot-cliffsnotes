import concurrent.futures
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

tokenizer = AutoTokenizer.from_pretrained("deepset/deberta-v3-large-squad2")

model = AutoModelForQuestionAnswering.from_pretrained("deepset/deberta-v3-large-squad2")

# Define a function that generates an answer for a single segment
def generate_answer_for_segment(question, segment):
    inputs = tokenizer.encode_plus(question, segment, add_special_tokens=True, return_tensors="pt")
    answer_start_scores, answer_end_scores = model(**inputs).values()
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))
    if answer != '[CLS]':
        if answer != '':
            score = torch.max(torch.softmax(answer_start_scores, dim=1)) * torch.max(torch.softmax(answer_end_scores, dim=1))
            return (answer, score.item())
    return None

def generate_answer(question, context):
    # Split the input sequence into smaller segments that are less than 512 tokens each
    max_length = 512 - len(tokenizer.encode(question, add_special_tokens=False)) - 3
    segments = [context[i:i+max_length] for i in range(0, len(context), max_length)]

    # Generate an answer for each segment using multiple threads
    answers = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_segment = {executor.submit(generate_answer_for_segment, question, segment): segment for segment in segments}
        for future in concurrent.futures.as_completed(future_to_segment):
            result = future.result()
            if result:
                answers.append(result)

    # Concatenate the answers to get the final answer for the entire input sequence
    answers = sorted(answers, key=lambda x: x[1], reverse=True)
    if len(answers) == 0 or answers[0][0] == '':
        return 'Sorry I do not know the answer, kindly rephrase your question'
    return answers[0][0]
