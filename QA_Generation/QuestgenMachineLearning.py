import spacy
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

nlp = spacy.load("en_core_web_sm")
qa_generator = pipeline("question-answering", model="valhalla/t5-base-qa-qg-hl")
qa_model = AutoModelForSeq2SeqLM.from_pretrained("valhalla/t5-base-qa-qg-hl")
qa_tokenizer = AutoTokenizer.from_pretrained("valhalla/t5-base-qa-qg-hl")

def generate_qa_pairs(text):
    doc = nlp(text)
    qa_pairs = []

    # Rule-based approach to generate simple fact-based questions
    for sent in doc.sents:
        if sent.text.strip() and sent.text[-1] == '.':
            if sent.start != sent.end:
                sentence = sent.text.strip()
                if '?' not in sentence:
                    question = 'What is ' + sentence.split(' is ')[0].strip() + '?'
                    answer = sentence.split(' is ')[1].strip()
                    qa_pairs.append({'question': question, 'answer': answer})

    # Fine-tuned machine learning-based approach to generate more complex questions
    for sent in doc.sents:
        if sent.text.strip() and sent.start != sent.end:
            inputs = qa_tokenizer.encode("generate questions: " + sent.text.strip(), return_tensors="pt")
            outputs = qa_model.generate(inputs, max_length=128, num_beams=4, early_stopping=True)
            question = qa_tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = qa_generator(question=question, context=text)['answer']
            qa_pairs.append({'question': question, 'answer': answer})

    return qa_pairs
