from nltk.tokenize import sent_tokenize
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation

nltk.download('punkt')
nltk.download('stopwords')

class TextRank:
    def __init__(self, text, lang='english', summarize_ratio=0.2):
        self.sentences = sent_tokenize(text)
        self.words = word_tokenize(text.lower())
        self.words = [word for word in self.words if word not in stopwords.words(lang) and word not in punctuation]
        self.word_count = len(self.words)
        self.summarize_count = int(self.word_count * summarize_ratio)
        self.word_weights = {}
        self.build_word_weights()
        self.sentence_scores = {}
        self.build_sentence_scores()

    def build_word_weights(self):
        for word in self.words:
            if word not in self.word_weights:
                self.word_weights[word] = self.words.count(word) / self.word_count

    def build_sentence_scores(self):
        for i, sentence in enumerate(self.sentences):
            sentence_words = [word.lower() for word in word_tokenize(sentence) if word.lower() not in stopwords.words('english') and word not in punctuation]
            sentence_word_count = len(sentence_words)
            sentence_weight = sum([self.word_weights[word] for word in sentence_words])
            self.sentence_scores[i] = sentence_weight / sentence_word_count if sentence_word_count > 0 else 0

    def get_top_sentences(self):
        top_sentence_indices = sorted(self.sentence_scores, key=self.sentence_scores.get, reverse=True)[:self.summarize_count]
        top_sentence_indices.sort()
        return [self.sentences[i] for i in top_sentence_indices]

    def get_summary(self):
        top_sentences = self.get_top_sentences()
        summary = ' '.join(top_sentences)
        return summary
