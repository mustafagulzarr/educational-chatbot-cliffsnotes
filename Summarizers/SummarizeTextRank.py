# Import necessary modules and download required resources
from nltk.tokenize import sent_tokenize
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation

nltk.download('punkt')
nltk.download('stopwords')

# Class that uses TextRank algorithm to summarize text
class TextRank:
    # Initialize the object and preprocess the text
    def __init__(self, text, lang='english', summarize_ratio=0.2):
        # Tokenize text into sentences and words
        self.sentences = sent_tokenize(text)
        self.words = word_tokenize(text.lower())
        # Remove stop words and punctuation from words
        self.words = [word for word in self.words if word not in stopwords.words(lang) and word not in punctuation]
        # Calculate word count and summarize count based on word count and summarize ratio
        self.word_count = len(self.words)
        self.summarize_count = int(self.word_count * summarize_ratio)
        # Create dictionary to store word weights
        self.word_weights = {}
        self.build_word_weights()
        # Create dictionary to store sentence scores
        self.sentence_scores = {}
        self.build_sentence_scores()

    # Calculate word weights based on word frequency
    def build_word_weights(self):
        for word in self.words:
            if word not in self.word_weights:
                self.word_weights[word] = self.words.count(word) / self.word_count

    # Calculate sentence scores based on word weights
    def build_sentence_scores(self):
        for i, sentence in enumerate(self.sentences):
            # Tokenize sentence into words and remove stop words and punctuation
            sentence_words = [word.lower() for word in word_tokenize(sentence) if word.lower() not in stopwords.words('english') and word not in punctuation]
            sentence_word_count = len(sentence_words)
            # Calculate sentence weight as the sum of the weights of its words
            sentence_weight = sum([self.word_weights[word] for word in sentence_words])
            # Normalize sentence weight by sentence word count to prevent bias towards longer sentences
            self.sentence_scores[i] = sentence_weight / sentence_word_count if sentence_word_count > 0 else 0

    # Get top sentences based on their scores
    def get_top_sentences(self):
        # Sort sentences by score and get top 'summarize_count' sentences
        top_sentence_indices = sorted(self.sentence_scores, key=self.sentence_scores.get, reverse=True)[:self.summarize_count]
        # Sort indices in ascending order to preserve the original order of sentences in the text
        top_sentence_indices.sort()
        return [self.sentences[i] for i in top_sentence_indices]

    # Get summary of text
    def get_summary(self):
        # Get top sentences and join them into a summary string
        top_sentences = self.get_top_sentences()
        summary = ' '.join(top_sentences)
        return summary
