import string
import json

from nltk import word_tokenize
from nltk import pos_tag

class ContextAwareMarkovBot():
    def __init__(self, 
                 ngram_len=5, 
                 size_reweight=True,
                 punctuation_dataset=None,
                 style_dataset=None):
        self.ngram_len = ngram_len
        self.size_reweight = int(size_reweight)
        self.punctuation_dataset = punctuation_dataset
        self.style_dataset = style_dataset 

    def train_punctuation(self):
        # Initialize POS graph
        self.punctuation_graph = {} 

        def _add_message_to_punctuation(message):
            # Remove contractions and potentially other characters
            message = \
                "".join([ch for ch in message if ch not in "'"])	

            words = word_tokenize(message)
            tagged_words = pos_tag(words)

            for gram_len in range(1, self.ngram_len):
                # The minus one is to ensure that we always have a word
                # right after the gram
                for i in range(len(tagged_words)-gram_len-1):
                    gram = tagged_words[i:i+gram_len]
                    
                    # Turn the gram into a hashable string.
                    tags = " ".join([t[1] for t in gram])

                    # Identify the type of the word that comes after the gram
                    next_word = tagged_words[i+gram_len][1]

                    if tags not in self.punctuation_graph:
                        self.punctuation_graph[tags] = {}

                    if next_word not in self.punctuation_graph[tags]:
                        self.punctuation_graph[tags][next_word] = 0

                    self.punctuation_graph[tags][next_word] += 1
                    
        # Need to turn the text into the right format
        messages = self.extract_messages(self.punctuation_dataset)

        for message in messages:
            _add_message_to_punctuation(message)

    def train_style(self):
        # Initialize POS graph
        self.style_graph = {} 

        def _add_message_to_style(message):
            # Remove contractions and potentially other characters
            message = \
                "".join([ch for ch in message if ch not in "'"])	

            words = word_tokenize(message)
            tagged_words = pos_tag(words)

            for gram_len in range(1, self.ngram_len):
                # The minus one is to ensure that we always have a word
                # right after the gram
                for i in range(len(tagged_words)-gram_len-1):
                    gram = tagged_words[i:i+gram_len]
                    
                    # Turn the gram into a hashable tuple.
                    words = " ".join([t[0] for t in gram])
                    tags = " ".join([t[1] for t in gram])
                    gram_tuple = (words,tags)

                    # Identify the the word that comes after the gram
                    next_word = tagged_words[i+gram_len]

                    if gram_tuple not in self.style_graph:
                        self.style_graph[gram_tuple] = {}

                    if next_word not in self.style_graph[gram_tuple]:
                        self.style_graph[gram_tuple][next_word] = 0

                    self.style_graph[gram_tuple][next_word] += 1
                    
        # Need to turn the text into the right format
        messages = self.extract_messages(self.style_dataset)

        for message in messages:
            _add_message_to_style(message)

    def extract_messages(self, filename):
        messages = []
        with open(filename) as f:
            for i in range(10000):
                message = json.loads(f.next())
                messages.append(message['body'])
        return messages

if __name__ == '__main__':
    cmb = ContextAwareMarkovBot(punctuation_dataset='RC_2015-01', 
                                style_dataset='RC_2015-01')
    cmb.train_style()
    cmb.train_punctuation()
    import pdb; pdb.set_trace()
