import string
import json
import random

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

    def generate_message(self, prompt):
        prompt = \
            "".join([ch for ch in prompt if ch not in "'"])

        words = word_tokenize(prompt)
        chain = pos_tag(words)

        while chain[-1][1] != '.':
            potential_next_words = {}
            for depth in range(1, self.ngram_len):
                gram = chain[-depth:]

                # Turn the gram into a hashable tuple to read from the style graph
                words = " ".join([t[0] for t in gram])
                tags = " ".join([t[1] for t in gram])
                gram_tuple = (words,tags)

                # If this chain of words has never occurred before, continue
                if gram_tuple not in self.style_graph:
                    continue

                # Potential next words. Take the top twenty.
                all_word_scores = self.style_graph[gram_tuple]
                all_words = all_word_scores.keys()
                top_words = \
                    sorted(all_words, key=lambda w: -all_word_scores[w])[:20]
                word_scores = {word: all_word_scores[word] for word in top_words}

                # Use the part of speech tag information to determine the next POS 
                if tags not in self.punctuation_graph:
                    continue

                all_pos_scores = self.punctuation_graph[tags]
                all_pos = all_pos_scores.keys()
                top_pos = sorted(all_pos, key=lambda p: -all_pos_scores[p])
                pos_scores = {pos: all_pos_scores[pos] for pos in top_pos}

                word_scores = \
                    {word: 1.0*(word_scores[word]+pos_scores[word[1]])/2 
                        for word in top_words if word[1] in pos_scores}

                # Update master word list
                for word,score in word_scores.items():
                    if word not in potential_next_words:
                        potential_next_words[word] = 0

                    potential_next_words[word] = \
                        max(potential_next_words[word],
                            score * depth ** self.size_reweight)

            # Only consider the top 50 words
            all_words = potential_next_words.keys()
            top_words = \
                sorted(all_words, key=lambda w: -potential_next_words[w])[:100]
            potential_next_words = \
                {word: potential_next_words[word] for word in top_words}
        
            import pdb; pdb.set_trace()

            # Choose the next word proportional to its score
            scores_sum = sum(potential_next_words.values())
            choice = random.random()

            for word,score in potential_next_words.items():
                choice -= score*1.0/scores_sum 
                if choice <= 0:
                    chain.append(word)
                    break
                    
        return " ".join([word[0] for word in chain]) 

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
    cmb.generate_message("I think")
    import pdb; pdb.set_trace()
