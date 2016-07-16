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
                 style_dataset=None,
                 subreddit=None):
        self.ngram_len = ngram_len
        self.size_reweight = int(size_reweight)
        self.punctuation_dataset = punctuation_dataset
        self.style_dataset = style_dataset
        self.subreddit = subreddit

    def generate_message(self, prompt):
        def _add_to_back(chain):
            potential_next_words = {}
            for depth in range(1, min(self.ngram_len, len(chain))+1):
                # import pdb; pdb.set_trace()
                try:
                    gram = chain[-depth:]
                except:
                    break

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
                        potential_next_words[word] + \
                        score * (depth ** 2*self.size_reweight)

            # Only consider the top 50 words
            all_words = potential_next_words.keys()
            top_words = \
                sorted(all_words, key=lambda w: -potential_next_words[w])[:50]

            potential_next_words = \
                {word: potential_next_words[word] for word in top_words}

            # Choose the next word proportional to its score
            choice = random.random()
            scores_sum = sum(potential_next_words.values())

            if len(potential_next_words.keys()) == 0:
                return None,0

            for word,score in potential_next_words.items():
                choice -= score*1.0/scores_sum
                if word[1] == '.' or choice <= 0:
                  return word,score

        def _add_to_front(chain):
            potential_next_words = {}
            for depth in range(1, min(self.ngram_len, len(chain))+1):
                gram = chain[-depth:]

                # Turn the gram into a hashable tuple to read from the style graph
                words = " ".join([t[0] for t in gram])
                tags = " ".join([t[1] for t in gram])
                gram_tuple = (words,tags)

                # If this chain of words has never occurred before, continue
                if gram_tuple not in self.rstyle_graph:
                    continue

                # Potential next words. Take the top twenty.
                all_word_scores = self.rstyle_graph[gram_tuple]
                all_words = all_word_scores.keys()
                top_words = \
                    sorted(all_words, key=lambda w: -all_word_scores[w])[:20]
                word_scores = {word: all_word_scores[word] for word in top_words}

                # Use the part of speech tag information to determine the next POS
                if tags not in self.rpunctuation_graph:
                    continue

                all_pos_scores = self.rpunctuation_graph[tags]
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
                        potential_next_words[word] + \
                        score * (depth ** 2*self.size_reweight)

            # Only consider the top 50 words
            all_words = potential_next_words.keys()
            top_words = \
                sorted(all_words, key=lambda w: -potential_next_words[w])[:50]

            potential_next_words = \
                {word: potential_next_words[word] for word in top_words}

            # Choose the next word proportional to its score
            choice = random.random()
            scores_sum = sum(potential_next_words.values())

            if len(potential_next_words.keys()) == 0:
                return None,0

            for word,score in potential_next_words.items():
                choice -= score*1.0/scores_sum
                if word[1] == '.' or choice <= 0:
                  return word,score

        prompt = prompt.lower()
        prompt = \
            "".join([ch for ch in prompt if ch not in "'"])

        words = word_tokenize(prompt)
        chain = pos_tag(words)

        delete_len = 0
        while len(chain) < 30:
            back_word,bscore = _add_to_back(chain)
            front_word,fscore = _add_to_front(chain[::-1])

            if bscore > fscore and chain[-1][1] != '.':
                chain.append(back_word)
            elif chain[0][1] != '.':
                chain = [front_word] + chain
            else:
                break

        return " ".join([word[0] for word in chain])

    def train_punctuation(self):
        # Initialize POS graph
        self.punctuation_graph = {}

        def _add_message_to_punctuation(message):
            score = message[1]
            message = message[0]

            # Remove contractions and potentially other characters
            message = \
                "".join([ch for ch in message if ch not in "'"])

            words = word_tokenize(message)
            tagged_words = pos_tag(words)

            for gram_len in range(1, self.ngram_len+1):
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

                    self.punctuation_graph[tags][next_word] += score

        # Need to turn the text into the right format
        messages = self.extract_messages(self.punctuation_dataset)

        for message in messages:
            _add_message_to_punctuation(message)

    def reverse_train_punctuation(self):
        # Initialize POS graph
        self.rpunctuation_graph = {}

        def _add_message_to_punctuation(message):
            score = message[1]
            message = message[0]

            # Remove contractions and potentially other characters
            message = \
                "".join([ch for ch in message if ch not in "'"])

            words = ['.'] + word_tokenize(message)[::-1]
            tagged_words = pos_tag(words)

            for gram_len in range(1, self.ngram_len+1):
                # The minus one is to ensure that we always have a word
                # right after the gram
                for i in range(len(tagged_words)-gram_len-1):
                    gram = tagged_words[i:i+gram_len]

                    # Turn the gram into a hashable string.
                    tags = " ".join([t[1] for t in gram])

                    # Identify the type of the word that comes after the gram
                    next_word = tagged_words[i+gram_len][1]

                    if tags not in self.rpunctuation_graph:
                        self.rpunctuation_graph[tags] = {}

                    if next_word not in self.rpunctuation_graph[tags]:
                        self.rpunctuation_graph[tags][next_word] = 0

                    self.rpunctuation_graph[tags][next_word] += score

        # Need to turn the text into the right format
        messages = self.extract_messages(self.punctuation_dataset)

        for message in messages:
            _add_message_to_punctuation(message)


    def train_style(self):
        # Initialize POS graph
        self.style_graph = {}

        def _add_message_to_style(message):
            score = message[1]
            message = message[0]

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

                    self.style_graph[gram_tuple][next_word] += score

        # Need to turn the text into the right format
        messages = self.extract_messages(self.style_dataset)

        for message in messages:
            _add_message_to_style(message)

    def reverse_train_style(self):
        # Initialize POS graph
        self.rstyle_graph = {}

        def _add_message_to_style(message):
            score = message[1]
            message = message[0]

            # Remove contractions and potentially other characters
            message = \
                "".join([ch for ch in message if ch not in "'"])

            words = ['.'] + word_tokenize(message)[::-1]
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

                    if gram_tuple not in self.rstyle_graph:
                        self.rstyle_graph[gram_tuple] = {}

                    if next_word not in self.rstyle_graph[gram_tuple]:
                        self.rstyle_graph[gram_tuple][next_word] = 0

                    self.rstyle_graph[gram_tuple][next_word] += score

        # Need to turn the text into the right format
        messages = self.extract_messages(self.style_dataset)

        for message in messages:
            _add_message_to_style(message)

    def extract_messages(self, filename):
        messages = []
        with open(filename) as f:
            for i in range(10000):
                try:
                    message = f.next()
                except:
                    break
                message = json.loads(message)
                messages.append((message['body'].lower(), message['score']))
        return messages

if __name__ == '__main__':
    cmb = ContextAwareMarkovBot(ngram_len=10,
                                punctuation_dataset='AskReddit',
                                style_dataset='AskReddit',
                                subreddit='AskReddit')
    cmb.train_style()
    cmb.train_punctuation()
    cmb.reverse_train_style()
    cmb.reverse_train_punctuation()
    import pdb; pdb.set_trace()
    cmb.generate_message("I wonder")
