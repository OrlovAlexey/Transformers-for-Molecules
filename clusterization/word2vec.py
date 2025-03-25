from collections import Counter
from gensim.models import Word2Vec
import time
import numpy as np

class Word2VecWrapper:
    def __init__(self, vocab_size, text):
        self.vocab = {} 
        self.inverse_vocab = {}
        self.vocab_size = vocab_size
        self.text = text
        self.unique_symbols_list = []
        self.self.text_splitted_by_space = None
        self.model = None

    def preprocess(self):
        text_splitted_by_dot = self.text.split(" . ")
        self.text_splitted_by_space = [i.split(" ") for i in text_splitted_by_dot]
        self.text_splitted_by_space = list(filter(lambda a: a != "", self.text_splitted_by_space))
        for mol in self.text_splitted_by_space:
            for symbol in mol:
                if symbol == "." or symbol == "" or symbol == " ":
                    continue
                self.unique_symbols_list.append(symbol)
        
        unique_symbols = set(self.unique_symbols_list)
        
        self.vocab = {i: symbol for i, symbol in enumerate(unique_symbols)}
        self.inverse_vocab = {symbol: i for i, symbol in self.vocab.items()}
        
    def remove_low_frequency_tokens(self, threshold=5, constant_value=-1):
        dict_of_freq = Counter(self.unique_symbols_list)
        sorted_dict = dict(sorted(dict_of_freq.items(), key=lambda item: item[1]))
        
        updated_inverse_vocab = {
            k: (constant_value if sorted_dict[k] < threshold else v)
            for k, v in self.inverse_vocab.items()
        }
        updated_inverse_vocab = dict(sorted(updated_inverse_vocab.items(), key=lambda item: item[1]))
        self.inverse_vocab = updated_inverse_vocab
        
    def get_tokens_ids(self):
        self.text_splitted_by_space = [[word for word in sublist if word != ""] for sublist in self.text_splitted_by_space]
        token_ids = [[self.inverse_vocab[symbol] for symbol in mol] for mol in self.text_splitted_by_space]
        
        print(len(set(list(self.inverse_vocab.values()))))
        return token_ids
    
    def train(self):
        token_ids = self.get_tokens_ids()
        sentences = [
            [str(token) for token in sentence]  # Оставляем все токены, включая UNK (-1)
            for sentence in token_ids
        ]
        
        st = time.time()
        
        self.model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=1)

        print(time.time()-st)

    def get_embedding(self, token):
        if token in self.model.wv:
            return self.model.wv[token]
        else:
            return np.zeros(self.model.vector_size)  # Если токен не найден, возвращаем нулевой вектор
